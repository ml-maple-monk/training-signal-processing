"""
Phase 1: stream unified_document_texts, compute MinHash bands,
bulk-COPY into lsh_candidate_bands.

Parallel mode: splits the doc_id range into cfg.num_workers chunks.
The last chunk is open-ended (no upper bound) so docs ingested while
the pipeline is running are naturally picked up by the last worker.
Each worker opens its own DB connection and writes independently.

Incremental mode (default): each batch filters with NOT EXISTS against
lsh_candidate_bands, so only docs not yet banded are processed. This is
correct regardless of doc_id ordering — unlike a MAX(doc_id) watermark,
which silently skips unprocessed docs when coverage is non-contiguous
(e.g. high-id rows ingested while low-id rows are still pending). Safe to
re-run as the source table grows and safe to restart after a crash.

Memory peak per worker: O(batch_size × num_permutations × 4 bytes)
e.g. 10K docs × 128 perms × 4B ≈ 5 MB per worker.
"""
from __future__ import annotations

import hashlib
import io
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import psycopg2
from datasketch import MinHash

from .config import NearDedupConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure helpers (module-level so ProcessPoolExecutor can pickle them)
# ---------------------------------------------------------------------------


def _word_trigrams(text: str, max_words: int) -> set[str]:
    words = text.split()
    if len(words) < 3:
        return set()
    if len(words) > max_words:
        words = words[:max_words]
    return {f"{words[i]} {words[i + 1]} {words[i + 2]}" for i in range(len(words) - 2)}


def _minhash_bands(text: str, cfg: NearDedupConfig) -> list[tuple[int, int]]:
    """Return [(band_index, band_hash), ...] for all bands, or [] if text is too short."""
    shingles = _word_trigrams(text, cfg.max_words_per_doc)
    if not shingles:
        return []
    m = MinHash(num_perm=cfg.num_permutations)
    for s in shingles:
        m.update(s.encode("utf8"))
    hashvalues = m.hashvalues  # numpy uint32 array, length = num_permutations
    bands: list[tuple[int, int]] = []
    for b in range(cfg.num_bands):
        start = b * cfg.rows_per_band
        end = start + cfg.rows_per_band
        # .astype('<u4') forces little-endian, matching int.from_bytes(..., "little") below
        # and producing the same bytes as struct.pack(f"{r}I", ...) on any host byte order.
        # Avoids cfg.rows_per_band int() casts + a generator per band.
        raw = hashvalues[start:end].astype("<u4").view(np.uint8).tobytes()
        band_hash = int.from_bytes(hashlib.sha256(raw).digest()[:8], "little", signed=True)
        bands.append((b, band_hash))
    return bands


def _flush_buffer(
    conn: psycopg2.extensions.connection, rows: list[tuple[int, int, int]]
) -> int:
    payload = "\n".join(f"{d}\t{b}\t{h}" for d, b, h in rows) + "\n"
    buf = io.StringIO(payload)
    with conn.cursor() as cur:
        cur.copy_from(buf, "lsh_candidate_bands", columns=("doc_id", "band_index", "band_hash"))
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Per-worker function (module-level for pickling by ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def compute_bands_for_range(
    conn_str: str, cfg: NearDedupConfig, start_id: int, end_id: int, worker_id: int
) -> int:
    """
    Process docs where start_id < doc_id <= end_id.
    Pass end_id=9223372036854775807 (INT8_MAX) for the last worker to make it open-ended —
    it will terminate naturally when it fetches an empty batch.
    Opens its own DB connection; safe to run in parallel with other workers.
    Returns number of band rows inserted.
    """
    worker_log = logging.getLogger(f"{__name__}.worker{worker_id}")
    conn = psycopg2.connect(conn_str)
    try:
        total_inserted = 0
        docs_processed = 0
        buffer: list[tuple[int, int, int]] = []
        batch_num = 0
        t_start = time.monotonic()

        open_ended = end_id == 9223372036854775807
        # Correctness comes from the per-batch anti-join (NOT EXISTS against
        # lsh_candidate_bands in the SELECT below), so coverage need not be
        # contiguous and a MAX(doc_id) watermark is unsafe: docs processed out
        # of doc_id order (e.g. high-id rows ingested while low-id rows are
        # still pending) would push the watermark past unprocessed docs and
        # silently skip them. last_doc_id is only an intra-run forward cursor —
        # it starts at the range low and advances past returned rows each batch
        # so un-flushed docs are not re-fetched within a run. Already-processed
        # docs anywhere in the range are excluded by the anti-join, making
        # restarts safe without re-processing.
        last_doc_id = start_id
        worker_log.info(
            "worker %d  start  range=(%d, %s]",
            worker_id, start_id, "∞" if open_ended else end_id,
        )

        while True:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT t.doc_id, t.cleaned_text
                    FROM   unified_document_texts t
                    JOIN   unified_documents      d USING (doc_id)
                    WHERE  t.doc_id > %s
                      AND  t.doc_id <= %s
                      AND  d.cleaning_is_dropped = FALSE
                      AND  t.cleaned_text IS NOT NULL
                      AND  char_length(t.cleaned_text) > %s
                      AND  NOT EXISTS (
                               SELECT 1 FROM lsh_candidate_bands b
                               WHERE  b.doc_id = t.doc_id
                           )
                    ORDER BY t.doc_id
                    LIMIT  %s
                    """,
                    (last_doc_id, end_id, cfg.min_text_length, cfg.batch_size),
                )
                rows = cur.fetchall()

            if not rows:
                break

            for doc_id, text in rows:
                for band_index, band_hash in _minhash_bands(text, cfg):
                    buffer.append((doc_id, band_index, band_hash))
                    if len(buffer) >= cfg.copy_buffer_rows:
                        total_inserted += _flush_buffer(conn, buffer)
                        buffer.clear()

            docs_processed += len(rows)
            last_doc_id = rows[-1][0]
            batch_num += 1

            elapsed = time.monotonic() - t_start
            rate = docs_processed / elapsed if elapsed > 0 else 0
            worker_log.info(
                "worker %d  docs=%d  rate=%.0f doc/s  elapsed=%.0fs",
                worker_id, docs_processed, rate, elapsed,
            )

        if buffer:
            total_inserted += _flush_buffer(conn, buffer)

        elapsed = time.monotonic() - t_start
        rate = docs_processed / elapsed if elapsed > 0 else 0
        worker_log.info(
            "worker %d  done   docs=%d  band_rows=%d  elapsed=%.0fs  avg=%.0f doc/s",
            worker_id, docs_processed, total_inserted, elapsed, rate,
        )
        return total_inserted
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def compute_bands(conn_str: str, cfg: NearDedupConfig, *, incremental: bool = True) -> int:
    """
    Compute MinHash bands for all eligible docs and write to lsh_candidate_bands.

    incremental=True (default):
        Splits the doc_id space evenly across num_workers ranges. Each worker
        filters every batch with NOT EXISTS against lsh_candidate_bands, so
        already-banded docs are skipped no matter where they fall — correct
        even when coverage is non-contiguous. Backed by lsh_candidate_bands_doc_id_idx.

    incremental=False (full):
        TRUNCATEs lsh_candidate_bands first, then processes all docs.
        Use this for the first-ever run or to force a full recompute.

    Returns total band rows inserted in this run (0 if nothing new).
    """
    INT8_MAX = 9223372036854775807
    ranges: list[tuple[int, int]] = []

    conn = psycopg2.connect(conn_str)
    try:
        if incremental:
            log.info(
                "Incremental mode: workers skip already-processed docs via a "
                "per-batch anti-join against lsh_candidate_bands"
            )
        else:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE lsh_candidate_bands")
            conn.commit()
            log.info("Full mode: lsh_candidate_bands truncated, starting from doc_id=0")

        start_id = 0

        with conn.cursor() as cur:
            # Fast snapshot for range-splitting only — workers apply the real
            # filters (including the anti-join) per batch.
            cur.execute("SELECT COALESCE(MAX(doc_id), 0) FROM unified_document_texts")
            snapshot_max_id: int = cur.fetchone()[0]

        if snapshot_max_id == 0:
            log.info("No docs in unified_document_texts — nothing to process")
            return 0

        # Even split of [0, snapshot_max] into num_workers contiguous ranges.
        # Coverage-correctness no longer depends on how ranges are drawn — the
        # per-batch anti-join skips any doc already in lsh_candidate_bands — so a
        # plain even split is sufficient. The last range is open-ended so docs
        # ingested after the snapshot are still picked up.
        doc_count = snapshot_max_id - start_id
        chunk = (doc_count + cfg.num_workers - 1) // cfg.num_workers
        for i in range(cfg.num_workers):
            lo = start_id + i * chunk
            hi = start_id + (i + 1) * chunk if i < cfg.num_workers - 1 else INT8_MAX
            if lo < snapshot_max_id:
                ranges.append((lo, hi))

        log.info(
            "Doc-id space [0, %d] split into %d worker ranges (anti-join incremental)",
            snapshot_max_id, len(ranges),
        )

        # The per-batch anti-join needs an index on lsh_candidate_bands.doc_id,
        # otherwise each NOT EXISTS degrades to a seq scan over all band rows.
        # Idempotent and non-destructive (adds an index only).
        with conn.cursor() as cur:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS lsh_candidate_bands_doc_id_idx "
                "ON lsh_candidate_bands (doc_id)"
            )
        conn.commit()
        log.info("Ensured lsh_candidate_bands_doc_id_idx exists for anti-join")
    finally:
        conn.close()

    total_inserted = 0
    with ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
        futures = {
            executor.submit(compute_bands_for_range, conn_str, cfg, lo, hi, i): (i, lo, hi)
            for i, (lo, hi) in enumerate(ranges)
        }
        for future in as_completed(futures):
            worker_id, lo, hi = futures[future]
            n = future.result()  # re-raises any worker exception
            total_inserted += n
            label = "∞" if hi == INT8_MAX else str(hi)
            log.info("Worker %d (%d, %s]: %d rows", worker_id, lo, label, n)

    log.info("All workers done: %d total band rows inserted; rebuilding index…", total_inserted)

    # Rebuild index after all workers have written (much faster than incremental maintenance)
    conn = psycopg2.connect(conn_str)
    try:
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS lsh_candidate_bands_band_idx")
            cur.execute(
                "CREATE INDEX lsh_candidate_bands_band_idx "
                "ON lsh_candidate_bands (band_index, band_hash)"
            )
        conn.commit()
        log.info("Index rebuilt on lsh_candidate_bands")
    finally:
        conn.close()

    return total_inserted
