"""
Language detection pipeline using Malaya fasttext.

Streams unified_document_texts, runs Malaya document-level language
detection, and writes (doc_id, language_label, confidence) to
document_language_detection.

Two modes:
  incremental (default): orchestrator probes per-chunk watermarks, detects
      coverage gaps (including poisoned watermarks from prior runs with
      different range boundaries), and dispatches workers starting from the
      first uncovered doc in each chunk. ON CONFLICT DO NOTHING makes
      re-runs idempotent.
  full (--full): TRUNCATEs document_language_detection then processes all docs.
      TRUNCATE happens only after the Malaya model loads successfully.

Workers each load the Malaya model independently. Use --workers 1 (default)
on a cold cache to avoid concurrent downloads. With a warm cache, --workers N
is safe.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import psycopg2
import psycopg2.extras

from .config import LangDetectConfig

log = logging.getLogger(__name__)


@dataclass
class _MalayaRuntime:
    fasttext_model: Any


_runtime: _MalayaRuntime | None = None


def _get_runtime(*, quantized: bool = True) -> _MalayaRuntime:
    global _runtime
    if _runtime is None:
        import malaya
        fasttext_model = malaya.language_detection.fasttext(quantized=quantized)
        _runtime = _MalayaRuntime(fasttext_model=fasttext_model)
    return _runtime


def _detect_doc(text: str) -> tuple[str, float]:
    """Return (language_label, confidence). Returns ('', 0.0) for empty text."""
    if not text.strip():
        return "", 0.0
    rt = _get_runtime()
    scores_by_label: dict[str, float] = rt.fasttext_model.predict_proba([text])[0]
    if not scores_by_label:
        return "", 0.0
    label = max(scores_by_label, key=scores_by_label.__getitem__)
    return label, float(scores_by_label[label])


def detect_for_range(
    conn_str: str,
    cfg: LangDetectConfig,
    start_id: int,
    end_id: int,
    worker_id: int,
) -> int:
    """
    Detect language for docs where start_id < doc_id <= end_id.
    Pass end_id=9223372036854775807 (INT8_MAX) for the last open-ended worker.
    Opens its own DB connection. Returns number of rows written.
    """
    worker_log = logging.getLogger(f"{__name__}.worker{worker_id}")
    conn = psycopg2.connect(conn_str)
    try:
        open_ended = end_id == 9223372036854775807
        # start_id is the orchestrator-computed watermark (actual first gap - 1).
        # We do NOT self-resume from MAX(LID) here — that watermark can be poisoned
        # by previous runs with different range boundaries that wrote a high doc_id
        # without covering earlier docs in this range. ON CONFLICT DO NOTHING handles
        # any already-covered docs the worker re-encounters.
        last_doc_id: int = start_id
        worker_log.info(
            "worker %d  start  range=(%d, %s]",
            worker_id, start_id, "∞" if open_ended else end_id,
        )

        total_written = 0
        docs_processed = 0
        buffer: list[tuple[int, str, float]] = []
        t_start = time.monotonic()
        last_log_at = t_start
        last_log_docs = 0

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
                    ORDER BY t.doc_id
                    LIMIT  %s
                    """,
                    (last_doc_id, end_id, cfg.min_text_length, cfg.batch_size),
                )
                rows = cur.fetchall()

            if not rows:
                break

            docs_processed += len(rows)
            last_doc_id = rows[-1][0]

            for doc_id, text in rows:
                label, confidence = _detect_doc(str(text))
                if not label:
                    continue
                buffer.append((doc_id, label, confidence))
                if len(buffer) >= cfg.write_buffer_rows:
                    total_written += _flush_buffer(conn, buffer)
                    buffer.clear()
                now = time.monotonic()
                if now - last_log_at >= cfg.log_every_secs:
                    interval = now - last_log_at
                    instant_rate = (docs_processed - last_log_docs) / interval
                    worker_log.info(
                        "worker %d  docs=%d  written=%d  rate=%.0f doc/s",
                        worker_id, docs_processed, total_written, instant_rate,
                    )
                    last_log_at = now
                    last_log_docs = docs_processed

        if buffer:
            total_written += _flush_buffer(conn, buffer)

        elapsed = time.monotonic() - t_start
        rate = docs_processed / elapsed if elapsed > 0 else 0
        worker_log.info(
            "worker %d  done  docs=%d  written=%d  elapsed=%.0fs  avg=%.0f doc/s",
            worker_id, docs_processed, total_written, elapsed, rate,
        )
        return total_written
    finally:
        conn.close()


def _flush_buffer(
    conn: psycopg2.extensions.connection,
    rows: list[tuple[int, str, float]],
) -> int:
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO document_language_detection (doc_id, language_label, confidence)
            VALUES %s
            ON CONFLICT (doc_id) DO NOTHING
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def detect_language(conn_str: str, cfg: LangDetectConfig, *, incremental: bool = True) -> int:
    """
    Detect language for all eligible docs and write to document_language_detection.

    incremental=True (default): probes per-chunk watermarks, detects coverage gaps,
        and dispatches workers from the first uncovered doc in each chunk.
    incremental=False (--full): TRUNCATEs the table first (only after model pre-warm).
    """
    INT8_MAX = 9223372036854775807

    # Pre-warm model before forking so download happens once and errors fail fast.
    log.info("Pre-warming Malaya model...")
    _get_runtime()
    log.info("Malaya model ready.")

    conn = psycopg2.connect(conn_str)
    try:
        if not incremental:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE document_language_detection")
            conn.commit()
            log.info("Full mode: document_language_detection truncated")

        with conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(doc_id), 0) FROM unified_document_texts")
            snapshot_max_id: int = cur.fetchone()[0]

        if snapshot_max_id == 0:
            log.info("No docs in unified_document_texts — nothing to process")
            return 0

        # Step 1: coarse split to probe per-chunk watermarks.
        coarse_chunk = (snapshot_max_id + cfg.num_workers - 1) // cfg.num_workers
        coarse: list[tuple[int, int]] = [
            (
                i * coarse_chunk,
                (i + 1) * coarse_chunk if i < cfg.num_workers - 1 else INT8_MAX,
            )
            for i in range(cfg.num_workers)
            if i * coarse_chunk < snapshot_max_id
        ]

        # Step 2: skip fully-covered chunks; collect intervals with remaining work.
        # Fast path: MAX(LID doc_id) < effective_hi → range has work from watermark.
        # Slow path: MAX >= effective_hi (appears covered) → verify with NOT EXISTS to
        #   catch "poisoned watermarks" — a previous run with different range boundaries
        #   may have written a doc at effective_hi without covering earlier eligible docs.
        remaining: list[tuple[int, int]] = []
        for lo, hi in coarse:
            effective_hi = snapshot_max_id if hi == INT8_MAX else hi
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COALESCE(MAX(doc_id), %s) FROM document_language_detection "
                    "WHERE doc_id > %s AND doc_id <= %s",
                    (lo, lo, effective_hi),
                )
                watermark: int = cur.fetchone()[0]
            if watermark < effective_hi:
                remaining.append((watermark, hi))
            else:
                # Appears fully covered — check for gaps before accepting.
                # ORDER BY + LIMIT 1 lets PG terminate early once it finds the
                # first uncovered doc (much faster than MIN over the full range).
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT t.doc_id - 1
                        FROM   unified_document_texts t
                        JOIN   unified_documents      d USING (doc_id)
                        WHERE  t.doc_id > %s AND t.doc_id <= %s
                          AND  d.cleaning_is_dropped = FALSE
                          AND  t.cleaned_text IS NOT NULL
                          AND  char_length(t.cleaned_text) > %s
                          AND  NOT EXISTS (
                              SELECT 1 FROM document_language_detection l
                              WHERE l.doc_id = t.doc_id
                          )
                        ORDER BY t.doc_id
                        LIMIT 1
                        """,
                        (lo, effective_hi, cfg.min_text_length),
                    )
                    row = cur.fetchone()
                actual_watermark = row[0] if row else effective_hi
                if actual_watermark < effective_hi:
                    log.info(
                        "Chunk (%d, %s]: poisoned watermark detected — "
                        "gap starts at doc_id %d",
                        lo, "∞" if hi == INT8_MAX else str(hi), actual_watermark + 1,
                    )
                    remaining.append((actual_watermark, hi))
    finally:
        conn.close()

    if not remaining:
        log.info("All ranges fully covered — nothing to process")
        return 0

    # Step 3: redistribute remaining intervals into num_workers balanced ranges.
    total_remaining = sum(
        (snapshot_max_id if hi == INT8_MAX else hi) - lo for lo, hi in remaining
    )
    target = max(1, total_remaining // cfg.num_workers)
    ranges: list[tuple[int, int]] = []
    for idx, (seg_lo, seg_hi) in enumerate(remaining):
        effective_hi = snapshot_max_id if seg_hi == INT8_MAX else seg_hi
        seg_size = effective_hi - seg_lo
        is_last = idx == len(remaining) - 1
        slots_left = cfg.num_workers - len(ranges)
        segs_left = len(remaining) - idx
        if is_last:
            n_splits = slots_left
        else:
            max_splits = max(1, slots_left - (segs_left - 1))
            n_splits = max(1, min(max_splits, round(seg_size / target)))
        chunk = max(1, seg_size // n_splits)
        for j in range(n_splits):
            lo = seg_lo + j * chunk
            hi = seg_hi if j == n_splits - 1 else seg_lo + (j + 1) * chunk
            ranges.append((lo, hi))
        if len(ranges) >= cfg.num_workers:
            break
    if ranges:
        ranges[-1] = (ranges[-1][0], INT8_MAX)

    log.info(
        "Remaining ~%d doc-ids redistributed into %d balanced worker range(s) (snapshot max=%d)",
        total_remaining, len(ranges), snapshot_max_id,
    )

    total_written = 0
    with ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
        futures = {
            executor.submit(detect_for_range, conn_str, cfg, lo, hi, i): (i, lo, hi)
            for i, (lo, hi) in enumerate(ranges)
        }
        for future in as_completed(futures):
            worker_id, lo, hi = futures[future]
            n = future.result()
            total_written += n
            label = "∞" if hi == INT8_MAX else str(hi)
            log.info("Worker %d (%d, %s]: %d rows written", worker_id, lo, label, n)

    return total_written
