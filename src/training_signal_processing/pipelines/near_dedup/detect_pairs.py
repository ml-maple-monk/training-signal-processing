"""
Phase 2: band-sharded parallel scan of lsh_candidate_bands → candidate pairs.
Phase 3: Union-Find closure so every canonical_doc_id points to the cluster root.
Phase 4: bulk-upsert into document_near_duplicates.
Phase 5: drop lsh_candidate_bands to reclaim disk.

Phase 2 is parallelised by band_index: a (band_index, band_hash) collision
group is wholly contained in one band_index, so the 16 bands shard cleanly
across cfg.num_workers processes. Each worker GROUP BYs only its bands,
pre-aggregates pair counts in a bounded local dict, and bulk-COPYs partial
(doc_a, doc_b, cnt) rows into the UNLOGGED work table near_dedup_pair_counts.
The parent then merges with a single SQL `GROUP BY doc_a, doc_b SUM(cnt)`
streamed through a server-side cursor — far cheaper than the original
single-process O(group²) expansion over every band group.

Memory: O(|candidate pairs| × ~200 bytes) + O(|distinct doc_ids| × 8 bytes)
in the parent for Phase 3; each worker holds only ≤ copy_buffer_rows
pre-aggregated pairs before flushing.
"""
from __future__ import annotations

import io
import logging
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import psycopg2
import psycopg2.extras

from .config import NearDedupConfig

log = logging.getLogger(__name__)

PAIR_COUNTS_TABLE = "near_dedup_pair_counts"


# ---------------------------------------------------------------------------
# Union-Find (path-halving; lowest index = canonical by construction in _union)
# ---------------------------------------------------------------------------


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]  # path halving
        x = parent[x]
    return x


def _union(parent: list[int], x: int, y: int) -> None:
    rx, ry = _find(parent, x), _find(parent, y)
    if rx == ry:
        return
    # Route smaller index to root so lowest doc_id always wins.
    if rx < ry:
        parent[ry] = rx
    else:
        parent[rx] = ry


# ---------------------------------------------------------------------------
# Phase 2 worker (module-level for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------


def _flush_pair_counts(
    conn: psycopg2.extensions.connection, local: dict[tuple[int, int], int]
) -> int:
    """COPY a worker's pre-aggregated pair counts into the work table."""
    if not local:
        return 0
    payload = "".join(f"{a}\t{b}\t{c}\n" for (a, b), c in local.items())
    with conn.cursor() as cur:
        cur.copy_from(
            io.StringIO(payload), PAIR_COUNTS_TABLE, columns=("doc_a", "doc_b", "cnt")
        )
    conn.commit()
    return len(local)


def _compute_partial_pairs(
    conn_str: str, cfg: NearDedupConfig, band_indexes: list[int], worker_id: int
) -> tuple[int, int]:
    """
    GROUP BY this worker's band_index subset, pre-aggregate pair counts in a
    bounded local dict, and bulk-COPY them into near_dedup_pair_counts.

    Pre-aggregating before flush collapses repeated pair collisions within the
    worker's bands; the parent's final SQL SUM(cnt) handles cross-worker and
    cross-flush merging, so flush boundaries do not affect correctness.

    Returns (band_groups_seen, pair_rows_written).
    """
    wlog = logging.getLogger(f"{__name__}.phase2w{worker_id}")
    conn = psycopg2.connect(conn_str)
    try:
        local: dict[tuple[int, int], int] = defaultdict(int)
        groups = 0
        rows_written = 0
        t0 = time.monotonic()

        with conn.cursor(name=f"band_groups_w{worker_id}") as cur:
            cur.itersize = 5_000
            cur.execute(
                """
                SELECT array_agg(doc_id ORDER BY doc_id) AS docs
                FROM   lsh_candidate_bands
                WHERE  band_index = ANY(%s)
                GROUP  BY band_index, band_hash
                HAVING count(*) > 1
                """,
                (band_indexes,),
            )
            for (docs,) in cur:
                for i in range(len(docs)):
                    di = docs[i]
                    for j in range(i + 1, len(docs)):
                        local[(di, docs[j])] += 1  # docs sorted → di < docs[j]
                groups += 1
                if len(local) >= cfg.copy_buffer_rows:
                    rows_written += _flush_pair_counts(conn, local)
                    local = defaultdict(int)
                if groups % 50_000 == 0:
                    elapsed = time.monotonic() - t0
                    rate = groups / elapsed if elapsed > 0 else 0
                    wlog.info(
                        "phase2 w%d  bands=%s  groups=%d  rows=%d  rate=%.0f grp/s",
                        worker_id, band_indexes, groups, rows_written, rate,
                    )

        rows_written += _flush_pair_counts(conn, local)
        wlog.info(
            "phase2 w%d done  bands=%s  groups=%d  pair_rows=%d  elapsed=%.0fs",
            worker_id, band_indexes, groups, rows_written, time.monotonic() - t0,
        )
        return groups, rows_written
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def detect_pairs(conn_str: str, cfg: NearDedupConfig, *, keep_bands: bool = True) -> int:
    """
    Reads lsh_candidate_bands, builds near-duplicate pairs via Union-Find,
    writes results to document_near_duplicates.

    keep_bands=True (default for incremental runs):
        lsh_candidate_bands is kept so future incremental runs can add new doc bands
        and re-run pair detection over the full accumulated table.

    keep_bands=False (full/one-shot mode):
        Drops lsh_candidate_bands after writing results to reclaim ~25 GB disk.

    Returns the number of rows written to document_near_duplicates.
    """
    conn = psycopg2.connect(conn_str)
    try:
        # --- Phase 2: band-sharded parallel scan → near_dedup_pair_counts ---
        # pair_bands[(smaller_doc_id, larger_doc_id)] = count of shared bands
        pair_bands: dict[tuple[int, int], int] = defaultdict(int)

        t2 = time.monotonic()
        # UNLOGGED work table: transient, always rebuilt; dropped after merge.
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE UNLOGGED TABLE IF NOT EXISTS {PAIR_COUNTS_TABLE} ("
                "  doc_a BIGINT NOT NULL,"
                "  doc_b BIGINT NOT NULL,"
                "  cnt   INTEGER NOT NULL)"
            )
            cur.execute(f"TRUNCATE {PAIR_COUNTS_TABLE}")
        conn.commit()

        # A (band_index, band_hash) group lives in exactly one band_index, so
        # the bands shard cleanly. Cannot shard finer than one band per worker.
        n_workers = min(cfg.num_workers, cfg.num_bands)
        shards: list[list[int]] = [[] for _ in range(n_workers)]
        for b in range(cfg.num_bands):
            shards[b % n_workers].append(b)
        shards = [s for s in shards if s]

        log.info(
            "Phase 2: %d workers over %d bands, shards=%s",
            len(shards), cfg.num_bands, shards,
        )
        total_groups = 0
        with ProcessPoolExecutor(max_workers=len(shards)) as ex:
            futs = {
                ex.submit(_compute_partial_pairs, conn_str, cfg, bands, i): i
                for i, bands in enumerate(shards)
            }
            for fut in as_completed(futs):
                g, _ = fut.result()  # re-raises worker exceptions
                total_groups += g

        # Merge: single SQL aggregation streamed through a server-side cursor.
        log.info("Phase 2: merging partial pair counts (SQL SUM)…")
        with conn.cursor(name="pair_merge") as cur:
            cur.itersize = 50_000
            cur.execute(
                f"SELECT doc_a, doc_b, SUM(cnt) "
                f"FROM {PAIR_COUNTS_TABLE} GROUP BY doc_a, doc_b"
            )
            for a, b, c in cur:
                pair_bands[(a, b)] = c

        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {PAIR_COUNTS_TABLE}")
        conn.commit()

        log.info(
            "Phase 2 done: %d candidate pairs  %d band groups  elapsed=%.0fs",
            len(pair_bands), total_groups, time.monotonic() - t2,
        )

        # --- Phase 3: Union-Find closure over candidate pairs ---
        t3 = time.monotonic()
        n_pairs = len(pair_bands)
        all_ids_set = {d for pair in pair_bands for d in pair}
        n_distinct = len(all_ids_set)
        log.info(
            "Phase 3: Union-Find on %d pairs  %d distinct docs…",
            n_pairs, n_distinct,
        )
        all_ids = sorted(all_ids_set)
        del all_ids_set
        id_to_idx = {d: i for i, d in enumerate(all_ids)}
        parent = list(range(len(all_ids)))

        for i, (a, b) in enumerate(pair_bands):
            _union(parent, id_to_idx[a], id_to_idx[b])
            if (i + 1) % 200_000 == 0:
                log.info(
                    "  phase3 union  %d/%d pairs  elapsed=%.0fs",
                    i + 1, n_pairs, time.monotonic() - t3,
                )

        # Build final rows: one per non-root doc_id.
        # shared_bands = direct band count to cluster root (0 for transitively resolved docs).
        # id_to_idx is still used in the union loop above; here we use idx from enumerate
        # directly to avoid a redundant dict lookup per doc.
        final_rows: list[tuple[int, int, int]] = []
        for idx, doc_id in enumerate(all_ids):
            root_idx = _find(parent, idx)
            if root_idx == idx:
                continue  # this doc IS the cluster root
            root = all_ids[root_idx]
            lo, hi = min(doc_id, root), max(doc_id, root)
            shared = pair_bands.get((lo, hi), 0)  # 0 means reached root transitively
            final_rows.append((doc_id, root, shared))
            if (idx + 1) % 200_000 == 0:
                log.info(
                    "  phase3 rows   %d/%d docs scanned  %d dup rows so far  elapsed=%.0fs",
                    idx + 1, n_distinct, len(final_rows), time.monotonic() - t3,
                )

        log.info(
            "Phase 3 done: %d near-duplicate rows  elapsed=%.0fs",
            len(final_rows), time.monotonic() - t3,
        )

        # --- Phase 4: bulk upsert into document_near_duplicates ---
        log.info("Phase 4: upserting into document_near_duplicates…")
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO document_near_duplicates (doc_id, canonical_doc_id, shared_bands)
                VALUES %s
                ON CONFLICT (doc_id) DO UPDATE
                    SET canonical_doc_id = EXCLUDED.canonical_doc_id,
                        shared_bands     = EXCLUDED.shared_bands,
                        detected_at      = now()
                """,
                final_rows,
                page_size=10_000,
            )
        conn.commit()
        log.info("Phase 4 done: %d rows upserted", len(final_rows))

        # --- Phase 5: optionally drop work table to reclaim disk ---
        if not keep_bands:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS lsh_candidate_bands")
            conn.commit()
            log.info("Phase 5: lsh_candidate_bands dropped (keep_bands=False)")
        else:
            log.info("Phase 5: keeping lsh_candidate_bands for future incremental runs")

        return len(final_rows)
    finally:
        conn.close()
