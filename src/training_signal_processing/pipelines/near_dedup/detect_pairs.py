"""
Phase 2: group lsh_candidate_bands → candidate pair_bands dict.
Phase 3: Union-Find closure so every canonical_doc_id points to the cluster root.
Phase 4: bulk-upsert into document_near_duplicates.
Phase 5: drop lsh_candidate_bands to reclaim disk.

Memory: O(|candidate pairs| × ~200 bytes) + O(|distinct doc_ids| × 8 bytes).
Worst case for 1M pairs ≈ ~220 MB — well within typical server memory.

A server-side named cursor is used so PostgreSQL streams band groups without
materialising the full 608M-row result set on the client.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict

import psycopg2
import psycopg2.extras

from .config import NearDedupConfig

log = logging.getLogger(__name__)


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
        # --- Phase 2: collect candidate pairs from band collisions ---
        # pair_bands[(smaller_doc_id, larger_doc_id)] = count of shared bands
        pair_bands: dict[tuple[int, int], int] = defaultdict(int)

        log.info("Phase 2: streaming band groups from lsh_candidate_bands…")
        t2 = time.monotonic()
        with conn.cursor(name="band_groups") as cur:
            cur.itersize = 5_000
            cur.execute(
                """
                SELECT array_agg(doc_id ORDER BY doc_id) AS docs
                FROM   lsh_candidate_bands
                GROUP  BY band_index, band_hash
                HAVING count(*) > 1
                """
            )
            group_count = 0
            for (docs,) in cur:
                for i in range(len(docs)):
                    for j in range(i + 1, len(docs)):
                        key = (docs[i], docs[j])  # docs[i] < docs[j] due to ORDER BY
                        pair_bands[key] += 1
                group_count += 1
                if group_count % 10_000 == 0:
                    elapsed = time.monotonic() - t2
                    rate = group_count / elapsed if elapsed > 0 else 0
                    log.info(
                        "  phase2  groups=%d  pairs=%d  elapsed=%.0fs  rate=%.0f grp/s",
                        group_count, len(pair_bands), elapsed, rate,
                    )

        log.info(
            "Phase 2 done: %d candidate pairs  %d band groups  elapsed=%.0fs",
            len(pair_bands), group_count, time.monotonic() - t2,
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
