"""
Near-duplicate detection pipeline using MinHash LSH.

Usage:
    python -m training_signal_processing.pipelines.near_dedup.run \\
        --conn "postgresql://user:pass@host/db" \\
        [--batch-size 10000] \\
        [--num-permutations 128] \\
        [--num-bands 16]

Prerequisites:
    1. Apply the SQL migration:
       psql $DATABASE_URL -f src/data-storage/scripts/migrations/add_near_dedup_table.sql
    2. Install dependencies:
       uv sync --group near_dedup

Two phases are run sequentially:

  Phase 1 (compute_bands):  Stream all eligible docs, compute MinHash
      bands, COPY into lsh_candidate_bands. Takes several hours on 38M docs.
      Safe to restart — the table is TRUNCATED at the start of each run.

  Phase 2 (detect_pairs):  Group bands to find candidate pairs, run
      Union-Find closure, upsert into document_near_duplicates, drop
      lsh_candidate_bands.

Verify results:
    -- All canonicals have lower doc_id (should return 0):
    SELECT count(*) FROM document_near_duplicates WHERE canonical_doc_id > doc_id;

    -- No chains (canonical itself is not a duplicate, should return 0):
    SELECT count(*)
    FROM document_near_duplicates d
    JOIN document_near_duplicates c ON d.canonical_doc_id = c.doc_id;
"""
from __future__ import annotations

import argparse
import logging
import time

from .compute_bands import compute_bands
from .config import NearDedupConfig
from .detect_pairs import detect_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MinHash LSH near-duplicate detection for unified_document_texts."
    )
    parser.add_argument("--conn", required=True, help="PostgreSQL connection string")
    parser.add_argument("--batch-size", type=int, default=10_000,
                        help="Docs per DB fetch per worker (default: 10000)")
    parser.add_argument("--num-permutations", type=int, default=128,
                        help="MinHash permutations (default: 128)")
    parser.add_argument("--num-bands", type=int, default=16,
                        help="LSH bands; rows_per_band = num_permutations // num_bands (default: 16)"  # noqa: E501
                        )
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel band-compute workers (default: 4)")
    parser.add_argument("--full", action="store_true",
                        help="Full recompute: TRUNCATE bands and process all docs. "
                             "Default is incremental (only new docs since last run).")
    args = parser.parse_args()

    incremental = not args.full
    rows_per_band = args.num_permutations // args.num_bands
    cfg = NearDedupConfig(
        batch_size=args.batch_size,
        num_permutations=args.num_permutations,
        num_bands=args.num_bands,
        rows_per_band=rows_per_band,
        num_workers=args.workers,
    )
    log.info(
        "Config: num_permutations=%d, num_bands=%d, rows_per_band=%d, "
        "batch_size=%d, workers=%d, incremental=%s",
        cfg.num_permutations, cfg.num_bands, cfg.rows_per_band,
        cfg.batch_size, cfg.num_workers, incremental,
    )

    log.info("=== Phase 1: computing MinHash bands (%s mode) ===",
             "incremental" if incremental else "full")
    t0 = time.monotonic()
    n_bands = compute_bands(args.conn, cfg, incremental=incremental)
    log.info("Phase 1 done: %d band rows written in %.0fs", n_bands, time.monotonic() - t0)

    log.info("=== Phase 2+3+4+5: pair detection, Union-Find, upsert ===")
    t1 = time.monotonic()
    n_pairs = detect_pairs(args.conn, cfg, keep_bands=incremental)
    log.info("Phase 2-5 done: %d near-duplicate rows in %.0fs", n_pairs, time.monotonic() - t1)

    log.info("Total elapsed: %.0fs", time.monotonic() - t0)


if __name__ == "__main__":
    main()
