"""
Document-level language detection pipeline using Malaya fasttext.

Usage:
    python -m training_signal_processing.pipelines.lang_detect.run \\
        --conn "postgresql://user:pass@host/db" \\
        [--batch-size 10000] \\
        [--workers 1]

Prerequisites:
    1. Apply the SQL migration:
       psql $DATABASE_URL -f \
           src/data-storage/scripts/migrations/add_document_language_detection_table.sql
    2. Dependencies (malaya is in the default dependency group — no extra uv sync needed).
       First run downloads ~300 MB Malaya model to ~/.cache/malaya/.
       Use --workers 1 (default) until the cache is warm.

Two modes:

  incremental (default): workers self-resume from the highest doc_id already
      written in their assigned range. Safe to re-run as the corpus grows.

  full (--full): TRUNCATEs document_language_detection then processes all docs.
      TRUNCATE happens only after the Malaya model loads successfully.

Verify results:
    SELECT language_label, COUNT(*), ROUND(AVG(confidence)::numeric, 3) AS avg_conf
    FROM document_language_detection
    GROUP BY 1 ORDER BY 2 DESC;
"""
from __future__ import annotations

import argparse
import logging
import time

from .config import LangDetectConfig
from .detect import detect_language

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Malaya fasttext document-level language detection for unified_document_texts."
    )
    parser.add_argument("--conn", required=True, help="PostgreSQL connection string")
    parser.add_argument(
        "--batch-size", type=int, default=10_000,
        help="Docs per DB fetch per worker (default: 10000)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel detection workers (default: 1; increase only with a warm Malaya cache)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full recompute: TRUNCATE result table and process all docs. "
             "Default is incremental (only new docs since last run).",
    )
    args = parser.parse_args()

    cfg = LangDetectConfig(
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    incremental = not args.full
    log.info(
        "Config: batch_size=%d, workers=%d, min_text_length=%d, mode=%s",
        cfg.batch_size, cfg.num_workers, cfg.min_text_length,
        "incremental" if incremental else "full",
    )

    t0 = time.monotonic()
    n_written = detect_language(args.conn, cfg, incremental=incremental)
    log.info("Done: %d rows written in %.0fs", n_written, time.monotonic() - t0)


if __name__ == "__main__":
    main()
