"""
Ingest SEA-PILE-v2 Malay subset from HuggingFace into sea_pile_malay_documents.
Uses asyncpg with unnest INSERT for bulk throughput.

Raw source ingest only — does NOT touch unified_documents or unified_document_texts.
Those are populated after the cleaning + LID pipeline runs.

Usage:
  python3 ingest_sea_pile_malay.py [--dry-run] [--workers 2] [--batch-size 5000]
  python3 ingest_sea_pile_malay.py --workers 1 --batch-size 100   # single-shard test
"""

import argparse
import asyncio
import ctypes
import gc
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import asyncpg
import duckdb
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s INFO %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None

HF_REPO_ID = "aisingapore/SEA-PILE-v2"
HF_LANG_PATH = "ms"

DEFAULT_DSN = (
    "postgresql://corpus:corpus_secret@localhost:5432/corpus"
)
DEFAULT_PROGRESS_FILE = "sea_pile_malay_progress.json"
DEFAULT_BATCH_SIZE = 5000
DEFAULT_WORKERS = 2


# ── helpers ──────────────────────────────────────────────────────────────────


def _malloc_trim() -> None:
    if _libc:
        _libc.malloc_trim(0)


def load_progress(path: str) -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return {"completed": []}


def save_progress_atomic(path: str, progress: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f)
    os.rename(tmp, path)


# ── shard listing ─────────────────────────────────────────────────────────────


def list_malay_shards(hf_token: str | None) -> list[tuple[str, str]]:
    """Return [(fname, hf_repo_path), ...] for all ms/*.parquet shards.

    hf_repo_path is the repo-relative path (e.g. 'ms/train-00000-of-00042.parquet').
    Shards are downloaded locally via hf_hub_download before DuckDB reads them to
    avoid the DuckDB v1.5.2 httpfs bug with negative Content-Range headers from HF CDN.
    """
    from huggingface_hub import list_repo_tree

    items = list(
        list_repo_tree(
            HF_REPO_ID,
            path_in_repo=HF_LANG_PATH,
            repo_type="dataset",
            token=hf_token,
            expand=False,
        )
    )
    shards = []
    for item in items:
        if not item.path.endswith(".parquet"):
            continue
        fname = item.path.split("/")[-1]
        shards.append((fname, item.path))
    shards.sort()
    return shards


# ── per-shard async worker ────────────────────────────────────────────────────


async def _insert_batch_async(
    conn: asyncpg.Connection,
    df: pd.DataFrame,
    dry_run: bool,
) -> tuple[int, int]:
    def _ts(v):
        if pd.isna(v):
            return None
        try:
            return pd.Timestamp(v).to_pydatetime()
        except Exception:
            return None

    warc_ids = list(df["warc_record_id"])
    urls = [v or None for v in df["url"]]
    dumps = [v or None for v in df["dump"]]
    timestamps = [_ts(v) for v in df["timestamp"]]
    contents = [v or None for v in df["text"]]

    tr = conn.transaction()
    await tr.start()
    inserted = 0
    try:
        tag = await conn.execute(
            """
            INSERT INTO sea_pile_malay_documents
                (warc_record_id, url, dump, crawl_timestamp, content)
            SELECT * FROM unnest(
                $1::TEXT[], $2::TEXT[], $3::TEXT[], $4::TIMESTAMPTZ[], $5::TEXT[]
            ) ON CONFLICT (warc_record_id) DO NOTHING
            """,
            warc_ids, urls, dumps, timestamps, contents,
        )
        # tag is "INSERT 0 N" — parse the row count
        inserted = int(tag.split()[-1])
        if dry_run:
            await tr.rollback()
        else:
            await tr.commit()
    except Exception:
        await tr.rollback()
        raise

    skipped = len(warc_ids) - inserted
    return inserted, skipped


async def ingest_shard_async(
    fname: str,
    hf_path: str,
    dsn: str,
    batch_size: int,
    dry_run: bool,
    hf_token: str | None,
) -> tuple[str, int]:
    from huggingface_hub import hf_hub_download

    log.info("Downloading %s ...", fname)
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=hf_path,
        repo_type="dataset",
        token=hf_token,
    )
    log.info("Downloaded %s → %s", fname, local_path)

    conn = await asyncpg.connect(dsn)
    await conn.execute("SET TIME ZONE 'UTC'")
    total_rows = 0
    total_skipped = 0

    try:
        loop = asyncio.get_running_loop()
        duck = duckdb.connect()
        duck.execute("SET memory_limit='3GB'")
        duck.execute("SET threads=2")
        cursor = duck.execute(
            f'SELECT "warc-record-id" AS warc_record_id, url, dump, "timestamp", text '
            f"FROM read_parquet('{local_path}')"
        )
        columns = [d[0] for d in cursor.description]
        batch_idx = 0
        while True:
            rows = await loop.run_in_executor(None, cursor.fetchmany, batch_size)
            if not rows:
                break
            df = pd.DataFrame(rows, columns=columns)
            null_count = df["warc_record_id"].isna().sum()
            assert null_count == 0, (
                f"{fname} batch {batch_idx}: {null_count} NULL warc_record_id rows"
            )
            n_inserted, n_skipped = await _insert_batch_async(conn, df, dry_run)
            total_rows += n_inserted
            total_skipped += n_skipped
            log.info(
                "  %s batch %d: %d inserted, %d skipped",
                fname, batch_idx, n_inserted, n_skipped,
            )
            del df, rows
            gc.collect()
            _malloc_trim()
            batch_idx += 1
        duck.close()
    finally:
        await conn.close()

    log.info("Done %s — %d rows inserted, %d skipped", fname, total_rows, total_skipped)
    return fname, total_rows


def ingest_shard(
    fname: str,
    hf_path: str,
    dsn: str,
    batch_size: int,
    dry_run: bool,
    hf_token: str | None,
) -> tuple[str, int]:
    """Sync wrapper — each ProcessPoolExecutor worker runs its own event loop."""
    return asyncio.run(
        ingest_shard_async(fname, hf_path, dsn, batch_size, dry_run, hf_token)
    )


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest SEA-PILE-v2 Malay shards")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--progress-file", default=DEFAULT_PROGRESS_FILE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--hf-token", default=None,
        help="HuggingFace token (needed if repo is gated)",
    )
    parser.add_argument(
        "--limit-shards", type=int, default=None,
        help="Process at most N shards (for testing)",
    )
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY RUN — no rows will be committed")

    log.info("Listing Malay shards from %s/%s ...", HF_REPO_ID, HF_LANG_PATH)
    shards = list_malay_shards(args.hf_token)
    log.info("Found %d shards", len(shards))

    progress = load_progress(args.progress_file)
    completed_set = set(progress["completed"])
    pending = [(fname, hf_path) for fname, hf_path in shards if fname not in completed_set]

    if args.limit_shards:
        pending = pending[: args.limit_shards]

    total_shards = len(shards)
    log.info(
        "%d shards total, %d completed, %d pending — %d workers",
        total_shards, len(completed_set), len(pending), args.workers,
    )

    if not pending:
        log.info("Nothing to do.")
        return

    total_rows = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                ingest_shard,
                fname,
                hf_path,
                args.dsn,
                args.batch_size,
                args.dry_run,
                args.hf_token,
            ): fname
            for fname, hf_path in pending
        }
        for future in as_completed(futures):
            fname = futures[future]
            try:
                _, rows = future.result()
                total_rows += rows
                if not args.dry_run:
                    progress["completed"].append(fname)
                    save_progress_atomic(args.progress_file, progress)
                log.info(
                    "Progress: %d/%d shards done",
                    len(progress["completed"]), total_shards,
                )
            except Exception as exc:
                log.error("FAILED %s: %s", fname, exc, exc_info=True)

    log.info("Ingest complete — %d rows total", total_rows)


if __name__ == "__main__":
    main()
