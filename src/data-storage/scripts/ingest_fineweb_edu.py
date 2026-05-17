"""
Ingest HuggingFace FineWeb-Edu and promote it directly into the unified layer.

Per batch, in a single transaction (rolled back on --dry-run):
  1. fineweb_edu_documents      raw row (source of truth, kept for provenance)
  2. unified_documents          metadata row (no cleaning: cleaned == original)
  3. unified_document_texts     cleaned_text == raw text (…_fineweb_edu partition)
  4. document_language_detection language_label='standard-english', confidence=1.0

FineWeb-Edu is a curated, language-filtered English corpus, so the Malaya LID
model is intentionally NOT run — every doc is assigned the 'standard-english'
label (the exact Malaya fasttext vocabulary value) directly. All four inserts
use ON CONFLICT DO NOTHING, so the ingest is resume-safe.

Usage:
  python3 ingest_fineweb_edu.py [--dry-run] [--workers 2] [--batch-size 5000]
  python3 ingest_fineweb_edu.py --subset sample/10BT --workers 1 --batch-size 100
"""

import argparse
import asyncio
import ctypes
import gc
import hashlib
import json
import logging
import os
import sys
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

HF_REPO_ID = "HuggingFaceFW/fineweb-edu"
DEFAULT_SUBSET = "sample/100BT"

CLEANING_SOURCE = "fineweb-edu"  # == data_sources.source_name; partition key
TEXT_COLUMN = "text"
LID_LABEL = "standard-english"  # Malaya fasttext label 0 (curated EN corpus)
LID_CONFIDENCE = 1.0  # direct assignment, no model run
HF_RESOLVE_URL = "https://huggingface.co/datasets/" + HF_REPO_ID + "/resolve/main/"

DEFAULT_DSN = (
    "postgresql://corpus:corpus_secret@localhost:5432/corpus"
)
DEFAULT_PROGRESS_FILE = "fineweb_edu_progress.json"
DEFAULT_BATCH_SIZE = 5000
DEFAULT_WORKERS = 2


# ── helpers ──────────────────────────────────────────────────────────────────


def _malloc_trim() -> None:
    if _libc:
        _libc.malloc_trim(0)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sample_key(h: str) -> int:
    # Verbatim from scripts/ingest.py:174-175 — deterministic 60-bit sample key.
    return int(h[:15], 16) % (2**62)


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


def list_shards(subset: str, hf_token: str | None) -> list[tuple[str, str]]:
    """Return [(fname, hf_repo_path), ...] for all *.parquet shards under subset.

    hf_repo_path is the repo-relative path (e.g. 'sample/100BT/000_00000.parquet').
    Shards are downloaded locally via hf_hub_download before DuckDB reads them to
    avoid the DuckDB httpfs bug with negative Content-Range headers from HF CDN.
    """
    from huggingface_hub import list_repo_tree

    items = list(
        list_repo_tree(
            HF_REPO_ID,
            path_in_repo=subset,
            repo_type="dataset",
            token=hf_token,
            expand=False,
            recursive=True,
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
    source_id: int,
    hf_path: str,
    row_offset: int,
    dry_run: bool,
) -> tuple[int, int]:
    """Raw + unified + text + direct-LID inserts for one batch (one txn)."""

    def _int(v):
        if pd.isna(v):
            return None
        return int(v)

    def _float(v):
        if pd.isna(v):
            return None
        return float(v)

    hf_ids = list(df["hf_id"])
    urls = [v or None for v in df["url"]]
    dumps = [v or None for v in df["dump"]]
    file_paths = [v or None for v in df["file_path"]]
    languages = [v or None for v in df["language"]]
    language_scores = [_float(v) for v in df["language_score"]]
    token_counts = [_int(v) for v in df["token_count"]]
    edu_scores = [_float(v) for v in df["score"]]
    edu_int_scores = [_int(v) for v in df["int_score"]]
    texts = [("" if (v is None or pd.isna(v)) else str(v)) for v in df["text"]]
    contents = [t or None for t in texts]

    # Unified-layer derived fields (no cleaning: cleaned == original).
    sample_uids = [f"{CLEANING_SOURCE}://{hid}" for hid in hf_ids]
    sample_uid_hashes = [_sha256(u) for u in sample_uids]
    sample_keys = [_sample_key(h) for h in sample_uid_hashes]
    text_sha256s = [_sha256(t) for t in texts]
    char_counts = [len(t) for t in texts]
    approx_tokens = token_counts  # dataset's own token_count
    src_parquet_url = HF_RESOLVE_URL + hf_path
    row_indexes = [row_offset + i for i in range(len(hf_ids))]

    tr = conn.transaction()
    await tr.start()
    inserted = 0
    try:
        # 1. raw provenance table
        tag = await conn.execute(
            """
            INSERT INTO fineweb_edu_documents
                (hf_id, url, dump, file_path, language,
                 language_score, source_token_count, edu_score,
                 edu_int_score, content)
            SELECT * FROM unnest(
                $1::TEXT[], $2::TEXT[], $3::TEXT[], $4::TEXT[],
                $5::TEXT[], $6::REAL[], $7::BIGINT[], $8::REAL[],
                $9::SMALLINT[], $10::TEXT[]
            ) ON CONFLICT (hf_id) DO NOTHING
            """,
            hf_ids, urls, dumps, file_paths, languages,
            language_scores, token_counts, edu_scores, edu_int_scores,
            contents,
        )
        # tag is "INSERT 0 N" — parse the row count
        inserted = int(tag.split()[-1])

        # 2. unified_documents (column list/order matches ingest.py:523-563)
        result = await conn.fetch(
            """
            INSERT INTO unified_documents
                (sample_uid, sample_uid_hash, source_id, cleaning_source,
                 source_bucket, source_object_key, source_parquet_url, text_column,
                 source_row_group_index, source_row_index, row_index_in_row_group,
                 original_text_sha256, cleaned_text_sha256,
                 original_char_count, cleaned_char_count, removed_char_count,
                 approximate_original_token_count, approximate_cleaned_token_count,
                 approximate_removed_token_count, cleaning_is_dropped,
                 cleaning_rules_triggered,
                 cleaned_o200k_token_count, cleaned_o200k_tokenizer, sample_key)
            SELECT
                t.sample_uid, t.sample_uid_hash, $1::SMALLINT, $2::TEXT,
                NULL, $3::TEXT, $4::TEXT, $5::TEXT,
                NULL, t.source_row_index, NULL,
                t.text_sha256, t.text_sha256,
                t.char_count, t.char_count, 0,
                t.approx_tokens, t.approx_tokens,
                0, FALSE, NULL::TEXT[],
                NULL, NULL, t.sample_key
            FROM unnest(
                $6::TEXT[], $7::TEXT[], $8::BIGINT[], $9::TEXT[],
                $10::BIGINT[], $11::BIGINT[], $12::BIGINT[]
            ) AS t(sample_uid, sample_uid_hash, source_row_index,
                   text_sha256, char_count, approx_tokens, sample_key)
            ON CONFLICT (sample_uid) DO NOTHING
            RETURNING doc_id, sample_uid
            """,
            source_id, CLEANING_SOURCE, hf_path, src_parquet_url, TEXT_COLUMN,
            sample_uids, sample_uid_hashes, row_indexes, text_sha256s,
            char_counts, approx_tokens, sample_keys,
        )
        doc_id_map = {r["sample_uid"]: r["doc_id"] for r in result}
        # Rows skipped by ON CONFLICT (resume) — fetch their existing doc_ids
        # (mirrors ingest.py:576-582).
        missing = [u for u in sample_uids if u not in doc_id_map]
        if missing:
            existing = await conn.fetch(
                "SELECT doc_id, sample_uid FROM unified_documents "
                "WHERE sample_uid = ANY($1::TEXT[])",
                missing,
            )
            doc_id_map.update({r["sample_uid"]: r["doc_id"] for r in existing})

        doc_ids = [doc_id_map[u] for u in sample_uids]

        # 3. unified_document_texts — cleaned_text == raw text (no cleaning)
        await conn.execute(
            """
            INSERT INTO unified_document_texts (doc_id, cleaning_source, cleaned_text)
            SELECT t.doc_id, $1::TEXT, t.cleaned_text
            FROM unnest($2::BIGINT[], $3::TEXT[]) AS t(doc_id, cleaned_text)
            ON CONFLICT (cleaning_source, doc_id) DO NOTHING
            """,
            CLEANING_SOURCE, doc_ids, contents,
        )

        # 4. direct LID — no model run; curated English corpus
        await conn.execute(
            """
            INSERT INTO document_language_detection
                (doc_id, language_label, confidence)
            SELECT doc_id, $1::TEXT, $2::REAL
            FROM unnest($3::BIGINT[]) AS t(doc_id)
            ON CONFLICT (doc_id) DO NOTHING
            """,
            LID_LABEL, LID_CONFIDENCE, doc_ids,
        )

        if dry_run:
            await tr.rollback()
        else:
            await tr.commit()
    except Exception:
        await tr.rollback()
        raise

    skipped = len(hf_ids) - inserted
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
    src_row = await conn.fetchrow(
        "SELECT source_id FROM data_sources WHERE source_name = $1",
        CLEANING_SOURCE,
    )
    if src_row is None:
        await conn.close()
        raise RuntimeError(
            f"data_sources has no '{CLEANING_SOURCE}' row — "
            "apply scripts/migrations/add_fineweb_edu.sql first"
        )
    source_id = src_row["source_id"]
    total_rows = 0
    total_skipped = 0
    row_offset = 0

    try:
        loop = asyncio.get_running_loop()
        duck = duckdb.connect()
        duck.execute("SET memory_limit='3GB'")
        duck.execute("SET threads=2")
        cursor = duck.execute(
            f"SELECT id AS hf_id, url, dump, file_path, language, "
            f"language_score, token_count, score, int_score, text "
            f"FROM read_parquet('{local_path}')"
        )
        columns = [d[0] for d in cursor.description]
        batch_idx = 0
        while True:
            rows = await loop.run_in_executor(None, cursor.fetchmany, batch_size)
            if not rows:
                break
            df = pd.DataFrame(rows, columns=columns)
            null_count = df["hf_id"].isna().sum()
            assert null_count == 0, (
                f"{fname} batch {batch_idx}: {null_count} NULL hf_id rows"
            )
            n_inserted, n_skipped = await _insert_batch_async(
                conn, df, source_id, hf_path, row_offset, dry_run
            )
            row_offset += len(df)
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
    parser = argparse.ArgumentParser(description="Ingest FineWeb-Edu shards")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--progress-file", default=DEFAULT_PROGRESS_FILE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--subset", default=DEFAULT_SUBSET,
        help="In-repo path to ingest: sample/100BT (default), sample/10BT, "
             "sample/350BT, data/CC-MAIN-2024-51, etc.",
    )
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

    log.info("Listing shards from %s/%s ...", HF_REPO_ID, args.subset)
    shards = list_shards(args.subset, args.hf_token)
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
