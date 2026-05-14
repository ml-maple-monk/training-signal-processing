#!/usr/bin/env python3
"""
Ingest unified-data parquet files into the corpus PostgreSQL 18 database.

Usage:
    python3 ingest.py [--parts-dir PATH] [--dsn DSN] [--progress-file PATH]
                      [--limit-files N] [--dry-run]

Populates:
    unified_documents, unified_document_texts, lid_metadata,
    lowyat_threads, lowyat_posts, cari_threads, cari_posts,
    reddit_indonesia_posts, hplt_malay_documents, hplt_indonesia_documents, ocr_books

Untested stubs (sources absent from current 78-file dataset):
    reddit_bolehland_posts, fineweb_documents
"""

import argparse
import ctypes
import gc
import json
import logging
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import duckdb

# Force-return freed heap pages to the OS after each batch.
# Without this, Python's malloc never shrinks the process RSS even after gc.collect().
try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import pyarrow.parquet as pq

DEFAULT_DSN = "postgresql://corpus:corpus_secret@localhost:5432/corpus"
DEFAULT_PARTS_DIR = (
    "/home/geeyang/workspace/minimind-mfu-working/data/unified_data"
    "/final-completed-20260430T160615Z/parts"
)
DEFAULT_PROGRESS_FILE = "ingest_progress.json"
DEFAULT_BATCH_SIZE = 10  # rows per transaction; small batches = trivial per-batch memory

# Sources with tested ingest branches. Sources not in this set will log a warning and skip.
TESTED_SOURCES = {
    "lowyat",
    "cari",
    "reddit-bolehland",
    "reddit-indonesia",
    "hplt-malay",
    "hplt-indonesia",
    "books-ocr",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Type conversion helpers ──────────────────────────────────────────────────

class NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if math.isnan(float(obj)) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _nan(v):
    """Return None if v is any kind of NA/NaN, else v."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _str(v):
    """String ID column: pass through or None."""
    v = _nan(v)
    return v if v is not None else None


_PG_INT_MIN = -2_147_483_648
_PG_INT_MAX =  2_147_483_647


def _int(v, pg_int=False):
    """Numeric-string column → int or None.

    Set pg_int=True for columns typed INTEGER (32-bit) in PostgreSQL.
    Out-of-range values (corrupt data) are silently set to None.
    """
    v = _nan(v)
    if v is None or v in ("", "None"):
        return None
    try:
        result = int(v)
    except (ValueError, TypeError):
        return None
    if pg_int and not (_PG_INT_MIN <= result <= _PG_INT_MAX):
        return None
    return result


def _float(v):
    """Numeric-string column → float or None."""
    v = _nan(v)
    if v is None or v in ("", "None"):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _bool(v):
    v = _nan(v)
    if v is None:
        return False
    return bool(v)


def _jsonb(v):
    v = _nan(v)
    if v is None:
        return None
    return json.dumps(v, cls=NumpySafeEncoder)


def _parse_cari_ts(v):
    """Parse cari posted_at: '13-1-2025 11:25 AM' → datetime(UTC) or None."""
    v = _nan(v)
    if v is None:
        return None
    for fmt in ("%d-%m-%Y %I:%M %p", "%d-%m-%Y %H:%M"):
        try:
            return datetime.strptime(v, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _parse_hplt_ts(v):
    """Parse HPLT timestamp ISO string → datetime(UTC) or None."""
    v = _nan(v)
    if v is None:
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


def _sample_key(h: str) -> int:
    """Deterministic 60-bit positive integer from first 15 hex chars of SHA-256."""
    return int(h[:15], 16) % (2**62)


def _list_or_none(v):
    """Return list as-is (psycopg2 adapts to TEXT[]) or None."""
    v = _nan(v)
    return v if isinstance(v, list) else None


# ── Progress file ────────────────────────────────────────────────────────────

def load_progress(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed": [], "current_file": None, "current_rg": -1}


def save_progress_atomic(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.rename(tmp, path)


# ── DB helpers ───────────────────────────────────────────────────────────────

def load_source_id_map(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT source_name, source_id FROM data_sources")
        return {row[0]: row[1] for row in cur.fetchall()}


def preflight_checks(conn, parts_dir: Path, source_id_map: dict):
    log.info("Running pre-flight checks...")

    # 1. Source coverage already verified by scan_sources()

    # 2. Assert unified_documents is empty
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM unified_documents")
        count = cur.fetchone()[0]
        if count > 0:
            raise SystemExit(
                f"FATAL: unified_documents is not empty ({count} rows). "
                "This script is a fresh-load script. Truncate the DB first."
            )

    # 3. Assert HPLT/fineweb source tables are empty
    identity_pk_tables = [
        "hplt_malay_documents",
        "hplt_indonesia_documents",
        "fineweb_documents",
    ]
    with conn.cursor() as cur:
        for tbl in identity_pk_tables:
            cur.execute(f"SELECT COUNT(*) FROM {tbl}")
            count = cur.fetchone()[0]
            if count > 0:
                raise SystemExit(
                    f"FATAL: {tbl} is not empty ({count} rows). "
                    "Re-ingest into IDENTITY PK tables requires full truncation first."
                )

    log.info("Pre-flight passed.")


# ── Per-source insert functions ──────────────────────────────────────────────

def _upsert(cur, sql: str, rows: list, label: str):
    if not rows:
        return
    psycopg2.extras.execute_values(cur, sql, rows, page_size=2000)
    inserted = cur.rowcount
    if inserted != len(rows):
        log.debug("%s: %d rows, %d inserted (%d skipped as duplicates)",
                  label, len(rows), inserted, len(rows) - inserted)


def insert_lowyat_threads(cur, df: pd.DataFrame):
    seen = {}
    for r in df.itertuples(index=False):
        tid = _str(r.thread_id)
        if tid is None or tid in seen:
            continue
        seen[tid] = (
            tid,
            _str(r.thread_title),
            _str(r.thread_url),
            _str(r.forum),
            _int(r.thread_total_pages, pg_int=True),
            _str(r.thread_status),
            None,  # fetched_at: relative string, unparseable
        )
    _upsert(
        cur,
        """INSERT INTO lowyat_threads
           (thread_id, thread_title, thread_url, forum, thread_total_pages, thread_status, fetched_at)
           VALUES %s ON CONFLICT (thread_id) DO NOTHING""",
        list(seen.values()),
        "lowyat_threads",
    )


def insert_lowyat_posts(cur, df: pd.DataFrame):
    rows = [
        (
            _str(r.post_id),
            _str(r.thread_id),
            _int(r.post_floor, pg_int=True),
            _int(r.page_number, pg_int=True),
            _int(r.page_offset, pg_int=True),
            _str(r.author),
            _str(r.author_id),
            None,  # posted_at: lowyat uses relative strings ("Today, 06:35 PM")
            _nan(r.cleaned_text),  # body_text = cleaned version (raw not in unified parquet)
            _str(r.body_html),
            _str(r.quoted_post_id),
            None,  # fetched_at
            _str(r.error_reason),
        )
        for r in df.itertuples(index=False)
        if _str(r.post_id) is not None
    ]
    _upsert(
        cur,
        """INSERT INTO lowyat_posts
           (post_id, thread_id, post_floor, page_number, page_offset, author, author_id,
            posted_at, body_text, body_html, quoted_post_id, fetched_at, error_reason)
           VALUES %s ON CONFLICT (post_id) DO NOTHING""",
        rows,
        "lowyat_posts",
    )


def insert_cari_threads(cur, df: pd.DataFrame):
    seen = {}
    for r in df.itertuples(index=False):
        tid = _str(r.thread_id)
        if tid is None or tid in seen:
            continue
        seen[tid] = (
            tid,
            _str(r.thread_title),
            _str(r.thread_url),
            _str(r.forum),
            _int(r.thread_total_pages, pg_int=True),
            _str(r.thread_status),
            None,  # fetched_at
        )
    _upsert(
        cur,
        """INSERT INTO cari_threads
           (thread_id, thread_title, thread_url, forum, thread_total_pages, thread_status, fetched_at)
           VALUES %s ON CONFLICT (thread_id) DO NOTHING""",
        list(seen.values()),
        "cari_threads",
    )


def insert_cari_posts(cur, df: pd.DataFrame):
    rows = [
        (
            _str(r.post_id),
            _str(r.thread_id),
            _int(r.post_floor, pg_int=True),
            _int(r.page_number, pg_int=True),
            _int(r.page_offset, pg_int=True),
            _str(r.author),
            _str(r.author_id),
            _parse_cari_ts(_nan(r.posted_at)),
            _nan(r.cleaned_text),
            _str(r.body_html),
            _str(r.quoted_post_id),
            None,  # fetched_at
            _str(r.error_reason),
        )
        for r in df.itertuples(index=False)
        if _str(r.post_id) is not None
    ]
    _upsert(
        cur,
        """INSERT INTO cari_posts
           (post_id, thread_id, post_floor, page_number, page_offset, author, author_id,
            posted_at, body_text, body_html, quoted_post_id, fetched_at, error_reason)
           VALUES %s ON CONFLICT (post_id) DO NOTHING""",
        rows,
        "cari_posts",
    )


def insert_reddit_posts(cur, df: pd.DataFrame, table: str):
    rows = [
        (
            _str(r.post_id),
            _str(r.post_kind),
            _str(r.submission_id),
            _str(r.parent_id),
            _str(r.subreddit),
            _str(r.author),
            _str(r.title),
            _int(r.score, pg_int=True),
            _int(r.num_comments, pg_int=True),
            _nan(r.created_utc),  # already a datetime or None from pandas
            _str(r.permalink),
            _str(r.url),
            _str(r.month),
            _nan(r.cleaned_text),  # body = cleaned text
        )
        for r in df.itertuples(index=False)
        if _str(r.post_id) is not None
    ]
    _upsert(
        cur,
        f"""INSERT INTO {table}
           (post_id, post_kind, submission_id, parent_id, subreddit, author, title,
            score, num_comments, created_utc, permalink, url, month, body)
           VALUES %s ON CONFLICT (post_id) DO NOTHING""",
        rows,
        table,
    )


def insert_hplt(cur, df: pd.DataFrame, table: str):
    rows = [
        (
            _str(r.id),          # hplt_id
            _str(r.url),
            _parse_hplt_ts(_nan(r.timestamp)),  # crawl_timestamp
            _str(r.crawl_id),
            _str(r.source_shard),
            _str(r.language),
            _str(r.row_language_code),
            _float(r.row_language_prob),
        )
        for r in df.itertuples(index=False)
    ]
    # Plain INSERT — no UPSERT (no unique key on hplt_id); pre-flight guarantees empty table
    # content omitted: cleaned_text already stored in unified_document_texts
    if rows:
        psycopg2.extras.execute_values(
            cur,
            f"""INSERT INTO {table}
               (hplt_id, url, crawl_timestamp, crawl_id, source_shard, language,
                row_language_code, row_language_prob)
               VALUES %s""",
            rows,
            page_size=2000,
        )


def insert_ocr_books(cur, df: pd.DataFrame):
    rows = [
        (
            _str(r.document_id),
            _str(r.source_run_id),
            _str(r.source_format),
            _str(r.markdown_file_name),
            _str(r.markdown_rel_path),
            _str(r.markdown_sha256),
            _int(r.markdown_char_count, pg_int=True),
            _int(r.markdown_byte_count, pg_int=True),
            _nan(r.cleaned_text),  # markdown_text = cleaned version
        )
        for r in df.itertuples(index=False)
        if _str(r.document_id) is not None
    ]
    _upsert(
        cur,
        """INSERT INTO ocr_books
           (document_id, source_run_id, source_format, markdown_file_name, markdown_rel_path,
            markdown_sha256, markdown_char_count, markdown_byte_count, markdown_text)
           VALUES %s ON CONFLICT (document_id) DO NOTHING""",
        rows,
        "ocr_books",
    )


def insert_fineweb(cur, df: pd.DataFrame):  # UNTESTED STUB
    rows = [
        (
            None,  # hf_id not in parquet
            _str(r.url),
            None,  # date
            _str(r.month),
            _str(r.language),
            _nan(r.cleaned_text),
        )
        for r in df.itertuples(index=False)
    ]
    if rows:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO fineweb_documents
               (hf_id, url, date, month, language, content)
               VALUES %s""",
            rows,
            page_size=2000,
        )


# ── Unified tables ───────────────────────────────────────────────────────────

def insert_unified_documents(cur, df: pd.DataFrame, source_id_map: dict) -> dict:
    rows = []
    for r in df.itertuples(index=False):
        src = r.cleaning_source
        rows.append((
            r.sample_uid,
            r.sample_uid_hash,
            source_id_map[src],
            src,
            _str(r.source_bucket),
            _str(r.source_object_key),
            _str(r.source_parquet_url),
            _str(r.text_column),
            _nan(r.source_row_group_index),
            _nan(r.source_row_index),
            _nan(r.row_index_in_row_group),
            _str(r.original_text_sha256),
            _str(r.cleaned_text_sha256),
            _nan(r.original_char_count),
            _nan(r.cleaned_char_count),
            _nan(r.removed_char_count),
            _nan(r.approximate_original_token_count),
            _nan(r.approximate_cleaned_token_count),
            _nan(r.approximate_removed_token_count),
            _bool(r.cleaning_is_dropped),
            _list_or_none(r.cleaning_rules_triggered),
            _nan(r.cleaned_o200k_token_count),
            _str(r.cleaned_o200k_tokenizer),
            _sample_key(r.sample_uid_hash),
        ))

    result = psycopg2.extras.execute_values(
        cur,
        """INSERT INTO unified_documents
           (sample_uid, sample_uid_hash, source_id, cleaning_source,
            source_bucket, source_object_key, source_parquet_url, text_column,
            source_row_group_index, source_row_index, row_index_in_row_group,
            original_text_sha256, cleaned_text_sha256,
            original_char_count, cleaned_char_count, removed_char_count,
            approximate_original_token_count, approximate_cleaned_token_count,
            approximate_removed_token_count, cleaning_is_dropped, cleaning_rules_triggered,
            cleaned_o200k_token_count, cleaned_o200k_tokenizer, sample_key)
           VALUES %s RETURNING doc_id, sample_uid""",
        rows,
        fetch=True,
        page_size=500,
    )
    # Build order-independent lookup dict
    return {sample_uid: doc_id for doc_id, sample_uid in result}


def insert_unified_texts(cur, df: pd.DataFrame, doc_id_map: dict):
    rows = [
        (doc_id_map[r.sample_uid], r.cleaning_source, _nan(r.cleaned_text))
        for r in df.itertuples(index=False)
        if r.sample_uid in doc_id_map
    ]
    if rows:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO unified_document_texts (doc_id, cleaning_source, cleaned_text) VALUES %s",
            rows,
            page_size=len(rows),
        )


def insert_lid_metadata(cur, df: pd.DataFrame, doc_id_map: dict):
    rows = [
        (
            doc_id_map[r.sample_uid],
            _nan(r.lid_cleaned_token_count),
            _nan(r.reference_removed),
            _str(r.reference_removal_method),
            _nan(r.removed_reference_char_count),
            _str(r.lingua_primary_language),
            _jsonb(_nan(r.lingua_spans)),
            _str(r.malaya_document_label),
            _jsonb(_nan(r.malaya_document_scores)),
            _jsonb(_nan(r.malaya_word_detections)),
            _jsonb(_nan(r.malaya_word_label_counts)),
        )
        for r in df.itertuples(index=False)
        if r.sample_uid in doc_id_map and _str(r.lingua_primary_language) is not None
    ]
    if rows:
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO lid_metadata
               (doc_id, lid_cleaned_token_count, reference_removed, reference_removal_method,
                removed_reference_char_count, lingua_primary_language, lingua_spans,
                malaya_document_label, malaya_document_scores,
                malaya_word_detections, malaya_word_label_counts)
               VALUES %s ON CONFLICT (doc_id) DO NOTHING""",
            rows,
            page_size=2000,
        )


# ── Row group processor ──────────────────────────────────────────────────────

def process_row_group(conn, df: pd.DataFrame, source_id_map: dict, dry_run: bool):
    with conn.cursor() as cur:
        # Per-source inserts
        for src, grp in df.groupby("cleaning_source"):
            if src not in TESTED_SOURCES:
                log.warning("Skipping untested source: %s (%d rows)", src, len(grp))
                continue

            if src == "lowyat":
                insert_lowyat_threads(cur, grp)
                insert_lowyat_posts(cur, grp)
            elif src == "cari":
                insert_cari_threads(cur, grp)
                insert_cari_posts(cur, grp)
            elif src == "reddit-bolehland":
                insert_reddit_posts(cur, grp, "reddit_bolehland_posts")
            elif src == "reddit-indonesia":
                insert_reddit_posts(cur, grp, "reddit_indonesia_posts")
            elif src == "hplt-malay":
                insert_hplt(cur, grp, "hplt_malay_documents")
            elif src == "hplt-indonesia":
                insert_hplt(cur, grp, "hplt_indonesia_documents")
            elif src == "books-ocr":
                insert_ocr_books(cur, grp)
            elif src == "fineweb":
                insert_fineweb(cur, grp)

        # Unified tables
        doc_id_map = insert_unified_documents(cur, df, source_id_map)
        insert_unified_texts(cur, df, doc_id_map)
        insert_lid_metadata(cur, df, doc_id_map)

    if not dry_run:
        conn.commit()
    else:
        conn.rollback()


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Ingest unified parquet files into corpus DB")
    p.add_argument("--parts-dir", default=DEFAULT_PARTS_DIR)
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--progress-file", default=DEFAULT_PROGRESS_FILE)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help="Rows per transaction (default: 5000)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel file workers (default: 4)")
    p.add_argument("--limit-files", type=int, default=None, help="Process at most N files")
    p.add_argument("--dry-run", action="store_true", help="Read and parse but do not commit")
    p.add_argument("--force", action="store_true",
                   help="Skip pre-flight emptiness checks (use when resuming a partial ingest)")
    return p.parse_args()


def scan_sources(parts_dir: Path, source_id_map: dict):
    all_sources = set()
    for f in sorted(parts_dir.glob("*.parquet")):
        pf = pq.ParquetFile(f)
        batch = next(pf.iter_batches(batch_size=5, columns=["cleaning_source"]))
        all_sources.update(batch.to_pydict()["cleaning_source"])
    missing = all_sources - set(source_id_map.keys())
    if missing:
        sys.exit(f"FATAL: cleaning_source values not in data_sources: {missing}")
    stubs = all_sources - TESTED_SOURCES
    if stubs:
        log.warning("Sources NOT in TESTED_SOURCES (will skip): %s", stubs)
    log.info("Sources found: %s", sorted(all_sources))


def _open_conn(dsn: str) -> psycopg2.extensions.connection:
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute("SET TIME ZONE 'UTC'")
    conn.commit()
    return conn


def ingest_file(
    pf_path: Path,
    dsn: str,
    source_id_map: dict,
    batch_size: int,
    dry_run: bool,
) -> tuple[str, int]:
    """Process one parquet file in an isolated OS process (ProcessPoolExecutor).

    When this process exits, ALL memory is returned to the OS — DuckDB buffer
    pool, pandas heap, Python objects — preventing the RSS accumulation that
    caused OOM with threads.
    """
    fname = pf_path.name
    log.info("Starting %s", fname)
    conn = _open_conn(dsn)
    total_rows = 0

    try:
        duck = duckdb.connect()
        duck.execute("SET memory_limit='3GB'")
        duck.execute("SET threads=2")
        cursor = duck.execute(f"SELECT * FROM read_parquet('{pf_path}')")
        columns = [d[0] for d in cursor.description]
        batch_idx = 0
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            df = pd.DataFrame(rows, columns=columns)
            n = len(df)
            process_row_group(conn, df, source_id_map, dry_run)
            del df, rows
            gc.collect()
            if _libc:
                _libc.malloc_trim(0)
            total_rows += n
            log.info("  %s batch %d: %d rows committed", fname, batch_idx, n)
            batch_idx += 1
        duck.close()
    finally:
        conn.close()

    log.info("Done %s — %d rows", fname, total_rows)
    return fname, total_rows


def main():
    args = parse_args()
    parts_dir = Path(args.parts_dir)
    if not parts_dir.exists():
        sys.exit(f"parts-dir not found: {parts_dir}")

    # Orchestrator connection: used only for pre-flight checks, not for ingest
    conn = _open_conn(args.dsn)
    source_id_map = load_source_id_map(conn)
    scan_sources(parts_dir, source_id_map)

    if args.dry_run or args.force:
        log.info("Pre-flight emptiness checks skipped (%s)",
                 "--dry-run" if args.dry_run else "--force")
    else:
        preflight_checks(conn, parts_dir, source_id_map)
    conn.close()

    progress = load_progress(args.progress_file)
    completed_set = set(progress.get("completed", []))

    parquet_files = sorted(parts_dir.glob("*.parquet"))
    if args.limit_files:
        parquet_files = parquet_files[: args.limit_files]

    pending = [f for f in parquet_files if f.name not in completed_set]
    total_files = len(parquet_files)
    log.info(
        "%d files total, %d completed, %d pending — %d workers",
        total_files, len(completed_set), len(pending), args.workers,
    )

    total_rows = 0
    # ProcessPoolExecutor: each file runs in its own OS process.
    # When a process exits, ALL memory is released — no RSS accumulation.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                ingest_file,
                pf_path,
                args.dsn,
                source_id_map,
                args.batch_size,
                args.dry_run,
            ): pf_path
            for pf_path in pending
        }
        for future in as_completed(futures):
            pf_path = futures[future]
            try:
                fname, rows = future.result()
                total_rows += rows
                if not args.dry_run:
                    progress["completed"].append(fname)
                    save_progress_atomic(args.progress_file, progress)
                    log.info("Progress: %d/%d files done", len(progress["completed"]), total_files)
            except Exception as exc:
                log.error("FAILED %s: %s", pf_path.name, exc, exc_info=True)

    log.info("Ingest complete. Total rows processed: %d", total_rows)
    if args.dry_run:
        log.info("--dry-run: no data was committed")


if __name__ == "__main__":
    main()
