#!/usr/bin/env python3
"""
Ingest unified-data parquet files into the corpus PostgreSQL 18 database.
Uses asyncpg with COPY/unnest for bulk throughput (5-10x faster than psycopg2).

Usage:
    python3 ingest.py [--parts-dir PATH] [--dsn DSN] [--progress-file PATH]
                      [--batch-size N] [--workers N] [--limit-files N] [--dry-run] [--force]

Populates:
    unified_documents, unified_document_texts, lid_metadata,
    lowyat_threads, lowyat_posts, cari_threads, cari_posts,
    reddit_indonesia_posts, hplt_malay_documents, hplt_indonesia_documents, ocr_books

Untested stubs (sources absent from current 78-file dataset):
    reddit_bolehland_posts, fineweb_documents
"""

import argparse
import asyncio
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

import asyncpg
import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None

DEFAULT_DSN = "postgresql://corpus:corpus_secret@localhost:5432/corpus"
DEFAULT_PARTS_DIR = (
    "/home/geeyang/workspace/minimind-mfu-working/data/unified_data"
    "/final-completed-20260430T160615Z/parts"
)
DEFAULT_PROGRESS_FILE = "ingest_progress.json"
DEFAULT_BATCH_SIZE = 5000

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
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _str(v):
    v = _nan(v)
    return v if v is not None else None


_PG_INT_MIN = -2_147_483_648
_PG_INT_MAX = 2_147_483_647


def _int(v, pg_int=False):
    """Convert to int or None. pg_int=True: clamp out-of-range values to None for INTEGER cols."""
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


def _bool_or_none(v):
    v = _nan(v)
    return None if v is None else bool(v)


def _jsonb(v):
    """Serialize to JSON string for asyncpg TEXT[] parameter; cast to JSONB in SQL.

    asyncpg can't encode list[dict] as JSONB[] (non-homogeneous array error).
    Passing as TEXT[] and casting in SQL avoids this entirely.
    """
    v = _nan(v)
    if v is None:
        return None
    return json.dumps(v, cls=NumpySafeEncoder)


def _parse_cari_ts(v):
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
    v = _nan(v)
    if v is None:
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


def _sample_key(h: str) -> int:
    return int(h[:15], 16) % (2**62)


def _list_or_none(v):
    v = _nan(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return list(v) if isinstance(v, list) else None


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


async def load_source_id_map(conn: asyncpg.Connection) -> dict:
    rows = await conn.fetch("SELECT source_name, source_id FROM data_sources")
    return {r["source_name"]: r["source_id"] for r in rows}


async def preflight_checks(conn: asyncpg.Connection, source_id_map: dict):
    log.info("Running pre-flight checks...")
    count = await conn.fetchval("SELECT COUNT(*) FROM unified_documents")
    if count > 0:
        raise SystemExit(
            f"FATAL: unified_documents is not empty ({count} rows). "
            "This script is a fresh-load script. Truncate the DB first."
        )
    for tbl in ["hplt_malay_documents", "hplt_indonesia_documents", "fineweb_documents"]:
        count = await conn.fetchval(f"SELECT COUNT(*) FROM {tbl}")
        if count > 0:
            raise SystemExit(
                f"FATAL: {tbl} is not empty ({count} rows). "
                "Re-ingest into IDENTITY PK tables requires full truncation first."
            )
    log.info("Pre-flight passed.")


# ── Per-source async insert functions ────────────────────────────────────────
# UPSERT tables  → unnest arrays + ON CONFLICT DO NOTHING
# Plain INSERT   → copy_records_to_table (COPY protocol, 3-5× faster)


async def insert_lowyat_threads(conn: asyncpg.Connection, df: pd.DataFrame):
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
    if not seen:
        return
    v = list(seen.values())
    await conn.execute(
        """
        INSERT INTO lowyat_threads
            (thread_id, thread_title, thread_url, forum,
             thread_total_pages, thread_status, fetched_at)
        SELECT * FROM unnest(
            $1::TEXT[], $2::TEXT[], $3::TEXT[], $4::TEXT[],
            $5::INTEGER[], $6::TEXT[], $7::TIMESTAMPTZ[]
        ) ON CONFLICT (thread_id) DO NOTHING
        """,
        [x[0] for x in v], [x[1] for x in v], [x[2] for x in v], [x[3] for x in v],
        [x[4] for x in v], [x[5] for x in v], [x[6] for x in v],
    )


async def insert_lowyat_posts(conn: asyncpg.Connection, df: pd.DataFrame):
    rows = [
        (
            _str(r.post_id), _str(r.thread_id),
            _int(r.post_floor, pg_int=True), _int(r.page_number, pg_int=True),
            _int(r.page_offset, pg_int=True), _str(r.author), _str(r.author_id),
            None,  # posted_at
            _nan(r.cleaned_text), _str(r.body_html), _str(r.quoted_post_id),
            None,  # fetched_at
            _str(r.error_reason),
        )
        for r in df.itertuples(index=False)
        if _str(r.post_id) is not None
    ]
    if not rows:
        return
    await conn.execute(
        """
        INSERT INTO lowyat_posts
            (post_id, thread_id, post_floor, page_number, page_offset,
             author, author_id, posted_at, body_text, body_html,
             quoted_post_id, fetched_at, error_reason)
        SELECT * FROM unnest(
            $1::TEXT[], $2::TEXT[], $3::INTEGER[], $4::INTEGER[], $5::INTEGER[],
            $6::TEXT[], $7::TEXT[], $8::TIMESTAMPTZ[], $9::TEXT[], $10::TEXT[],
            $11::TEXT[], $12::TIMESTAMPTZ[], $13::TEXT[]
        ) ON CONFLICT (post_id) DO NOTHING
        """,
        [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
        [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows],
        [r[6] for r in rows], [r[7] for r in rows], [r[8] for r in rows],
        [r[9] for r in rows], [r[10] for r in rows], [r[11] for r in rows],
        [r[12] for r in rows],
    )


async def insert_cari_threads(conn: asyncpg.Connection, df: pd.DataFrame):
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
    if not seen:
        return
    v = list(seen.values())
    await conn.execute(
        """
        INSERT INTO cari_threads
            (thread_id, thread_title, thread_url, forum,
             thread_total_pages, thread_status, fetched_at)
        SELECT * FROM unnest(
            $1::TEXT[], $2::TEXT[], $3::TEXT[], $4::TEXT[],
            $5::INTEGER[], $6::TEXT[], $7::TIMESTAMPTZ[]
        ) ON CONFLICT (thread_id) DO NOTHING
        """,
        [x[0] for x in v], [x[1] for x in v], [x[2] for x in v], [x[3] for x in v],
        [x[4] for x in v], [x[5] for x in v], [x[6] for x in v],
    )


async def insert_cari_posts(conn: asyncpg.Connection, df: pd.DataFrame):
    rows = [
        (
            _str(r.post_id), _str(r.thread_id),
            _int(r.post_floor, pg_int=True), _int(r.page_number, pg_int=True),
            _int(r.page_offset, pg_int=True), _str(r.author), _str(r.author_id),
            _parse_cari_ts(_nan(r.posted_at)),
            _nan(r.cleaned_text), _str(r.body_html), _str(r.quoted_post_id),
            None,  # fetched_at
            _str(r.error_reason),
        )
        for r in df.itertuples(index=False)
        if _str(r.post_id) is not None
    ]
    if not rows:
        return
    await conn.execute(
        """
        INSERT INTO cari_posts
            (post_id, thread_id, post_floor, page_number, page_offset,
             author, author_id, posted_at, body_text, body_html,
             quoted_post_id, fetched_at, error_reason)
        SELECT * FROM unnest(
            $1::TEXT[], $2::TEXT[], $3::INTEGER[], $4::INTEGER[], $5::INTEGER[],
            $6::TEXT[], $7::TEXT[], $8::TIMESTAMPTZ[], $9::TEXT[], $10::TEXT[],
            $11::TEXT[], $12::TIMESTAMPTZ[], $13::TEXT[]
        ) ON CONFLICT (post_id) DO NOTHING
        """,
        [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
        [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows],
        [r[6] for r in rows], [r[7] for r in rows], [r[8] for r in rows],
        [r[9] for r in rows], [r[10] for r in rows], [r[11] for r in rows],
        [r[12] for r in rows],
    )


async def insert_reddit_posts(conn: asyncpg.Connection, df: pd.DataFrame, table: str):
    rows = [
        (
            _str(r.post_id), _str(r.post_kind), _str(r.submission_id), _str(r.parent_id),
            _str(r.subreddit), _str(r.author), _str(r.title),
            _int(r.score, pg_int=True), _int(r.num_comments, pg_int=True),
            _nan(r.created_utc), _str(r.permalink), _str(r.url), _str(r.month),
            _nan(r.cleaned_text),
        )
        for r in df.itertuples(index=False)
        if _str(r.post_id) is not None
    ]
    if not rows:
        return
    await conn.execute(
        f"""
        INSERT INTO {table}
            (post_id, post_kind, submission_id, parent_id, subreddit, author, title,
             score, num_comments, created_utc, permalink, url, month, body)
        SELECT * FROM unnest(
            $1::TEXT[], $2::TEXT[], $3::TEXT[], $4::TEXT[], $5::TEXT[], $6::TEXT[], $7::TEXT[],
            $8::INTEGER[], $9::INTEGER[], $10::TIMESTAMPTZ[], $11::TEXT[], $12::TEXT[], $13::TEXT[],
            $14::TEXT[]
        ) ON CONFLICT (post_id) DO NOTHING
        """,
        [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
        [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows],
        [r[6] for r in rows], [r[7] for r in rows], [r[8] for r in rows],
        [r[9] for r in rows], [r[10] for r in rows], [r[11] for r in rows],
        [r[12] for r in rows], [r[13] for r in rows],
    )


async def insert_hplt(conn: asyncpg.Connection, df: pd.DataFrame, table: str):
    """Plain INSERT via COPY — pre-flight guarantees empty table. No content column."""
    records = [
        (
            _str(r.id),
            _str(r.url),
            _parse_hplt_ts(_nan(r.timestamp)),
            _str(r.crawl_id),
            _str(r.source_shard),
            _str(r.language),
            _str(r.row_language_code),
            _float(r.row_language_prob),
        )
        for r in df.itertuples(index=False)
    ]
    if not records:
        return
    await conn.copy_records_to_table(
        table,
        records=records,
        columns=["hplt_id", "url", "crawl_timestamp", "crawl_id", "source_shard",
                 "language", "row_language_code", "row_language_prob"],
    )


async def insert_ocr_books(conn: asyncpg.Connection, df: pd.DataFrame):
    rows = [
        (
            _str(r.document_id), _str(r.source_run_id), _str(r.source_format),
            _str(r.markdown_file_name), _str(r.markdown_rel_path), _str(r.markdown_sha256),
            _int(r.markdown_char_count, pg_int=True), _int(r.markdown_byte_count, pg_int=True),
            _nan(r.cleaned_text),
        )
        for r in df.itertuples(index=False)
        if _str(r.document_id) is not None
    ]
    if not rows:
        return
    await conn.execute(
        """
        INSERT INTO ocr_books
            (document_id, source_run_id, source_format, markdown_file_name,
             markdown_rel_path, markdown_sha256, markdown_char_count,
             markdown_byte_count, markdown_text)
        SELECT * FROM unnest(
            $1::TEXT[], $2::TEXT[], $3::TEXT[], $4::TEXT[], $5::TEXT[],
            $6::TEXT[], $7::INTEGER[], $8::INTEGER[], $9::TEXT[]
        ) ON CONFLICT (document_id) DO NOTHING
        """,
        [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
        [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows],
        [r[6] for r in rows], [r[7] for r in rows], [r[8] for r in rows],
    )


async def insert_fineweb(conn: asyncpg.Connection, df: pd.DataFrame):  # UNTESTED STUB
    records = [
        (None, _str(r.url), None, _str(r.month), _str(r.language), _nan(r.cleaned_text))
        for r in df.itertuples(index=False)
    ]
    if not records:
        return
    await conn.copy_records_to_table(
        "fineweb_documents",
        records=records,
        columns=["hf_id", "url", "date", "month", "language", "content"],
    )


# ── Unified tables ────────────────────────────────────────────────────────────


async def insert_unified_documents(
    conn: asyncpg.Connection, df: pd.DataFrame, source_id_map: dict
) -> dict:
    """Insert rows and return {sample_uid: doc_id} via unnest + RETURNING.

    cleaning_rules_triggered (TEXT[]) is encoded as JSON text per row to avoid
    the TEXT[][] rectangular-array constraint; decoded back to TEXT[] in SQL.
    """
    s_uid, s_hash, src_ids, srcs = [], [], [], []
    buckets, obj_keys, pq_urls, txt_cols = [], [], [], []
    rg_idxs, row_idxs, row_in_rg = [], [], []
    orig_sha, clean_sha = [], []
    orig_cc, clean_cc, rem_cc = [], [], []
    approx_orig, approx_clean, approx_rem = [], [], []
    is_dropped, rules_json = [], []
    o200k_cnt, o200k_tok, skeys = [], [], []

    for r in df.itertuples(index=False):
        src = r.cleaning_source
        s_uid.append(r.sample_uid)
        s_hash.append(r.sample_uid_hash)
        src_ids.append(source_id_map[src])
        srcs.append(src)
        buckets.append(_str(r.source_bucket))
        obj_keys.append(_str(r.source_object_key))
        pq_urls.append(_str(r.source_parquet_url))
        txt_cols.append(_str(r.text_column))
        rg_idxs.append(_int(r.source_row_group_index))
        row_idxs.append(_int(r.source_row_index))
        row_in_rg.append(_int(r.row_index_in_row_group))
        orig_sha.append(_str(r.original_text_sha256))
        clean_sha.append(_str(r.cleaned_text_sha256))
        orig_cc.append(_int(r.original_char_count))
        clean_cc.append(_int(r.cleaned_char_count))
        rem_cc.append(_int(r.removed_char_count))
        approx_orig.append(_int(r.approximate_original_token_count))
        approx_clean.append(_int(r.approximate_cleaned_token_count))
        approx_rem.append(_int(r.approximate_removed_token_count))
        is_dropped.append(_bool(r.cleaning_is_dropped))
        lst = _list_or_none(r.cleaning_rules_triggered)
        rules_json.append(json.dumps(lst) if lst is not None else None)
        o200k_cnt.append(_int(r.cleaned_o200k_token_count))
        o200k_tok.append(_str(r.cleaned_o200k_tokenizer))
        skeys.append(_sample_key(r.sample_uid_hash))

    result = await conn.fetch(
        """
        INSERT INTO unified_documents
            (sample_uid, sample_uid_hash, source_id, cleaning_source,
             source_bucket, source_object_key, source_parquet_url, text_column,
             source_row_group_index, source_row_index, row_index_in_row_group,
             original_text_sha256, cleaned_text_sha256,
             original_char_count, cleaned_char_count, removed_char_count,
             approximate_original_token_count, approximate_cleaned_token_count,
             approximate_removed_token_count, cleaning_is_dropped, cleaning_rules_triggered,
             cleaned_o200k_token_count, cleaned_o200k_tokenizer, sample_key)
        SELECT
            t.sample_uid, t.sample_uid_hash, t.source_id, t.cleaning_source,
            t.source_bucket, t.source_object_key, t.source_parquet_url, t.text_column,
            t.source_row_group_index, t.source_row_index, t.row_index_in_row_group,
            t.original_text_sha256, t.cleaned_text_sha256,
            t.original_char_count, t.cleaned_char_count, t.removed_char_count,
            t.approximate_original_token_count, t.approximate_cleaned_token_count,
            t.approximate_removed_token_count, t.cleaning_is_dropped,
            CASE WHEN t.rules_json IS NULL THEN NULL::TEXT[]
                 ELSE ARRAY(SELECT jsonb_array_elements_text(t.rules_json::JSONB)) END,
            t.cleaned_o200k_token_count, t.cleaned_o200k_tokenizer, t.sample_key
        FROM unnest(
            $1::TEXT[], $2::TEXT[], $3::SMALLINT[], $4::TEXT[],
            $5::TEXT[], $6::TEXT[], $7::TEXT[], $8::TEXT[],
            $9::BIGINT[], $10::BIGINT[], $11::BIGINT[],
            $12::TEXT[], $13::TEXT[],
            $14::BIGINT[], $15::BIGINT[], $16::BIGINT[],
            $17::BIGINT[], $18::BIGINT[], $19::BIGINT[],
            $20::BOOLEAN[], $21::TEXT[],
            $22::BIGINT[], $23::TEXT[], $24::BIGINT[]
        ) AS t(
            sample_uid, sample_uid_hash, source_id, cleaning_source,
            source_bucket, source_object_key, source_parquet_url, text_column,
            source_row_group_index, source_row_index, row_index_in_row_group,
            original_text_sha256, cleaned_text_sha256,
            original_char_count, cleaned_char_count, removed_char_count,
            approximate_original_token_count, approximate_cleaned_token_count,
            approximate_removed_token_count, cleaning_is_dropped, rules_json,
            cleaned_o200k_token_count, cleaned_o200k_tokenizer, sample_key
        )
        ON CONFLICT (sample_uid) DO NOTHING
        RETURNING doc_id, sample_uid
        """,
        s_uid, s_hash, src_ids, srcs,
        buckets, obj_keys, pq_urls, txt_cols,
        rg_idxs, row_idxs, row_in_rg,
        orig_sha, clean_sha,
        orig_cc, clean_cc, rem_cc,
        approx_orig, approx_clean, approx_rem,
        is_dropped, rules_json,
        o200k_cnt, o200k_tok, skeys,
    )
    doc_id_map = {row["sample_uid"]: row["doc_id"] for row in result}
    # For rows skipped by ON CONFLICT (resume case), fetch their existing doc_ids.
    missing = [uid for uid in s_uid if uid not in doc_id_map]
    if missing:
        existing = await conn.fetch(
            "SELECT doc_id, sample_uid FROM unified_documents WHERE sample_uid = ANY($1::TEXT[])",
            missing,
        )
        doc_id_map.update({row["sample_uid"]: row["doc_id"] for row in existing})
    return doc_id_map


async def insert_unified_texts(
    conn: asyncpg.Connection, df: pd.DataFrame, doc_id_map: dict
):
    """INSERT into the partitioned text store; ON CONFLICT DO NOTHING for safe resume."""
    rows = [
        (doc_id_map[r.sample_uid], r.cleaning_source, _nan(r.cleaned_text))
        for r in df.itertuples(index=False)
        if r.sample_uid in doc_id_map
    ]
    if not rows:
        return
    await conn.execute(
        """
        INSERT INTO unified_document_texts (doc_id, cleaning_source, cleaned_text)
        SELECT * FROM unnest($1::BIGINT[], $2::TEXT[], $3::TEXT[])
        ON CONFLICT (cleaning_source, doc_id) DO NOTHING
        """,
        [r[0] for r in rows],
        [r[1] for r in rows],
        [r[2] for r in rows],
    )


async def insert_lid_metadata(
    conn: asyncpg.Connection, df: pd.DataFrame, doc_id_map: dict
):
    rows = [
        (
            doc_id_map[r.sample_uid],
            _int(r.lid_cleaned_token_count),
            _bool_or_none(r.reference_removed),
            _str(r.reference_removal_method),
            _int(r.removed_reference_char_count),
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
    if not rows:
        return
    # JSONB columns (r[6], r[8], r[9], r[10]) are passed as JSON strings (TEXT[])
    # and cast to JSONB in the SELECT — asyncpg can't encode list[dict] as JSONB[].
    await conn.execute(
        """
        INSERT INTO lid_metadata
            (doc_id, lid_cleaned_token_count, reference_removed, reference_removal_method,
             removed_reference_char_count, lingua_primary_language, lingua_spans,
             malaya_document_label, malaya_document_scores,
             malaya_word_detections, malaya_word_label_counts)
        SELECT
            t.doc_id, t.lid_cleaned_token_count, t.reference_removed, t.reference_removal_method,
            t.removed_reference_char_count, t.lingua_primary_language,
            t.lingua_spans::JSONB,
            t.malaya_document_label,
            t.malaya_document_scores::JSONB,
            t.malaya_word_detections::JSONB,
            t.malaya_word_label_counts::JSONB
        FROM unnest(
            $1::BIGINT[], $2::BIGINT[], $3::BOOLEAN[], $4::TEXT[],
            $5::BIGINT[], $6::TEXT[], $7::TEXT[],
            $8::TEXT[], $9::TEXT[], $10::TEXT[], $11::TEXT[]
        ) AS t(
            doc_id, lid_cleaned_token_count, reference_removed, reference_removal_method,
            removed_reference_char_count, lingua_primary_language, lingua_spans,
            malaya_document_label, malaya_document_scores, malaya_word_detections, malaya_word_label_counts
        )
        ON CONFLICT (doc_id) DO NOTHING
        """,
        [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows],
        [r[3] for r in rows], [r[4] for r in rows], [r[5] for r in rows],
        [r[6] for r in rows], [r[7] for r in rows], [r[8] for r in rows],
        [r[9] for r in rows], [r[10] for r in rows],
    )


# ── Batch processor ───────────────────────────────────────────────────────────


async def process_batch(
    conn: asyncpg.Connection, df: pd.DataFrame, source_id_map: dict, dry_run: bool
):
    tr = conn.transaction()
    await tr.start()
    try:
        for src, grp in df.groupby("cleaning_source"):
            if src not in TESTED_SOURCES:
                log.warning("Skipping untested source: %s (%d rows)", src, len(grp))
                continue
            if src == "lowyat":
                await insert_lowyat_threads(conn, grp)
                await insert_lowyat_posts(conn, grp)
            elif src == "cari":
                await insert_cari_threads(conn, grp)
                await insert_cari_posts(conn, grp)
            elif src == "reddit-bolehland":
                await insert_reddit_posts(conn, grp, "reddit_bolehland_posts")
            elif src == "reddit-indonesia":
                await insert_reddit_posts(conn, grp, "reddit_indonesia_posts")
            elif src == "hplt-malay":
                await insert_hplt(conn, grp, "hplt_malay_documents")
            elif src == "hplt-indonesia":
                await insert_hplt(conn, grp, "hplt_indonesia_documents")
            elif src == "books-ocr":
                await insert_ocr_books(conn, grp)
            elif src == "fineweb":
                await insert_fineweb(conn, grp)

        doc_id_map = await insert_unified_documents(conn, df, source_id_map)
        await insert_unified_texts(conn, df, doc_id_map)
        await insert_lid_metadata(conn, df, doc_id_map)

        if dry_run:
            await tr.rollback()
        else:
            await tr.commit()
    except Exception:
        await tr.rollback()
        raise


# ── File ingestion ────────────────────────────────────────────────────────────


async def ingest_file_async(
    pf_path: Path,
    dsn: str,
    source_id_map: dict,
    batch_size: int,
    dry_run: bool,
) -> tuple[str, int]:
    fname = pf_path.name
    log.info("Starting %s", fname)

    conn = await asyncpg.connect(dsn)
    await conn.execute("SET TIME ZONE 'UTC'")
    total_rows = 0

    try:
        loop = asyncio.get_running_loop()
        duck = duckdb.connect()
        duck.execute("SET memory_limit='3GB'")
        duck.execute("SET threads=2")
        cursor = duck.execute(f"SELECT * FROM read_parquet('{pf_path}')")
        columns = [d[0] for d in cursor.description]
        batch_idx = 0
        while True:
            rows = await loop.run_in_executor(None, cursor.fetchmany, batch_size)
            if not rows:
                break
            df = pd.DataFrame(rows, columns=columns)
            n = len(df)
            await process_batch(conn, df, source_id_map, dry_run)
            del df, rows
            gc.collect()
            if _libc:
                _libc.malloc_trim(0)
            total_rows += n
            log.info("  %s batch %d: %d rows", fname, batch_idx, n)
            batch_idx += 1
        duck.close()
    finally:
        await conn.close()

    log.info("Done %s — %d rows", fname, total_rows)
    return fname, total_rows


def ingest_file(
    pf_path: Path,
    dsn: str,
    source_id_map: dict,
    batch_size: int,
    dry_run: bool,
) -> tuple[str, int]:
    """Sync wrapper — each ProcessPoolExecutor worker runs its own event loop."""
    return asyncio.run(ingest_file_async(pf_path, dsn, source_id_map, batch_size, dry_run))


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Ingest unified parquet files into corpus DB")
    p.add_argument("--parts-dir", default=DEFAULT_PARTS_DIR)
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--progress-file", default=DEFAULT_PROGRESS_FILE)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--force",
        action="store_true",
        help="Skip pre-flight emptiness checks (use when resuming a partial ingest)",
    )
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


async def main_async(args):
    parts_dir = Path(args.parts_dir)
    if not parts_dir.exists():
        sys.exit(f"parts-dir not found: {parts_dir}")

    conn = await asyncpg.connect(args.dsn)
    await conn.execute("SET TIME ZONE 'UTC'")
    source_id_map = await load_source_id_map(conn)
    scan_sources(parts_dir, source_id_map)

    if args.dry_run or args.force:
        log.info(
            "Pre-flight emptiness checks skipped (%s)",
            "--dry-run" if args.dry_run else "--force",
        )
    else:
        await preflight_checks(conn, source_id_map)
    await conn.close()

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
                    log.info(
                        "Progress: %d/%d files done",
                        len(progress["completed"]), total_files,
                    )
            except Exception as exc:
                log.error("FAILED %s: %s", pf_path.name, exc, exc_info=True)

    log.info("Ingest complete. Total rows processed: %d", total_rows)
    if args.dry_run:
        log.info("--dry-run: no data was committed")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
