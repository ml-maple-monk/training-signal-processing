#!/usr/bin/env python3
"""Refresh the FineWeb2-Lite retained document index."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import os
from pathlib import Path

import asyncpg

DEFAULT_DSN = "postgresql://corpus:corpus_secret@localhost:5432/corpus"
DEFAULT_PROFILE_NAME = "fineweb2_lite_all_lid_v1"
DEFAULT_RUN_ID = "fineweb2-lite-metadata-full-20260515T183115Z"
DEFAULT_QUERY_PATH = Path(
    "src/data-storage/scripts/queries/fineweb2_lite_filtered_document_index.sql"
)
OVERLAP_HOURS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL", DEFAULT_DSN))
    parser.add_argument("--profile-name", default=DEFAULT_PROFILE_NAME)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--mode", choices=["full", "incremental"], default="incremental")
    parser.add_argument("--overlap-hours", type=float, default=OVERLAP_HOURS)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional smoke-test limit. Refused for default profile full refresh.",
    )
    return parser.parse_args()


def read_query(path: Path) -> tuple[str, str]:
    text = path.read_text()
    query = text.strip()
    if query.endswith(";"):
        query = query[:-1]
    return query, hashlib.sha256(text.encode("utf-8")).hexdigest()


def affected_rows(command_tag: str) -> int:
    try:
        return int(command_tag.rsplit(" ", 1)[-1])
    except (IndexError, ValueError):
        return 0


async def latest_lower_watermark(
    conn: asyncpg.Connection,
    *,
    profile_name: str,
    run_id: str,
    filter_sql_sha256: str,
    overlap_hours: float,
) -> dt.datetime | None:
    upper_watermark = await conn.fetchval(
        """
        SELECT max(upper_watermark)
        FROM fineweb2_lite_filtered_document_index_refreshes
        WHERE profile_name = $1
          AND run_id = $2
          AND filter_sql_sha256 = $3
          AND status = 'success'
        """,
        profile_name,
        run_id,
        filter_sql_sha256,
    )
    if upper_watermark is None:
        return None
    return upper_watermark - dt.timedelta(hours=overlap_hours)


def build_insert_sql(query: str, *, incremental: bool, limit: int) -> str:
    watermark_filter = ""
    if incremental:
        watermark_filter = """
        WHERE ($4::timestamptz IS NULL OR greatest(
                coalesce(metadata_created_at, '-infinity'::timestamptz),
                coalesce(lid_detected_at, '-infinity'::timestamptz)
            ) >= $4::timestamptz)
          AND greatest(
                coalesce(metadata_created_at, '-infinity'::timestamptz),
                coalesce(lid_detected_at, '-infinity'::timestamptz)
            ) <= $5::timestamptz
        """
    limit_clause = f"LIMIT {int(limit)}" if limit > 0 else ""
    return f"""
        INSERT INTO fineweb2_lite_filtered_document_index (
            profile_name,
            run_id,
            doc_id,
            source_doc_id,
            source_domain,
            lid_label,
            lid_confidence,
            metadata_created_at,
            lid_detected_at,
            filter_sql_sha256,
            indexed_at
        )
        SELECT
            profile_name,
            run_id,
            doc_id,
            source_doc_id,
            source_domain,
            lid_label,
            lid_confidence,
            metadata_created_at,
            lid_detected_at,
            $3::text AS filter_sql_sha256,
            now() AS indexed_at
        FROM ({query}) retained
        {watermark_filter}
        {limit_clause}
        ON CONFLICT (profile_name, run_id, doc_id) DO UPDATE SET
            source_doc_id = EXCLUDED.source_doc_id,
            source_domain = EXCLUDED.source_domain,
            lid_label = EXCLUDED.lid_label,
            lid_confidence = EXCLUDED.lid_confidence,
            metadata_created_at = EXCLUDED.metadata_created_at,
            lid_detected_at = EXCLUDED.lid_detected_at,
            filter_sql_sha256 = EXCLUDED.filter_sql_sha256,
            indexed_at = EXCLUDED.indexed_at
    """


async def refresh(args: argparse.Namespace) -> None:
    if args.limit < 0:
        raise ValueError("--limit must be non-negative.")
    if args.overlap_hours < 0:
        raise ValueError("--overlap-hours must be non-negative.")
    if args.limit and args.mode == "full" and args.profile_name == DEFAULT_PROFILE_NAME:
        raise ValueError("Refusing limited full refresh of the default profile.")

    query, filter_hash = read_query(args.query)
    conn = await asyncpg.connect(args.dsn)
    refresh_id = None
    started_at = dt.datetime.now(dt.UTC)
    upper_watermark = None
    lower_watermark = None
    try:
        upper_watermark = await conn.fetchval("SELECT now()")
        if args.mode == "incremental":
            lower_watermark = await latest_lower_watermark(
                conn,
                profile_name=args.profile_name,
                run_id=args.run_id,
                filter_sql_sha256=filter_hash,
                overlap_hours=args.overlap_hours,
            )
        refresh_id = await conn.fetchval(
            """
            INSERT INTO fineweb2_lite_filtered_document_index_refreshes (
                profile_name,
                run_id,
                mode,
                status,
                started_at,
                lower_watermark,
                upper_watermark,
                filter_sql_sha256
            )
            VALUES ($1, $2, $3, 'running', $4, $5, $6, $7)
            RETURNING refresh_id
            """,
            args.profile_name,
            args.run_id,
            args.mode,
            started_at,
            lower_watermark,
            upper_watermark,
            filter_hash,
        )

        rows_deleted = 0
        insert_sql = build_insert_sql(
            query,
            incremental=args.mode == "incremental",
            limit=args.limit,
        )
        async with conn.transaction():
            if args.mode == "full":
                rows_deleted = affected_rows(
                    await conn.execute(
                        """
                        DELETE FROM fineweb2_lite_filtered_document_index
                        WHERE profile_name = $1
                          AND run_id = $2
                        """,
                        args.profile_name,
                        args.run_id,
                    )
                )
                insert_tag = await conn.execute(
                    insert_sql,
                    args.profile_name,
                    args.run_id,
                    filter_hash,
                )
            else:
                # Retraction: a doc indexed as valid in a prior refresh can
                # later enter document_near_duplicates (Phase 2 dedup runs
                # asynchronously). Its metadata/LID timestamps do not change,
                # so the watermark upsert below never re-examines it. Without
                # this DELETE, stale duplicates accumulate in the index
                # forever (only --mode full purged them). Unconditional and
                # idempotent: removes any row for this profile/run whose
                # doc_id is now a known near-duplicate.
                rows_deleted = affected_rows(
                    await conn.execute(
                        """
                        DELETE FROM fineweb2_lite_filtered_document_index fi
                        WHERE fi.profile_name = $1
                          AND fi.run_id = $2
                          AND EXISTS (
                              SELECT 1 FROM document_near_duplicates nd
                              WHERE nd.doc_id = fi.doc_id
                          )
                        """,
                        args.profile_name,
                        args.run_id,
                    )
                )
                insert_tag = await conn.execute(
                    insert_sql,
                    args.profile_name,
                    args.run_id,
                    filter_hash,
                    lower_watermark,
                    upper_watermark,
                )
        rows_upserted = affected_rows(insert_tag)
        await conn.execute(
            """
            UPDATE fineweb2_lite_filtered_document_index_refreshes
            SET status = 'success',
                completed_at = now(),
                rows_upserted = $2,
                rows_deleted = $3
            WHERE refresh_id = $1
            """,
            refresh_id,
            rows_upserted,
            rows_deleted,
        )
        print(f"REFRESH_ID={refresh_id}")
        print(f"PROFILE={args.profile_name}")
        print(f"RUN_ID={args.run_id}")
        print(f"MODE={args.mode}")
        print(f"FILTER_SQL_SHA256={filter_hash}")
        print(f"LOWER_WATERMARK={lower_watermark}")
        print(f"UPPER_WATERMARK={upper_watermark}")
        print(f"ROWS_DELETED={rows_deleted}")
        print(f"ROWS_UPSERTED={rows_upserted}")
    except Exception as exc:
        if refresh_id is not None:
            await conn.execute(
                """
                UPDATE fineweb2_lite_filtered_document_index_refreshes
                SET status = 'failed',
                    completed_at = now(),
                    error_text = $2
                WHERE refresh_id = $1
                """,
                refresh_id,
                str(exc),
            )
        raise
    finally:
        await conn.close()


def main() -> None:
    asyncio.run(refresh(parse_args()))


if __name__ == "__main__":
    main()
