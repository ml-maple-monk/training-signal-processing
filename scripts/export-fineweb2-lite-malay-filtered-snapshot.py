#!/usr/bin/env python3
"""Export the FineWeb2-Lite Malay filtered snapshot query to parquet shards."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import re
import shutil
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

import asyncpg
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_DSN = "postgresql://corpus:corpus_secret@localhost:5432/corpus"
DEFAULT_QUERY_PATH = Path(
    "src/data-storage/scripts/queries/fineweb2_lite_malay_filtered_snapshot.sql"
)
DEFAULT_OUTPUT_DIR = Path("exports/fineweb2_lite_malay_filtered_snapshot")
DEFAULT_ROWS_PER_SHARD = 100_000
DEFAULT_PREFETCH = 10_000
DEFAULT_PROFILE_NAME = "fineweb2_lite_all_lid_v1"
DEFAULT_RUN_ID = "fineweb2-lite-metadata-full-20260515T183115Z"
DEFAULT_INDEX_TABLE = "fineweb2_lite_filtered_document_index"
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("doc_id", pa.int64()),
        ("source_domain", pa.string()),
        ("lid_label", pa.string()),
        ("lid_confidence", pa.float64()),
        ("completed_text", pa.large_string()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL", DEFAULT_DSN))
    parser.add_argument("--from-index", action="store_true")
    parser.add_argument("--index-table", default=DEFAULT_INDEX_TABLE)
    parser.add_argument("--profile-name", default=DEFAULT_PROFILE_NAME)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--source", action="append", dest="sources", default=[])
    parser.add_argument("--source-concurrency", type=int, default=2)
    parser.add_argument("--rows-per-shard", type=int, default=DEFAULT_ROWS_PER_SHARD)
    parser.add_argument("--prefetch", type=int, default=DEFAULT_PREFETCH)
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["brotli", "gzip", "lz4", "none", "snappy", "zstd"],
    )
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional row limit for smoke tests. Zero exports the full query.",
    )
    return parser.parse_args()


def validate_identifier(value: str) -> str:
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return value


def clean_part(value: str) -> str:
    value = value or "unknown"
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)


def read_query(path: Path, limit: int) -> str:
    query = path.read_text().strip()
    if query.endswith(";"):
        query = query[:-1]
    if limit > 0:
        query = f"select * from ({query}) export_query_limit limit {int(limit)}"
    return query


def table_from_rows(rows: list[asyncpg.Record]) -> pa.Table:
    columns: dict[str, list[Any]] = {name: [] for name in SCHEMA.names}
    for row in rows:
        confidence = row["lid_confidence"]
        if isinstance(confidence, Decimal):
            confidence = float(confidence)
        columns["run_id"].append(row["run_id"])
        columns["doc_id"].append(row["doc_id"])
        columns["source_domain"].append(row["source_domain"])
        columns["lid_label"].append(row["lid_label"])
        columns["lid_confidence"].append(confidence)
        columns["completed_text"].append(row["completed_text"])
    return pa.table(columns, schema=SCHEMA)


def write_shard(
    *,
    output_root: Path,
    partition_key: tuple[str, str, str],
    rows: list[asyncpg.Record],
    shard_indexes: defaultdict[tuple[str, str, str], int],
    compression: str,
    compression_level: int,
) -> Path:
    run_id, source_domain, lid_label = partition_key
    output_dir = (
        output_root
        / f"run_id={clean_part(run_id)}"
        / f"source_domain={clean_part(source_domain)}"
        / f"lid_label={clean_part(lid_label)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_index = shard_indexes[partition_key]
    shard_indexes[partition_key] += 1
    output_path = output_dir / f"part-{shard_index:05d}.parquet"
    pq.write_table(
        table_from_rows(rows),
        output_path,
        compression=None if compression == "none" else compression,
        compression_level=compression_level if compression != "none" else None,
        use_dictionary=True,
        write_statistics=True,
    )
    return output_path


def swap_output(temp_dir: Path, output_dir: Path) -> None:
    old_dir = output_dir.with_name(f".{output_dir.name}.previous")
    if old_dir.exists():
        shutil.rmtree(old_dir)
    if output_dir.exists():
        output_dir.rename(old_dir)
    temp_dir.rename(output_dir)
    if old_dir.exists():
        shutil.rmtree(old_dir)


async def export(args: argparse.Namespace) -> None:
    query = read_query(args.query, args.limit)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp-{timestamp}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    buffers: defaultdict[tuple[str, str, str], list[asyncpg.Record]] = defaultdict(list)
    shard_indexes: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    partition_counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    total_rows = 0
    total_files = 0

    conn = await asyncpg.connect(args.dsn)
    try:
        await conn.execute("SET statement_timeout = '0'")
        async with conn.transaction(readonly=True):
            async for row in conn.cursor(query, prefetch=args.prefetch):
                key = (row["run_id"], row["source_domain"], row["lid_label"])
                buffers[key].append(row)
                partition_counts[key] += 1
                total_rows += 1
                if len(buffers[key]) >= args.rows_per_shard:
                    path = write_shard(
                        output_root=temp_dir,
                        partition_key=key,
                        rows=buffers[key],
                        shard_indexes=shard_indexes,
                        compression=args.compression,
                        compression_level=args.compression_level,
                    )
                    total_files += 1
                    buffers[key].clear()
                    print(f"wrote {path} total_rows={total_rows}", flush=True)
                if total_rows and total_rows % 250_000 == 0:
                    print(f"progress rows={total_rows} files={total_files}", flush=True)

        for key in sorted(buffers):
            if not buffers[key]:
                continue
            path = write_shard(
                output_root=temp_dir,
                partition_key=key,
                rows=buffers[key],
                shard_indexes=shard_indexes,
                compression=args.compression,
                compression_level=args.compression_level,
            )
            total_files += 1
            buffers[key].clear()
            print(f"wrote {path} total_rows={total_rows}", flush=True)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    finally:
        await conn.close()

    swap_output(temp_dir, output_dir)
    print(f"EXPORT_ROOT={output_dir}")
    print(f"ROWS={total_rows}")
    print(f"FILES={total_files}")
    for key in sorted(partition_counts):
        print(
            "PARTITION "
            f"run_id={key[0]} source_domain={key[1]} lid_label={key[2]} "
            f"rows={partition_counts[key]}"
        )


async def list_index_sources(args: argparse.Namespace) -> list[str]:
    if args.sources:
        return sorted(set(args.sources))
    index_table = validate_identifier(args.index_table)
    conn = await asyncpg.connect(args.dsn)
    try:
        rows = await conn.fetch(
            f"""
            SELECT DISTINCT source_domain
            FROM {index_table}
            WHERE profile_name = $1
              AND run_id = $2
            ORDER BY source_domain
            """,
            args.profile_name,
            args.run_id,
        )
    finally:
        await conn.close()
    return [str(row["source_domain"]) for row in rows]


def build_index_export_query(index_table: str, source_limit: int) -> str:
    limit_clause = f"LIMIT {int(source_limit)}" if source_limit > 0 else ""
    return f"""
        SELECT
            i.run_id,
            i.doc_id,
            i.source_domain,
            i.lid_label,
            i.lid_confidence,
            t.cleaned_text AS completed_text
        FROM {index_table} i
        JOIN unified_document_texts t
          ON t.cleaning_source = i.source_domain
         AND t.doc_id = i.doc_id
        WHERE i.profile_name = $1
          AND i.run_id = $2
          AND i.source_domain = $3
          AND t.cleaned_text IS NOT NULL
        {limit_clause}
    """


async def export_index_source(
    *,
    args: argparse.Namespace,
    temp_dir: Path,
    source_domain: str,
) -> tuple[int, int, dict[tuple[str, str, str], int]]:
    index_table = validate_identifier(args.index_table)
    query = build_index_export_query(index_table, args.limit)
    buffers: defaultdict[tuple[str, str, str], list[asyncpg.Record]] = defaultdict(list)
    shard_indexes: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    partition_counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    total_rows = 0
    total_files = 0

    conn = await asyncpg.connect(args.dsn)
    try:
        await conn.execute("SET statement_timeout = '0'")
        async with conn.transaction(readonly=True):
            async for row in conn.cursor(
                query,
                args.profile_name,
                args.run_id,
                source_domain,
                prefetch=args.prefetch,
            ):
                key = (row["run_id"], row["source_domain"], row["lid_label"])
                buffers[key].append(row)
                partition_counts[key] += 1
                total_rows += 1
                if len(buffers[key]) >= args.rows_per_shard:
                    path = write_shard(
                        output_root=temp_dir,
                        partition_key=key,
                        rows=buffers[key],
                        shard_indexes=shard_indexes,
                        compression=args.compression,
                        compression_level=args.compression_level,
                    )
                    total_files += 1
                    buffers[key].clear()
                    print(
                        f"wrote {path} source={source_domain} source_rows={total_rows}",
                        flush=True,
                    )
                if total_rows and total_rows % 250_000 == 0:
                    print(
                        f"progress source={source_domain} rows={total_rows} files={total_files}",
                        flush=True,
                    )

        for key in sorted(buffers):
            if not buffers[key]:
                continue
            path = write_shard(
                output_root=temp_dir,
                partition_key=key,
                rows=buffers[key],
                shard_indexes=shard_indexes,
                compression=args.compression,
                compression_level=args.compression_level,
            )
            total_files += 1
            buffers[key].clear()
            print(
                f"wrote {path} source={source_domain} source_rows={total_rows}",
                flush=True,
            )
    finally:
        await conn.close()

    return total_rows, total_files, dict(partition_counts)


async def export_from_index(args: argparse.Namespace) -> None:
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp-{timestamp}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    sources = await list_index_sources(args)
    if not sources:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ValueError("No source domains found in the filtered document index.")

    total_rows = 0
    total_files = 0
    partition_counts: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    semaphore = asyncio.Semaphore(args.source_concurrency)

    async def run_source(source: str) -> tuple[int, int, dict[tuple[str, str, str], int]]:
        async with semaphore:
            return await export_index_source(
                args=args,
                temp_dir=temp_dir,
                source_domain=source,
            )

    try:
        for result in await asyncio.gather(*(run_source(source) for source in sources)):
            rows, files, counts = result
            total_rows += rows
            total_files += files
            for key, count in counts.items():
                partition_counts[key] += count
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    swap_output(temp_dir, output_dir)
    print(f"EXPORT_ROOT={output_dir}")
    print(f"ROWS={total_rows}")
    print(f"FILES={total_files}")
    for key in sorted(partition_counts):
        print(
            "PARTITION "
            f"run_id={key[0]} source_domain={key[1]} lid_label={key[2]} "
            f"rows={partition_counts[key]}"
        )


def main() -> None:
    args = parse_args()
    if args.rows_per_shard <= 0:
        raise ValueError("--rows-per-shard must be positive.")
    if args.prefetch <= 0:
        raise ValueError("--prefetch must be positive.")
    if args.source_concurrency <= 0:
        raise ValueError("--source-concurrency must be positive.")
    if args.from_index:
        asyncio.run(export_from_index(args))
    else:
        asyncio.run(export(args))


if __name__ == "__main__":
    main()
