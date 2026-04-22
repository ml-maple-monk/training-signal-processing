#!/usr/bin/env python3
"""Query parquet files in Cloudflare R2 using DuckDB.

Examples:

    uv run python example/duckdb_r2_query.py --dataset local_ztd --describe
    uv run python example/duckdb_r2_query.py \\
      --dataset local_ztd \\
      --count \\
      --where "selection_bucket = 'high_diff' AND qwen_token_count >= 1000"
    uv run python example/duckdb_r2_query.py \\
      --dataset books \\
      --select "title, year, metadata_source" \\
      --where "year >= 2020" \\
      --limit 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "r2"


DATASETS: dict[str, dict[str, str | bool]] = {
    "books": {
        "pattern": "books/*.parquet",
        "hive_partitioning": False,
        "select_prefix": "'books' AS dataset_name",
    },
    "local_ztd": {
        "pattern": "local_ztd/*/selection_bucket=*/*.parquet",
        "hive_partitioning": True,
        "select_prefix": (
            "'local_ztd' AS dataset_name, "
            "regexp_extract(filename, '.*/local_ztd/([^/]+)/selection_bucket=.*', 1) AS path_lang"
        ),
    },
    "seapile_v2": {
        "pattern": "seapile_v2/*/selection_bucket=*/*.parquet",
        "hive_partitioning": True,
        "select_prefix": (
            "'seapile_v2' AS dataset_name, "
            "regexp_extract(filename, '.*/seapile_v2/([^/]+)/selection_bucket=.*', 1) AS path_lang"
        ),
    },
    "subtitle": {
        "pattern": "subtitle/*.parquet",
        "hive_partitioning": False,
        "select_prefix": "'subtitle' AS dataset_name",
    },
    "zhihu_article": {
        "pattern": "zhihu_article/*.parquet",
        "hive_partitioning": False,
        "select_prefix": "'zhihu_article' AS dataset_name",
    },
    "zhihu_qa": {
        "pattern": "zhihu_qa/*.parquet",
        "hive_partitioning": False,
        "select_prefix": "'zhihu_qa' AS dataset_name",
    },
}


def parse_config_file(path: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()
    return config


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def connect_to_r2(config_path: Path) -> duckdb.DuckDBPyConnection:
    config = parse_config_file(config_path)

    required = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "MLFLOW_S3_ENDPOINT_URL",
    ]
    missing = [key for key in required if not config.get(key)]
    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(f"Missing required config keys in {config_path}: {missing_list}")

    endpoint = (
        config["MLFLOW_S3_ENDPOINT_URL"]
        .removeprefix("https://")
        .removeprefix("http://")
        .rstrip("/")
    )

    con = duckdb.connect()
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute(
        f"""
        CREATE OR REPLACE SECRET r2_secret (
            TYPE s3,
            PROVIDER config,
            KEY_ID {sql_literal(config['AWS_ACCESS_KEY_ID'])},
            SECRET {sql_literal(config['AWS_SECRET_ACCESS_KEY'])},
            REGION {sql_literal(config['AWS_DEFAULT_REGION'])},
            ENDPOINT {sql_literal(endpoint)},
            URL_STYLE 'path'
        )
        """
    )
    return con


def build_dataset_scan(dataset: str, bucket: str, prefix: str) -> str:
    spec = DATASETS[dataset]
    full_path = f"s3://{bucket}/{prefix.rstrip('/')}/{spec['pattern']}"
    hive_partitioning = "true" if spec["hive_partitioning"] else "false"
    return f"""
        SELECT
            {spec['select_prefix']},
            *
        FROM read_parquet(
            {sql_literal(full_path)},
            filename = true,
            union_by_name = true,
            hive_partitioning = {hive_partitioning}
        )
    """


def print_result(con: duckdb.DuckDBPyConnection, query: str) -> None:
    rows = con.execute(query).fetchall()
    headers = [column[0] for column in con.description]
    print("\t".join(headers))
    for row in rows:
        print("\t".join("" if value is None else str(value) for value in row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query one parquet dataset family from Cloudflare R2 with DuckDB."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bucket", default="gpu-poor")
    parser.add_argument("--prefix", default="dataset/raw")
    parser.add_argument("--select", default="*")
    parser.add_argument("--where", default="")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--show-sql", action="store_true")
    args = parser.parse_args()

    con = connect_to_r2(args.config)
    scan_sql = build_dataset_scan(args.dataset, args.bucket, args.prefix)

    if args.describe:
        query = f"DESCRIBE SELECT * FROM ({scan_sql}) AS dataset_scan"
    elif args.count:
        where_clause = f"WHERE {args.where}" if args.where else ""
        query = f"""
            SELECT count(*) AS row_count
            FROM ({scan_sql}) AS dataset_scan
            {where_clause}
        """
    else:
        where_clause = f"WHERE {args.where}" if args.where else ""
        query = f"""
            SELECT {args.select}
            FROM ({scan_sql}) AS dataset_scan
            {where_clause}
            LIMIT {args.limit}
        """

    if args.show_sql:
        print(query.strip())
        print()

    print_result(con, query)


if __name__ == "__main__":
    main()

