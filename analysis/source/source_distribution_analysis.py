from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from numbers import Number
from pathlib import Path
from time import sleep
from typing import Any

REQUIRED_COLUMNS = (
    "source_name",
    "cleaning_source",
    "cleaned_text",
    "cleaned_o200k_token_count",
    "cleaning_is_dropped",
    "cleaned_char_count",
    "lingua_primary_language",
    "lingua_spans",
    "malaya_document_label",
    "malaya_word_label_counts",
)
ACCOUNTING_TABLE_HEADER = (
    "source",
    "cleaning_source",
    "final_r2_prefix",
    "text_column",
    "token_count",
    "byte_count",
    "sample_count",
    "dropped_sample_count",
    "source_object_count",
    "original_source_glob",
    "filters",
)
WORD_REGEX = r"[\p{L}][\p{L}\p{N}'_-]*"
DEFAULT_FIGURE_FORMATS = ("pdf", "png")
DEFAULT_STOPWORDS_FILE = Path(__file__).with_name("stopwords.txt")


@dataclass(frozen=True)
class SourceAccounting:
    source: str
    cleaning_source: str
    final_r2_prefix: str
    text_column: str
    token_count: int
    byte_count: int
    sample_count: int
    dropped_sample_count: int
    source_object_count: int | None
    original_source_glob: str
    filters: str


@dataclass(frozen=True)
class AccountingReport:
    final_r2_url: str
    final_bucket: str
    final_dataset_prefix: str
    final_parts_prefix: str
    parquet_parts: int
    cleaned_text_column: str
    token_column: str
    sources: tuple[SourceAccounting, ...]
    totals: dict[str, int]


@dataclass(frozen=True)
class RunInputs:
    parquet_paths: tuple[str, ...]
    parquet_keys: tuple[str, ...]
    source: str
    bucket: str
    prefix: str


@dataclass
class FigureArtifact:
    stem: str
    title: str
    paths: list[str]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    figure_formats = tuple(
        item.strip().lower() for item in str(args.figure_formats).split(",") if item.strip()
    )
    if not figure_formats:
        raise ValueError("--figure-formats must include at least one format.")
    if args.self_test:
        run_self_test(output_dir=output_dir, figure_formats=figure_formats)
        return 0
    run_pipeline(
        accounting_md=resolve_existing_path(Path(args.accounting_md)),
        r2_config=Path(args.r2_config),
        output_dir=output_dir,
        figure_formats=figure_formats,
        metadata_only=bool(args.metadata_only),
        local_parquet_glob=args.local_parquet_glob,
        strict_accounting=not bool(args.no_strict_accounting),
        top_words=int(args.top_words),
        wordcloud_words=int(args.wordcloud_words),
        sample_rows_per_source=int(args.sample_rows_per_source),
        min_word_chars=int(args.min_word_chars),
        max_parts=int(args.max_parts),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze source, LID, token, and word distributions for the final dataset."
    )
    parser.add_argument("--accounting-md", default="analysis/final_merged_dataset_accounting.md")
    parser.add_argument("--r2-config", default="r2")
    parser.add_argument("--output-dir", default="analysis/source/outputs")
    parser.add_argument("--figure-formats", default=",".join(DEFAULT_FIGURE_FORMATS))
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--local-parquet-glob",
        default="",
        help="Use local parquet files instead of R2, mainly for fixture validation.",
    )
    parser.add_argument("--top-words", type=int, default=40)
    parser.add_argument("--wordcloud-words", type=int, default=250)
    parser.add_argument("--sample-rows-per-source", type=int, default=5)
    parser.add_argument("--min-word-chars", type=int, default=2)
    parser.add_argument("--max-parts", type=int, default=0, help="Debug/smoke-test limit.")
    parser.add_argument("--no-strict-accounting", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def run_pipeline(
    *,
    accounting_md: Path,
    r2_config: Path,
    output_dir: Path,
    figure_formats: tuple[str, ...],
    metadata_only: bool,
    local_parquet_glob: str,
    strict_accounting: bool,
    top_words: int,
    wordcloud_words: int,
    sample_rows_per_source: int,
    min_word_chars: int,
    max_parts: int,
) -> None:
    import duckdb

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    work_dir = output_dir / "work"
    for directory in (tables_dir, figures_dir, work_dir):
        directory.mkdir(parents=True, exist_ok=True)

    accounting = parse_accounting_markdown(accounting_md)
    inputs = resolve_run_inputs(
        accounting=accounting,
        r2_config=r2_config,
        local_parquet_glob=local_parquet_glob,
    )
    if max_parts > 0:
        inputs = RunInputs(
            parquet_paths=inputs.parquet_paths[:max_parts],
            parquet_keys=inputs.parquet_keys[:max_parts],
            source=inputs.source,
            bucket=inputs.bucket,
            prefix=inputs.prefix,
        )
    if max_parts <= 0 and len(inputs.parquet_paths) != accounting.parquet_parts:
        raise ValueError(
            f"Discovered {len(inputs.parquet_paths)} parquet parts but accounting "
            f"declares {accounting.parquet_parts}."
        )

    db_path = work_dir / "source_distribution.duckdb"
    if db_path.exists():
        db_path.unlink()
    connection = duckdb.connect(str(db_path))
    try:
        if inputs.source == "r2":
            configure_duckdb_s3(connection, read_r2_env(resolve_existing_path(r2_config)))
        schema = describe_relation(connection, inputs.parquet_paths[0])
        write_dict_rows(
            tables_dir / "schema_columns.csv",
            [{"column": name, "duckdb_type": dtype} for name, dtype in schema.items()],
        )
        validate_schema(schema)
        write_json(
            output_dir / "run_metadata.json",
            {
                "accounting_md": str(accounting_md),
                "input_source": inputs.source,
                "bucket": inputs.bucket,
                "prefix": inputs.prefix,
                "parquet_part_count": len(inputs.parquet_paths),
                "first_parquet": inputs.parquet_paths[0],
                "last_parquet": inputs.parquet_paths[-1],
                "metadata_only": metadata_only,
                "max_parts": max_parts,
                "word_regex": WORD_REGEX,
                "figure_formats": list(figure_formats),
                "required_columns": list(REQUIRED_COLUMNS),
            },
        )
        if metadata_only:
            print(
                f"Validated {len(inputs.parquet_paths)} parquet parts and "
                f"{len(schema)} schema columns. Metadata-only run complete.",
                flush=True,
            )
            return

        source_order = [source.cleaning_source for source in accounting.sources]
        stopwords = load_stopwords(DEFAULT_STOPWORDS_FILE)
        word_query_limit = max(top_words, wordcloud_words)
        (
            macro,
            doc_lid_malaya,
            doc_lid_lingua,
            span_lid,
            word_lid,
            samples,
            raw_top_words,
            filtered_top_words,
        ) = compute_analysis_tables_partwise(
            connection=connection,
            inputs=inputs,
            r2_env=read_r2_env(resolve_existing_path(r2_config)) if inputs.source == "r2" else {},
            staging_dir=work_dir / "staged_parts",
            sample_rows_per_source=sample_rows_per_source,
            top_words=word_query_limit,
            stopwords=stopwords,
            min_word_chars=min_word_chars,
        )
        macro = order_sources(macro, source_order)
        comparison = compare_accounting(macro, accounting)
        write_dataframe(macro, tables_dir / "macro_source_stats.csv")
        write_dataframe(comparison, tables_dir / "accounting_comparison.csv")
        if strict_accounting:
            assert_accounting_matches(comparison)

        doc_lid_malaya = order_sources(doc_lid_malaya, source_order)
        doc_lid_lingua = order_sources(doc_lid_lingua, source_order)
        span_lid = order_sources(span_lid, source_order)
        word_lid = order_sources(word_lid, source_order)
        samples = order_sources(samples, source_order)

        write_dataframe(doc_lid_malaya, tables_dir / "document_lid_malaya.csv")
        write_dataframe(doc_lid_lingua, tables_dir / "document_lid_lingua.csv")
        write_dataframe(span_lid, tables_dir / "lingua_span_char_distribution.csv")
        write_dataframe(word_lid, tables_dir / "malaya_word_label_distribution.csv")
        write_dataframe(samples, tables_dir / "sample_rows.csv")

        raw_top_words = order_sources(raw_top_words, source_order)
        filtered_top_words = order_sources(filtered_top_words, source_order)
        write_dataframe(raw_top_words, tables_dir / "top_words_raw.csv")
        write_dataframe(filtered_top_words, tables_dir / "top_words_stopword_filtered.csv")

        figure_artifacts = build_figures(
            output_dir=figures_dir,
            figure_formats=figure_formats,
            macro=macro,
            doc_lid_malaya=doc_lid_malaya,
            doc_lid_lingua=doc_lid_lingua,
            span_lid=span_lid,
            word_lid=word_lid,
            filtered_top_words=filtered_top_words,
            wordcloud_words=wordcloud_words,
        )
        write_figure_manifest(output_dir / "figure_manifest.json", figure_artifacts)
        write_report(
            output_dir / "report.md",
            accounting=accounting,
            inputs=inputs,
            macro=macro,
            figure_artifacts=figure_artifacts,
        )
        print(
            f"Analysis complete: {len(figure_artifacts)} figure groups written to {figures_dir}",
            flush=True,
        )
    finally:
        connection.close()


def parse_accounting_markdown(path: Path) -> AccountingReport:
    text = path.read_text(encoding="utf-8")
    final_r2_url = required_regex(
        r"Final R2 prefix:\s*\n`([^`]+)`",
        text,
        "Final R2 prefix",
    )
    parquet_parts = parse_number(
        required_regex(r"Parquet parts:\s*`?([0-9,]+)`?", text, "Parquet parts")
    )
    cleaned_text_column = required_regex(
        r"Cleaned text column:\s*`([^`]+)`",
        text,
        "Cleaned text column",
    )
    token_column = required_regex(r"Token column:\s*`([^`]+)`", text, "Token column")
    sources, total_row = parse_source_table(text)
    if not sources:
        raise ValueError(f"No source rows found in {path}.")
    totals = parse_additional_totals(text)
    totals.setdefault("sample_count", total_row.sample_count)
    totals.setdefault("dropped_sample_count", total_row.dropped_sample_count)
    totals.setdefault("cleaned_o200k_token_count", total_row.token_count)
    totals.setdefault("cleaned_text_byte_count", total_row.byte_count)
    final_bucket, final_dataset_prefix = parse_r2_url(final_r2_url)
    final_parts_prefix = sources[0].final_r2_prefix
    return AccountingReport(
        final_r2_url=final_r2_url,
        final_bucket=final_bucket,
        final_dataset_prefix=final_dataset_prefix,
        final_parts_prefix=final_parts_prefix,
        parquet_parts=parquet_parts,
        cleaned_text_column=cleaned_text_column,
        token_column=token_column,
        sources=tuple(sources),
        totals=totals,
    )


def parse_source_table(text: str) -> tuple[list[SourceAccounting], SourceAccounting]:
    lines = text.splitlines()
    header_index = None
    for index, line in enumerate(lines):
        cells = markdown_cells(line)
        if tuple(cells) == ACCOUNTING_TABLE_HEADER:
            header_index = index
            break
    if header_index is None:
        raise ValueError("Could not find source accounting table.")

    sources: list[SourceAccounting] = []
    total_row: SourceAccounting | None = None
    for line in lines[header_index + 2 :]:
        if not line.strip().startswith("|"):
            break
        cells = markdown_cells(line)
        if len(cells) != len(ACCOUNTING_TABLE_HEADER):
            continue
        row = SourceAccounting(
            source=strip_markdown(cells[0]),
            cleaning_source=strip_markdown(cells[1]),
            final_r2_prefix=strip_markdown(cells[2]),
            text_column=strip_markdown(cells[3]),
            token_count=parse_number(cells[4]),
            byte_count=parse_number(cells[5]),
            sample_count=parse_number(cells[6]),
            dropped_sample_count=parse_number(cells[7]),
            source_object_count=parse_optional_number(cells[8]),
            original_source_glob=strip_markdown(cells[9]),
            filters=strip_markdown(cells[10]),
        )
        if row.source.lower() == "total":
            total_row = row
        else:
            sources.append(row)
    if total_row is None:
        raise ValueError("Source accounting table is missing Total row.")
    return sources, total_row


def parse_additional_totals(text: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    in_section = False
    for line in text.splitlines():
        if line.strip() == "## Additional Totals":
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section or not line.strip().startswith("|"):
            continue
        cells = markdown_cells(line)
        if len(cells) != 2 or cells[0] in {"metric", "---"}:
            continue
        if cells[0].startswith("---"):
            continue
        totals[strip_markdown(cells[0])] = parse_number(cells[1])
    return totals


def markdown_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def strip_markdown(value: str) -> str:
    value = value.strip()
    if value.startswith("**") and value.endswith("**"):
        value = value[2:-2]
    if value.startswith("`") and value.endswith("`"):
        value = value[1:-1]
    if value == "``":
        return ""
    return value.strip()


def parse_number(value: str) -> int:
    cleaned = strip_markdown(value).replace(",", "").strip()
    if not cleaned:
        return 0
    return int(cleaned)


def parse_optional_number(value: str) -> int | None:
    cleaned = strip_markdown(value).replace(",", "").strip()
    return int(cleaned) if cleaned else None


def required_regex(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not parse {label}.")
    return match.group(1).strip()


def parse_r2_url(url: str) -> tuple[str, str]:
    if not url.startswith("r2://"):
        raise ValueError(f"Expected r2:// URL, got: {url}")
    rest = url.removeprefix("r2://")
    authority, _, prefix = rest.partition("/")
    bucket = authority.rsplit(":", 1)[-1]
    if not bucket or not prefix:
        raise ValueError(f"Could not parse R2 bucket and prefix from {url}.")
    return bucket, prefix


def resolve_run_inputs(
    *,
    accounting: AccountingReport,
    r2_config: Path,
    local_parquet_glob: str,
) -> RunInputs:
    if local_parquet_glob:
        paths = tuple(sorted(str(Path(path).resolve()) for path in glob.glob(local_parquet_glob)))
        if not paths:
            raise ValueError(f"No local parquet files matched {local_parquet_glob!r}.")
        return RunInputs(
            parquet_paths=paths,
            parquet_keys=tuple(Path(path).name for path in paths),
            source="local",
            bucket="",
            prefix=local_parquet_glob,
        )
    env = read_r2_env(resolve_existing_path(r2_config))
    bucket = accounting.final_bucket or env.get("R2_BUCKET_NAME", "")
    prefix = accounting.final_parts_prefix
    keys = discover_r2_parquet_keys(env=env, bucket=bucket, prefix=prefix)
    paths = tuple(f"s3://{bucket}/{key}" for key in keys)
    return RunInputs(
        parquet_paths=paths,
        parquet_keys=tuple(keys),
        source="r2",
        bucket=bucket,
        prefix=prefix,
    )


def read_r2_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def discover_r2_parquet_keys(*, env: dict[str, str], bucket: str, prefix: str) -> list[str]:
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        aws_access_key_id=env_value(env, "AWS_ACCESS_KEY_ID", "R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env_value(env, "AWS_SECRET_ACCESS_KEY", "R2_SECRET_ACCESS_KEY"),
        region_name=env_value(env, "AWS_DEFAULT_REGION", "R2_REGION"),
        endpoint_url=env_value(env, "MLFLOW_S3_ENDPOINT_URL", "R2_ENDPOINT_URL"),
        config=Config(connect_timeout=10, read_timeout=60, retries={"max_attempts": 3}),
    )
    keys: list[str] = []
    continuation_token = ""
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        response = client.list_objects_v2(**kwargs)
        keys.extend(
            item["Key"]
            for item in response.get("Contents", [])
            if str(item["Key"]).endswith(".parquet")
        )
        if not response.get("IsTruncated"):
            break
        continuation_token = str(response.get("NextContinuationToken", ""))
        if not continuation_token:
            raise ValueError("R2 listing was truncated but no continuation token was returned.")
    return sorted(keys)


def env_value(env: dict[str, str], *names: str) -> str:
    for name in names:
        value = env.get(name, "")
        if value:
            return value
    return ""


def configure_duckdb_s3(connection: Any, env: dict[str, str]) -> None:
    try:
        connection.execute("LOAD httpfs")
    except Exception:
        connection.execute("INSTALL httpfs")
        connection.execute("LOAD httpfs")
    endpoint = env_value(env, "MLFLOW_S3_ENDPOINT_URL", "R2_ENDPOINT_URL")
    endpoint = endpoint.removeprefix("https://").removeprefix("http://")
    settings = {
        "s3_region": env_value(env, "AWS_DEFAULT_REGION", "R2_REGION"),
        "s3_access_key_id": env_value(env, "AWS_ACCESS_KEY_ID", "R2_ACCESS_KEY_ID"),
        "s3_secret_access_key": env_value(env, "AWS_SECRET_ACCESS_KEY", "R2_SECRET_ACCESS_KEY"),
        "s3_endpoint": endpoint,
    }
    for key, value in settings.items():
        if not value:
            raise ValueError(f"R2 config is missing {key}.")
        connection.execute(f"SET {key} = {sql_literal(value)}")
    connection.execute("SET s3_url_style = 'path'")
    connection.execute("SET s3_use_ssl = true")
    connection.execute("SET enable_http_metadata_cache = true")
    connection.execute("SET enable_external_file_cache = false")
    connection.execute("SET disable_parquet_prefetching = true")
    connection.execute("SET http_timeout = 600")
    connection.execute("SET http_retries = 8")
    connection.execute("SET http_retry_wait_ms = 1000")
    connection.execute("SET http_retry_backoff = 2")
    connection.execute("SET httpfs_connection_caching = true")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET threads = 4")


def describe_relation(connection: Any, parquet_path: str) -> dict[str, str]:
    rows = connection.execute(
        f"DESCRIBE SELECT * FROM {parquet_relation([parquet_path])}"
    ).fetchall()
    return {str(row[0]): str(row[1]) for row in rows}


def validate_schema(schema: dict[str, str]) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in schema]
    if missing:
        raise ValueError("Final parquet schema is missing required columns: " + ", ".join(missing))


def parquet_relation(paths: Sequence[str]) -> str:
    if len(paths) == 1:
        return f"read_parquet({sql_literal(paths[0])}, union_by_name=true)"
    path_list = ", ".join(sql_literal(path) for path in paths)
    return f"read_parquet([{path_list}], union_by_name=true)"


def compute_analysis_tables_partwise(
    *,
    connection: Any,
    inputs: RunInputs,
    r2_env: dict[str, str],
    staging_dir: Path,
    sample_rows_per_source: int,
    top_words: int,
    stopwords: set[str],
    min_word_chars: int,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    initialize_distribution_work_tables(connection)
    initialize_word_count_table(connection)
    total_parts = len(inputs.parquet_paths)
    for index, path in enumerate_materialized_part_paths(
        inputs=inputs,
        r2_env=r2_env,
        staging_dir=staging_dir,
    ):
        relation = parquet_relation([path])
        insert_doc_lengths(connection, relation)
        insert_document_lid_part(
            connection,
            relation,
            kind="malaya",
            column_name="malaya_document_label",
        )
        insert_document_lid_part(
            connection,
            relation,
            kind="lingua",
            column_name="lingua_primary_language",
        )
        insert_lingua_span_part(connection, relation)
        insert_malaya_word_label_part(connection, relation)
        insert_sample_candidates(connection, relation, sample_rows_per_source)
        insert_word_count_part(connection, relation, min_word_chars)
        print(f"[part] processed aggregate+word counts {index}/{total_parts}", flush=True)
    raw_top_words, filtered_top_words = finalize_word_counts(
        connection=connection,
        top_words=top_words,
        stopwords=stopwords,
    )
    return (
        compute_macro_stats_from_doc_lengths(connection),
        finalize_document_lid(connection, kind="malaya"),
        finalize_document_lid(connection, kind="lingua"),
        finalize_lingua_span_distribution(connection),
        finalize_malaya_word_distribution(connection),
        finalize_samples(connection, sample_rows_per_source),
        raw_top_words,
        filtered_top_words,
    )


def enumerate_materialized_part_paths(
    *,
    inputs: RunInputs,
    r2_env: dict[str, str],
    staging_dir: Path,
) -> Iterable[tuple[int, str]]:
    if inputs.source != "r2":
        for index, path in enumerate(inputs.parquet_paths, start=1):
            yield index, path
        return

    import boto3
    from botocore.config import Config

    staging_dir.mkdir(parents=True, exist_ok=True)
    client = boto3.client(
        "s3",
        aws_access_key_id=env_value(r2_env, "AWS_ACCESS_KEY_ID", "R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env_value(r2_env, "AWS_SECRET_ACCESS_KEY", "R2_SECRET_ACCESS_KEY"),
        region_name=env_value(r2_env, "AWS_DEFAULT_REGION", "R2_REGION"),
        endpoint_url=env_value(r2_env, "MLFLOW_S3_ENDPOINT_URL", "R2_ENDPOINT_URL"),
        config=Config(connect_timeout=10, read_timeout=600, retries={"max_attempts": 8}),
    )
    for index, key in enumerate(inputs.parquet_keys, start=1):
        local_path = staging_dir / Path(key).name
        temp_path = local_path.with_suffix(local_path.suffix + ".download")
        for stale_path in staging_dir.glob(local_path.name + ".download*"):
            if stale_path != temp_path:
                stale_path.unlink()
        if local_path.exists():
            local_path.unlink()
        print(f"[download] {index}/{len(inputs.parquet_keys)} {key}", flush=True)
        download_s3_object_with_ranges(
            client=client,
            bucket=inputs.bucket,
            key=key,
            destination=temp_path,
        )
        temp_path.replace(local_path)
        try:
            yield index, str(local_path)
        finally:
            local_path.unlink(missing_ok=True)


def download_s3_object_with_ranges(
    *,
    client: Any,
    bucket: str,
    key: str,
    destination: Path,
    range_bytes: int = 256 * 1024 * 1024,
    stream_bytes: int = 4 * 1024 * 1024,
    max_attempts: int = 8,
) -> None:
    size = int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
    downloaded = destination.stat().st_size if destination.exists() else 0
    if downloaded > size:
        destination.unlink()
        downloaded = 0
    with destination.open("ab") as handle:
        while downloaded < size:
            start = downloaded
            end = min(size - 1, start + range_bytes - 1)
            for attempt in range(1, max_attempts + 1):
                try:
                    response = client.get_object(
                        Bucket=bucket,
                        Key=key,
                        Range=f"bytes={start}-{end}",
                    )
                    for chunk in response["Body"].iter_chunks(chunk_size=stream_bytes):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                    handle.flush()
                    break
                except Exception:
                    if downloaded > start:
                        break
                    if attempt == max_attempts:
                        raise
                    sleep(min(2**attempt, 30))
            pct = downloaded / max(size, 1)
            print(
                f"[download-progress] {Path(key).name} {downloaded:,}/{size:,} "
                f"({pct:.1%})",
                flush=True,
            )


def initialize_distribution_work_tables(connection: Any) -> None:
    for table_name in (
        "doc_lengths",
        "document_lid_parts",
        "lingua_span_parts",
        "malaya_word_label_parts",
        "sample_candidates",
    ):
        connection.execute(f"DROP TABLE IF EXISTS {table_name}")
    connection.execute(
        """
        CREATE TABLE doc_lengths(
            source_name VARCHAR,
            cleaning_source VARCHAR,
            token_count BIGINT,
            char_count BIGINT,
            is_dropped BOOLEAN
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE document_lid_parts(
            kind VARCHAR,
            source_name VARCHAR,
            cleaning_source VARCHAR,
            label VARCHAR,
            document_count BIGINT,
            token_count BIGINT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE lingua_span_parts(
            source_name VARCHAR,
            cleaning_source VARCHAR,
            label VARCHAR,
            span_count BIGINT,
            char_count BIGINT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE malaya_word_label_parts(
            source_name VARCHAR,
            cleaning_source VARCHAR,
            label VARCHAR,
            word_count BIGINT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE sample_candidates(
            source_name VARCHAR,
            cleaning_source VARCHAR,
            sample_uid VARCHAR,
            document_id VARCHAR,
            source_object_key VARCHAR,
            cleaned_o200k_token_count BIGINT,
            lingua_primary_language VARCHAR,
            malaya_document_label VARCHAR,
            cleaned_text_preview VARCHAR
        )
        """
    )


def insert_doc_lengths(connection: Any, relation: str) -> None:
    connection.execute(
        f"""
        INSERT INTO doc_lengths
        SELECT
            source_name,
            cleaning_source,
            COALESCE(cleaned_o200k_token_count, 0)::BIGINT AS token_count,
            COALESCE(cleaned_char_count, 0)::BIGINT AS char_count,
            COALESCE(cleaning_is_dropped, false) AS is_dropped
        FROM {relation}
        """
    )


def insert_document_lid_part(
    connection: Any,
    relation: str,
    *,
    kind: str,
    column_name: str,
) -> None:
    connection.execute(
        f"""
        INSERT INTO document_lid_parts
        SELECT
            {sql_literal(kind)} AS kind,
            source_name,
            cleaning_source,
            COALESCE(NULLIF({quote_identifier(column_name)}, ''), 'missing') AS label,
            COUNT(*)::BIGINT AS document_count,
            SUM(COALESCE(cleaned_o200k_token_count, 0))::BIGINT AS token_count
        FROM {relation}
        GROUP BY source_name, cleaning_source, label
        """
    )


def insert_lingua_span_part(connection: Any, relation: str) -> None:
    connection.execute(
        f"""
        INSERT INTO lingua_span_parts
        WITH spans AS (
            SELECT
                t.source_name,
                t.cleaning_source,
                COALESCE(NULLIF(span.language_label, ''), 'missing') AS label,
                GREATEST(
                    CAST(span.end_index AS BIGINT) - CAST(span.start_index AS BIGINT),
                    0
                ) AS char_count
            FROM {relation} AS t
            CROSS JOIN UNNEST(t.lingua_spans) AS u(span)
        )
        SELECT
            source_name,
            cleaning_source,
            label,
            COUNT(*)::BIGINT AS span_count,
            SUM(char_count)::BIGINT AS char_count
        FROM spans
        GROUP BY source_name, cleaning_source, label
        """
    )


def insert_malaya_word_label_part(connection: Any, relation: str) -> None:
    connection.execute(
        f"""
        INSERT INTO malaya_word_label_parts
        WITH word_labels AS (
            SELECT
                t.source_name,
                t.cleaning_source,
                COALESCE(NULLIF(label_count."label", ''), 'missing') AS label,
                CAST(label_count."count" AS BIGINT) AS word_count
            FROM {relation} AS t
            CROSS JOIN UNNEST(t.malaya_word_label_counts) AS u(label_count)
        )
        SELECT
            source_name,
            cleaning_source,
            label,
            SUM(word_count)::BIGINT AS word_count
        FROM word_labels
        GROUP BY source_name, cleaning_source, label
        """
    )


def insert_sample_candidates(
    connection: Any,
    relation: str,
    sample_rows_per_source: int,
) -> None:
    connection.execute(
        f"""
        INSERT INTO sample_candidates
        WITH ranked AS (
            SELECT
                source_name,
                cleaning_source,
                sample_uid,
                document_id,
                source_object_key,
                COALESCE(cleaned_o200k_token_count, 0)::BIGINT
                    AS cleaned_o200k_token_count,
                lingua_primary_language,
                malaya_document_label,
                substr(
                    regexp_replace(COALESCE(cleaned_text, ''), '\\s+', ' ', 'g'),
                    1,
                    500
                ) AS cleaned_text_preview,
                row_number() OVER (
                    PARTITION BY cleaning_source
                    ORDER BY COALESCE(cleaned_o200k_token_count, 0) DESC, sample_uid
                ) AS source_sample_rank
            FROM {relation}
            WHERE NOT COALESCE(cleaning_is_dropped, false)
                AND length(COALESCE(cleaned_text, '')) > 0
        )
        SELECT
            source_name,
            cleaning_source,
            sample_uid,
            document_id,
            source_object_key,
            cleaned_o200k_token_count,
            lingua_primary_language,
            malaya_document_label,
            cleaned_text_preview
        FROM ranked
        WHERE source_sample_rank <= {int(sample_rows_per_source)}
        """
    )


def compute_macro_stats_from_doc_lengths(connection: Any) -> Any:
    frame = connection.execute(
        """
        SELECT
            source_name,
            cleaning_source,
            COUNT(*)::BIGINT AS sample_count,
            SUM(CASE WHEN is_dropped THEN 1 ELSE 0 END)::BIGINT
                AS dropped_sample_count,
            SUM(CASE WHEN NOT is_dropped THEN 1 ELSE 0 END)::BIGINT
                AS kept_sample_count,
            SUM(token_count)::BIGINT AS token_count,
            SUM(char_count)::BIGINT AS char_count,
            AVG(token_count)::DOUBLE AS avg_tokens_per_doc,
            median(token_count)::DOUBLE AS median_tokens_per_doc,
            quantile_cont(token_count, 0.90)::DOUBLE AS p90_tokens_per_doc,
            quantile_cont(token_count, 0.95)::DOUBLE AS p95_tokens_per_doc,
            quantile_cont(token_count, 0.99)::DOUBLE AS p99_tokens_per_doc,
            MAX(token_count)::BIGINT AS max_tokens_per_doc,
            AVG(CASE WHEN token_count > 0 THEN char_count::DOUBLE / token_count END)::DOUBLE
                AS avg_chars_per_token
        FROM doc_lengths
        GROUP BY source_name, cleaning_source
        """
    ).df()
    total_docs = frame["sample_count"].sum()
    total_tokens = frame["token_count"].sum()
    frame["doc_share"] = safe_divide(frame["sample_count"], total_docs)
    frame["token_share"] = safe_divide(frame["token_count"], total_tokens)
    frame["drop_rate"] = safe_divide(frame["dropped_sample_count"], frame["sample_count"])
    return frame


def finalize_document_lid(connection: Any, *, kind: str) -> Any:
    frame = connection.execute(
        f"""
        SELECT
            source_name,
            cleaning_source,
            label,
            SUM(document_count)::BIGINT AS document_count,
            SUM(token_count)::BIGINT AS token_count
        FROM document_lid_parts
        WHERE kind = {sql_literal(kind)}
        GROUP BY source_name, cleaning_source, label
        """
    ).df()
    frame["document_share_within_source"] = grouped_share(
        frame, group_column="cleaning_source", value_column="document_count"
    )
    frame["token_share_within_source"] = grouped_share(
        frame, group_column="cleaning_source", value_column="token_count"
    )
    return frame


def finalize_lingua_span_distribution(connection: Any) -> Any:
    frame = connection.execute(
        """
        SELECT
            source_name,
            cleaning_source,
            label,
            SUM(span_count)::BIGINT AS span_count,
            SUM(char_count)::BIGINT AS char_count
        FROM lingua_span_parts
        GROUP BY source_name, cleaning_source, label
        """
    ).df()
    frame["char_share_within_source"] = grouped_share(
        frame, group_column="cleaning_source", value_column="char_count"
    )
    return frame


def finalize_malaya_word_distribution(connection: Any) -> Any:
    frame = connection.execute(
        """
        SELECT
            source_name,
            cleaning_source,
            label,
            SUM(word_count)::BIGINT AS word_count
        FROM malaya_word_label_parts
        GROUP BY source_name, cleaning_source, label
        """
    ).df()
    frame["word_share_within_source"] = grouped_share(
        frame, group_column="cleaning_source", value_column="word_count"
    )
    return frame


def finalize_samples(connection: Any, sample_rows_per_source: int) -> Any:
    return connection.execute(
        f"""
        WITH ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY cleaning_source
                    ORDER BY cleaned_o200k_token_count DESC, sample_uid
                ) AS source_sample_rank
            FROM sample_candidates
        )
        SELECT *
        FROM ranked
        WHERE source_sample_rank <= {int(sample_rows_per_source)}
        ORDER BY cleaning_source, source_sample_rank
        """
    ).df()


def initialize_word_count_table(connection: Any) -> None:
    connection.execute("DROP TABLE IF EXISTS word_counts_raw")
    connection.execute(
        "CREATE TABLE word_counts_raw(cleaning_source VARCHAR, word VARCHAR, count BIGINT)"
    )


def insert_word_count_part(connection: Any, relation: str, min_word_chars: int) -> None:
    connection.execute(
        f"""
        INSERT INTO word_counts_raw
        SELECT cleaning_source, word, COUNT(*)::BIGINT AS count
        FROM (
            SELECT
                cleaning_source,
                UNNEST(
                    regexp_extract_all(
                        lower(COALESCE(cleaned_text, '')),
                        {sql_literal(WORD_REGEX)}
                    )
                ) AS word
            FROM {relation}
            WHERE NOT COALESCE(cleaning_is_dropped, false)
                AND cleaned_text IS NOT NULL
        ) words
        WHERE length(word) >= {int(min_word_chars)}
        GROUP BY cleaning_source, word
        """
    )


def finalize_word_counts(
    *,
    connection: Any,
    top_words: int,
    stopwords: set[str],
) -> tuple[Any, Any]:
    connection.execute("DROP TABLE IF EXISTS word_counts")
    connection.execute(
        """
        CREATE TABLE word_counts AS
        SELECT cleaning_source, word, SUM(count)::BIGINT AS count
        FROM word_counts_raw
        GROUP BY cleaning_source, word
        """
    )
    connection.execute("CREATE OR REPLACE TEMP TABLE stopwords(word VARCHAR)")
    if stopwords:
        connection.executemany(
            "INSERT INTO stopwords VALUES (?)",
            [(word,) for word in sorted(stopwords)],
        )
    raw = connection.execute(top_words_query("word_counts", top_words)).df()
    filtered = connection.execute(
        f"""
        WITH filtered AS (
            SELECT wc.*
            FROM word_counts wc
            LEFT JOIN stopwords sw ON wc.word = sw.word
            WHERE sw.word IS NULL
        )
        {top_words_select("filtered", top_words)}
        """
    ).df()
    return raw, filtered


def compute_macro_stats(connection: Any, relation: str) -> Any:
    query = f"""
        WITH base AS (
            SELECT
                source_name,
                cleaning_source,
                COALESCE(cleaned_o200k_token_count, 0)::BIGINT AS token_count,
                COALESCE(cleaned_char_count, 0)::BIGINT AS char_count,
                COALESCE(cleaning_is_dropped, false) AS is_dropped
            FROM {relation}
        )
        SELECT
            source_name,
            cleaning_source,
            COUNT(*)::BIGINT AS sample_count,
            SUM(CASE WHEN is_dropped THEN 1 ELSE 0 END)::BIGINT AS dropped_sample_count,
            SUM(CASE WHEN NOT is_dropped THEN 1 ELSE 0 END)::BIGINT AS kept_sample_count,
            SUM(token_count)::BIGINT AS token_count,
            SUM(char_count)::BIGINT AS char_count,
            AVG(token_count)::DOUBLE AS avg_tokens_per_doc,
            median(token_count)::DOUBLE AS median_tokens_per_doc,
            quantile_cont(token_count, 0.90)::DOUBLE AS p90_tokens_per_doc,
            quantile_cont(token_count, 0.95)::DOUBLE AS p95_tokens_per_doc,
            quantile_cont(token_count, 0.99)::DOUBLE AS p99_tokens_per_doc,
            MAX(token_count)::BIGINT AS max_tokens_per_doc,
            AVG(CASE WHEN token_count > 0 THEN char_count::DOUBLE / token_count END)::DOUBLE
                AS avg_chars_per_token
        FROM base
        GROUP BY source_name, cleaning_source
    """
    frame = connection.execute(query).df()
    total_docs = frame["sample_count"].sum()
    total_tokens = frame["token_count"].sum()
    frame["doc_share"] = safe_divide(frame["sample_count"], total_docs)
    frame["token_share"] = safe_divide(frame["token_count"], total_tokens)
    frame["drop_rate"] = safe_divide(frame["dropped_sample_count"], frame["sample_count"])
    return frame


def compute_document_lid(connection: Any, relation: str, column_name: str) -> Any:
    query = f"""
        SELECT
            source_name,
            cleaning_source,
            COALESCE(NULLIF({quote_identifier(column_name)}, ''), 'missing') AS label,
            COUNT(*)::BIGINT AS document_count,
            SUM(COALESCE(cleaned_o200k_token_count, 0))::BIGINT AS token_count
        FROM {relation}
        GROUP BY source_name, cleaning_source, label
    """
    frame = connection.execute(query).df()
    frame["document_share_within_source"] = grouped_share(
        frame, group_column="cleaning_source", value_column="document_count"
    )
    frame["token_share_within_source"] = grouped_share(
        frame, group_column="cleaning_source", value_column="token_count"
    )
    return frame


def compute_lingua_span_distribution(connection: Any, relation: str) -> Any:
    query = f"""
        WITH spans AS (
            SELECT
                t.source_name,
                t.cleaning_source,
                COALESCE(NULLIF(span.language_label, ''), 'missing') AS label,
                GREATEST(
                    CAST(span.end_index AS BIGINT) - CAST(span.start_index AS BIGINT),
                    0
                ) AS char_count
            FROM {relation} AS t
            CROSS JOIN UNNEST(t.lingua_spans) AS u(span)
        )
        SELECT
            source_name,
            cleaning_source,
            label,
            COUNT(*)::BIGINT AS span_count,
            SUM(char_count)::BIGINT AS char_count
        FROM spans
        GROUP BY source_name, cleaning_source, label
    """
    frame = connection.execute(query).df()
    frame["char_share_within_source"] = grouped_share(
        frame, group_column="cleaning_source", value_column="char_count"
    )
    return frame


def compute_malaya_word_distribution(connection: Any, relation: str) -> Any:
    query = f"""
        WITH word_labels AS (
            SELECT
                t.source_name,
                t.cleaning_source,
                COALESCE(NULLIF(label_count."label", ''), 'missing') AS label,
                CAST(label_count."count" AS BIGINT) AS word_count
            FROM {relation} AS t
            CROSS JOIN UNNEST(t.malaya_word_label_counts) AS u(label_count)
        )
        SELECT
            source_name,
            cleaning_source,
            label,
            SUM(word_count)::BIGINT AS word_count
        FROM word_labels
        GROUP BY source_name, cleaning_source, label
    """
    frame = connection.execute(query).df()
    frame["word_share_within_source"] = grouped_share(
        frame, group_column="cleaning_source", value_column="word_count"
    )
    return frame


def compute_samples(connection: Any, relation: str, rows_per_source: int) -> Any:
    query = f"""
        WITH ranked AS (
            SELECT
                source_name,
                cleaning_source,
                sample_uid,
                document_id,
                source_object_key,
                cleaned_o200k_token_count,
                lingua_primary_language,
                malaya_document_label,
                substr(
                    regexp_replace(COALESCE(cleaned_text, ''), '\\s+', ' ', 'g'),
                    1,
                    500
                ) AS cleaned_text_preview,
                row_number() OVER (
                    PARTITION BY cleaning_source
                    ORDER BY COALESCE(cleaned_o200k_token_count, 0) DESC, sample_uid
                ) AS source_sample_rank
            FROM {relation}
            WHERE NOT COALESCE(cleaning_is_dropped, false)
                AND length(COALESCE(cleaned_text, '')) > 0
        )
        SELECT *
        FROM ranked
        WHERE source_sample_rank <= {int(rows_per_source)}
        ORDER BY cleaning_source, source_sample_rank
    """
    return connection.execute(query).df()


def compute_word_counts(
    *,
    connection: Any,
    parquet_paths: Sequence[str],
    top_words: int,
    stopwords: set[str],
    min_word_chars: int,
) -> tuple[Any, Any]:
    connection.execute("DROP TABLE IF EXISTS word_counts_raw")
    connection.execute(
        "CREATE TABLE word_counts_raw(cleaning_source VARCHAR, word VARCHAR, count BIGINT)"
    )
    for index, path in enumerate(parquet_paths, start=1):
        relation = parquet_relation([path])
        query = f"""
            INSERT INTO word_counts_raw
            SELECT cleaning_source, word, COUNT(*)::BIGINT AS count
            FROM (
                SELECT
                    cleaning_source,
                    UNNEST(
                        regexp_extract_all(
                            lower(COALESCE(cleaned_text, '')),
                            {sql_literal(WORD_REGEX)}
                        )
                    ) AS word
                FROM {relation}
                WHERE NOT COALESCE(cleaning_is_dropped, false)
                    AND cleaned_text IS NOT NULL
            ) words
            WHERE length(word) >= {int(min_word_chars)}
            GROUP BY cleaning_source, word
        """
        connection.execute(query)
        print(f"[word-scan] processed part {index}/{len(parquet_paths)}", flush=True)
    connection.execute("DROP TABLE IF EXISTS word_counts")
    connection.execute(
        """
        CREATE TABLE word_counts AS
        SELECT cleaning_source, word, SUM(count)::BIGINT AS count
        FROM word_counts_raw
        GROUP BY cleaning_source, word
        """
    )
    connection.execute("CREATE OR REPLACE TEMP TABLE stopwords(word VARCHAR)")
    if stopwords:
        connection.executemany(
            "INSERT INTO stopwords VALUES (?)",
            [(word,) for word in sorted(stopwords)],
        )
    raw = connection.execute(top_words_query("word_counts", top_words)).df()
    filtered = connection.execute(
        f"""
        WITH filtered AS (
            SELECT wc.*
            FROM word_counts wc
            LEFT JOIN stopwords sw ON wc.word = sw.word
            WHERE sw.word IS NULL
        )
        {top_words_select("filtered", top_words)}
        """
    ).df()
    return raw, filtered


def top_words_query(table: str, limit: int) -> str:
    return top_words_select(table, limit)


def top_words_select(table: str, limit: int) -> str:
    return f"""
        SELECT cleaning_source, word, count, rank
        FROM (
            SELECT
                cleaning_source,
                word,
                count,
                row_number() OVER (
                    PARTITION BY cleaning_source
                    ORDER BY count DESC, word ASC
                ) AS rank
            FROM {table}
        )
        WHERE rank <= {int(limit)}
        ORDER BY cleaning_source, rank
    """


def compare_accounting(macro: Any, accounting: AccountingReport) -> Any:
    import pandas as pd

    expected = pd.DataFrame(
        [
            {
                "cleaning_source": source.cleaning_source,
                "expected_source": source.source,
                "expected_sample_count": source.sample_count,
                "expected_dropped_sample_count": source.dropped_sample_count,
                "expected_token_count": source.token_count,
                "expected_byte_count": source.byte_count,
            }
            for source in accounting.sources
        ]
    )
    observed = macro[
        [
            "source_name",
            "cleaning_source",
            "sample_count",
            "dropped_sample_count",
            "token_count",
            "char_count",
        ]
    ].rename(
        columns={
            "source_name": "observed_source",
            "sample_count": "observed_sample_count",
            "dropped_sample_count": "observed_dropped_sample_count",
            "token_count": "observed_token_count",
            "char_count": "observed_char_count",
        }
    )
    comparison = expected.merge(observed, on="cleaning_source", how="outer")
    for name in ("sample_count", "dropped_sample_count", "token_count"):
        comparison[f"{name}_delta"] = (
            comparison[f"observed_{name}"] - comparison[f"expected_{name}"]
        )
    comparison["matches_accounting"] = (
        (comparison["sample_count_delta"] == 0)
        & (comparison["dropped_sample_count_delta"] == 0)
        & (comparison["token_count_delta"] == 0)
    )
    return comparison


def assert_accounting_matches(comparison: Any) -> None:
    failures = comparison[~comparison["matches_accounting"].fillna(False)]
    if not failures.empty:
        details = failures[
            [
                "cleaning_source",
                "sample_count_delta",
                "dropped_sample_count_delta",
                "token_count_delta",
            ]
        ].to_dict(orient="records")
        raise ValueError("Aggregated parquet totals do not match accounting: " + repr(details))


def build_figures(
    *,
    output_dir: Path,
    figure_formats: tuple[str, ...],
    macro: Any,
    doc_lid_malaya: Any,
    doc_lid_lingua: Any,
    span_lid: Any,
    word_lid: Any,
    filtered_top_words: Any,
    wordcloud_words: int,
) -> list[FigureArtifact]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter
    from wordcloud import WordCloud

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    artifacts: list[FigureArtifact] = []
    macro_sorted = macro.sort_values("token_count", ascending=True)
    source_labels = macro_sorted["source_name"].tolist()
    colors = [
        "#2364AA",
        "#3DA5D9",
        "#73BFB8",
        "#FEC601",
        "#EA7317",
        "#8A5A44",
        "#5C677D",
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.barh(source_labels, macro_sorted["sample_count"], color=colors[: len(source_labels)])
    ax.xaxis.set_major_formatter(FuncFormatter(compact_number))
    ax.set_xlabel("Documents")
    ax.set_title("Document Count by Source")
    artifacts.append(
        save_figure(fig, output_dir, "source_document_counts", "Document counts", figure_formats)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.barh(source_labels, macro_sorted["token_count"], color=colors[: len(source_labels)])
    ax.xaxis.set_major_formatter(FuncFormatter(compact_number))
    ax.set_xlabel("o200k_base tokens")
    ax.set_title("Token Count by Source")
    artifacts.append(
        save_figure(fig, output_dir, "source_token_counts", "Token counts", figure_formats)
    )

    share = macro.sort_values("token_count", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    x_positions = range(len(share))
    width = 0.38
    ax.bar([x - width / 2 for x in x_positions], share["doc_share"], width, label="Documents")
    ax.bar([x + width / 2 for x in x_positions], share["token_share"], width, label="Tokens")
    ax.set_xticks(list(x_positions), share["source_name"], rotation=30, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Share of final dataset")
    ax.set_title("Document Share vs Token Share")
    ax.legend(frameon=False)
    artifacts.append(
        save_figure(
            fig,
            output_dir,
            "source_doc_token_share",
            "Document/token share",
            figure_formats,
        )
    )

    length = macro.sort_values("avg_tokens_per_doc", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.barh(length["source_name"], length["avg_tokens_per_doc"], color="#3DA5D9")
    ax.xaxis.set_major_formatter(FuncFormatter(compact_number))
    ax.set_xlabel("Average tokens per document")
    ax.set_title("Average Document Length by Source")
    artifacts.append(
        save_figure(
            fig,
            output_dir,
            "source_average_doc_length",
            "Average document length",
            figure_formats,
        )
    )

    pct = macro.sort_values("p95_tokens_per_doc", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    x = range(len(pct))
    ax.plot(x, pct["median_tokens_per_doc"], marker="o", label="median")
    ax.plot(x, pct["p90_tokens_per_doc"], marker="o", label="p90")
    ax.plot(x, pct["p95_tokens_per_doc"], marker="o", label="p95")
    ax.plot(x, pct["p99_tokens_per_doc"], marker="o", label="p99")
    ax.set_xticks(list(x), pct["source_name"], rotation=30, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(compact_number))
    ax.set_ylabel("Tokens per document")
    ax.set_title("Document Length Percentiles")
    ax.legend(frameon=False, ncols=4)
    artifacts.append(
        save_figure(
            fig,
            output_dir,
            "source_doc_length_percentiles",
            "Document length percentiles",
            figure_formats,
        )
    )

    dropped = macro.sort_values("drop_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    ax.barh(dropped["source_name"], dropped["drop_rate"], color="#EA7317")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Dropped row rate")
    ax.set_title("Dropped Rows by Source")
    artifacts.append(save_figure(fig, output_dir, "source_drop_rate", "Drop rate", figure_formats))

    artifacts.append(
        heatmap_figure(
            output_dir=output_dir,
            figure_formats=figure_formats,
            frame=doc_lid_malaya,
            value_column="document_share_within_source",
            label_column="label",
            stem="document_lid_malaya_heatmap",
            title="Document LID Distribution: Malaya",
            colorbar_label="Document share within source",
        )
    )
    artifacts.append(
        heatmap_figure(
            output_dir=output_dir,
            figure_formats=figure_formats,
            frame=doc_lid_lingua,
            value_column="document_share_within_source",
            label_column="label",
            stem="document_lid_lingua_heatmap",
            title="Document LID Distribution: Lingua",
            colorbar_label="Document share within source",
        )
    )
    artifacts.append(
        heatmap_figure(
            output_dir=output_dir,
            figure_formats=figure_formats,
            frame=span_lid,
            value_column="char_share_within_source",
            label_column="label",
            stem="lingua_span_char_heatmap",
            title="Lingua Span Character Distribution",
            colorbar_label="Character share within source",
        )
    )
    artifacts.append(
        heatmap_figure(
            output_dir=output_dir,
            figure_formats=figure_formats,
            frame=word_lid,
            value_column="word_share_within_source",
            label_column="label",
            stem="malaya_word_lid_heatmap",
            title="Word-Level LID Distribution: Malaya",
            colorbar_label="Word share within source",
        )
    )

    for cleaning_source, group in filtered_top_words.groupby("cleaning_source", sort=False):
        group = group.sort_values("count", ascending=True).tail(20)
        source_name = source_name_for_cleaning_source(macro, cleaning_source)
        fig, ax = plt.subplots(figsize=(8.5, 6), constrained_layout=True)
        ax.barh(group["word"], group["count"], color="#2364AA")
        ax.xaxis.set_major_formatter(FuncFormatter(compact_number))
        ax.set_xlabel("Exact word count")
        ax.set_title(f"Top Words: {source_name}")
        stem = f"top_words_{slugify(cleaning_source)}"
        artifacts.append(
            save_figure(fig, output_dir, stem, f"Top words: {source_name}", figure_formats)
        )

        frequencies = {
            str(word): int(count)
            for word, count in group.sort_values("count", ascending=False)
            .head(wordcloud_words)
            [["word", "count"]]
            .itertuples(index=False, name=None)
        }
        if frequencies:
            cloud = WordCloud(
                width=2400,
                height=1600,
                background_color="white",
                max_words=wordcloud_words,
                collocations=False,
                prefer_horizontal=0.95,
                color_func=publication_word_color,
                random_state=42,
            ).generate_from_frequencies(frequencies)
            fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
            ax.imshow(cloud, interpolation="bilinear")
            ax.set_axis_off()
            ax.set_title(f"Word Cloud: {source_name}", pad=12)
            stem = f"wordcloud_{slugify(cleaning_source)}"
            artifacts.append(
                save_figure(fig, output_dir, stem, f"Word cloud: {source_name}", figure_formats)
            )

    verify_figure_artifacts(artifacts)
    return artifacts


def heatmap_figure(
    *,
    output_dir: Path,
    figure_formats: tuple[str, ...],
    frame: Any,
    value_column: str,
    label_column: str,
    stem: str,
    title: str,
    colorbar_label: str,
) -> FigureArtifact:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    pivot = frame.pivot_table(
        index="source_name",
        columns=label_column,
        values=value_column,
        aggfunc="sum",
        fill_value=0.0,
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    fig_width = max(8, 1.05 * len(pivot.columns) + 3)
    fig_height = max(4.5, 0.55 * len(pivot.index) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    colorbar.set_label(colorbar_label)
    for row_index, source_name in enumerate(pivot.index):
        del source_name
        for col_index, label in enumerate(pivot.columns):
            value = float(pivot.iloc[row_index, col_index])
            if value >= 0.05:
                ax.text(
                    col_index,
                    row_index,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if value < 0.55 else "white",
                )
            del label
    return save_figure(fig, output_dir, stem, title, figure_formats)


def save_figure(
    fig: Any,
    output_dir: Path,
    stem: str,
    title: str,
    formats: tuple[str, ...],
) -> FigureArtifact:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return FigureArtifact(stem=stem, title=title, paths=paths)


def verify_figure_artifacts(artifacts: Sequence[FigureArtifact]) -> None:
    missing = [
        path
        for artifact in artifacts
        for path in artifact.paths
        if not Path(path).is_file() or Path(path).stat().st_size == 0
    ]
    if missing:
        raise ValueError("Missing or empty figure artifacts: " + ", ".join(missing))


def write_figure_manifest(path: Path, artifacts: Sequence[FigureArtifact]) -> None:
    verify_figure_artifacts(artifacts)
    write_json(
        path,
        {
            "figure_count": len(artifacts),
            "artifacts": [asdict(artifact) for artifact in artifacts],
        },
    )


def write_report(
    path: Path,
    *,
    accounting: AccountingReport,
    inputs: RunInputs,
    macro: Any,
    figure_artifacts: Sequence[FigureArtifact],
) -> None:
    total_docs = int(macro["sample_count"].sum())
    total_tokens = int(macro["token_count"].sum())
    total_dropped = int(macro["dropped_sample_count"].sum())
    top_sources = macro.sort_values("token_count", ascending=False)[
        ["source_name", "sample_count", "token_count", "avg_tokens_per_doc", "drop_rate"]
    ]
    lines = [
        "# Source Distribution Analysis Report",
        "",
        "## Dataset",
        "",
        f"- Accounting file final R2 URL: `{accounting.final_r2_url}`",
        f"- Input source: `{inputs.source}`",
        f"- Parquet parts scanned: `{len(inputs.parquet_paths)}`",
        f"- Documents: `{total_docs:,}`",
        f"- Tokens: `{total_tokens:,}`",
        f"- Dropped rows: `{total_dropped:,}`",
        f"- Word regex for lexical counts: `{WORD_REGEX}`",
        "",
        "## Source Summary",
        "",
        dataframe_to_markdown(top_sources),
        "",
        "## Figures",
        "",
    ]
    for artifact in figure_artifacts:
        lines.append(f"- `{artifact.stem}`: {artifact.title}")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Existing unified-data LID columns are treated as authoritative; "
            "no language ID is rerun.",
            "- Word clouds and top-word tables are exact for the documented regex "
            "tokenizer, not for o200k token IDs.",
            "- Stopword-filtered lexical figures use `analysis/source/stopwords.txt`; "
            "raw top-word tables are unfiltered.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_self_test(*, output_dir: Path, figure_formats: tuple[str, ...]) -> None:
    fixture_dir = output_dir / "_fixture"
    run_dir = output_dir / "run"
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    accounting_path = write_fixture_dataset(fixture_dir)
    run_pipeline(
        accounting_md=accounting_path,
        r2_config=Path("r2"),
        output_dir=run_dir,
        figure_formats=figure_formats,
        metadata_only=False,
        local_parquet_glob=str(fixture_dir / "parts" / "*.parquet"),
        strict_accounting=True,
        top_words=20,
        wordcloud_words=50,
        sample_rows_per_source=2,
        min_word_chars=2,
        max_parts=0,
    )
    print(f"Self-test complete: {run_dir}", flush=True)


def write_fixture_dataset(fixture_dir: Path) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parts_dir = fixture_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "source_name": "Alpha Source",
            "cleaning_source": "alpha",
            "sample_uid": "alpha-1",
            "document_id": "a1",
            "source_object_key": "fixture/alpha.parquet",
            "cleaned_text": "Saya suka bahasa Melayu and clean training data.",
            "cleaned_o200k_token_count": 9,
            "cleaning_is_dropped": False,
            "cleaned_char_count": 47,
            "lingua_primary_language": "MALAY",
            "lingua_spans": [
                {"start_index": 0, "end_index": 24, "language_label": "MALAY"},
                {"start_index": 25, "end_index": 47, "language_label": "ENGLISH"},
            ],
            "malaya_document_label": "standard-malay",
            "malaya_word_label_counts": [
                {"label": "MS", "count": 4},
                {"label": "EN", "count": 3},
            ],
        },
        {
            "source_name": "Alpha Source",
            "cleaning_source": "alpha",
            "sample_uid": "alpha-2",
            "document_id": "a2",
            "source_object_key": "fixture/alpha.parquet",
            "cleaned_text": "",
            "cleaned_o200k_token_count": 0,
            "cleaning_is_dropped": True,
            "cleaned_char_count": 0,
            "lingua_primary_language": "missing",
            "lingua_spans": [],
            "malaya_document_label": "missing",
            "malaya_word_label_counts": [],
        },
        {
            "source_name": "Beta Source",
            "cleaning_source": "beta",
            "sample_uid": "beta-1",
            "document_id": "b1",
            "source_object_key": "fixture/beta.parquet",
            "cleaned_text": "Indonesia data has bahasa Indonesia tokens for model pretraining.",
            "cleaned_o200k_token_count": 10,
            "cleaning_is_dropped": False,
            "cleaned_char_count": 63,
            "lingua_primary_language": "INDONESIAN",
            "lingua_spans": [
                {"start_index": 0, "end_index": 35, "language_label": "INDONESIAN"},
                {"start_index": 36, "end_index": 63, "language_label": "ENGLISH"},
            ],
            "malaya_document_label": "standard-indonesian",
            "malaya_word_label_counts": [
                {"label": "ID", "count": 4},
                {"label": "EN", "count": 5},
            ],
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows[:2]), parts_dir / "part-000000.parquet")
    pq.write_table(pa.Table.from_pylist(rows[2:]), parts_dir / "part-000001.parquet")
    source_rows = [
        ("Alpha Source", "alpha", 9, 47, 2, 1),
        ("Beta Source", "beta", 10, 63, 1, 0),
    ]
    total_tokens = sum(row[2] for row in source_rows)
    total_bytes = sum(row[3] for row in source_rows)
    total_samples = sum(row[4] for row in source_rows)
    total_dropped = sum(row[5] for row in source_rows)
    accounting = [
        "# Fixture Accounting",
        "",
        "Final R2 prefix:",
        "`r2://fixture:fixture-bucket/fixture/final`",
        "",
        "Parquet parts: `2`",
        "",
        "Cleaned text column: `cleaned_text`",
        "",
        "Token column: `cleaned_o200k_token_count`",
        "",
        "| source | cleaning_source | final_r2_prefix | text_column | token_count | "
        "byte_count | sample_count | dropped_sample_count | source_object_count | "
        "original_source_glob | filters |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for source, cleaning_source, tokens, bytes_, samples, dropped in source_rows:
        accounting.append(
            f"| {source} | `{cleaning_source}` | `fixture/parts/` | `cleaned_text` | "
            f"{tokens:,} | {bytes_:,} | {samples:,} | {dropped:,} | 1 | `fixture` | `` |"
        )
    accounting.extend(
        [
            f"| **Total** | `` | `fixture/parts/` | `cleaned_text` | {total_tokens:,} | "
            f"{total_bytes:,} | {total_samples:,} | {total_dropped:,} |  |  |  |",
            "",
            "## Additional Totals",
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| cleaned_text_byte_count | {total_bytes:,} |",
            f"| cleaned_o200k_token_count | {total_tokens:,} |",
            f"| sample_count | {total_samples:,} |",
            f"| dropped_sample_count | {total_dropped:,} |",
        ]
    )
    path = fixture_dir / "fixture_accounting.md"
    path.write_text("\n".join(accounting) + "\n", encoding="utf-8")
    return path


def load_stopwords(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def order_sources(frame: Any, source_order: Sequence[str]) -> Any:
    if "cleaning_source" not in frame.columns:
        return frame
    order = {source: index for index, source in enumerate(source_order)}
    ordered = frame.copy()
    ordered["_source_order"] = ordered["cleaning_source"].map(order).fillna(len(order))
    sort_columns = ["_source_order"]
    if "rank" in ordered.columns:
        sort_columns.append("rank")
        ascending = [True, True]
    elif "token_count" in ordered.columns:
        sort_columns.append("token_count")
        ascending = [True, False]
    else:
        ascending = [True]
    ordered = ordered.sort_values(sort_columns, ascending=ascending)
    return ordered.drop(columns=["_source_order"])


def grouped_share(frame: Any, *, group_column: str, value_column: str) -> Any:
    totals = frame.groupby(group_column)[value_column].transform("sum")
    return safe_divide(frame[value_column], totals)


def safe_divide(numerator: Any, denominator: Any) -> Any:
    import pandas as pd

    if isinstance(denominator, Number):
        if denominator == 0:
            return 0
        return numerator / denominator
    return numerator / denominator.replace({0: pd.NA})


def source_name_for_cleaning_source(macro: Any, cleaning_source: str) -> str:
    rows = macro.loc[macro["cleaning_source"] == cleaning_source, "source_name"]
    return str(rows.iloc[0]) if len(rows) else cleaning_source


def compact_number(value: float, _position: int | None = None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    abs_value = abs(float(value))
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def publication_word_color(
    word: str,
    font_size: int,
    position: tuple[int, int],
    orientation: Any,
    random_state: Any = None,
    **kwargs: Any,
) -> str:
    color_index = hash(str(position)) % 6
    del word, font_size, position, orientation, random_state, kwargs
    palette = ["#2364AA", "#3DA5D9", "#73BFB8", "#FEC601", "#EA7317", "#5C677D"]
    return palette[color_index]


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "source"


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def resolve_existing_path(path: Path) -> Path:
    if path.is_file():
        return path
    repo_relative = Path.cwd() / path
    if repo_relative.is_file():
        return repo_relative
    script_relative = Path(__file__).resolve().parents[2] / path
    if script_relative.is_file():
        return script_relative
    raise FileNotFoundError(path)


def write_dataframe(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_dict_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dataframe_to_markdown(frame: Any) -> str:
    columns = [str(column) for column in frame.columns]
    rows = []
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            elif isinstance(value, int):
                values.append(f"{value:,}")
            else:
                values.append(str(value))
        rows.append(values)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in rows:
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
