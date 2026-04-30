from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import PurePosixPath
from time import perf_counter
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ...core.storage import ObjectStore, resolve_runtime_object_store, strip_endpoint_scheme
from ...ops.builtin import RowWiseMapperOp, SourcePreparationOp
from .models import (
    PARQUET_COMPRESSIONS_WITH_LEVEL,
    SUPPORTED_TOKENIZER_ENCODING,
    MergeSegment,
    UnifiedDataPartResult,
    UnifiedDataPartTask,
    part_source_key,
)

TOKEN_TEXT_BATCH_SIZE = 8192
RAW_TEXT_COLUMNS = ("markdown_text", "body_text", "body", "text")
LID_DATA_COLUMNS = (
    "cleaned_token_count",
    "reference_removed",
    "reference_removal_method",
    "removed_reference_char_count",
    "lingua_primary_language",
    "lingua_spans",
    "malaya_document_label",
    "malaya_document_scores",
    "malaya_word_detections",
    "malaya_word_label_counts",
)
LID_COLUMN_RENAMES = {
    "cleaned_token_count": "lid_cleaned_token_count",
}
LID_VALIDATION_COLUMNS = (
    "sample_uid_hash",
    "source_row_group_index",
    "source_row_index",
    "row_index_in_row_group",
    "original_text_sha256",
)
LID_REQUIRED_COLUMNS = ("sample_uid", *LID_VALIDATION_COLUMNS)
_TIKTOKEN_ENCODINGS: dict[str, Any] = {}


class PrepareUnifiedDataPartOp(SourcePreparationOp):
    op_name = "prepare_unified_data_part"

    def process_row(self, row: dict[str, object]) -> dict[str, object] | None:
        runtime = self.require_runtime()
        task = UnifiedDataPartTask.from_dict(row)
        completed = part_source_key(task) in runtime.completed_source_keys
        if completed and not runtime.allow_overwrite:
            return None
        return task.to_dict()


class WriteUnifiedDataPartOp(RowWiseMapperOp):
    op_name = "write_unified_data_part"
    op_stage = "transform"

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        task = UnifiedDataPartTask.from_dict(row)
        try:
            object_store = resolve_runtime_object_store(self.require_runtime())
            metrics = write_unified_data_part(task=task, object_store=object_store)
            return UnifiedDataPartResult.success_from_task(
                task=task,
                metrics=metrics,
                duration_sec=perf_counter() - started,
            ).to_dict()
        except Exception as exc:
            return UnifiedDataPartResult.failed_from_task(
                task=task,
                error_message=str(exc),
                duration_sec=perf_counter() - started,
            ).to_dict()


def write_unified_data_part(
    *,
    task: UnifiedDataPartTask,
    object_store: ObjectStore,
) -> dict[str, Any]:
    started = perf_counter()
    tables: list[pa.Table] = []
    validation_counts = Counter()
    connection = build_duckdb_connection(object_store=object_store, threads=task.duckdb_threads)
    try:
        for segment in task.segments:
            segment_table, segment_counts = build_segment_table(
                object_store=object_store,
                connection=connection,
                segment=segment,
                tokenizer_encoding=task.tokenizer_encoding,
                tokenizer_threads=task.tokenizer_threads,
            )
            tables.append(segment_table)
            validation_counts.update(segment_counts)
    finally:
        connection.close()
    if not tables:
        raise ValueError(f"Unified data part {task.part_index} has no merge segments.")
    part_table = pa.concat_tables(tables, promote_options="default")
    if part_table.num_rows != task.expected_row_count:
        raise ValueError(
            f"Unified data part {task.part_index} expected {task.expected_row_count} rows "
            f"but built {part_table.num_rows} rows."
        )
    write_parquet_frame(
        object_store=object_store,
        key=task.part_key,
        table=part_table,
        rows_per_row_group=task.rows_per_row_group,
        compression=task.parquet_compression,
        compression_level=task.parquet_compression_level,
    )
    return build_part_metrics(
        task=task,
        table=part_table,
        validation_counts=validation_counts,
        duration_sec=perf_counter() - started,
    )


def build_segment_table(
    *,
    object_store: ObjectStore,
    connection: duckdb.DuckDBPyConnection,
    segment: MergeSegment,
    tokenizer_encoding: str,
    tokenizer_threads: int,
) -> tuple[pa.Table, Counter[str]]:
    cleaning = read_cleaning_frame(
        object_store=object_store,
        connection=connection,
        key=segment.cleaning_unified_shard_key,
        row_offset=segment.row_offset,
        row_count=segment.row_count,
    )
    lid = prepare_lid_table(
        read_lid_table(
            object_store=object_store,
            connection=connection,
            key=segment.lid_shard_key,
            row_offset=segment.row_offset,
            row_count=segment.row_count,
        )
    )
    validate_aligned_segment(
        segment=segment,
        cleaning=cleaning,
        lid=lid,
    )
    if cleaning.num_rows != segment.row_count or lid.num_rows != segment.row_count:
        raise ValueError(
            f"Segment {segment.source_object_key} row_group={segment.source_row_group_index} "
            f"offset={segment.row_offset} expected {segment.row_count} rows but "
            f"built cleaning={cleaning.num_rows} lid={lid.num_rows} rows."
        )
    lid_data = lid.select(list(renamed_lid_data_columns()))
    merged = append_table_columns(cleaning, lid_data)
    merged = add_exact_token_counts(
        table=merged,
        tokenizer_encoding=tokenizer_encoding,
        tokenizer_threads=tokenizer_threads,
    )
    counts = Counter(
        {
            "cleaning_input_rows": cleaning.num_rows,
            "lid_input_rows": lid.num_rows,
            "joined_rows": merged.num_rows,
            "output_rows": merged.num_rows,
        }
    )
    return merged, counts


def build_duckdb_connection(
    *,
    object_store: ObjectStore,
    threads: int,
) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(database=":memory:")
    connection.execute(f"SET threads = {int(threads)}")
    connection.execute("SET disable_parquet_prefetching = true")
    connection.execute("SET parquet_metadata_cache = true")
    config = getattr(object_store, "config", None)
    if config is not None:
        connection.execute("LOAD httpfs")
        connection.execute(f"SET s3_region = {sql_string_literal(config.region)}")
        connection.execute(f"SET s3_access_key_id = {sql_string_literal(config.access_key_id)}")
        connection.execute(
            f"SET s3_secret_access_key = {sql_string_literal(config.secret_access_key)}"
        )
        connection.execute(
            f"SET s3_endpoint = {sql_string_literal(strip_endpoint_scheme(config.endpoint_url))}"
        )
        connection.execute("SET s3_url_style = 'path'")
        connection.execute("SET s3_use_ssl = true")
    return connection


def read_cleaning_frame(
    *,
    object_store: ObjectStore,
    connection: duckdb.DuckDBPyConnection,
    key: str,
    row_offset: int,
    row_count: int,
) -> pa.Table:
    parquet_file = pq.ParquetFile(
        f"{object_store.bucket}/{key}",
        filesystem=object_store.build_pyarrow_filesystem(),
    )
    columns = [
        column for column in parquet_file.schema_arrow.names if column not in RAW_TEXT_COLUMNS
    ]
    return read_parquet_slice(
        object_store=object_store,
        connection=connection,
        parquet_file=parquet_file,
        key=key,
        columns=columns,
        row_offset=row_offset,
        row_count=row_count,
        prefer_direct_r2=False,
    )


def read_lid_table(
    *,
    object_store: ObjectStore,
    connection: duckdb.DuckDBPyConnection,
    key: str,
    row_offset: int,
    row_count: int,
) -> pa.Table:
    parquet_file = pq.ParquetFile(
        f"{object_store.bucket}/{key}",
        filesystem=object_store.build_pyarrow_filesystem(),
    )
    schema_names = set(parquet_file.schema_arrow.names)
    missing = sorted(set(LID_REQUIRED_COLUMNS) - schema_names)
    if missing:
        raise ValueError("LID shard is missing required columns: " + ", ".join(missing))
    columns = [
        column
        for column in (*LID_REQUIRED_COLUMNS, *LID_DATA_COLUMNS)
        if column in schema_names
    ]
    return read_parquet_slice(
        object_store=object_store,
        connection=connection,
        parquet_file=parquet_file,
        key=key,
        columns=columns,
        row_offset=row_offset,
        row_count=row_count,
        prefer_direct_r2=False,
    )


def read_parquet_slice(
    *,
    object_store: ObjectStore,
    connection: duckdb.DuckDBPyConnection,
    parquet_file: pq.ParquetFile,
    key: str,
    columns: Sequence[str],
    row_offset: int,
    row_count: int,
    prefer_direct_r2: bool,
) -> pa.Table:
    if row_offset < 0 or row_count < 0:
        raise ValueError("Parquet slice offsets and counts must be non-negative.")
    if getattr(object_store, "config", None) is not None and not prefer_direct_r2:
        table = parquet_file.read(columns=list(columns))
        connection.register("parquet_slice_source", table)
        try:
            return connection.execute(
                "SELECT "
                + ", ".join(select_columns(columns))
                + f" FROM parquet_slice_source LIMIT {int(row_count)} OFFSET {int(row_offset)}"
            ).to_arrow_table()
        finally:
            connection.unregister("parquet_slice_source")
    path = duckdb_parquet_path(object_store=object_store, key=key)
    sql = (
        "SELECT "
        + ", ".join(select_columns(columns))
        + f" FROM read_parquet({sql_string_literal(path)})"
        + f" LIMIT {int(row_count)} OFFSET {int(row_offset)}"
    )
    return connection.execute(sql).to_arrow_table()


def duckdb_parquet_path(*, object_store: ObjectStore, key: str) -> str:
    storage_root = getattr(object_store, "storage_root", None)
    if storage_root is not None:
        return str(storage_root / key)
    config = getattr(object_store, "config", None)
    if config is not None:
        return f"s3://{object_store.bucket}/{key.lstrip('/')}"
    raise ValueError("DuckDB parquet reads require an R2 object store or local storage_root.")


def select_columns(columns: Sequence[str]) -> list[str]:
    return [quote_identifier(column) for column in columns]


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def prepare_lid_table(lid: pa.Table) -> pa.Table:
    column_names = set(lid.column_names)
    missing = sorted(set(LID_REQUIRED_COLUMNS) - column_names)
    if missing:
        raise ValueError("LID shard is missing required columns: " + ", ".join(missing))
    arrays: list[pa.ChunkedArray | pa.Array] = [lid["sample_uid"]]
    names: list[str] = ["sample_uid"]
    for column in LID_VALIDATION_COLUMNS:
        arrays.append(lid[column])
        names.append(f"__lid_{column}")
    for column in LID_DATA_COLUMNS:
        output_column = LID_COLUMN_RENAMES.get(column, column)
        if column in column_names:
            arrays.append(lid[column])
        else:
            arrays.append(pa.nulls(lid.num_rows))
        names.append(output_column)
    return pa.table(arrays, names=names)


def validate_aligned_segment(
    *,
    segment: MergeSegment,
    cleaning: pa.Table,
    lid: pa.Table,
) -> None:
    if cleaning.num_rows != lid.num_rows:
        raise ValueError(
            f"Join row count mismatch for {segment.source_object_key} "
            f"row_group={segment.source_row_group_index}: "
            f"cleaning={cleaning.num_rows} lid={lid.num_rows}"
        )
    mismatch_mask: pa.ChunkedArray | None = None
    for cleaning_column, lid_column in (
        ("sample_uid", "sample_uid"),
        ("sample_uid_hash", "__lid_sample_uid_hash"),
        ("source_row_group_index", "__lid_source_row_group_index"),
        ("source_row_index", "__lid_source_row_index"),
        ("row_index_in_row_group", "__lid_row_index_in_row_group"),
        ("original_text_sha256", "__lid_original_text_sha256"),
    ):
        equal = pc.fill_null(pc.equal(cleaning[cleaning_column], lid[lid_column]), False)
        mismatch = pc.invert(equal)
        mismatch_mask = mismatch if mismatch_mask is None else pc.or_(mismatch_mask, mismatch)
    mismatch_count = 0
    if mismatch_mask is not None:
        mismatch_count = int(pc.sum(pc.cast(mismatch_mask, pa.int64())).as_py() or 0)
    if mismatch_count:
        raise ValueError(
            f"Join validation failed for {segment.source_object_key} "
            f"row_group={segment.source_row_group_index}: {mismatch_count} "
            "positionally mismatched rows."
        )


def add_exact_token_counts(
    *,
    table: pa.Table,
    tokenizer_encoding: str,
    tokenizer_threads: int,
) -> pa.Table:
    if tokenizer_encoding != SUPPORTED_TOKENIZER_ENCODING:
        raise ValueError(
            f"Unsupported tokenizer encoding {tokenizer_encoding!r}; "
            f"expected {SUPPORTED_TOKENIZER_ENCODING!r}."
        )
    encoding = get_tiktoken_encoding(tokenizer_encoding)
    token_counts: list[int] = []
    for text_batch in iter_text_batches(table["cleaned_text"], TOKEN_TEXT_BATCH_SIZE):
        token_counts.extend(
            len(tokens)
            for tokens in encoding.encode_ordinary_batch(
                text_batch,
                num_threads=tokenizer_threads,
            )
        )
    return table.append_column(
        "cleaned_o200k_token_count",
        pa.array(token_counts, type=pa.int64()),
    ).append_column(
        "cleaned_o200k_tokenizer",
        pa.repeat(pa.scalar(tokenizer_encoding, type=pa.string()), table.num_rows),
    )


def get_tiktoken_encoding(tokenizer_encoding: str) -> Any:
    encoding = _TIKTOKEN_ENCODINGS.get(tokenizer_encoding)
    if encoding is None:
        import tiktoken

        encoding = tiktoken.get_encoding(tokenizer_encoding)
        _TIKTOKEN_ENCODINGS[tokenizer_encoding] = encoding
    return encoding


def append_table_columns(left: pa.Table, right: pa.Table) -> pa.Table:
    table = left
    for name in right.column_names:
        table = table.append_column(name, right[name])
    return table


def iter_text_batches(
    values: pa.ChunkedArray,
    batch_size: int,
) -> list[str]:
    total = len(values)
    for offset in range(0, total, batch_size):
        batch = values.slice(offset, min(batch_size, total - offset)).to_pylist()
        yield ["" if value is None else str(value) for value in batch]


def renamed_lid_data_columns() -> tuple[str, ...]:
    return tuple(LID_COLUMN_RENAMES.get(column, column) for column in LID_DATA_COLUMNS)


def write_parquet_frame(
    *,
    object_store: ObjectStore,
    key: str,
    table: pa.Table,
    rows_per_row_group: int,
    compression: str,
    compression_level: int,
) -> None:
    filesystem = object_store.build_pyarrow_filesystem()
    path = f"{object_store.bucket}/{key}"
    parent = str(PurePosixPath(path).parent)
    filesystem.create_dir(parent, recursive=True)
    resolved_compression = None if compression == "none" else compression
    resolved_level = (
        compression_level if compression in PARQUET_COMPRESSIONS_WITH_LEVEL else None
    )
    with filesystem.open_output_stream(path) as sink:
        pq.write_table(
            table,
            sink,
            row_group_size=rows_per_row_group,
            compression=resolved_compression,
            compression_level=resolved_level,
        )


def build_part_metrics(
    *,
    task: UnifiedDataPartTask,
    table: pa.Table,
    validation_counts: Counter[str],
    duration_sec: float,
) -> dict[str, Any]:
    source_counts = arrow_value_counts(table["cleaning_source"])
    dropped_rows = int(pc.sum(pc.cast(table["cleaning_is_dropped"], pa.int64())).as_py() or 0)
    removed_chars = int(pc.sum(table["removed_char_count"]).as_py() or 0)
    exact_tokens = int(pc.sum(table["cleaned_o200k_token_count"]).as_py() or 0)
    input_lid_shard_keys = sorted({segment.lid_shard_key for segment in task.segments})
    input_cleaning_shard_keys = sorted(
        {segment.cleaning_unified_shard_key for segment in task.segments}
    )
    return {
        "part_index": task.part_index,
        "part_key": task.part_key,
        "row_count": table.num_rows,
        "rows_per_row_group": task.rows_per_row_group,
        "tokenizer_encoding": task.tokenizer_encoding,
        "cleaned_o200k_token_count": exact_tokens,
        "dropped_row_count": dropped_rows,
        "removed_char_count": removed_chars,
        "source_counts": dict(sorted(source_counts.items())),
        "input_lid_shard_keys": input_lid_shard_keys,
        "input_cleaning_unified_shard_keys": input_cleaning_shard_keys,
        "input_segment_count": len(task.segments),
        "join_validation_counts": dict(sorted(validation_counts.items())),
        "duration_sec": duration_sec,
        "rows_per_sec": table.num_rows / max(duration_sec, 1e-9),
    }


def arrow_value_counts(values: pa.ChunkedArray) -> dict[str, int]:
    counts = pc.value_counts(values).to_pylist()
    return {str(item["values"]): int(item["counts"]) for item in counts}
