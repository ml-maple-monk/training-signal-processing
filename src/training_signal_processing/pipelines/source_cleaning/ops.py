from __future__ import annotations

from time import perf_counter
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ...core.storage import ObjectStore, resolve_runtime_object_store
from ...ops.builtin import RowWiseMapperOp, SourcePreparationOp
from ...ops.source_uninformative_cleaning import clean_polars_frame_for_source
from .models import (
    SourceCleaningRowGroupTask,
    SourceCleaningShardResult,
    row_group_source_key,
    sha256_text,
)

ORIGINAL_TEXT_COLUMN = "__source_cleaning_original_text"

TRACE_STRING_COLUMNS = (
    "sample_uid",
    "sample_uid_hash",
    "source_name",
    "cleaning_source",
    "source_bucket",
    "source_object_key",
    "source_parquet_url",
    "text_column",
)
TRACE_INT_COLUMNS = (
    "source_row_group_index",
    "source_row_index",
    "row_index_in_row_group",
)
CLEANING_INT_COLUMNS = (
    "original_char_count",
    "cleaned_char_count",
    "removed_char_count",
    "approximate_original_token_count",
    "approximate_cleaned_token_count",
    "approximate_removed_token_count",
)
CLEANING_COLUMNS = (
    "cleaned_text",
    "original_text_sha256",
    "cleaned_text_sha256",
    *CLEANING_INT_COLUMNS,
    "cleaning_is_dropped",
    "cleaning_rules_triggered",
)
UNIFIED_SOURCE_COLUMNS = (
    "document_id",
    "source_run_id",
    "source_format",
    "markdown_file_name",
    "markdown_rel_path",
    "markdown_sha256",
    "markdown_char_count",
    "markdown_byte_count",
    "markdown_text",
    "thread_id",
    "thread_title",
    "thread_url",
    "forum",
    "thread_total_pages",
    "thread_status",
    "page_number",
    "page_offset",
    "post_id",
    "post_floor",
    "author",
    "author_id",
    "posted_at",
    "body_html",
    "quoted_post_id",
    "fetched_at",
    "error_reason",
    "body_text",
    "post_kind",
    "submission_id",
    "parent_id",
    "subreddit",
    "title",
    "score",
    "num_comments",
    "created_utc",
    "permalink",
    "url",
    "month",
    "body",
    "id",
    "timestamp",
    "crawl_id",
    "native_source",
    "source_shard",
    "language",
    "row_language_code",
    "row_language_prob",
    "text",
)
UNIFIED_COLUMNS = (
    *TRACE_STRING_COLUMNS,
    *TRACE_INT_COLUMNS,
    *CLEANING_COLUMNS,
    *UNIFIED_SOURCE_COLUMNS,
)
UNIFIED_SOURCE_COLUMN_ALIASES = {
    "native_source": "source",
}


class PrepareSourceCleaningRowGroupOp(SourcePreparationOp):
    op_name = "prepare_source_cleaning_row_group"

    def process_row(self, row: dict[str, object]) -> dict[str, object] | None:
        runtime = self.require_runtime()
        task = SourceCleaningRowGroupTask.from_dict(row)
        completed = row_group_source_key(task) in runtime.completed_source_keys
        if completed and not runtime.allow_overwrite:
            return None
        return task.to_dict()


class CleanSourceRowGroupOp(RowWiseMapperOp):
    op_name = "clean_source_row_group"
    op_stage = "transform"

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        task = SourceCleaningRowGroupTask.from_dict(row)
        try:
            object_store = resolve_runtime_object_store(self.require_runtime())
            metrics = clean_source_row_group(task=task, object_store=object_store)
            return SourceCleaningShardResult.success_from_task(
                task=task,
                metrics=metrics,
                duration_sec=perf_counter() - started,
            ).to_dict()
        except Exception as exc:
            return SourceCleaningShardResult.failed_from_task(
                task=task,
                error_message=str(exc),
                duration_sec=perf_counter() - started,
            ).to_dict()


def clean_source_row_group(
    *,
    task: SourceCleaningRowGroupTask,
    object_store: ObjectStore,
) -> dict[str, Any]:
    started = perf_counter()
    table = read_task_row_group(task=task, object_store=object_store)
    frame = pl.from_arrow(table)
    frame = add_row_trace_columns(frame=frame, task=task)
    frame = apply_filters(frame=frame, filters=task.filters)
    cleaned = clean_frame(frame=frame, task=task)
    source_shard = build_source_shard_frame(cleaned)
    unified_shard = build_unified_shard_frame(source_shard)
    object_store.write_bytes(task.source_shard_key, dataframe_to_parquet_bytes(source_shard))
    object_store.write_bytes(task.unified_shard_key, dataframe_to_parquet_bytes(unified_shard))
    return build_metrics(
        task=task,
        cleaned=cleaned,
        source_shard=source_shard,
        duration_sec=perf_counter() - started,
    )


def read_task_row_group(
    *,
    task: SourceCleaningRowGroupTask,
    object_store: ObjectStore,
) -> pa.Table:
    parquet_file = pq.ParquetFile(
        f"{object_store.bucket}/{task.source_object_key}",
        filesystem=object_store.build_pyarrow_filesystem(),
    )
    schema_names = set(parquet_file.schema_arrow.names)
    if task.text_column not in schema_names:
        raise ValueError(
            f"Parquet source '{task.source_object_key}' is missing text column "
            f"'{task.text_column}'."
        )
    missing_filter_columns = sorted(set(task.filters) - schema_names)
    if missing_filter_columns:
        raise ValueError(
            f"Parquet source '{task.source_object_key}' is missing filter columns: "
            + ", ".join(missing_filter_columns)
        )
    return parquet_file.read_row_group(task.source_row_group_index)


def add_row_trace_columns(
    *,
    frame: pl.DataFrame,
    task: SourceCleaningRowGroupTask,
) -> pl.DataFrame:
    row_index = pl.int_range(0, pl.len(), dtype=pl.Int64)
    source_row_index = row_index + task.source_row_group_start_index
    uid_prefix = f"r2://{task.source_bucket}/{task.source_object_key.lstrip('/')}#row="
    return frame.with_columns(
        row_index.alias("row_index_in_row_group"),
        source_row_index.alias("source_row_index"),
    ).with_columns(
        pl.concat_str([pl.lit(uid_prefix), pl.col("source_row_index").cast(pl.String)]).alias(
            "sample_uid"
        ),
        pl.col(task.text_column).cast(pl.String).fill_null("").alias(ORIGINAL_TEXT_COLUMN),
        pl.lit(task.source_name).alias("source_name"),
        pl.lit(task.cleaning_source).alias("cleaning_source"),
        pl.lit(task.source_bucket).alias("source_bucket"),
        pl.lit(task.source_object_key).alias("source_object_key"),
        pl.lit(task.source_parquet_url).alias("source_parquet_url"),
        pl.lit(task.source_row_group_index).alias("source_row_group_index"),
        pl.lit(task.text_column).alias("text_column"),
    )


def apply_filters(*, frame: pl.DataFrame, filters: dict[str, str]) -> pl.DataFrame:
    if not filters:
        return frame
    predicate = pl.lit(True)
    for column_name, expected_value in filters.items():
        predicate = predicate & (pl.col(column_name).cast(pl.String) == expected_value)
    return frame.filter(predicate)


def clean_frame(*, frame: pl.DataFrame, task: SourceCleaningRowGroupTask) -> pl.DataFrame:
    cleaned, _metrics = clean_polars_frame_for_source(
        frame,
        source=task.cleaning_source,
        keep_metadata=True,
        keep_dropped=True,
    )
    cleaning = pl.col("_cleaning").struct
    cleaned_text = pl.col(task.text_column).cast(pl.String).fill_null("")
    return cleaned.with_columns(
        cleaned_text.alias("cleaned_text"),
        pl.col("sample_uid").map_elements(sha256_text, return_dtype=pl.String).alias(
            "sample_uid_hash"
        ),
        pl.col(ORIGINAL_TEXT_COLUMN)
        .map_elements(sha256_text, return_dtype=pl.String)
        .alias("original_text_sha256"),
        cleaned_text.map_elements(sha256_text, return_dtype=pl.String).alias(
            "cleaned_text_sha256"
        ),
        cleaning.field("dropped").alias("cleaning_is_dropped"),
        cleaning.field("rules_triggered").alias("cleaning_rules_triggered"),
        cleaning.field("original_char_count").alias("original_char_count"),
        cleaning.field("cleaned_char_count").alias("cleaned_char_count"),
        cleaning.field("removed_char_count").alias("removed_char_count"),
        cleaning.field("approximate_original_token_count").alias(
            "approximate_original_token_count"
        ),
        cleaning.field("approximate_cleaned_token_count").alias(
            "approximate_cleaned_token_count"
        ),
        cleaning.field("approximate_removed_token_count").alias(
            "approximate_removed_token_count"
        ),
    )


def build_source_shard_frame(frame: pl.DataFrame) -> pl.DataFrame:
    drop_columns = [
        column for column in ("_cleaning", ORIGINAL_TEXT_COLUMN) if column in frame.columns
    ]
    return frame.drop(drop_columns) if drop_columns else frame


def build_unified_shard_frame(frame: pl.DataFrame) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for column in TRACE_STRING_COLUMNS:
        expressions.append(required_column(column).cast(pl.String).alias(column))
    for column in TRACE_INT_COLUMNS:
        expressions.append(required_column(column).cast(pl.Int64).alias(column))
    expressions.extend(
        [
            required_column("cleaned_text").cast(pl.String).alias("cleaned_text"),
            required_column("original_text_sha256").cast(pl.String).alias("original_text_sha256"),
            required_column("cleaned_text_sha256").cast(pl.String).alias("cleaned_text_sha256"),
        ]
    )
    for column in CLEANING_INT_COLUMNS:
        expressions.append(required_column(column).cast(pl.Int64).alias(column))
    expressions.append(
        required_column("cleaning_is_dropped")
        .cast(pl.Boolean)
        .alias("cleaning_is_dropped")
    )
    expressions.append(
        required_column("cleaning_rules_triggered").alias("cleaning_rules_triggered")
    )
    for column in UNIFIED_SOURCE_COLUMNS:
        source_column = UNIFIED_SOURCE_COLUMN_ALIASES.get(column, column)
        if source_column in frame.columns:
            expressions.append(pl.col(source_column).cast(pl.String).alias(column))
        else:
            expressions.append(pl.lit(None, dtype=pl.String).alias(column))
    return frame.select(expressions)


def required_column(column: str) -> pl.Expr:
    return pl.col(column)


def dataframe_to_parquet_bytes(frame: pl.DataFrame) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(frame.to_arrow(), sink)
    return sink.getvalue().to_pybytes()


def build_metrics(
    *,
    task: SourceCleaningRowGroupTask,
    cleaned: pl.DataFrame,
    source_shard: pl.DataFrame,
    duration_sec: float,
) -> dict[str, Any]:
    summary = cleaned.select(
        pl.len().alias("row_count"),
        pl.col("cleaning_is_dropped").sum().alias("dropped_row_count"),
        (pl.col("cleaning_is_dropped") == False).sum().alias("retained_row_count"),  # noqa: E712
        pl.col("original_char_count").sum().alias("original_total_characters"),
        pl.col("cleaned_char_count").sum().alias("cleaned_total_characters"),
        pl.col("removed_char_count").sum().alias("removed_total_characters"),
        pl.col("approximate_original_token_count").sum().alias("approximate_original_tokens"),
        pl.col("approximate_cleaned_token_count").sum().alias("approximate_cleaned_tokens"),
        pl.col("approximate_removed_token_count").sum().alias("approximate_tokens_removed"),
    ).to_dicts()[0]
    row_count = int(summary["row_count"] or 0)
    elapsed = max(duration_sec, 1e-9)
    return {
        "source_name": task.source_name,
        "cleaning_source": task.cleaning_source,
        "source_object_key": task.source_object_key,
        "source_row_group_index": task.source_row_group_index,
        "source_row_group_num_rows": task.source_row_group_num_rows,
        "filtered_row_count": row_count,
        "output_row_count": source_shard.height,
        "dropped_row_count": int(summary["dropped_row_count"] or 0),
        "retained_row_count": int(summary["retained_row_count"] or 0),
        "original_total_characters": int(summary["original_total_characters"] or 0),
        "cleaned_total_characters": int(summary["cleaned_total_characters"] or 0),
        "removed_total_characters": int(summary["removed_total_characters"] or 0),
        "approximate_original_tokens": int(summary["approximate_original_tokens"] or 0),
        "approximate_cleaned_tokens": int(summary["approximate_cleaned_tokens"] or 0),
        "approximate_tokens_removed": int(summary["approximate_tokens_removed"] or 0),
        "duration_sec": duration_sec,
        "rows_per_sec": row_count / elapsed,
        "source_shard_key": task.source_shard_key,
        "unified_shard_key": task.unified_shard_key,
        "metrics_key": task.metrics_key,
        "done_key": task.done_key,
    }
