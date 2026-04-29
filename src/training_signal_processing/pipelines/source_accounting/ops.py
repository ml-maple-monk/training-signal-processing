from __future__ import annotations

from fnmatch import fnmatchcase
from time import perf_counter
from typing import Iterable

import pyarrow.parquet as pq
import tiktoken

from ...core.storage import ObjectStore, resolve_runtime_object_store
from ...core.utils import join_s3_key
from ...ops.builtin import SourcePreparationOp
from .models import (
    SourceAccountingResult,
    SourceAccountingTask,
    SourceSpec,
    source_slug,
)


class PrepareSourceAccountingSourceOp(SourcePreparationOp):
    op_name = "prepare_source_accounting_source"

    def process_row(self, row: dict[str, object]) -> dict[str, object] | None:
        runtime = self.require_runtime()
        source = SourceSpec.from_dict(row)
        if source.name in runtime.completed_source_keys and not runtime.allow_overwrite:
            return None
        output_key = build_source_row_r2_key(runtime.output_root_key, source.name)
        source_order = int(row["source_order"])
        return SourceAccountingTask(
            source_order=source_order,
            source=source.name,
            format=source.format,
            r2_relative_glob_path=source.r2_relative_glob_path,
            text_column=source.text_column,
            parquet_batch_size=source.parquet_batch_size,
            count_concurrency=source.count_concurrency,
            filters=source.filters,
            token_encoding=runtime.config.tokenizer.encoding,
            source_row_r2_key=output_key,
            table_r2_key=build_table_r2_key(runtime.output_root_key),
        ).to_dict()


class CountSourceAccountingSourceOp(SourcePreparationOp):
    op_name = "count_source_accounting_source"
    op_stage = "transform"

    def process_row(self, row: dict[str, object]) -> dict[str, object]:
        started = perf_counter()
        task = SourceAccountingTask.from_dict(row)
        try:
            object_store = resolve_runtime_object_store(self.require_runtime())
            result = count_source(task=task, object_store=object_store)
            result.duration_sec = perf_counter() - started
            return result.to_dict()
        except Exception as exc:
            return SourceAccountingResult.failed_from_task(
                task=task,
                error_message=str(exc),
                duration_sec=perf_counter() - started,
            ).to_dict()


def build_source_row_r2_key(output_root_key: str, source: str) -> str:
    return join_s3_key(output_root_key, f"sources/{source_slug(source)}.json")


def build_table_r2_key(output_root_key: str) -> str:
    return join_s3_key(output_root_key, "source-accounting.md")


def count_source(
    *,
    task: SourceAccountingTask,
    object_store: ObjectStore,
) -> SourceAccountingResult:
    encoding = tiktoken.get_encoding(task.token_encoding)
    keys = list_matching_keys(object_store, task.r2_relative_glob_path)
    if not keys:
        raise ValueError(f"No R2 objects matched: {task.r2_relative_glob_path}")
    token_count = 0
    word_count = 0
    byte_count = 0
    document_count = 0
    metadata_columns: list[str] = []
    if task.format == "parquet":
        metadata_columns = resolve_parquet_metadata_columns(
            object_store=object_store,
            keys=keys,
            text_column=task.text_column,
        )
        text_batches = iter_parquet_text_batches(
            object_store=object_store,
            keys=keys,
            text_column=task.text_column,
            batch_size=task.parquet_batch_size,
            filters=task.filters,
        )
    elif task.format in {"markdown", "text"}:
        metadata_columns = ["object_key"]
        text_batches = ([text] for text in iter_text_objects(object_store=object_store, keys=keys))
    else:
        raise ValueError(f"Unsupported source format: {task.format}")
    for texts in text_batches:
        batch_metrics = count_text_batch(
            encoding=encoding,
            texts=texts,
            count_concurrency=task.count_concurrency,
        )
        token_count += batch_metrics[0]
        word_count += batch_metrics[1]
        byte_count += batch_metrics[2]
        document_count += batch_metrics[3]
    return SourceAccountingResult(
        source_order=task.source_order,
        source=task.source,
        token_count=token_count,
        word_count=word_count,
        byte_count=byte_count,
        document_count=document_count,
        r2_relative_glob_path=task.r2_relative_glob_path,
        filters=task.filters,
        metadata_columns=metadata_columns,
        source_row_r2_key=task.source_row_r2_key,
        table_r2_key=task.table_r2_key,
    )


def list_matching_keys(object_store: ObjectStore, r2_relative_glob_path: str) -> list[str]:
    prefix = glob_listing_prefix(r2_relative_glob_path)
    keys = object_store.list_keys(prefix)
    return sorted(key for key in keys if fnmatchcase(key, r2_relative_glob_path))


def glob_listing_prefix(pattern: str) -> str:
    wildcard_positions = [
        position
        for token in ("*", "?", "[")
        if (position := pattern.find(token)) != -1
    ]
    if not wildcard_positions:
        slash_index = pattern.rfind("/")
        return pattern[: slash_index + 1] if slash_index >= 0 else pattern
    prefix = pattern[: min(wildcard_positions)]
    slash_index = prefix.rfind("/")
    return prefix[: slash_index + 1] if slash_index >= 0 else ""


def iter_text_objects(
    *,
    object_store: ObjectStore,
    keys: list[str],
) -> Iterable[str]:
    for key in keys:
        yield object_store.read_bytes(key).decode("utf-8")


def count_text_batch(
    *,
    encoding,
    texts: list[str],
    count_concurrency: int,
) -> tuple[int, int, int, int]:  # type: ignore[no-untyped-def]
    if not texts:
        return (0, 0, 0, 0)
    tokenized = encoding.encode_batch(texts, num_threads=count_concurrency)
    return (
        sum(len(tokens) for tokens in tokenized),
        sum(len(text.split()) for text in texts),
        sum(len(text.encode("utf-8")) for text in texts),
        len(texts),
    )


def iter_parquet_text_batches(
    *,
    object_store: ObjectStore,
    keys: list[str],
    text_column: str,
    batch_size: int,
    filters: dict[str, str],
) -> Iterable[list[str]]:
    filesystem = object_store.build_pyarrow_filesystem()
    for key in keys:
        parquet_file = pq.ParquetFile(f"{object_store.bucket}/{key}", filesystem=filesystem)
        if text_column not in parquet_file.schema_arrow.names:
            raise ValueError(f"Parquet source '{key}' is missing text column '{text_column}'.")
        missing_filter_columns = sorted(set(filters) - set(parquet_file.schema_arrow.names))
        if missing_filter_columns:
            raise ValueError(
                f"Parquet source '{key}' is missing filter columns: "
                + ", ".join(missing_filter_columns)
            )
        columns = [text_column, *filters]
        for batch in parquet_file.iter_batches(
            columns=columns,
            batch_size=batch_size,
        ):
            text_values = batch.column(0).to_pylist()
            filter_values_by_column = {
                column_name: batch.column(index).to_pylist()
                for index, column_name in enumerate(columns[1:], start=1)
            }
            texts: list[str] = []
            for row_index, value in enumerate(text_values):
                if row_matches_filters(
                    filters=filters,
                    filter_values_by_column=filter_values_by_column,
                    row_index=row_index,
                ):
                    texts.append("" if value is None else str(value))
            if texts:
                yield texts


def resolve_parquet_metadata_columns(
    *,
    object_store: ObjectStore,
    keys: list[str],
    text_column: str,
) -> list[str]:
    filesystem = object_store.build_pyarrow_filesystem()
    columns: list[str] = []
    for key in keys:
        parquet_file = pq.ParquetFile(f"{object_store.bucket}/{key}", filesystem=filesystem)
        for column_name in parquet_file.schema_arrow.names:
            if column_name != text_column and column_name not in columns:
                columns.append(column_name)
    return columns


def row_matches_filters(
    *,
    filters: dict[str, str],
    filter_values_by_column: dict[str, list[object]],
    row_index: int,
) -> bool:
    for column_name, expected_value in filters.items():
        actual_value = filter_values_by_column[column_name][row_index]
        if str(actual_value) != expected_value:
            return False
    return True
