from __future__ import annotations

import hashlib
import tempfile
from collections import Counter
from pathlib import PurePosixPath
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ...core.storage import ObjectStore, resolve_runtime_object_store
from ...ops.builtin import RowWiseMapperOp, SourcePreparationOp
from .models import (
    PARQUET_COMPRESSIONS_WITH_LEVEL,
    DatasetTokenizationResult,
    DatasetTokenizationTask,
    part_source_key,
)

PROVENANCE_COLUMNS = (
    "sample_uid",
    "sample_uid_hash",
    "source_name",
    "cleaning_source",
    "source_bucket",
    "source_object_key",
    "source_parquet_url",
    "text_column",
    "source_row_group_index",
    "source_row_index",
    "row_index_in_row_group",
    "document_id",
    "source_run_id",
    "source_format",
    "cleaned_text_sha256",
)
INT64_PROVENANCE_COLUMNS = {
    "source_row_group_index",
    "source_row_index",
    "row_index_in_row_group",
}
TOKENIZER_CACHE: dict[str, Any] = {}


class PrepareDatasetTokenizationPartOp(SourcePreparationOp):
    op_name = "prepare_dataset_tokenization_part"

    def process_row(self, row: dict[str, object]) -> dict[str, object] | None:
        runtime = self.require_runtime()
        task = DatasetTokenizationTask.from_dict(row)
        completed = part_source_key(task) in runtime.completed_source_keys
        if completed and not runtime.allow_overwrite:
            return None
        return task.to_dict()


class WriteDatasetTokenizationPartOp(RowWiseMapperOp):
    op_name = "write_dataset_tokenization_part"
    op_stage = "transform"

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        task = DatasetTokenizationTask.from_dict(row)
        try:
            object_store = resolve_runtime_object_store(self.require_runtime())
            metrics = write_dataset_tokenization_part(task=task, object_store=object_store)
            return DatasetTokenizationResult.success_from_task(
                task=task,
                metrics=metrics,
                duration_sec=perf_counter() - started,
            ).to_dict()
        except Exception as exc:
            return DatasetTokenizationResult.failed_from_task(
                task=task,
                error_message=str(exc),
                duration_sec=perf_counter() - started,
            ).to_dict()


def write_dataset_tokenization_part(
    *,
    task: DatasetTokenizationTask,
    object_store: ObjectStore,
) -> dict[str, Any]:
    started = perf_counter()
    tokenizer = load_tokenizer(task=task, object_store=object_store)
    filesystem = object_store.build_pyarrow_filesystem()
    input_file = pq.ParquetFile(
        f"{object_store.bucket}/{task.source_part_key}",
        filesystem=filesystem,
    )
    schema_names = set(input_file.schema_arrow.names)
    if task.text_column not in schema_names:
        raise ValueError(
            f"Input part {task.source_part_key} is missing text column {task.text_column!r}."
        )
    read_columns = [
        column
        for column in (*PROVENANCE_COLUMNS, task.text_column, task.dropped_column)
        if column in schema_names
    ]
    if task.text_column not in read_columns:
        read_columns.append(task.text_column)

    output_path = f"{object_store.bucket}/{task.part_key}"
    filesystem.create_dir(str(PurePosixPath(output_path).parent), recursive=True)
    schema = tokenized_output_schema()
    compression = None if task.parquet_compression == "none" else task.parquet_compression
    compression_level = (
        task.parquet_compression_level
        if task.parquet_compression in PARQUET_COMPRESSIONS_WITH_LEVEL
        else None
    )

    metrics = Counter()
    token_count_total = 0
    text_byte_count_total = 0
    dropped_count_total = 0
    with filesystem.open_output_stream(output_path) as sink:
        writer = pq.ParquetWriter(
            sink,
            schema=schema,
            compression=compression,
            compression_level=compression_level,
        )
        try:
            wrote_any = False
            for batch in input_file.iter_batches(
                batch_size=task.read_batch_rows,
                columns=read_columns,
            ):
                table = pa.Table.from_batches([batch])
                output_table = tokenize_table_batch(
                    table=table,
                    task=task,
                    tokenizer=tokenizer,
                    schema=schema,
                )
                writer.write_table(
                    output_table,
                    row_group_size=task.rows_per_row_group,
                )
                wrote_any = True
                token_count_total += int(pc.sum(output_table["token_count"]).as_py() or 0)
                text_byte_count_total += int(
                    pc.sum(output_table["text_byte_count"]).as_py() or 0
                )
                dropped_count_total += int(
                    pc.sum(pc.cast(output_table["cleaning_is_dropped"], pa.int64())).as_py()
                    or 0
                )
                metrics["row_count"] += output_table.num_rows
            if not wrote_any:
                writer.write_table(pa.Table.from_pylist([], schema=schema))
        finally:
            writer.close()

    duration_sec = perf_counter() - started
    return {
        "task_index": task.task_index,
        "source_group": task.source_group,
        "source_part_key": task.source_part_key,
        "part_key": task.part_key,
        "row_count": int(metrics["row_count"]),
        "token_count": token_count_total,
        "text_byte_count": text_byte_count_total,
        "dropped_row_count": dropped_count_total,
        "tokenizer_name": task.tokenizer_name,
        "tokenizer_json_sha256": task.tokenizer_json_sha256,
        "duration_sec": duration_sec,
        "rows_per_sec": int(metrics["row_count"]) / max(duration_sec, 1e-9),
        "tokens_per_sec": token_count_total / max(duration_sec, 1e-9),
    }


def tokenize_table_batch(
    *,
    table: pa.Table,
    task: DatasetTokenizationTask,
    tokenizer: Any,
    schema: pa.Schema,
) -> pa.Table:
    texts = ["" if value is None else str(value) for value in table[task.text_column].to_pylist()]
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    token_ids = [encoding.ids for encoding in encodings]
    token_counts = [len(ids) for ids in token_ids]
    text_bytes = [len(text.encode("utf-8")) for text in texts]
    cleaned_hashes = column_values(table, "cleaned_text_sha256", default="")
    text_hashes = [
        str(existing) if existing else hashlib.sha256(text.encode("utf-8")).hexdigest()
        for existing, text in zip(cleaned_hashes, texts, strict=True)
    ]

    arrays: list[pa.Array | pa.ChunkedArray] = [
        pa.repeat(pa.scalar(task.source_group, type=pa.string()), table.num_rows),
        pa.repeat(pa.scalar(task.source_part_key, type=pa.string()), table.num_rows),
    ]
    names = ["source_group", "source_part_key"]
    for column in PROVENANCE_COLUMNS:
        if column == "cleaned_text_sha256":
            continue
        values = column_values(table, column, default=None)
        if column in INT64_PROVENANCE_COLUMNS:
            arrays.append(pa.array(values, type=pa.int64()))
        else:
            arrays.append(pa.array(values, type=pa.string()))
        names.append(column)
    arrays.extend(
        [
            pa.array(text_hashes, type=pa.string()),
            pa.array(text_bytes, type=pa.int64()),
            pa.array(column_values(table, task.dropped_column, default=False), type=pa.bool_()),
            pa.array(token_ids, type=pa.list_(pa.int32())),
            pa.array(token_counts, type=pa.int32()),
            pa.repeat(pa.scalar(task.tokenizer_name, type=pa.string()), table.num_rows),
            pa.repeat(pa.scalar(task.tokenizer_json_sha256, type=pa.string()), table.num_rows),
        ]
    )
    names.extend(
        [
            "text_sha256",
            "text_byte_count",
            "cleaning_is_dropped",
            "token_ids",
            "token_count",
            "tokenizer_name",
            "tokenizer_json_sha256",
        ]
    )
    return pa.table(arrays, names=names).cast(schema)


def column_values(table: pa.Table, column: str, default: Any) -> list[Any]:
    if column not in table.column_names:
        return [default] * table.num_rows
    return table[column].to_pylist()


def load_tokenizer(
    *,
    task: DatasetTokenizationTask,
    object_store: ObjectStore,
) -> Any:
    tokenizer = TOKENIZER_CACHE.get(task.tokenizer_json_sha256)
    if tokenizer is not None:
        return tokenizer
    payload = object_store.read_bytes(task.tokenizer_object_key)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != task.tokenizer_json_sha256:
        raise ValueError(
            f"Tokenizer SHA mismatch for {task.tokenizer_object_key}: "
            f"expected {task.tokenizer_json_sha256}, got {digest}."
        )
    from tokenizers import Tokenizer

    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        handle.write(payload)
        handle.flush()
        tokenizer = Tokenizer.from_file(handle.name)
    TOKENIZER_CACHE[task.tokenizer_json_sha256] = tokenizer
    return tokenizer


def tokenized_output_schema() -> pa.Schema:
    fields = [
        pa.field("source_group", pa.string()),
        pa.field("source_part_key", pa.string()),
    ]
    for column in PROVENANCE_COLUMNS:
        if column == "cleaned_text_sha256":
            continue
        if column in INT64_PROVENANCE_COLUMNS:
            fields.append(pa.field(column, pa.int64()))
        else:
            fields.append(pa.field(column, pa.string()))
    fields.extend(
        [
            pa.field("text_sha256", pa.string()),
            pa.field("text_byte_count", pa.int64()),
            pa.field("cleaning_is_dropped", pa.bool_()),
            pa.field("token_ids", pa.list_(pa.int32())),
            pa.field("token_count", pa.int32()),
            pa.field("tokenizer_name", pa.string()),
            pa.field("tokenizer_json_sha256", pa.string()),
        ]
    )
    return pa.schema(fields)
