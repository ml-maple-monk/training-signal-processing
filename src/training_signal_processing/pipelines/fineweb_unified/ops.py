from __future__ import annotations

import hashlib
import math
import os
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import PurePosixPath
from time import perf_counter
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ...core.storage import ObjectStore, resolve_runtime_object_store
from ...ops.builtin import RowWiseMapperOp, SourcePreparationOp
from .models import (
    FINAL_BASE_COLUMNS,
    FINAL_UNIFIED_COLUMNS,
    LID_COLUMNS,
    PARQUET_COMPRESSIONS_WITH_LEVEL,
    SUPPORTED_TOKENIZER_ENCODING,
    FineWebPartResult,
    FineWebPartTask,
    part_source_key,
)

TOKEN_TEXT_BATCH_SIZE = 8192
_TIKTOKEN_ENCODINGS: dict[str, Any] = {}


class PrepareFineWebUnifiedPartOp(SourcePreparationOp):
    op_name = "prepare_fineweb_unified_part"

    def process_row(self, row: dict[str, object]) -> dict[str, object] | None:
        runtime = self.require_runtime()
        task = FineWebPartTask.from_dict(row)
        completed = part_source_key(task) in runtime.completed_source_keys
        if completed and not runtime.allow_overwrite:
            return None
        return task.to_dict()


class WriteFineWebUnifiedPartOp(RowWiseMapperOp):
    op_name = "write_fineweb_unified_part"
    op_stage = "transform"

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        task = FineWebPartTask.from_dict(row)
        object_store: ObjectStore | None = None
        try:
            object_store = resolve_runtime_object_store(self.require_runtime())
            metrics = write_fineweb_unified_part(task=task, object_store=object_store)
            duration_sec = perf_counter() - started
            write_completion_artifacts(
                task=task,
                metrics=metrics,
                duration_sec=duration_sec,
                object_store=object_store,
            )
            return FineWebPartResult.success_from_task(
                task=task,
                metrics=metrics,
                duration_sec=duration_sec,
            ).to_dict()
        except Exception as exc:
            duration_sec = perf_counter() - started
            if object_store is not None:
                write_error_artifact(
                    task=task,
                    error_message=str(exc),
                    duration_sec=duration_sec,
                    object_store=object_store,
                )
            return FineWebPartResult.failed_from_task(
                task=task,
                error_message=str(exc),
                duration_sec=duration_sec,
            ).to_dict()


def write_fineweb_unified_part(
    *,
    task: FineWebPartTask,
    object_store: ObjectStore,
) -> dict[str, Any]:
    started = perf_counter()
    schema = final_unified_schema()
    filesystem = object_store.build_pyarrow_filesystem()
    path = f"{object_store.bucket}/{task.part_key}"
    filesystem.create_dir(str(PurePosixPath(path).parent), recursive=True)
    compression = None if task.parquet_compression == "none" else task.parquet_compression
    compression_level = (
        task.parquet_compression_level
        if task.parquet_compression in PARQUET_COMPRESSIONS_WITH_LEVEL
        else None
    )
    metrics = Counter()
    language_counts: Counter[str] = Counter()
    accepted_rows: list[dict[str, Any]] = []
    accepted_byte_count = 0
    accepted_row_count = 0
    exact_token_count = 0
    source_row_index = 0

    with filesystem.open_output_stream(path) as sink:
        writer = pq.ParquetWriter(
            sink,
            schema=schema,
            compression=compression,
            compression_level=compression_level,
        )
        try:
            for raw_row in iter_task_fineweb_rows(task):
                metrics["streamed_row_count"] += 1
                if task.enforce_month_filter and fineweb_month(raw_row) != task.month:
                    continue
                text = str(raw_row.get(task.text_column) or "")
                text_bytes = len(text.encode("utf-8"))
                if text_bytes > task.byte_quota - accepted_byte_count:
                    metrics["quota_stop_row_count"] += 1
                    break
                row = fineweb_row_to_unified_row(
                    raw_row=raw_row,
                    task=task,
                    text=text,
                    source_row_index=source_row_index,
                    row_index_in_part=accepted_row_count,
                )
                accepted_rows.append(row)
                accepted_byte_count += text_bytes
                accepted_row_count += 1
                source_row_index += 1
                language_counts[str(raw_row.get("language") or "")] += 1
                if len(accepted_rows) >= task.write_batch_rows:
                    exact_token_count += write_unified_rows(
                        writer=writer,
                        rows=accepted_rows,
                        schema=schema,
                        compute_exact_token_counts=task.compute_exact_token_counts,
                        tokenizer_encoding=task.tokenizer_encoding,
                        tokenizer_threads=task.tokenizer_threads,
                        rows_per_row_group=task.rows_per_row_group,
                    )
                    accepted_rows = []
            if accepted_rows:
                exact_token_count += write_unified_rows(
                    writer=writer,
                    rows=accepted_rows,
                    schema=schema,
                    compute_exact_token_counts=task.compute_exact_token_counts,
                    tokenizer_encoding=task.tokenizer_encoding,
                    tokenizer_threads=task.tokenizer_threads,
                    rows_per_row_group=task.rows_per_row_group,
                )
            if accepted_row_count == 0:
                writer.write_table(pa.Table.from_pylist([], schema=schema))
        finally:
            writer.close()

    duration_sec = perf_counter() - started
    return {
        "part_index": task.part_index,
        "month": task.month,
        "part_key": task.part_key,
        "row_count": accepted_row_count,
        "byte_quota": task.byte_quota,
        "cleaned_text_byte_count": accepted_byte_count,
        "cleaned_o200k_token_count": exact_token_count,
        "tokenizer_encoding": task.tokenizer_encoding,
        "compute_exact_token_counts": task.compute_exact_token_counts,
        "exact_tokenization_skipped": not task.compute_exact_token_counts,
        "enforce_month_filter": task.enforce_month_filter,
        "source_counts": {task.cleaning_source: accepted_row_count},
        "language_counts": dict(sorted(language_counts.items())),
        "streamed_row_count": int(metrics["streamed_row_count"]),
        "quota_stop_row_count": int(metrics["quota_stop_row_count"]),
        "duration_sec": duration_sec,
        "rows_per_sec": accepted_row_count / max(duration_sec, 1e-9),
    }


def write_completion_artifacts(
    *,
    task: FineWebPartTask,
    metrics: dict[str, Any],
    duration_sec: float,
    object_store: ObjectStore,
) -> None:
    object_store.write_json(task.metrics_key, metrics)
    object_store.write_json(
        task.done_key,
        {
            "status": "success",
            "part_index": task.part_index,
            "month": task.month,
            "part_key": task.part_key,
            "metrics_key": task.metrics_key,
            "row_count": int(metrics.get("row_count", 0)),
            "cleaned_text_byte_count": int(metrics.get("cleaned_text_byte_count", 0)),
            "byte_quota": task.byte_quota,
            "duration_sec": duration_sec,
        },
    )


def write_error_artifact(
    *,
    task: FineWebPartTask,
    error_message: str,
    duration_sec: float,
    object_store: ObjectStore,
) -> None:
    object_store.write_json(
        task.error_key,
        {
            "status": "failed",
            "part_index": task.part_index,
            "month": task.month,
            "part_key": task.part_key,
            "byte_quota": task.byte_quota,
            "error_message": error_message,
            "duration_sec": duration_sec,
        },
    )


def iter_task_fineweb_rows(task: FineWebPartTask) -> Iterator[dict[str, Any]]:
    for config_index, config_name in enumerate(task.dataset_configs):
        dataset = load_fineweb_stream(
            dataset_name=task.dataset_name,
            config_name=config_name,
            split=task.split,
            hf_token_env_var=task.hf_token_env_var,
        )
        seed = task.shuffle_seed + task.part_index + (config_index * 10_000)
        shard_count = min(
            max(task.month_part_count, 1),
            max(task.stream_shards_per_config, 1),
        )
        available_shards = getattr(dataset, "num_shards", None)
        if isinstance(available_shards, int) and available_shards > 0:
            shard_count = min(shard_count, available_shards)
        if shard_count > 1:
            dataset = dataset.shard(
                num_shards=shard_count,
                index=task.month_part_index % shard_count,
            )
        dataset = dataset.shuffle(seed=seed, buffer_size=task.shuffle_buffer_size)
        for row in dataset:
            if isinstance(row, dict):
                yield row


def load_fineweb_stream(
    *,
    dataset_name: str,
    config_name: str,
    split: str,
    hf_token_env_var: str,
) -> Any:
    from datasets import load_dataset

    token = resolve_hf_token(hf_token_env_var)
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if token:
        kwargs["token"] = token
    return load_dataset(dataset_name, config_name, **kwargs)


def resolve_hf_token(hf_token_env_var: str) -> str:
    return (
        os.environ.get(hf_token_env_var.strip())
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or ""
    )


def fineweb_month(row: dict[str, Any]) -> str:
    value = str(row.get("date") or "")
    if len(value) >= 7 and value[4] == "-":
        return value[:7]
    return "unknown"


def discover_fineweb_months(
    *,
    dataset_name: str,
    dataset_configs: Iterable[str],
    split: str,
    sample_rows_per_config: int,
    shuffle_seed: int,
    shuffle_buffer_size: int,
    hf_token_env_var: str,
) -> list[str]:
    months: set[str] = set()
    for config_index, config_name in enumerate(dataset_configs):
        dataset = load_fineweb_stream(
            dataset_name=dataset_name,
            config_name=config_name,
            split=split,
            hf_token_env_var=hf_token_env_var,
        )
        dataset = dataset.shuffle(
            seed=shuffle_seed + config_index,
            buffer_size=shuffle_buffer_size,
        )
        for row_index, row in enumerate(dataset):
            if row_index >= sample_rows_per_config:
                break
            if isinstance(row, dict):
                months.add(fineweb_month(row))
    if not months:
        raise ValueError("FineWeb month discovery produced no date buckets.")
    return sorted(months)


def build_month_byte_quotas(*, months: list[str], byte_cap: int) -> dict[str, int]:
    if not months:
        raise ValueError("months must contain at least one value.")
    base = byte_cap // len(months)
    remainder = byte_cap % len(months)
    return {
        month: base + (1 if index < remainder else 0)
        for index, month in enumerate(sorted(months))
    }


def split_month_quota(*, quota: int, part_target_bytes: int) -> list[int]:
    if quota <= 0:
        return []
    part_count = max(1, math.ceil(quota / part_target_bytes))
    base = quota // part_count
    remainder = quota % part_count
    return [base + (1 if index < remainder else 0) for index in range(part_count)]


def fineweb_row_to_unified_row(
    *,
    raw_row: dict[str, Any],
    task: FineWebPartTask,
    text: str,
    source_row_index: int,
    row_index_in_part: int,
) -> dict[str, Any]:
    text_hash = sha256_text(text)
    row_id = str(raw_row.get("id") or f"{task.month}-{task.part_index}-{row_index_in_part}")
    sample_uid = (
        f"hf://datasets/{task.dataset_name}/{','.join(task.dataset_configs)}"
        f"#{row_id}"
    )
    char_count = len(text)
    row = {column: None for column in FINAL_UNIFIED_COLUMNS}
    row.update(
        {
            "sample_uid": sample_uid,
            "sample_uid_hash": sha256_text(sample_uid),
            "source_name": task.source_name,
            "cleaning_source": task.cleaning_source,
            "source_bucket": "huggingface",
            "source_object_key": f"{task.dataset_name}/{','.join(task.dataset_configs)}",
            "source_parquet_url": f"https://huggingface.co/datasets/{task.dataset_name}",
            "text_column": task.text_column,
            "source_row_group_index": task.part_index,
            "source_row_index": source_row_index,
            "row_index_in_row_group": row_index_in_part,
            "cleaned_text": text,
            "original_text_sha256": text_hash,
            "cleaned_text_sha256": text_hash,
            "original_char_count": char_count,
            "cleaned_char_count": char_count,
            "removed_char_count": 0,
            "approximate_original_token_count": int(raw_row.get("token_count") or 0),
            "approximate_cleaned_token_count": int(raw_row.get("token_count") or 0),
            "approximate_removed_token_count": 0,
            "cleaning_is_dropped": False,
            "cleaning_rules_triggered": [],
            "document_id": row_id,
            "source_run_id": ",".join(task.dataset_configs),
            "source_format": "hf-streaming-parquet",
            "url": as_optional_text(raw_row.get("url")),
            "month": task.month,
            "id": row_id,
            "timestamp": as_optional_text(raw_row.get("date")),
            "crawl_id": as_optional_text(raw_row.get("dump")),
            "native_source": task.dataset_name,
            "source_shard": as_optional_text(raw_row.get("file_path")),
            "language": as_optional_text(raw_row.get("language")),
            "row_language_code": as_optional_text(raw_row.get("language")),
            "row_language_prob": as_optional_text(raw_row.get("language_score")),
            "cleaned_o200k_tokenizer": task.tokenizer_encoding,
        }
    )
    return row


def as_optional_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def write_unified_rows(
    *,
    writer: pq.ParquetWriter,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
    compute_exact_token_counts: bool,
    tokenizer_encoding: str,
    tokenizer_threads: int,
    rows_per_row_group: int,
) -> int:
    if compute_exact_token_counts:
        rows = add_exact_token_counts(
            rows=rows,
            tokenizer_encoding=tokenizer_encoding,
            tokenizer_threads=tokenizer_threads,
        )
    table = pa.Table.from_pylist(rows, schema=schema)
    writer.write_table(table, row_group_size=rows_per_row_group)
    if not compute_exact_token_counts:
        return 0
    return int(pc.sum(table["cleaned_o200k_token_count"]).as_py() or 0)


def add_exact_token_counts(
    *,
    rows: list[dict[str, Any]],
    tokenizer_encoding: str,
    tokenizer_threads: int,
) -> list[dict[str, Any]]:
    if tokenizer_encoding != SUPPORTED_TOKENIZER_ENCODING:
        raise ValueError(
            f"Unsupported tokenizer encoding {tokenizer_encoding!r}; "
            f"expected {SUPPORTED_TOKENIZER_ENCODING!r}."
        )
    encoding = get_tiktoken_encoding(tokenizer_encoding)
    texts = [str(row.get("cleaned_text") or "") for row in rows]
    token_counts: list[int] = []
    for offset in range(0, len(texts), TOKEN_TEXT_BATCH_SIZE):
        token_counts.extend(
            len(tokens)
            for tokens in encoding.encode_ordinary_batch(
                texts[offset : offset + TOKEN_TEXT_BATCH_SIZE],
                num_threads=tokenizer_threads,
            )
        )
    for row, token_count in zip(rows, token_counts, strict=True):
        row["cleaned_o200k_token_count"] = token_count
        row["cleaned_o200k_tokenizer"] = tokenizer_encoding
    return rows


def get_tiktoken_encoding(tokenizer_encoding: str) -> Any:
    encoding = _TIKTOKEN_ENCODINGS.get(tokenizer_encoding)
    if encoding is None:
        import tiktoken

        encoding = tiktoken.get_encoding(tokenizer_encoding)
        _TIKTOKEN_ENCODINGS[tokenizer_encoding] = encoding
    return encoding


def final_unified_schema() -> pa.Schema:
    fields: list[pa.Field] = []
    string_columns = set(FINAL_BASE_COLUMNS) - {
        "source_row_group_index",
        "source_row_index",
        "row_index_in_row_group",
        "original_char_count",
        "cleaned_char_count",
        "removed_char_count",
        "approximate_original_token_count",
        "approximate_cleaned_token_count",
        "approximate_removed_token_count",
        "cleaning_is_dropped",
        "cleaning_rules_triggered",
    }
    int_columns = {
        "source_row_group_index",
        "source_row_index",
        "row_index_in_row_group",
        "original_char_count",
        "cleaned_char_count",
        "removed_char_count",
        "approximate_original_token_count",
        "approximate_cleaned_token_count",
        "approximate_removed_token_count",
        "lid_cleaned_token_count",
        "removed_reference_char_count",
        "cleaned_o200k_token_count",
    }
    bool_columns = {"cleaning_is_dropped", "reference_removed"}
    for column in FINAL_UNIFIED_COLUMNS:
        if column in string_columns or column in {
            "reference_removal_method",
            "lingua_primary_language",
            "malaya_document_label",
            "cleaned_o200k_tokenizer",
        }:
            fields.append(pa.field(column, pa.string()))
        elif column in int_columns:
            fields.append(pa.field(column, pa.int64()))
        elif column in bool_columns:
            fields.append(pa.field(column, pa.bool_()))
        elif column == "cleaning_rules_triggered":
            fields.append(pa.field(column, pa.list_(pa.string())))
        elif column == "lingua_spans":
            fields.append(
                pa.field(
                    column,
                    pa.list_(
                        pa.struct(
                            [
                                pa.field("start_index", pa.int64()),
                                pa.field("end_index", pa.int64()),
                                pa.field("language_label", pa.string()),
                            ]
                        )
                    ),
                )
            )
        elif column == "malaya_document_scores":
            fields.append(
                pa.field(
                    column,
                    pa.list_(
                        pa.struct(
                            [
                                pa.field("label", pa.string()),
                                pa.field("score", pa.float64()),
                            ]
                        )
                    ),
                )
            )
        elif column == "malaya_word_detections":
            fields.append(
                pa.field(
                    column,
                    pa.list_(
                        pa.struct(
                            [
                                pa.field("word_index", pa.int64()),
                                pa.field("start_index", pa.int64()),
                                pa.field("end_index", pa.int64()),
                                pa.field("token", pa.string()),
                                pa.field("label", pa.string()),
                            ]
                        )
                    ),
                )
            )
        elif column == "malaya_word_label_counts":
            fields.append(
                pa.field(
                    column,
                    pa.list_(
                        pa.struct(
                            [
                                pa.field("label", pa.string()),
                                pa.field("count", pa.int64()),
                            ]
                        )
                    ),
                )
            )
        elif column in LID_COLUMNS:
            fields.append(pa.field(column, pa.null()))
        else:
            raise ValueError(f"No FineWeb unified schema type for column: {column}")
    return pa.schema(fields)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
