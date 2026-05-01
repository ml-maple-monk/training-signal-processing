from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from ...core.models import (
    MlflowConfig,
    ObservabilityConfig,
    OpConfig,
    R2Config,
    RayConfig,
    RemoteRuntimeConfig,
    SshConfig,
)

SUPPORTED_TOKENIZER_ENCODING = "o200k_base"
PARQUET_COMPRESSIONS_WITH_LEVEL = {"brotli", "gzip", "zstd"}
RAW_TEXT_COLUMNS = frozenset({"markdown_text", "body_text", "body", "text"})
FINAL_BASE_COLUMNS = (
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
    "cleaned_text",
    "original_text_sha256",
    "cleaned_text_sha256",
    "original_char_count",
    "cleaned_char_count",
    "removed_char_count",
    "approximate_original_token_count",
    "approximate_cleaned_token_count",
    "approximate_removed_token_count",
    "cleaning_is_dropped",
    "cleaning_rules_triggered",
    "document_id",
    "source_run_id",
    "source_format",
    "markdown_file_name",
    "markdown_rel_path",
    "markdown_sha256",
    "markdown_char_count",
    "markdown_byte_count",
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
    "id",
    "timestamp",
    "crawl_id",
    "native_source",
    "source_shard",
    "language",
    "row_language_code",
    "row_language_prob",
)
LID_COLUMNS = (
    "lid_cleaned_token_count",
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
FINAL_UNIFIED_COLUMNS = (
    *FINAL_BASE_COLUMNS,
    *LID_COLUMNS,
    "cleaned_o200k_token_count",
    "cleaned_o200k_tokenizer",
)


@dataclass
class InputConfig:
    source_root_key: str
    dataset_name: str
    dataset_configs: list[str]
    split: str = "train"
    source_name: str = "FineWeb"
    cleaning_source: str = "fineweb"
    text_column: str = "text"
    hf_token_env_var: str = "HF_TOKEN"
    discovery_sample_rows_per_config: int = 50_000
    shuffle_seed: int = 42
    shuffle_buffer_size: int = 10_000
    stream_shards_per_config: int = 1024
    enforce_month_filter: bool = True

    def __post_init__(self) -> None:
        self.source_root_key = self.source_root_key.strip()
        self.dataset_name = self.dataset_name.strip()
        self.dataset_configs = [
            str(item).strip()
            for item in self.dataset_configs
            if str(item).strip()
        ]
        self.split = self.split.strip()
        self.source_name = self.source_name.strip()
        self.cleaning_source = self.cleaning_source.strip().lower().replace("_", "-")
        self.text_column = self.text_column.strip()
        self.hf_token_env_var = self.hf_token_env_var.strip()
        if not self.source_root_key:
            raise ValueError("input.source_root_key must be non-empty.")
        if not self.dataset_name:
            raise ValueError("input.dataset_name must be non-empty.")
        if not self.dataset_configs:
            raise ValueError("input.dataset_configs must contain at least one config.")
        if "default" in self.dataset_configs and len(self.dataset_configs) > 1:
            raise ValueError("input.dataset_configs must not combine 'default' with other configs.")
        if not self.split:
            raise ValueError("input.split must be non-empty.")
        if not self.source_name:
            raise ValueError("input.source_name must be non-empty.")
        if not self.cleaning_source:
            raise ValueError("input.cleaning_source must be non-empty.")
        if not self.text_column:
            raise ValueError("input.text_column must be non-empty.")
        if not self.hf_token_env_var:
            raise ValueError("input.hf_token_env_var must be non-empty.")
        if self.discovery_sample_rows_per_config <= 0:
            raise ValueError("input.discovery_sample_rows_per_config must be positive.")
        if self.shuffle_buffer_size <= 0:
            raise ValueError("input.shuffle_buffer_size must be positive.")
        if self.stream_shards_per_config <= 0:
            raise ValueError("input.stream_shards_per_config must be positive.")


@dataclass
class ExportConfig:
    byte_cap: int
    part_target_bytes: int
    rows_per_row_group: int = 500_000
    write_batch_rows: int = 8192
    compute_exact_token_counts: bool = True
    tokenizer_encoding: str = SUPPORTED_TOKENIZER_ENCODING
    tokenizer_threads: int = 8
    ray_num_cpus_per_worker: float = 1.0
    ray_memory_gib_per_worker: float = 16.0
    parquet_compression: str = "zstd"
    parquet_compression_level: int = 1

    def __post_init__(self) -> None:
        self.tokenizer_encoding = self.tokenizer_encoding.strip()
        self.parquet_compression = self.parquet_compression.strip().lower()
        if self.byte_cap <= 0:
            raise ValueError("export.byte_cap must be positive.")
        if self.part_target_bytes <= 0:
            raise ValueError("export.part_target_bytes must be positive.")
        if self.rows_per_row_group <= 0:
            raise ValueError("export.rows_per_row_group must be positive.")
        if self.write_batch_rows <= 0:
            raise ValueError("export.write_batch_rows must be positive.")
        if self.tokenizer_encoding != SUPPORTED_TOKENIZER_ENCODING:
            raise ValueError(
                "export.tokenizer_encoding must be "
                f"{SUPPORTED_TOKENIZER_ENCODING!r} for the fixed output schema."
            )
        if self.tokenizer_threads <= 0:
            raise ValueError("export.tokenizer_threads must be positive.")
        if self.ray_num_cpus_per_worker <= 0:
            raise ValueError("export.ray_num_cpus_per_worker must be positive.")
        if self.ray_memory_gib_per_worker <= 0:
            raise ValueError("export.ray_memory_gib_per_worker must be positive.")
        if not self.parquet_compression:
            raise ValueError("export.parquet_compression must be non-empty.")
        if self.parquet_compression_level <= 0:
            raise ValueError("export.parquet_compression_level must be positive.")


@dataclass
class ResumeConfig:
    strategy: str
    commit_every_batches: int
    resume_mode: str


@dataclass
class RecipeConfig:
    run_name: str
    config_version: int
    ssh: SshConfig
    remote: RemoteRuntimeConfig
    ray: RayConfig
    r2: R2Config
    input: InputConfig
    export: ExportConfig
    mlflow: MlflowConfig
    observability: ObservabilityConfig
    resumability: ResumeConfig
    ops: list[OpConfig]


@dataclass(frozen=True)
class FineWebPartTask:
    part_index: int
    month: str
    month_part_index: int
    month_part_count: int
    byte_quota: int
    dataset_name: str
    dataset_configs: tuple[str, ...]
    split: str
    source_name: str
    cleaning_source: str
    text_column: str
    hf_token_env_var: str
    shuffle_seed: int
    shuffle_buffer_size: int
    stream_shards_per_config: int
    enforce_month_filter: bool
    part_key: str
    metrics_key: str
    done_key: str
    error_key: str
    rows_per_row_group: int
    write_batch_rows: int
    tokenizer_encoding: str
    tokenizer_threads: int
    parquet_compression: str
    parquet_compression_level: int
    compute_exact_token_counts: bool = True

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "FineWebPartTask":
        return cls(
            part_index=int(row["part_index"]),
            month=str(row["month"]),
            month_part_index=int(row["month_part_index"]),
            month_part_count=int(row["month_part_count"]),
            byte_quota=int(row["byte_quota"]),
            dataset_name=str(row["dataset_name"]),
            dataset_configs=tuple(str(item) for item in row["dataset_configs"]),
            split=str(row["split"]),
            source_name=str(row["source_name"]),
            cleaning_source=str(row["cleaning_source"]),
            text_column=str(row["text_column"]),
            hf_token_env_var=str(row["hf_token_env_var"]),
            shuffle_seed=int(row["shuffle_seed"]),
            shuffle_buffer_size=int(row["shuffle_buffer_size"]),
            stream_shards_per_config=int(row["stream_shards_per_config"]),
            enforce_month_filter=bool(row.get("enforce_month_filter", True)),
            part_key=str(row["part_key"]),
            metrics_key=str(row["metrics_key"]),
            done_key=str(row["done_key"]),
            error_key=str(row["error_key"]),
            rows_per_row_group=int(row["rows_per_row_group"]),
            write_batch_rows=int(row["write_batch_rows"]),
            tokenizer_encoding=str(row["tokenizer_encoding"]),
            tokenizer_threads=int(row["tokenizer_threads"]),
            parquet_compression=str(row["parquet_compression"]),
            parquet_compression_level=int(row["parquet_compression_level"]),
            compute_exact_token_counts=bool(row.get("compute_exact_token_counts", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_configs"] = list(self.dataset_configs)
        return payload


@dataclass(frozen=True)
class FineWebPartResult:
    status: str
    part_index: int
    month: str
    part_key: str
    metrics_key: str
    done_key: str
    error_key: str
    byte_quota: int
    output_row_count: int
    output_byte_count: int
    duration_sec: float
    metrics_json: str = "{}"
    error_message: str = ""

    @classmethod
    def success_from_task(
        cls,
        *,
        task: FineWebPartTask,
        metrics: dict[str, Any],
        duration_sec: float,
    ) -> "FineWebPartResult":
        return cls(
            status="success",
            part_index=task.part_index,
            month=task.month,
            part_key=task.part_key,
            metrics_key=task.metrics_key,
            done_key=task.done_key,
            error_key=task.error_key,
            byte_quota=task.byte_quota,
            output_row_count=int(metrics.get("row_count", 0)),
            output_byte_count=int(metrics.get("cleaned_text_byte_count", 0)),
            duration_sec=duration_sec,
            metrics_json=json.dumps(metrics, ensure_ascii=False, sort_keys=True),
        )

    @classmethod
    def failed_from_task(
        cls,
        *,
        task: FineWebPartTask,
        error_message: str,
        duration_sec: float,
    ) -> "FineWebPartResult":
        return cls(
            status="failed",
            part_index=task.part_index,
            month=task.month,
            part_key=task.part_key,
            metrics_key=task.metrics_key,
            done_key=task.done_key,
            error_key=task.error_key,
            byte_quota=task.byte_quota,
            output_row_count=0,
            output_byte_count=0,
            duration_sec=duration_sec,
            error_message=error_message,
        )

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "FineWebPartResult":
        return cls(
            status=str(row["status"]),
            part_index=int(row["part_index"]),
            month=str(row["month"]),
            part_key=str(row["part_key"]),
            metrics_key=str(row["metrics_key"]),
            done_key=str(row["done_key"]),
            error_key=str(row["error_key"]),
            byte_quota=int(row["byte_quota"]),
            output_row_count=int(row.get("output_row_count", 0)),
            output_byte_count=int(row.get("output_byte_count", 0)),
            duration_sec=float(row.get("duration_sec", 0.0) or 0.0),
            metrics_json=str(row.get("metrics_json", "{}")),
            error_message=str(row.get("error_message", "")),
        )

    def metrics(self) -> dict[str, Any]:
        loaded = json.loads(self.metrics_json or "{}")
        if not isinstance(loaded, dict):
            raise ValueError("FineWeb result metrics_json must decode to an object.")
        return loaded

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def part_source_key(task: FineWebPartTask) -> str:
    return f"month={task.month}/part={task.month_part_index:06d}"
