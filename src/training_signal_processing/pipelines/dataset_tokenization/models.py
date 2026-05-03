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

PARQUET_COMPRESSIONS_WITH_LEVEL = {"brotli", "gzip", "zstd"}
DEFAULT_TOKENIZER_NAME = "native_superbpe_1m_rows_max4w"


@dataclass
class InputConfig:
    source_root_key: str
    final_parts_prefix: str
    fineweb_run_root: str
    text_column: str = "cleaned_text"
    dropped_column: str = "cleaning_is_dropped"

    def __post_init__(self) -> None:
        self.source_root_key = self.source_root_key.strip()
        self.final_parts_prefix = self.final_parts_prefix.strip().lstrip("/")
        self.fineweb_run_root = self.fineweb_run_root.strip().strip("/")
        self.text_column = self.text_column.strip()
        self.dropped_column = self.dropped_column.strip()
        if not self.source_root_key:
            raise ValueError("input.source_root_key must be non-empty.")
        if not self.final_parts_prefix:
            raise ValueError("input.final_parts_prefix must be non-empty.")
        if not self.fineweb_run_root:
            raise ValueError("input.fineweb_run_root must be non-empty.")
        if not self.text_column:
            raise ValueError("input.text_column must be non-empty.")
        if not self.dropped_column:
            raise ValueError("input.dropped_column must be non-empty.")


@dataclass
class TokenizerConfig:
    name: str = DEFAULT_TOKENIZER_NAME
    json_path: str = "tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json"

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.json_path = self.json_path.strip()
        if not self.name:
            raise ValueError("tokenizer.name must be non-empty.")
        if not self.json_path:
            raise ValueError("tokenizer.json_path must be non-empty.")


@dataclass
class ExportConfig:
    read_batch_rows: int = 2048
    rows_per_row_group: int = 8192
    ray_num_cpus_per_worker: float = 1.0
    ray_memory_gib_per_worker: float = 16.0
    parquet_compression: str = "zstd"
    parquet_compression_level: int = 1

    def __post_init__(self) -> None:
        self.parquet_compression = self.parquet_compression.strip().lower()
        if self.read_batch_rows <= 0:
            raise ValueError("export.read_batch_rows must be positive.")
        if self.rows_per_row_group <= 0:
            raise ValueError("export.rows_per_row_group must be positive.")
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
    tokenizer: TokenizerConfig
    export: ExportConfig
    mlflow: MlflowConfig
    observability: ObservabilityConfig
    resumability: ResumeConfig
    ops: list[OpConfig]


@dataclass(frozen=True)
class DatasetTokenizationTask:
    task_index: int
    source_group: str
    source_part_key: str
    part_key: str
    metrics_key: str
    done_key: str
    error_key: str
    text_column: str
    dropped_column: str
    tokenizer_name: str
    tokenizer_object_key: str
    tokenizer_json_sha256: str
    read_batch_rows: int
    rows_per_row_group: int
    parquet_compression: str
    parquet_compression_level: int

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DatasetTokenizationTask":
        return cls(
            task_index=int(row["task_index"]),
            source_group=str(row["source_group"]),
            source_part_key=str(row["source_part_key"]),
            part_key=str(row["part_key"]),
            metrics_key=str(row["metrics_key"]),
            done_key=str(row["done_key"]),
            error_key=str(row["error_key"]),
            text_column=str(row["text_column"]),
            dropped_column=str(row["dropped_column"]),
            tokenizer_name=str(row["tokenizer_name"]),
            tokenizer_object_key=str(row["tokenizer_object_key"]),
            tokenizer_json_sha256=str(row["tokenizer_json_sha256"]),
            read_batch_rows=int(row["read_batch_rows"]),
            rows_per_row_group=int(row["rows_per_row_group"]),
            parquet_compression=str(row["parquet_compression"]),
            parquet_compression_level=int(row["parquet_compression_level"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetTokenizationResult:
    status: str
    task_index: int
    source_group: str
    source_part_key: str
    part_key: str
    metrics_key: str
    done_key: str
    error_key: str
    output_row_count: int
    duration_sec: float
    metrics_json: str = "{}"
    error_message: str = ""

    @classmethod
    def success_from_task(
        cls,
        *,
        task: DatasetTokenizationTask,
        metrics: dict[str, Any],
        duration_sec: float,
    ) -> "DatasetTokenizationResult":
        return cls(
            status="success",
            task_index=task.task_index,
            source_group=task.source_group,
            source_part_key=task.source_part_key,
            part_key=task.part_key,
            metrics_key=task.metrics_key,
            done_key=task.done_key,
            error_key=task.error_key,
            output_row_count=int(metrics.get("row_count", 0)),
            duration_sec=duration_sec,
            metrics_json=json.dumps(metrics, ensure_ascii=False, sort_keys=True),
        )

    @classmethod
    def failed_from_task(
        cls,
        *,
        task: DatasetTokenizationTask,
        error_message: str,
        duration_sec: float,
    ) -> "DatasetTokenizationResult":
        return cls(
            status="failed",
            task_index=task.task_index,
            source_group=task.source_group,
            source_part_key=task.source_part_key,
            part_key=task.part_key,
            metrics_key=task.metrics_key,
            done_key=task.done_key,
            error_key=task.error_key,
            output_row_count=0,
            duration_sec=duration_sec,
            error_message=error_message,
        )

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DatasetTokenizationResult":
        return cls(
            status=str(row["status"]),
            task_index=int(row["task_index"]),
            source_group=str(row["source_group"]),
            source_part_key=str(row["source_part_key"]),
            part_key=str(row["part_key"]),
            metrics_key=str(row["metrics_key"]),
            done_key=str(row["done_key"]),
            error_key=str(row["error_key"]),
            output_row_count=int(row.get("output_row_count", 0)),
            duration_sec=float(row.get("duration_sec", 0.0) or 0.0),
            metrics_json=str(row.get("metrics_json", "{}")),
            error_message=str(row.get("error_message", "")),
        )

    def metrics(self) -> dict[str, Any]:
        loaded = json.loads(self.metrics_json or "{}")
        if not isinstance(loaded, dict):
            raise ValueError("Dataset tokenization metrics_json must decode to an object.")
        return loaded

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def part_source_key(task: DatasetTokenizationTask) -> str:
    return task.source_part_key
