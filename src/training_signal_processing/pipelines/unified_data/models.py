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


@dataclass
class InputConfig:
    source_root_key: str
    lid_metadata_output_prefix: str
    lid_run_id: str
    source_cleaning_output_prefix: str
    source_cleaning_run_id: str
    allow_partial_upstream: bool = False
    include_cleaning_sources: str = ""

    def __post_init__(self) -> None:
        self.source_root_key = self.source_root_key.strip()
        self.lid_metadata_output_prefix = self.lid_metadata_output_prefix.strip().strip("/")
        self.lid_run_id = self.lid_run_id.strip()
        self.source_cleaning_output_prefix = self.source_cleaning_output_prefix.strip().strip("/")
        self.source_cleaning_run_id = self.source_cleaning_run_id.strip()
        self.include_cleaning_sources = self.include_cleaning_sources.strip()
        if not self.source_root_key:
            raise ValueError("input.source_root_key must be non-empty.")
        if not self.lid_metadata_output_prefix:
            raise ValueError("input.lid_metadata_output_prefix must be non-empty.")
        if not self.lid_run_id:
            raise ValueError("input.lid_run_id must be non-empty.")
        if not self.source_cleaning_output_prefix:
            raise ValueError("input.source_cleaning_output_prefix must be non-empty.")
        if not self.source_cleaning_run_id:
            raise ValueError("input.source_cleaning_run_id must be non-empty.")


@dataclass
class ExportConfig:
    rows_per_row_group: int = 500_000
    tokenizer_encoding: str = SUPPORTED_TOKENIZER_ENCODING
    tokenizer_threads: int = 8
    ray_num_cpus_per_worker: float = 1.0
    merge_engine: str = "duckdb"
    duckdb_threads: int = 1
    parquet_compression: str = "zstd"
    parquet_compression_level: int = 1

    def __post_init__(self) -> None:
        self.tokenizer_encoding = self.tokenizer_encoding.strip()
        self.merge_engine = self.merge_engine.strip().lower()
        self.parquet_compression = self.parquet_compression.strip().lower()
        if self.rows_per_row_group <= 0:
            raise ValueError("export.rows_per_row_group must be positive.")
        if self.tokenizer_encoding != SUPPORTED_TOKENIZER_ENCODING:
            raise ValueError(
                "export.tokenizer_encoding must be "
                f"{SUPPORTED_TOKENIZER_ENCODING!r} for the fixed output schema."
            )
        if self.tokenizer_threads <= 0:
            raise ValueError("export.tokenizer_threads must be positive.")
        if self.ray_num_cpus_per_worker <= 0:
            raise ValueError("export.ray_num_cpus_per_worker must be positive.")
        if self.merge_engine not in {"duckdb"}:
            raise ValueError("export.merge_engine must be 'duckdb'.")
        if self.duckdb_threads <= 0:
            raise ValueError("export.duckdb_threads must be positive.")
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
class MergeSegment:
    source_name: str
    cleaning_source: str
    source_object_key: str
    source_row_group_index: int
    row_offset: int
    row_count: int
    lid_shard_key: str
    cleaning_unified_shard_key: str
    cleaning_metrics_key: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "MergeSegment":
        return cls(
            source_name=str(row["source_name"]),
            cleaning_source=str(row["cleaning_source"]),
            source_object_key=str(row["source_object_key"]),
            source_row_group_index=int(row["source_row_group_index"]),
            row_offset=int(row["row_offset"]),
            row_count=int(row["row_count"]),
            lid_shard_key=str(row["lid_shard_key"]),
            cleaning_unified_shard_key=str(row["cleaning_unified_shard_key"]),
            cleaning_metrics_key=str(row["cleaning_metrics_key"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnifiedDataPartTask:
    part_index: int
    part_key: str
    metrics_key: str
    done_key: str
    error_key: str
    expected_row_count: int
    rows_per_row_group: int
    tokenizer_encoding: str
    tokenizer_threads: int
    merge_engine: str
    duckdb_threads: int
    parquet_compression: str
    parquet_compression_level: int
    segments: tuple[MergeSegment, ...]

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "UnifiedDataPartTask":
        raw_segments = row.get("segments", [])
        if not isinstance(raw_segments, list):
            raise ValueError("Unified data part task segments must be a list.")
        return cls(
            part_index=int(row["part_index"]),
            part_key=str(row["part_key"]),
            metrics_key=str(row["metrics_key"]),
            done_key=str(row["done_key"]),
            error_key=str(row["error_key"]),
            expected_row_count=int(row["expected_row_count"]),
            rows_per_row_group=int(row["rows_per_row_group"]),
            tokenizer_encoding=str(row["tokenizer_encoding"]),
            tokenizer_threads=int(row["tokenizer_threads"]),
            merge_engine=str(row.get("merge_engine", "duckdb")),
            duckdb_threads=int(row.get("duckdb_threads", 1)),
            parquet_compression=str(row.get("parquet_compression", "zstd")),
            parquet_compression_level=int(row.get("parquet_compression_level", 1)),
            segments=tuple(MergeSegment.from_dict(item) for item in raw_segments),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["segments"] = [segment.to_dict() for segment in self.segments]
        return payload


@dataclass(frozen=True)
class UnifiedDataPartResult:
    status: str
    part_index: int
    part_key: str
    metrics_key: str
    done_key: str
    error_key: str
    expected_row_count: int
    output_row_count: int
    duration_sec: float
    metrics_json: str = "{}"
    error_message: str = ""

    @classmethod
    def success_from_task(
        cls,
        *,
        task: UnifiedDataPartTask,
        metrics: dict[str, Any],
        duration_sec: float,
    ) -> "UnifiedDataPartResult":
        return cls(
            status="success",
            part_index=task.part_index,
            part_key=task.part_key,
            metrics_key=task.metrics_key,
            done_key=task.done_key,
            error_key=task.error_key,
            expected_row_count=task.expected_row_count,
            output_row_count=int(metrics.get("row_count", 0)),
            duration_sec=duration_sec,
            metrics_json=json.dumps(metrics, ensure_ascii=False, sort_keys=True),
        )

    @classmethod
    def failed_from_task(
        cls,
        *,
        task: UnifiedDataPartTask,
        error_message: str,
        duration_sec: float,
    ) -> "UnifiedDataPartResult":
        return cls(
            status="failed",
            part_index=task.part_index,
            part_key=task.part_key,
            metrics_key=task.metrics_key,
            done_key=task.done_key,
            error_key=task.error_key,
            expected_row_count=task.expected_row_count,
            output_row_count=0,
            duration_sec=duration_sec,
            error_message=error_message,
        )

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "UnifiedDataPartResult":
        return cls(
            status=str(row["status"]),
            part_index=int(row["part_index"]),
            part_key=str(row["part_key"]),
            metrics_key=str(row["metrics_key"]),
            done_key=str(row["done_key"]),
            error_key=str(row["error_key"]),
            expected_row_count=int(row["expected_row_count"]),
            output_row_count=int(row.get("output_row_count", 0)),
            duration_sec=float(row.get("duration_sec", 0.0) or 0.0),
            metrics_json=str(row.get("metrics_json", "{}")),
            error_message=str(row.get("error_message", "")),
        )

    def metrics(self) -> dict[str, Any]:
        loaded = json.loads(self.metrics_json or "{}")
        if not isinstance(loaded, dict):
            raise ValueError("Unified data result metrics_json must decode to an object.")
        return loaded

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def part_source_key(task: UnifiedDataPartTask) -> str:
    return f"part={task.part_index:06d}"
