from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
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
from ...ops.source_uninformative_cleaning import SOURCE_CHOICES

SOURCE_FORMATS = {"parquet"}
SOURCE_ROW_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class InputConfig:
    source_root_key: str

    def __post_init__(self) -> None:
        self.source_root_key = self.source_root_key.strip()
        if not self.source_root_key:
            raise ValueError("input.source_root_key must be non-empty.")


@dataclass
class CleaningConfig:
    ray_num_cpus_per_worker: float = 1.0
    polars_max_threads: int = 1

    def __post_init__(self) -> None:
        if self.ray_num_cpus_per_worker <= 0:
            raise ValueError("cleaning.ray_num_cpus_per_worker must be positive.")
        if self.polars_max_threads <= 0:
            raise ValueError("cleaning.polars_max_threads must be positive.")


@dataclass
class SourceSpec:
    name: str
    format: str
    r2_relative_glob_path: str
    text_column: str
    cleaning_source: str
    filters: dict[str, str] = field(default_factory=dict)
    scheduling_weight: int = 1

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.format = self.format.strip()
        self.r2_relative_glob_path = self.r2_relative_glob_path.strip()
        self.text_column = self.text_column.strip()
        self.cleaning_source = self.cleaning_source.strip().lower().replace("_", "-")
        self.filters = {str(key).strip(): str(value) for key, value in self.filters.items()}
        if not self.name:
            raise ValueError("source.name must be non-empty.")
        if self.format not in SOURCE_FORMATS:
            raise ValueError(
                f"source '{self.name}' format must be one of {sorted(SOURCE_FORMATS)}."
            )
        if not self.r2_relative_glob_path:
            raise ValueError(f"source '{self.name}' r2_relative_glob_path must be non-empty.")
        if self.r2_relative_glob_path.startswith(("s3://", "r2://")):
            raise ValueError(
                f"source '{self.name}' r2_relative_glob_path must be bucket-relative."
            )
        if not self.text_column:
            raise ValueError(f"source '{self.name}' text_column is required.")
        if self.cleaning_source not in SOURCE_CHOICES:
            raise ValueError(
                f"source '{self.name}' cleaning_source must be one of {list(SOURCE_CHOICES)}."
            )
        if any(not key for key in self.filters):
            raise ValueError(f"source '{self.name}' filters must not contain empty column names.")
        if self.scheduling_weight <= 0:
            raise ValueError(f"source '{self.name}' scheduling_weight must be positive.")

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SourceSpec":
        return cls(
            name=str(row["name"]),
            format=str(row["format"]),
            r2_relative_glob_path=str(row["r2_relative_glob_path"]),
            text_column=str(row["text_column"]),
            cleaning_source=str(row["cleaning_source"]),
            filters=dict(row.get("filters", {})),
            scheduling_weight=int(row.get("scheduling_weight", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    cleaning: CleaningConfig
    mlflow: MlflowConfig
    observability: ObservabilityConfig
    resumability: ResumeConfig
    sources: list[SourceSpec]
    ops: list[OpConfig]


@dataclass
class SourceCleaningRowGroupTask:
    source_order: int
    source_name: str
    source_scheduling_weight: int
    cleaning_source: str
    source_bucket: str
    source_object_key: str
    source_parquet_url: str
    source_row_group_index: int
    source_row_group_start_index: int
    source_row_group_num_rows: int
    text_column: str
    filters: dict[str, str]
    source_shard_key: str
    unified_shard_key: str
    metrics_key: str
    done_key: str
    error_key: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SourceCleaningRowGroupTask":
        return cls(
            source_order=int(row["source_order"]),
            source_name=str(row["source_name"]),
            source_scheduling_weight=int(row.get("source_scheduling_weight", 1)),
            cleaning_source=str(row["cleaning_source"]),
            source_bucket=str(row["source_bucket"]),
            source_object_key=str(row["source_object_key"]),
            source_parquet_url=str(row["source_parquet_url"]),
            source_row_group_index=int(row["source_row_group_index"]),
            source_row_group_start_index=int(row["source_row_group_start_index"]),
            source_row_group_num_rows=int(row["source_row_group_num_rows"]),
            text_column=str(row["text_column"]),
            filters=normalize_filter_values(row.get("filters", {})),
            source_shard_key=str(row["source_shard_key"]),
            unified_shard_key=str(row["unified_shard_key"]),
            metrics_key=str(row["metrics_key"]),
            done_key=str(row["done_key"]),
            error_key=str(row["error_key"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceCleaningShardResult:
    source_name: str
    cleaning_source: str
    source_object_key: str
    source_row_group_index: int
    status: str
    source_shard_key: str
    unified_shard_key: str
    metrics_key: str
    done_key: str
    error_key: str
    metrics_json: str = "{}"
    error_message: str = ""
    duration_sec: float = 0.0

    @classmethod
    def success_from_task(
        cls,
        *,
        task: SourceCleaningRowGroupTask,
        metrics: dict[str, Any],
        duration_sec: float,
    ) -> "SourceCleaningShardResult":
        return cls(
            source_name=task.source_name,
            cleaning_source=task.cleaning_source,
            source_object_key=task.source_object_key,
            source_row_group_index=task.source_row_group_index,
            status="success",
            source_shard_key=task.source_shard_key,
            unified_shard_key=task.unified_shard_key,
            metrics_key=task.metrics_key,
            done_key=task.done_key,
            error_key=task.error_key,
            metrics_json=stable_json(metrics),
            duration_sec=duration_sec,
        )

    @classmethod
    def failed_from_task(
        cls,
        *,
        task: SourceCleaningRowGroupTask,
        error_message: str,
        duration_sec: float,
    ) -> "SourceCleaningShardResult":
        return cls(
            source_name=task.source_name,
            cleaning_source=task.cleaning_source,
            source_object_key=task.source_object_key,
            source_row_group_index=task.source_row_group_index,
            status="failed",
            source_shard_key=task.source_shard_key,
            unified_shard_key=task.unified_shard_key,
            metrics_key=task.metrics_key,
            done_key=task.done_key,
            error_key=task.error_key,
            error_message=error_message,
            duration_sec=duration_sec,
        )

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SourceCleaningShardResult":
        return cls(
            source_name=str(row["source_name"]),
            cleaning_source=str(row["cleaning_source"]),
            source_object_key=str(row["source_object_key"]),
            source_row_group_index=int(row["source_row_group_index"]),
            status=str(row["status"]),
            source_shard_key=str(row["source_shard_key"]),
            unified_shard_key=str(row["unified_shard_key"]),
            metrics_key=str(row["metrics_key"]),
            done_key=str(row["done_key"]),
            error_key=str(row["error_key"]),
            metrics_json=str(row.get("metrics_json", "{}")),
            error_message=str(row.get("error_message", "")),
            duration_sec=float(row.get("duration_sec", 0.0)),
        )

    def metrics(self) -> dict[str, Any]:
        import json

        loaded = json.loads(self.metrics_json or "{}")
        if not isinstance(loaded, dict):
            raise ValueError("Source cleaning result metrics_json must decode to an object.")
        return loaded

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_filter_values(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    filters: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key).strip()
        if not key or raw_value is None:
            continue
        filter_value = str(raw_value).strip()
        if filter_value:
            filters[key] = filter_value
    return filters


def row_group_source_key(task: SourceCleaningRowGroupTask) -> str:
    return (
        f"{task.source_object_key}#row_group={task.source_row_group_index}"
        f"#cleaning_source={task.cleaning_source}"
    )


def source_slug(source: str) -> str:
    slug = SOURCE_ROW_SAFE_PATTERN.sub("-", source.strip().lower()).strip("-")
    return slug or "source"


def sample_uid(*, bucket: str, source_object_key: str, source_row_index: int) -> str:
    return f"r2://{bucket}/{source_object_key.lstrip('/')}#row={source_row_index}"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)
