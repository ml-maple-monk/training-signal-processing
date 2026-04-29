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

SOURCE_FORMATS = {"parquet"}
DEFAULT_PASS_THROUGH_COLUMNS = (
    "document_id",
    "id",
    "url",
    "permalink",
    "thread_url",
    "subreddit",
    "language",
    "row_language_code",
)
DEFAULT_REFERENCE_HEADINGS = (
    "REFERENCES",
    "Bibliography",
    "Daftar Pustaka",
    "Rujukan",
)
SOURCE_ROW_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
LID_INNER_PARALLELISM_MODES = {"none", "thread_pool", "process_pool"}
MULTIPROCESSING_CONTEXTS = {"spawn"}


@dataclass
class ReferenceRemovalConfig:
    enabled: bool = False
    url_column: str = ""
    heading_names: tuple[str, ...] = DEFAULT_REFERENCE_HEADINGS
    books_ocr_cleanup_enabled: bool = False

    def __post_init__(self) -> None:
        self.url_column = self.url_column.strip()
        self.heading_names = tuple(
            heading.strip() for heading in self.heading_names if heading.strip()
        )
        if self.enabled and not self.heading_names:
            raise ValueError("reference_removal.heading_names must be non-empty when enabled.")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ReferenceRemovalConfig":
        values = dict(raw or {})
        headings = values.get("heading_names", DEFAULT_REFERENCE_HEADINGS)
        return cls(
            enabled=bool(values.get("enabled", False)),
            url_column=str(values.get("url_column", "")),
            heading_names=tuple(str(heading) for heading in headings),
            books_ocr_cleanup_enabled=bool(values.get("books_ocr_cleanup_enabled", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "url_column": self.url_column,
            "heading_names": list(self.heading_names),
            "books_ocr_cleanup_enabled": self.books_ocr_cleanup_enabled,
        }


@dataclass
class SourceSpec:
    name: str
    format: str
    r2_relative_glob_path: str
    text_column: str
    filters: dict[str, str] = field(default_factory=dict)
    pass_through_columns: tuple[str, ...] = DEFAULT_PASS_THROUGH_COLUMNS
    reference_removal: ReferenceRemovalConfig = field(default_factory=ReferenceRemovalConfig)
    parquet_batch_size: int = 1

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.format = self.format.strip()
        self.r2_relative_glob_path = self.r2_relative_glob_path.strip()
        self.text_column = self.text_column.strip()
        self.filters = {str(key).strip(): str(value) for key, value in self.filters.items()}
        self.pass_through_columns = tuple(
            str(column).strip() for column in self.pass_through_columns if str(column).strip()
        )
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
        if any(not key for key in self.filters):
            raise ValueError(f"source '{self.name}' filters must not contain empty column names.")
        if self.parquet_batch_size <= 0:
            raise ValueError(f"source '{self.name}' parquet_batch_size must be positive.")

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SourceSpec":
        return cls(
            name=str(row["name"]),
            format=str(row["format"]),
            r2_relative_glob_path=str(row["r2_relative_glob_path"]),
            text_column=str(row["text_column"]),
            filters=dict(row.get("filters", {})),
            pass_through_columns=tuple(
                row.get("pass_through_columns", DEFAULT_PASS_THROUGH_COLUMNS)
            ),
            reference_removal=ReferenceRemovalConfig.from_dict(
                row.get("reference_removal")
                if isinstance(row.get("reference_removal"), dict)
                else None
            ),
            parquet_batch_size=int(row.get("parquet_batch_size", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "format": self.format,
            "r2_relative_glob_path": self.r2_relative_glob_path,
            "text_column": self.text_column,
            "filters": dict(self.filters),
            "pass_through_columns": list(self.pass_through_columns),
            "reference_removal": self.reference_removal.to_dict(),
            "parquet_batch_size": self.parquet_batch_size,
        }


@dataclass
class InputConfig:
    source_root_key: str
    shard_rows_per_file: int = 0

    def __post_init__(self) -> None:
        self.source_root_key = self.source_root_key.strip()
        if not self.source_root_key:
            raise ValueError("input.source_root_key must be non-empty.")
        if self.shard_rows_per_file < 0:
            raise ValueError("input.shard_rows_per_file must be zero or positive.")


@dataclass
class LidConfig:
    lingua_languages: tuple[str, ...] = ("english", "malay", "indonesian")
    malaya_fasttext_quantized: bool = True
    tokenizer_encoding: str = "o200k_base"
    row_batch_size: int = 0
    ray_num_cpus_per_worker: float = 1.0
    inner_parallelism: str = "none"
    inner_workers: int = 1
    checkpoint_every_rows: int = 1000
    checkpoint_every_seconds: float = 30.0
    progress_log_every_seconds: float = 30.0
    experiment_name: str = "lid-metadata"
    variant_name: str = "baseline"
    multiprocessing_context: str = "spawn"
    process_pool_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        self.lingua_languages = tuple(
            language.strip().lower() for language in self.lingua_languages if language.strip()
        )
        required = {"english", "malay", "indonesian"}
        if set(self.lingua_languages) != required:
            raise ValueError("lid.lingua_languages must be exactly english, malay, indonesian.")
        self.tokenizer_encoding = self.tokenizer_encoding.strip()
        self.inner_parallelism = self.inner_parallelism.strip()
        self.experiment_name = self.experiment_name.strip()
        self.variant_name = self.variant_name.strip()
        self.multiprocessing_context = self.multiprocessing_context.strip()
        if not self.tokenizer_encoding:
            raise ValueError("lid.tokenizer_encoding must be non-empty.")
        if self.row_batch_size < 0:
            raise ValueError("lid.row_batch_size must be zero or positive.")
        if self.ray_num_cpus_per_worker <= 0:
            raise ValueError("lid.ray_num_cpus_per_worker must be positive.")
        if self.inner_parallelism not in LID_INNER_PARALLELISM_MODES:
            raise ValueError(
                "lid.inner_parallelism must be one of "
                + ", ".join(sorted(LID_INNER_PARALLELISM_MODES))
                + "."
            )
        if self.inner_workers <= 0:
            raise ValueError("lid.inner_workers must be positive.")
        if self.checkpoint_every_rows <= 0:
            raise ValueError("lid.checkpoint_every_rows must be positive.")
        if self.checkpoint_every_seconds <= 0:
            raise ValueError("lid.checkpoint_every_seconds must be positive.")
        if self.progress_log_every_seconds <= 0:
            raise ValueError("lid.progress_log_every_seconds must be positive.")
        if not self.experiment_name:
            raise ValueError("lid.experiment_name must be non-empty.")
        if not self.variant_name:
            raise ValueError("lid.variant_name must be non-empty.")
        if self.multiprocessing_context not in MULTIPROCESSING_CONTEXTS:
            raise ValueError(
                "lid.multiprocessing_context must be one of "
                + ", ".join(sorted(MULTIPROCESSING_CONTEXTS))
                + "."
            )
        if self.process_pool_timeout_seconds <= 0:
            raise ValueError("lid.process_pool_timeout_seconds must be positive.")


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
    lid: LidConfig
    mlflow: MlflowConfig
    observability: ObservabilityConfig
    resumability: ResumeConfig
    sources: list[SourceSpec]
    ops: list[OpConfig]


@dataclass
class ParquetRowGroupTask:
    source_order: int
    source_name: str
    source_bucket: str
    source_object_key: str
    source_parquet_url: str
    source_row_group_index: int
    source_row_group_start_index: int
    source_row_group_num_rows: int
    text_column: str
    filters: dict[str, str]
    pass_through_columns: tuple[str, ...]
    reference_removal: ReferenceRemovalConfig
    output_shard_key: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ParquetRowGroupTask":
        return cls(
            source_order=int(row["source_order"]),
            source_name=str(row["source_name"]),
            source_bucket=str(row["source_bucket"]),
            source_object_key=str(row["source_object_key"]),
            source_parquet_url=str(row["source_parquet_url"]),
            source_row_group_index=int(row["source_row_group_index"]),
            source_row_group_start_index=int(row["source_row_group_start_index"]),
            source_row_group_num_rows=int(row["source_row_group_num_rows"]),
            text_column=str(row["text_column"]),
            filters=normalize_filter_values(row.get("filters", {})),
            pass_through_columns=tuple(row.get("pass_through_columns", ())),
            reference_removal=ReferenceRemovalConfig.from_dict(
                row.get("reference_removal")
                if isinstance(row.get("reference_removal"), dict)
                else None
            ),
            output_shard_key=str(row["output_shard_key"]),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pass_through_columns"] = list(self.pass_through_columns)
        payload["reference_removal"] = self.reference_removal.to_dict()
        return payload


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


@dataclass
class LidMetadataShardResult:
    source_name: str
    source_object_key: str
    source_row_group_index: int
    output_shard_key: str
    status: str
    records: list[dict[str, Any]] = field(default_factory=list)
    error_message: str = ""
    duration_sec: float = 0.0
    row_count: int = 0
    success_count: int = 0
    failed_count: int = 0
    cleaned_token_count: int = 0
    tokens_per_sec: float = 0.0
    rows_per_sec: float = 0.0
    experiment_name: str = ""
    variant_name: str = ""
    inner_parallelism: str = ""
    inner_workers: int = 0
    row_batch_size: int = 0
    checkpoint_key: str = ""
    parallelism_fallback_reason: str = ""

    @classmethod
    def success_from_task(
        cls,
        *,
        task: ParquetRowGroupTask,
        records: list[dict[str, Any]],
        duration_sec: float,
        metrics: dict[str, Any] | None = None,
    ) -> "LidMetadataShardResult":
        values = dict(metrics or {})
        return cls(
            source_name=task.source_name,
            source_object_key=task.source_object_key,
            source_row_group_index=task.source_row_group_index,
            output_shard_key=task.output_shard_key,
            status="success",
            records=records,
            duration_sec=duration_sec,
            row_count=int(values.get("row_count", len(records))),
            success_count=int(values.get("success_count", len(records))),
            failed_count=int(values.get("failed_count", 0)),
            cleaned_token_count=int(values.get("cleaned_token_count", 0)),
            tokens_per_sec=float(values.get("tokens_per_sec", 0.0)),
            rows_per_sec=float(values.get("rows_per_sec", 0.0)),
            experiment_name=str(values.get("experiment_name", "")),
            variant_name=str(values.get("variant_name", "")),
            inner_parallelism=str(values.get("inner_parallelism", "")),
            inner_workers=int(values.get("inner_workers", 0)),
            row_batch_size=int(values.get("row_batch_size", 0)),
            checkpoint_key=str(values.get("checkpoint_key", "")),
            parallelism_fallback_reason=str(
                values.get("parallelism_fallback_reason", "")
            ),
        )

    @classmethod
    def failed_from_task(
        cls,
        *,
        task: ParquetRowGroupTask,
        error_message: str,
        duration_sec: float,
        metrics: dict[str, Any] | None = None,
    ) -> "LidMetadataShardResult":
        values = dict(metrics or {})
        return cls(
            source_name=task.source_name,
            source_object_key=task.source_object_key,
            source_row_group_index=task.source_row_group_index,
            output_shard_key=task.output_shard_key,
            status="failed",
            error_message=error_message,
            duration_sec=duration_sec,
            row_count=int(values.get("row_count", 0)),
            success_count=int(values.get("success_count", 0)),
            failed_count=int(values.get("failed_count", 1)),
            cleaned_token_count=int(values.get("cleaned_token_count", 0)),
            tokens_per_sec=float(values.get("tokens_per_sec", 0.0)),
            rows_per_sec=float(values.get("rows_per_sec", 0.0)),
            experiment_name=str(values.get("experiment_name", "")),
            variant_name=str(values.get("variant_name", "")),
            inner_parallelism=str(values.get("inner_parallelism", "")),
            inner_workers=int(values.get("inner_workers", 0)),
            row_batch_size=int(values.get("row_batch_size", 0)),
            checkpoint_key=str(values.get("checkpoint_key", "")),
            parallelism_fallback_reason=str(
                values.get("parallelism_fallback_reason", "")
            ),
        )

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "LidMetadataShardResult":
        return cls(
            source_name=str(row["source_name"]),
            source_object_key=str(row["source_object_key"]),
            source_row_group_index=int(row["source_row_group_index"]),
            output_shard_key=str(row["output_shard_key"]),
            status=str(row["status"]),
            records=list(row.get("records", [])),
            error_message=str(row.get("error_message", "")),
            duration_sec=float(row.get("duration_sec", 0.0)),
            row_count=int(row.get("row_count", 0)),
            success_count=int(row.get("success_count", 0)),
            failed_count=int(row.get("failed_count", 0)),
            cleaned_token_count=int(row.get("cleaned_token_count", 0)),
            tokens_per_sec=float(row.get("tokens_per_sec", 0.0)),
            rows_per_sec=float(row.get("rows_per_sec", 0.0)),
            experiment_name=str(row.get("experiment_name", "")),
            variant_name=str(row.get("variant_name", "")),
            inner_parallelism=str(row.get("inner_parallelism", "")),
            inner_workers=int(row.get("inner_workers", 0)),
            row_batch_size=int(row.get("row_batch_size", 0)),
            checkpoint_key=str(row.get("checkpoint_key", "")),
            parallelism_fallback_reason=str(row.get("parallelism_fallback_reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def source_slug(source: str) -> str:
    slug = SOURCE_ROW_SAFE_PATTERN.sub("-", source.strip().lower()).strip("-")
    return slug or "source"


def sample_uid(*, bucket: str, source_object_key: str, source_row_index: int) -> str:
    return f"r2://{bucket}/{source_object_key.lstrip('/')}#row={source_row_index}"


def sample_uid_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
