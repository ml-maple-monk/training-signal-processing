from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
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

SOURCE_FORMATS = {"markdown", "parquet", "text"}
DEFAULT_TOKEN_ENCODING = "o200k_base"


@dataclass
class TokenizerConfig:
    encoding: str = DEFAULT_TOKEN_ENCODING

    def __post_init__(self) -> None:
        if not self.encoding.strip():
            self.encoding = DEFAULT_TOKEN_ENCODING


@dataclass
class SourceAccountingInputConfig:
    plan_path: str
    source_root_key: str

    def __post_init__(self) -> None:
        if not self.plan_path.strip():
            raise ValueError("input.plan_path must be non-empty.")
        if not self.source_root_key.strip():
            raise ValueError("input.source_root_key must be non-empty.")


@dataclass
class SourceSpec:
    name: str
    format: str
    r2_relative_glob_path: str
    text_column: str = ""
    parquet_batch_size: int = 2048
    count_concurrency: int = 8
    filters: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        self.format = self.format.strip()
        self.r2_relative_glob_path = self.r2_relative_glob_path.strip()
        self.text_column = self.text_column.strip()
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
        if self.format == "parquet" and not self.text_column:
            raise ValueError(f"source '{self.name}' text_column is required for parquet.")
        if self.format in {"markdown", "text"} and self.text_column:
            raise ValueError(
                f"source '{self.name}' text_column is only supported for parquet sources."
            )
        if self.format in {"markdown", "text"} and self.filters:
            raise ValueError(
                f"source '{self.name}' filters are only supported for parquet sources."
            )
        if any(not key for key in self.filters):
            raise ValueError(f"source '{self.name}' filters must not contain empty column names.")
        if self.parquet_batch_size <= 0:
            raise ValueError(f"source '{self.name}' parquet_batch_size must be positive.")
        if self.count_concurrency <= 0:
            raise ValueError(f"source '{self.name}' count_concurrency must be positive.")

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SourceSpec":
        return cls(
            name=str(row["name"]),
            format=str(row["format"]),
            r2_relative_glob_path=str(row["r2_relative_glob_path"]),
            text_column=str(row.get("text_column", "")),
            parquet_batch_size=int(row.get("parquet_batch_size", 2048)),
            count_concurrency=int(row.get("count_concurrency", 8)),
            filters=dict(row.get("filters", {})),
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
    input: SourceAccountingInputConfig
    tokenizer: TokenizerConfig
    mlflow: MlflowConfig
    observability: ObservabilityConfig
    resumability: ResumeConfig
    sources: list[SourceSpec]
    ops: list[OpConfig]


@dataclass
class SourceAccountingTask:
    source_order: int
    source: str
    format: str
    r2_relative_glob_path: str
    text_column: str
    parquet_batch_size: int
    count_concurrency: int
    filters: dict[str, str]
    token_encoding: str
    source_row_r2_key: str
    table_r2_key: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SourceAccountingTask":
        return cls(
            source_order=int(row["source_order"]),
            source=str(row["source"]),
            format=str(row["format"]),
            r2_relative_glob_path=str(row["r2_relative_glob_path"]),
            text_column=str(row.get("text_column", "")),
            parquet_batch_size=int(row.get("parquet_batch_size", 2048)),
            count_concurrency=int(row.get("count_concurrency", 8)),
            filters=dict(row.get("filters", {})),
            token_encoding=str(row["token_encoding"]),
            source_row_r2_key=str(row["source_row_r2_key"]),
            table_r2_key=str(row["table_r2_key"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceAccountingResult:
    source_order: int
    source: str
    token_count: int
    word_count: int
    byte_count: int
    document_count: int
    r2_relative_glob_path: str
    filters: dict[str, str]
    metadata_columns: list[str]
    source_row_r2_key: str
    table_r2_key: str
    status: str = "success"
    error_message: str = ""
    duration_sec: float = 0.0

    @classmethod
    def failed_from_task(
        cls,
        *,
        task: SourceAccountingTask,
        error_message: str,
        duration_sec: float,
    ) -> "SourceAccountingResult":
        return cls(
            source_order=task.source_order,
            source=task.source,
            token_count=0,
            word_count=0,
            byte_count=0,
            document_count=0,
            r2_relative_glob_path=task.r2_relative_glob_path,
            filters=task.filters,
            metadata_columns=[],
            source_row_r2_key=task.source_row_r2_key,
            table_r2_key=task.table_r2_key,
            status="failed",
            error_message=error_message,
            duration_sec=duration_sec,
        )

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SourceAccountingResult":
        return cls(
            source_order=int(row["source_order"]),
            source=str(row["source"]),
            token_count=int(row["token_count"]),
            word_count=int(row["word_count"]),
            byte_count=int(row["byte_count"]),
            document_count=int(row["document_count"]),
            r2_relative_glob_path=str(row["r2_relative_glob_path"]),
            filters=dict(row.get("filters", {})),
            metadata_columns=list(row.get("metadata_columns", [])),
            source_row_r2_key=str(row["source_row_r2_key"]),
            table_r2_key=str(row["table_r2_key"]),
            status=str(row.get("status", "success")),
            error_message=str(row.get("error_message", "")),
            duration_sec=float(row.get("duration_sec", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SOURCE_ROW_SAFE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def source_slug(source: str) -> str:
    slug = SOURCE_ROW_SAFE_PATTERN.sub("-", source.strip().lower()).strip("-")
    return slug or "source"


def extract_required_sources_from_plan(plan_path: Path) -> list[str]:
    if not plan_path.is_file():
        raise ValueError(f"Source accounting plan file not found: {plan_path}")
    sources: list[str] = []
    section = ""
    parent = ""
    for raw_line in plan_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("Write a table"):
            section = ""
            parent = ""
            continue
        if stripped.startswith("Books + OCR"):
            append_unique(sources, "Books + OCR")
            section = ""
            parent = ""
            continue
        if stripped.startswith("Malay forums data"):
            section = "forums"
            parent = ""
            continue
        if stripped.startswith("HPLT data"):
            section = "hplt"
            parent = ""
            continue
        if section == "forums":
            if stripped == "Reddit":
                parent = "Reddit"
                continue
            if stripped in {"Lowyat", "Cari"}:
                append_unique(sources, stripped)
                parent = ""
                continue
            if parent == "Reddit":
                append_unique(sources, f"Reddit {stripped}")
                continue
        if section == "hplt" and stripped in {"Malay", "Indonesia"}:
            append_unique(sources, f"HPLT {stripped}")
    return sources


def append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def render_markdown_table(results: list[SourceAccountingResult]) -> str:
    ordered = sorted(results, key=lambda item: (item.source_order, item.source))
    lines = [
        "| source | token_count | word_count | byte_count | document_count | "
        "r2_relative_glob_path | filters | metadata_columns |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for result in ordered:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_markdown_cell(result.source),
                    format_count(result.token_count),
                    format_count(result.word_count),
                    format_count(result.byte_count),
                    format_count(result.document_count),
                    f"`{escape_markdown_cell(result.r2_relative_glob_path)}`",
                    f"`{escape_markdown_cell(render_filters(result.filters))}`",
                    f"`{escape_markdown_cell(', '.join(result.metadata_columns))}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def format_count(value: int) -> str:
    return f"{value:,}"


def escape_markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render_filters(filters: dict[str, str]) -> str:
    if not filters:
        return ""
    return ", ".join(f"{key}={value}" for key, value in sorted(filters.items()))
