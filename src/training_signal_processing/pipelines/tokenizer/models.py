from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ...models import (
    MlflowConfig,
    ObservabilityConfig,
    OpConfig,
    R2Config,
    RayConfig,
    RemoteRuntimeConfig,
    SshConfig,
)


@dataclass
class ParquetFamilySpec:
    name: str
    glob: str
    text_column: str
    id_columns: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ParquetFamilySpec":
        return cls(
            name=str(row["name"]),
            glob=str(row["glob"]),
            text_column=str(row["text_column"]),
            id_columns=[str(value) for value in row.get("id_columns", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenizerConfig:
    model_name: str
    output_compression: str = "gzip"


@dataclass
class InputConfig:
    source_prefix: str
    family_specs: list[ParquetFamilySpec]


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
    mlflow: MlflowConfig
    observability: ObservabilityConfig
    resumability: ResumeConfig
    ops: list[OpConfig]


@dataclass
class ParquetShardTask:
    family_name: str
    source_r2_key: str
    source_r2_url: str
    source_rel_path: str
    text_column: str
    id_columns: list[str]
    output_r2_key: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ParquetShardTask":
        return cls(
            family_name=str(row["family_name"]),
            source_r2_key=str(row["source_r2_key"]),
            source_r2_url=str(row["source_r2_url"]),
            source_rel_path=str(row["source_rel_path"]),
            text_column=str(row["text_column"]),
            id_columns=[str(value) for value in row.get("id_columns", [])],
            output_r2_key=str(row["output_r2_key"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenizedRowResult:
    source_family: str
    source_r2_key: str
    row_id: str
    text_column: str
    tokenizer_model: str
    token_count: int
    token_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenizedShardResult:
    run_id: str
    family_name: str
    source_r2_key: str
    source_rel_path: str
    output_r2_key: str
    status: str
    error_message: str
    text_column: str
    tokenizer_model: str
    row_count: int
    tokenized_row_count: int
    started_at: str
    finished_at: str
    duration_sec: float
    output_written: bool = False

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "TokenizedShardResult":
        return cls(
            run_id=str(row["run_id"]),
            family_name=str(row["family_name"]),
            source_r2_key=str(row["source_r2_key"]),
            source_rel_path=str(row["source_rel_path"]),
            output_r2_key=str(row["output_r2_key"]),
            status=str(row["status"]),
            error_message=str(row.get("error_message", "")),
            text_column=str(row["text_column"]),
            tokenizer_model=str(row["tokenizer_model"]),
            row_count=int(row.get("row_count", 0)),
            tokenized_row_count=int(row.get("tokenized_row_count", 0)),
            started_at=str(row.get("started_at", "")),
            finished_at=str(row.get("finished_at", "")),
            duration_sec=float(row.get("duration_sec", 0.0)),
            output_written=bool(row.get("output_written", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
