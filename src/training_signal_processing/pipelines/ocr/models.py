from __future__ import annotations

from dataclasses import asdict, dataclass
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
class InputConfig:
    local_pdf_root: str
    include_glob: str
    max_files: int | None = None


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
    mlflow: MlflowConfig
    observability: ObservabilityConfig
    resumability: ResumeConfig
    ops: list[OpConfig]


@dataclass
class PdfTask:
    source_r2_key: str
    relative_path: str
    source_size_bytes: int
    source_sha256: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "PdfTask":
        return cls(
            source_r2_key=str(row["source_r2_key"]),
            relative_path=str(row["relative_path"]),
            source_size_bytes=int(row["source_size_bytes"]),
            source_sha256=str(row["source_sha256"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeBindings:
    run_id: str
    input_manifest_key: str
    config_object_key: str = ""
    uploaded_documents: int = 0
    allow_overwrite: bool = False
    marker_binary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
