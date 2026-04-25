from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ...core.models import (
    MlflowConfig,
    ObservabilityConfig,
    OpConfig,
    R2Config,
    RayConfig,
    RayTransformResources,
    RemoteRuntimeConfig,
    SshConfig,
)


@dataclass
class OcrRayConfig(RayConfig):
    marker_ocr_resources: RayTransformResources


@dataclass
class InputConfig:
    local_pdf_root: str
    include_glob: str
    raw_pdf_prefix: str
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
    ray: OcrRayConfig
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
class DocumentResult:
    run_id: str
    source_r2_key: str
    relative_path: str
    markdown_r2_key: str
    status: str
    error_message: str
    source_sha256: str
    source_size_bytes: int
    started_at: str
    finished_at: str
    duration_sec: float
    marker_exit_code: int
    markdown_text: str = ""
    staged_pdf_path: str = ""
    diagnostics: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DocumentResult":
        return cls(
            run_id=str(row["run_id"]),
            source_r2_key=str(row["source_r2_key"]),
            relative_path=str(row["relative_path"]),
            markdown_r2_key=str(row.get("markdown_r2_key", "")),
            status=str(row["status"]),
            error_message=str(row.get("error_message", "")),
            source_sha256=str(row["source_sha256"]),
            source_size_bytes=int(row["source_size_bytes"]),
            started_at=str(row.get("started_at", "")),
            finished_at=str(row.get("finished_at", "")),
            duration_sec=float(row.get("duration_sec", 0.0)),
            marker_exit_code=int(row.get("marker_exit_code", 0)),
            markdown_text=str(row.get("markdown_text", "")),
            staged_pdf_path=str(row.get("staged_pdf_path", "")),
            diagnostics=(
                dict(row["diagnostics"])
                if isinstance(row.get("diagnostics"), dict)
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
