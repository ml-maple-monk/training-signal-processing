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

    def __post_init__(self) -> None:
        if self.marker_ocr_resources.num_gpus is None:
            raise ValueError("ray.marker_ocr_resources.num_gpus is required.")
        if self.marker_ocr_resources.num_cpus is None:
            raise ValueError("ray.marker_ocr_resources.num_cpus is required.")
        if self.marker_ocr_resources.num_gpus <= 0:
            raise ValueError("ray.marker_ocr_resources.num_gpus must be positive.")
        if self.marker_ocr_resources.num_cpus <= 0:
            raise ValueError("ray.marker_ocr_resources.num_cpus must be positive.")


@dataclass
class InputConfig:
    local_pdf_root: str
    include_glob: str
    raw_pdf_prefix: str
    upload_transfers: int
    upload_checkers: int
    max_files: int | None = None

    def __post_init__(self) -> None:
        if self.upload_transfers <= 0:
            raise ValueError("input.upload_transfers must be positive.")
        if self.upload_checkers <= 0:
            raise ValueError("input.upload_checkers must be positive.")
        if self.max_files is not None and self.max_files <= 0:
            raise ValueError("input.max_files must be positive when provided.")


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
    def pending_from_task(
        cls,
        *,
        task: PdfTask,
        run_id: str,
        markdown_r2_key: str,
    ) -> "DocumentResult":
        return cls(
            run_id=run_id,
            source_r2_key=task.source_r2_key,
            relative_path=task.relative_path,
            markdown_r2_key=markdown_r2_key,
            status="pending",
            error_message="",
            source_sha256=task.source_sha256,
            source_size_bytes=task.source_size_bytes,
            started_at="",
            finished_at="",
            duration_sec=0.0,
            marker_exit_code=0,
            markdown_text="",
        )

    @classmethod
    def success_from_row(
        cls,
        row: dict[str, Any],
        *,
        run_id: str,
        started_at: str,
        finished_at: str,
        duration_sec: float,
        markdown_text: str,
        diagnostics: dict[str, Any],
    ) -> "DocumentResult":
        base = cls.from_dict({**row, "run_id": run_id})
        return cls(
            run_id=run_id,
            source_r2_key=base.source_r2_key,
            relative_path=base.relative_path,
            markdown_r2_key=base.markdown_r2_key,
            status="success",
            error_message="",
            source_sha256=base.source_sha256,
            source_size_bytes=base.source_size_bytes,
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=duration_sec,
            marker_exit_code=0,
            markdown_text=markdown_text,
            staged_pdf_path="",
            diagnostics=diagnostics,
        )

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
