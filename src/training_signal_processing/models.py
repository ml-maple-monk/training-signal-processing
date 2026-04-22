from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


@dataclass
class SshConfig:
    host: str
    port: int
    user: str
    identity_file: str


@dataclass
class RemoteRuntimeConfig:
    root_dir: str
    python_version: str
    venv_dir: str = ".venv"


@dataclass
class RayConfig:
    executor_type: str
    batch_size: int
    concurrency: int
    target_num_blocks: int


@dataclass
class R2Config:
    config_file: str
    bucket: str
    raw_pdf_prefix: str
    output_prefix: str
    access_key_id: str = ""
    secret_access_key: str = ""
    region: str = ""
    endpoint_url: str = ""


@dataclass
class InputConfig:
    local_pdf_root: str
    include_glob: str
    max_files: int | None = None


@dataclass
class MlflowConfig:
    enabled: bool
    local_tracking_uri: str
    remote_tunnel_port: int
    experiment_name: str


@dataclass
class ObservabilityConfig:
    flush_interval_sec: int
    log_per_file_events: bool
    heartbeat_interval_sec: int


@dataclass
class ResumeConfig:
    strategy: str
    commit_every_batches: int
    resume_mode: str


@dataclass
class OpConfig:
    name: str
    type: str = ""
    options: dict[str, Any] = field(default_factory=dict)


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
class DocumentResult:
    run_id: str
    batch_id: str
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

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "DocumentResult":
        return cls(
            run_id=str(row["run_id"]),
            batch_id=str(row.get("batch_id", "")),
            source_r2_key=str(row["source_r2_key"]),
            relative_path=str(row["relative_path"]),
            markdown_r2_key=str(row.get("markdown_r2_key", "")),
            status=str(row["status"]),
            error_message=str(row.get("error_message", "")),
            source_sha256=str(row["source_sha256"]),
            source_size_bytes=int(row["source_size_bytes"]),
            started_at=str(row["started_at"]),
            finished_at=str(row["finished_at"]),
            duration_sec=float(row["duration_sec"]),
            marker_exit_code=int(row.get("marker_exit_code", 0)),
            markdown_text=str(row.get("markdown_text", "")),
        )

    def manifest_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("markdown_text", None)
        return row


@dataclass
class BatchCommit:
    batch_id: str
    row_count: int
    success_count: int
    failed_count: int
    skipped_count: int
    duration_sec: float
    manifest_key: str
    event_key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExportBatchResult:
    batch_id: str
    row_count: int
    output_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunState:
    run_id: str
    status: str
    total_documents: int
    pending_documents: int
    success_count: int
    failed_count: int
    skipped_count: int
    last_committed_batch: int
    started_at: str
    updated_at: str
    raw_prefix: str
    output_prefix: str
    mlflow_run_id: str
    error_message: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RunState":
        return cls(
            run_id=str(row["run_id"]),
            status=str(row["status"]),
            total_documents=int(row["total_documents"]),
            pending_documents=int(row["pending_documents"]),
            success_count=int(row["success_count"]),
            failed_count=int(row["failed_count"]),
            skipped_count=int(row["skipped_count"]),
            last_committed_batch=int(row["last_committed_batch"]),
            started_at=str(row["started_at"]),
            updated_at=str(row["updated_at"]),
            raw_prefix=str(row["raw_prefix"]),
            output_prefix=str(row["output_prefix"]),
            mlflow_run_id=str(row.get("mlflow_run_id", "")),
            error_message=str(row.get("error_message", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionLogEvent:
    level: str
    code: str
    message: str
    run_id: str
    details: dict[str, Any] = field(default_factory=dict)
    op_name: str = ""
    batch_id: str = ""

    def __post_init__(self) -> None:
        for field_name in ("level", "code", "message", "run_id"):
            value = getattr(self, field_name).strip()
            if not value:
                raise ValueError(f"ExecutionLogEvent.{field_name} must be non-empty")
            setattr(self, field_name, value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutorRunSummary:
    run_id: str
    status: str
    input_manifest_key: str
    resolved_op_names: list[str]
    exported_batches: int
    output_keys: list[str] = field(default_factory=list)
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OpTestResult:
    op_name: str
    batch_size: int
    input_row_count: int
    output_row_count: int
    rows: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("OpTestResult.batch_size must be positive")

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


@dataclass
class OpRuntimeContext:
    config: RecipeConfig
    run_id: str
    object_store: Any
    output_root_key: str
    raw_root_key: str
    completed_source_keys: set[str] = field(default_factory=set)
    allow_overwrite: bool = False
    logger: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output_root_key": self.output_root_key,
            "raw_root_key": self.raw_root_key,
            "completed_source_keys": sorted(self.completed_source_keys),
            "allow_overwrite": self.allow_overwrite,
        }

    def get_object_store(self):  # type: ignore[no-untyped-def]
        if self.object_store is not None:
            return self.object_store
        import os

        from .storage import R2ObjectStore

        if os.environ.get("R2_ACCESS_KEY_ID"):
            self.object_store = R2ObjectStore.from_environment(self.config.r2)
        else:
            self.object_store = R2ObjectStore.from_config_file(self.config.r2)
        return self.object_store

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["object_store"] = None
        state["logger"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


@dataclass
class RunArtifacts:
    run_id: str
    input_manifest_key: str
    config_object_key: str
    discovered_documents: int
    uploaded_documents: int = 0
    is_resume: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SubmissionPlan:
    run_id: str
    ssh_target: str
    remote_root: str
    bootstrap_command: str
    remote_command: str
    input_manifest_key: str
    config_object_key: str
    discovered_documents: int
    uploaded_documents: int = 0
    is_resume: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_safe_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["remote_command"] = "<redacted remote command>"
        return payload
