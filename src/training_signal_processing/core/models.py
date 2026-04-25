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
    remote_jobs_root: str
    pgid_wait_attempts: int
    pgid_wait_sleep_seconds: float
    sync_paths: tuple[str, ...]
    venv_dir: str = ".venv"

    def __post_init__(self) -> None:
        if not isinstance(self.sync_paths, (list, tuple)):
            raise ValueError("remote.sync_paths must be a non-empty list of paths.")
        sync_paths = tuple(str(path).strip() for path in self.sync_paths)
        if not sync_paths or any(not path for path in sync_paths):
            raise ValueError("remote.sync_paths must be a non-empty list of paths.")
        self.sync_paths = sync_paths


@dataclass
class RayConfig:
    executor_type: str
    batch_size: int
    concurrency: int
    target_num_blocks: int


@dataclass(frozen=True)
class RayTransformResources:
    concurrency: int | None = None
    num_gpus: float | None = None
    num_cpus: float | None = None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass
class R2Config:
    config_file: str
    bucket: str
    output_prefix: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    region: str = ""
    endpoint_url: str = ""


@dataclass
class MlflowConfig:
    enabled: bool
    experiment_name: str
    tracking_uri: str = ""


@dataclass
class ObservabilityConfig:
    flush_interval_sec: int
    log_per_file_events: bool
    heartbeat_interval_sec: int


@dataclass
class OpConfig:
    name: str
    type: str = ""
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchProgress:
    batch_id: str
    input_row_count: int
    output_row_count: int
    success_count: int
    failed_count: int
    skipped_count: int
    duration_sec: float

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
class RuntimeRunBindings:
    run_id: str
    input_manifest_key: str
    config_object_key: str = ""
    uploaded_items: int = 0
    allow_overwrite: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunArtifactLayout:
    source_root_key: str
    output_root_key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeTrackingContext:
    enabled: bool
    tracking_uri: str
    experiment_name: str
    run_name: str
    executor_type: str
    batch_size: int
    concurrency: int
    target_num_blocks: int | None = None
    extra_params: dict[str, int | float | str | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RunState:
    run_id: str
    status: str
    total_items: int
    pending_items: int
    success_count: int
    failed_count: int
    skipped_count: int
    last_committed_batch: int
    started_at: str
    updated_at: str
    source_root_key: str
    output_root_key: str
    tracking_run_id: str
    error_message: str = ""
    current_phase: str = ""
    last_phase_at: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RunState":
        return cls(
            run_id=str(row["run_id"]),
            status=str(row["status"]),
            total_items=int(row["total_items"]),
            pending_items=int(row["pending_items"]),
            success_count=int(row["success_count"]),
            failed_count=int(row["failed_count"]),
            skipped_count=int(row["skipped_count"]),
            last_committed_batch=int(row["last_committed_batch"]),
            started_at=str(row["started_at"]),
            updated_at=str(row["updated_at"]),
            source_root_key=str(row["source_root_key"]),
            output_root_key=str(row["output_root_key"]),
            tracking_run_id=str(row.get("tracking_run_id", "")),
            error_message=str(row.get("error_message", "")),
            current_phase=str(row.get("current_phase", "")),
            last_phase_at=str(row.get("last_phase_at", "")),
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
class OpRuntimeContext:
    config: Any
    run_id: str
    object_store: Any | None
    output_root_key: str
    source_root_key: str
    completed_source_keys: set[str] = field(default_factory=set)
    allow_overwrite: bool = False
    logger: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output_root_key": self.output_root_key,
            "source_root_key": self.source_root_key,
            "completed_source_keys": sorted(self.completed_source_keys),
            "allow_overwrite": self.allow_overwrite,
        }

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["object_store"] = None
        state["logger"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
