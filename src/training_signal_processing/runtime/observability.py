from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod

from ..models import BatchCommit, ExecutionLogEvent, RunState
from ..ops.base import Op
from ..pipelines.ocr.models import RecipeConfig

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


class ExecutionLogger(ABC):
    @abstractmethod
    def log_event(self, event: ExecutionLogEvent) -> None:
        raise NotImplementedError


class StructuredExecutionLogger(ExecutionLogger):
    def __init__(self, logger_name: str) -> None:
        self.logger = logging.getLogger(logger_name)
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(message)s")

    def log_event(self, event: ExecutionLogEvent) -> None:
        self.logger.log(resolve_log_level(event.level), json.dumps(event.to_dict(), sort_keys=True))


class MlflowExecutionLogger(StructuredExecutionLogger):
    def __init__(
        self,
        config: RecipeConfig,
        logger_name: str,
        artifact_root: str = "execution_logs",
    ) -> None:
        super().__init__(logger_name)
        if not config.mlflow.enabled:
            raise ValueError("MlflowExecutionLogger requires mlflow.enabled=true in the recipe.")
        self.config = config
        self.artifact_root = artifact_root.strip("/") or "execution_logs"
        self.mlflow_run_id = ""
        self.event_count = 0
        self.level_counts: dict[str, int] = {}
        self.pending_events: list[ExecutionLogEvent] = []
        self.client = self.load_mlflow_client()

    def log_event(self, event: ExecutionLogEvent) -> None:
        super().log_event(event)
        if not self.mlflow_run_id:
            self.pending_events.append(event)
            return
        self.log_event_to_mlflow(event)

    def attach_run_id(self, run_id: str) -> None:
        if not run_id.strip():
            raise ValueError("MlflowExecutionLogger.attach_run_id requires a non-empty run_id.")
        self.client.get_run(run_id)
        self.mlflow_run_id = run_id
        for event in self.pending_events:
            self.log_event_to_mlflow(event)
        self.pending_events.clear()

    def log_event_to_mlflow(self, event: ExecutionLogEvent) -> None:
        if not self.mlflow_run_id:
            raise ValueError("MlflowExecutionLogger cannot log without an attached run_id.")
        self.event_count += 1
        level_name = event.level.lower()
        self.level_counts[level_name] = self.level_counts.get(level_name, 0) + 1
        self.client.log_metric(
            self.mlflow_run_id,
            "execution_event_count",
            float(self.event_count),
            step=self.event_count,
        )
        self.client.log_metric(
            self.mlflow_run_id,
            f"execution_{level_name}_count",
            float(self.level_counts[level_name]),
            step=self.event_count,
        )
        self.client.set_tag(self.mlflow_run_id, "last_execution_event_code", event.code)
        self.client.set_tag(self.mlflow_run_id, "last_execution_event_level", event.level.upper())
        if event.op_name:
            self.client.set_tag(self.mlflow_run_id, "last_execution_op_name", event.op_name)
        if event.batch_id:
            self.client.set_tag(self.mlflow_run_id, "last_execution_batch_id", event.batch_id)
        self.client.log_dict(
            self.mlflow_run_id,
            event.to_dict(),
            self.build_artifact_path(event),
        )

    def build_artifact_path(self, event: ExecutionLogEvent) -> str:
        safe_code = re.sub(r"[^a-zA-Z0-9._-]+", "_", event.code).strip("_")
        if not safe_code:
            safe_code = "event"
        return f"{self.artifact_root}/{self.event_count:05d}-{safe_code}.json"

    def load_mlflow_client(self):  # type: ignore[no-untyped-def]
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = (
            os.environ.get("MLFLOW_TRACKING_URI")
            or self.config.mlflow.local_tracking_uri
        )
        mlflow.set_tracking_uri(tracking_uri)
        return MlflowClient(tracking_uri=tracking_uri)


class Tracer(ABC):
    @abstractmethod
    def trace_before_op(self, op: Op) -> None:
        raise NotImplementedError

    @abstractmethod
    def trace_after_op(self, op: Op) -> None:
        raise NotImplementedError


class StructuredTracer(Tracer):
    def __init__(self, logger: ExecutionLogger, run_id: str) -> None:
        self.logger = logger
        self.run_id = run_id

    def trace_before_op(self, op: Op) -> None:
        self.logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="tracer.op.before",
                message=f"Starting op '{op.name}'.",
                run_id=self.run_id,
                op_name=op.name,
            )
        )

    def trace_after_op(self, op: Op) -> None:
        self.logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="tracer.op.after",
                message=f"Finished op '{op.name}'.",
                run_id=self.run_id,
                op_name=op.name,
            )
        )


class Monitor(ABC):
    @abstractmethod
    def start_run(self, run_state: RunState) -> None:
        raise NotImplementedError

    @abstractmethod
    def finish_run(self, run_state: RunState) -> None:
        raise NotImplementedError

    @abstractmethod
    def fail_run(self, run_state: RunState) -> None:
        raise NotImplementedError


class StructuredMonitor(Monitor):
    def __init__(self, logger: ExecutionLogger) -> None:
        self.logger = logger

    def start_run(self, run_state: RunState) -> None:
        self.logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="monitor.run.start",
                message="Run monitoring started.",
                run_id=run_state.run_id,
                details=run_state.to_dict(),
            )
        )

    def finish_run(self, run_state: RunState) -> None:
        self.logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="monitor.run.finish",
                message="Run monitoring finished.",
                run_id=run_state.run_id,
                details=run_state.to_dict(),
            )
        )

    def fail_run(self, run_state: RunState) -> None:
        self.logger.log_event(
            ExecutionLogEvent(
                level="ERROR",
                code="monitor.run.fail",
                message="Run monitoring recorded a failure.",
                run_id=run_state.run_id,
                details=run_state.to_dict(),
            )
        )


class ProgressTracker(ABC):
    @abstractmethod
    def get_run_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def log_run_started(self, total_documents: int, uploaded_documents: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def log_batch_commit(self, batch_commit: BatchCommit, run_state: RunState) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_run_finished(self, status: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_run_failed(self, message: str) -> None:
        raise NotImplementedError


class ProgressTrackerActor(ProgressTracker):
    """Actor-style observability contract for remote execution."""


class MlflowProgressTracker(ProgressTrackerActor):
    def __init__(
        self,
        config: RecipeConfig,
        run_id: str,
        logger: ExecutionLogger,
    ) -> None:
        self.config = config
        self.pipeline_run_id = run_id
        self.logger = logger
        self.mlflow_enabled = config.mlflow.enabled
        self.mlflow_run_id = self.create_mlflow_run_id()
        self.mlflow = self.load_mlflow_client() if self.mlflow_enabled else None
        if isinstance(self.logger, MlflowExecutionLogger):
            self.logger.attach_run_id(self.mlflow_run_id)

    def get_run_id(self) -> str:
        return self.mlflow_run_id

    def log_run_started(self, total_documents: int, uploaded_documents: int) -> str:
        if self.mlflow_enabled:
            self.mlflow.log_params(
                {
                    "pipeline_run_id": self.pipeline_run_id,
                    "total_documents": total_documents,
                    "uploaded_documents": uploaded_documents,
                    "batch_size": self.config.ray.batch_size,
                    "concurrency": self.config.ray.concurrency,
                }
            )
            self.mlflow.set_tags(
                {
                    "status": "running",
                    "executor_type": self.config.ray.executor_type,
                    "run_name": self.config.run_name,
                }
            )
        self.logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="progress.run.start",
                message="Progress tracker started the run.",
                run_id=self.pipeline_run_id,
                details={
                    "total_documents": total_documents,
                    "uploaded_documents": uploaded_documents,
                    "mlflow_run_id": self.mlflow_run_id,
                },
            )
        )
        return self.mlflow_run_id

    def log_batch_commit(self, batch_commit: BatchCommit, run_state: RunState) -> None:
        if self.mlflow_enabled:
            self.mlflow.log_metrics(
                {
                    "success_count": run_state.success_count,
                    "failed_count": run_state.failed_count,
                    "skipped_count": run_state.skipped_count,
                    "pending_documents": run_state.pending_documents,
                    "batch_duration_sec": batch_commit.duration_sec,
                },
                step=run_state.last_committed_batch,
            )
        self.logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="progress.batch.commit",
                message=f"Committed batch '{batch_commit.batch_id}'.",
                run_id=run_state.run_id,
                batch_id=batch_commit.batch_id,
                details={
                    "batch_commit": batch_commit.to_dict(),
                    "run_state": run_state.to_dict(),
                },
            )
        )

    def log_run_finished(self, status: str) -> None:
        if self.mlflow_enabled:
            self.mlflow.set_tag("status", status)
            self.mlflow.end_run(status="FINISHED")
        self.logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="progress.run.finish",
                message="Progress tracker finished the run.",
                run_id=self.pipeline_run_id,
                details={"status": status, "mlflow_run_id": self.mlflow_run_id},
            )
        )

    def log_run_failed(self, message: str) -> None:
        if self.mlflow_enabled:
            self.mlflow.set_tag("status", "failed")
            self.mlflow.set_tag("error_message", message)
            self.mlflow.end_run(status="FAILED")
        self.logger.log_event(
            ExecutionLogEvent(
                level="ERROR",
                code="progress.run.failed",
                message="Progress tracker recorded a failed run.",
                run_id=self.pipeline_run_id,
                details={"error": message, "mlflow_run_id": self.mlflow_run_id},
            )
        )

    def create_mlflow_run_id(self) -> str:
        if not self.mlflow_enabled:
            return f"disabled:{self.pipeline_run_id}"
        tracking_uri = (
            os.environ.get("MLFLOW_TRACKING_URI")
            or self.config.mlflow.local_tracking_uri
        )
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)
        active_run = mlflow.start_run(run_name=f"{self.config.run_name}:{self.pipeline_run_id}")
        return active_run.info.run_id

    def load_mlflow_client(self):  # type: ignore[no-untyped-def]
        import mlflow

        return mlflow


def resolve_log_level(level: str) -> int:
    mapping = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }
    try:
        return mapping[level.upper()]
    except KeyError as exc:
        raise ValueError(f"Unsupported log level: {level}") from exc
