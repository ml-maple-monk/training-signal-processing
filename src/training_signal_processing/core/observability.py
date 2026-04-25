from __future__ import annotations

import json
import logging
import math
import sys
from abc import ABC, abstractmethod

from ..ops.base import Op
from .models import BatchProgress, ExecutionLogEvent, RunState, RuntimeTrackingContext

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
        tracking: RuntimeTrackingContext,
        logger_name: str,
    ) -> None:
        super().__init__(logger_name)
        if not tracking.enabled:
            raise ValueError("MlflowExecutionLogger requires tracking.enabled=true.")
        self.tracking = tracking
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

    def load_mlflow_client(self):  # type: ignore[no-untyped-def]
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = self.resolve_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        return MlflowClient(tracking_uri=tracking_uri)

    def resolve_tracking_uri(self) -> str:
        tracking_uri = self.tracking.tracking_uri.strip()
        if not tracking_uri:
            raise ValueError(
                "mlflow.tracking_uri is required when mlflow.enabled=true; "
                "reverse-tunnel MLflow was removed."
            )
        return tracking_uri


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
    def log_run_started(self, total_items: int, uploaded_items: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def log_batch_progress(self, batch_progress: BatchProgress, run_state: RunState) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_run_finished(self, status: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_run_failed(self, message: str) -> None:
        raise NotImplementedError


class ProgressTrackerActor(ProgressTracker):
    """Actor-style observability contract for remote execution."""


class ProgressReporter(ABC):
    @abstractmethod
    def start_run(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def report_phase(self, phase: str, detail: str = "") -> None:
        raise NotImplementedError

    @abstractmethod
    def start_batch(self, batch_id: str, batch_index: int, input_row_count: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def start_op(self, batch_id: str, op_name: str, input_row_count: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def finish_op(self, batch_id: str, op_name: str, output_row_count: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def finish_batch(self, batch_progress: BatchProgress, run_state: RunState) -> None:
        raise NotImplementedError

    @abstractmethod
    def finish_run(self, status: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def fail_run(self, message: str) -> None:
        raise NotImplementedError


class NullProgressReporter(ProgressReporter):
    def start_run(self) -> None:
        return None

    def report_phase(self, phase: str, detail: str = "") -> None:
        return None

    def start_batch(self, batch_id: str, batch_index: int, input_row_count: int) -> None:
        return None

    def start_op(self, batch_id: str, op_name: str, input_row_count: int) -> None:
        return None

    def finish_op(self, batch_id: str, op_name: str, output_row_count: int) -> None:
        return None

    def finish_batch(self, batch_progress: BatchProgress, run_state: RunState) -> None:
        return None

    def finish_run(self, status: str) -> None:
        return None

    def fail_run(self, message: str) -> None:
        return None


class TqdmProgressReporter(ProgressReporter):
    def __init__(
        self,
        *,
        run_id: str,
        total_items: int,
        pending_items: int,
        batch_size: int,
    ) -> None:
        self.run_id = run_id
        self.total_items = total_items
        self.pending_items = pending_items
        self.batch_size = batch_size
        self.total_batches = max(1, math.ceil(max(pending_items, 0) / max(batch_size, 1)))
        self.progress_bar = self.build_progress_bar()

    def build_progress_bar(self):  # type: ignore[no-untyped-def]
        try:
            from tqdm import tqdm
        except ImportError:
            return None
        return tqdm(
            total=self.total_batches,
            desc=f"run {self.run_id}",
            unit="batch",
            dynamic_ncols=True,
            file=sys.stderr,
        )

    def write_line(self, message: str) -> None:
        if self.progress_bar is not None:
            self.progress_bar.write(message)
            return
        print(message, file=sys.stderr)

    def start_run(self) -> None:
        self.write_line(
            "[run] "
            f"run_id={self.run_id} total_items={self.total_items} "
            f"pending_items={self.pending_items} batch_size={self.batch_size}"
        )

    def report_phase(self, phase: str, detail: str = "") -> None:
        rendered_detail = f" detail={detail}" if detail else ""
        self.write_line(f"[phase] run_id={self.run_id} phase={phase}{rendered_detail}")

    def start_batch(self, batch_id: str, batch_index: int, input_row_count: int) -> None:
        self.write_line(
            f"[batch:start] batch_id={batch_id} index={batch_index} input_rows={input_row_count}"
        )

    def start_op(self, batch_id: str, op_name: str, input_row_count: int) -> None:
        self.write_line(
            f"[op:start] batch_id={batch_id} op={op_name} input_rows={input_row_count}"
        )

    def finish_op(self, batch_id: str, op_name: str, output_row_count: int) -> None:
        self.write_line(
            f"[op:finish] batch_id={batch_id} op={op_name} output_rows={output_row_count}"
        )

    def finish_batch(self, batch_progress: BatchProgress, run_state: RunState) -> None:
        if self.progress_bar is not None:
            self.progress_bar.update(1)
            self.progress_bar.set_postfix(
                success=run_state.success_count,
                failed=run_state.failed_count,
                skipped=run_state.skipped_count,
                pending=run_state.pending_items,
            )
        self.write_line(
            "[batch:finish] "
            f"batch_id={batch_progress.batch_id} "
            f"success={batch_progress.success_count} "
            f"failed={batch_progress.failed_count} "
            f"skipped={batch_progress.skipped_count} "
            f"pending_items={run_state.pending_items}"
        )

    def finish_run(self, status: str) -> None:
        self.write_line(f"[run:finish] run_id={self.run_id} status={status}")
        if self.progress_bar is not None:
            self.progress_bar.close()

    def fail_run(self, message: str) -> None:
        self.write_line(f"[run:fail] run_id={self.run_id} error={message}")
        if self.progress_bar is not None:
            self.progress_bar.close()


class MlflowProgressTracker(ProgressTrackerActor):
    def __init__(
        self,
        tracking: RuntimeTrackingContext,
        run_id: str,
        logger: ExecutionLogger,
    ) -> None:
        self.tracking = tracking
        self.pipeline_run_id = run_id
        self.logger = logger
        self.mlflow_enabled = tracking.enabled
        self.mlflow_run_id = self.create_mlflow_run_id()
        self.mlflow = self.load_mlflow_client() if self.mlflow_enabled else None
        if isinstance(self.logger, MlflowExecutionLogger):
            self.logger.attach_run_id(self.mlflow_run_id)

    def get_run_id(self) -> str:
        return self.mlflow_run_id

    def log_run_started(self, total_items: int, uploaded_items: int) -> str:
        if self.mlflow_enabled:
            params: dict[str, int | float | str] = {
                "pipeline_run_id": self.pipeline_run_id,
                "total_items": total_items,
                "uploaded_items": uploaded_items,
                "batch_size": self.tracking.batch_size,
                "concurrency": self.tracking.concurrency,
            }
            if self.tracking.target_num_blocks is not None:
                params["target_num_blocks"] = self.tracking.target_num_blocks
            params.update(self.tracking.extra_params)
            self.mlflow.log_params(params)
            self.mlflow.set_tags(
                {
                    "status": "running",
                    "executor_type": self.tracking.executor_type,
                    "run_name": self.tracking.run_name,
                }
            )
        self.logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="progress.run.start",
                message="Progress tracker started the run.",
                run_id=self.pipeline_run_id,
                details={
                    "total_items": total_items,
                    "uploaded_items": uploaded_items,
                    "mlflow_run_id": self.mlflow_run_id,
                },
            )
        )
        return self.mlflow_run_id

    def log_batch_progress(self, batch_progress: BatchProgress, run_state: RunState) -> None:
        if self.mlflow_enabled:
            self.mlflow.log_metrics(
                {
                    "success_count": run_state.success_count,
                    "failed_count": run_state.failed_count,
                    "skipped_count": run_state.skipped_count,
                    "pending_items": run_state.pending_items,
                    "batch_duration_sec": batch_progress.duration_sec,
                },
                step=run_state.last_committed_batch,
            )
        self.logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="progress.batch.finish",
                message=f"Finished batch '{batch_progress.batch_id}'.",
                run_id=run_state.run_id,
                batch_id=batch_progress.batch_id,
                details={
                    "batch_progress": batch_progress.to_dict(),
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
        tracking_uri = self.resolve_tracking_uri()
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.tracking.experiment_name)
        active_run = mlflow.start_run(run_name=f"{self.tracking.run_name}:{self.pipeline_run_id}")
        return active_run.info.run_id

    def load_mlflow_client(self):  # type: ignore[no-untyped-def]
        import mlflow

        return mlflow

    def resolve_tracking_uri(self) -> str:
        tracking_uri = self.tracking.tracking_uri.strip()
        if not tracking_uri:
            raise ValueError(
                "mlflow.tracking_uri is required when mlflow.enabled=true; "
                "reverse-tunnel MLflow was removed."
            )
        return tracking_uri


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


__all__ = [
    "ExecutionLogger",
    "MlflowExecutionLogger",
    "MlflowProgressTracker",
    "Monitor",
    "NullProgressReporter",
    "ProgressReporter",
    "ProgressTracker",
    "ProgressTrackerActor",
    "StructuredExecutionLogger",
    "StructuredMonitor",
    "StructuredTracer",
    "Tracer",
    "TqdmProgressReporter",
    "resolve_log_level",
]
