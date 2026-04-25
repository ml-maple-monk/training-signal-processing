from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..ops.base import Batch, Op
from ..ops.registry import OpRegistry, RegisteredOpRegistry
from .dataset import ConfiguredRayDatasetBuilder, DatasetBuilder
from .models import (
    BatchProgress,
    ExecutionLogEvent,
    ExecutorRunSummary,
    ExportBatchResult,
    OpConfig,
    OpRuntimeContext,
    RayConfig,
    RayTransformResources,
    RunArtifactLayout,
    RunState,
    RuntimeRunBindings,
    RuntimeTrackingContext,
)
from .observability import (
    ExecutionLogger,
    MlflowExecutionLogger,
    MlflowProgressTracker,
    NullProgressReporter,
    ProgressReporter,
    StructuredExecutionLogger,
    StructuredMonitor,
    StructuredTracer,
    TqdmProgressReporter,
)
from .utils import join_s3_key, utc_isoformat

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


class PipelineContractError(ValueError):
    """Raised when a pipeline adapter violates the protected executor contract."""


class Executor(ABC):
    @abstractmethod
    def run(self) -> dict[str, object]:
        raise NotImplementedError


class OutputCompletionTracker(ABC):
    """Outputs-only resume contract.

    Public method:
        completed_source_keys() is called by the executor at run startup.

    Pipeline hooks:
        source_key_for_input(), output_key_for_input(), and output_listing_prefix()
        are the only customization points. Everything else is framework-owned.
    """

    def __init__(self, object_store: Any) -> None:
        self.object_store = object_store

    def completed_source_keys(
        self,
        *,
        input_rows: list[dict[str, Any]],
        artifact_layout: RunArtifactLayout,
        allow_overwrite: bool,
    ) -> set[str]:
        if allow_overwrite:
            return set()
        prefix = self.output_listing_prefix(artifact_layout)
        existing_output_keys = set(self.object_store.list_keys(prefix))
        completed_keys: set[str] = set()
        for row in input_rows:
            output_key = self.output_key_for_input(row, artifact_layout)
            if output_key in existing_output_keys:
                completed_keys.add(self.source_key_for_input(row))
        return completed_keys

    @abstractmethod
    def source_key_for_input(self, row: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def output_key_for_input(
        self,
        row: dict[str, Any],
        artifact_layout: RunArtifactLayout,
    ) -> str:
        raise NotImplementedError

    def output_listing_prefix(self, artifact_layout: RunArtifactLayout) -> str:
        return artifact_layout.output_root_key


class Exporter(ABC):
    @abstractmethod
    def export_batch(self, batch_id: str, rows: Batch) -> ExportBatchResult:
        raise NotImplementedError

    def finalize_run(self, run_state: RunState) -> None:
        return None


class RayExporter(Exporter):
    """Ray-only exporter contract for explicit batch materialization."""

    def __init__(self, object_store: Any) -> None:
        self.object_store = object_store

    def _put_bytes(self, key: str, body: bytes) -> None:
        self.object_store.write_bytes(key, body)


@dataclass(frozen=True)
class RunSpec:
    bindings: RuntimeRunBindings
    execution: RayConfig
    tracking: RuntimeTrackingContext
    artifact_layout: RunArtifactLayout
    op_configs: list[OpConfig]


@dataclass
class ExecutorRunContext:
    run_spec: RunSpec
    input_rows: list[dict[str, Any]]
    completed_source_keys: set[str]
    logger: ExecutionLogger
    progress_tracker: MlflowProgressTracker
    progress_reporter: ProgressReporter
    tracer: StructuredTracer
    monitor: StructuredMonitor
    resolved_pipeline: Any
    exporter: Exporter
    dataset_builder: DatasetBuilder
    run_state: RunState
    exported_batches: int = 0
    output_keys: list[str] = field(default_factory=list)


class PipelineRuntimeAdapter(ABC):
    @abstractmethod
    def get_run_bindings(self) -> RuntimeRunBindings:
        raise NotImplementedError

    @abstractmethod
    def get_execution_config(self) -> RayConfig:
        raise NotImplementedError

    @abstractmethod
    def get_tracking_context(self) -> RuntimeTrackingContext:
        raise NotImplementedError

    @abstractmethod
    def get_op_configs(self) -> list[OpConfig]:
        raise NotImplementedError

    @abstractmethod
    def get_artifact_layout(self) -> RunArtifactLayout:
        raise NotImplementedError

    @abstractmethod
    def load_input_rows(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def build_runtime_context(
        self,
        *,
        logger: ExecutionLogger,
        completed_source_keys: set[str],
    ) -> OpRuntimeContext:
        raise NotImplementedError

    @abstractmethod
    def build_op_registry(self, runtime_context: OpRuntimeContext) -> OpRegistry:
        raise NotImplementedError

    @abstractmethod
    def build_exporter(self) -> Exporter:
        raise NotImplementedError

    @abstractmethod
    def build_completion_tracker(self) -> OutputCompletionTracker:
        raise NotImplementedError

    def build_dataset_builder(self) -> DatasetBuilder:
        return ConfiguredRayDatasetBuilder(self.get_execution_config())

    def run_spec(self) -> RunSpec:
        return RunSpec(
            bindings=self.get_run_bindings(),
            execution=self.get_execution_config(),
            tracking=self.get_tracking_context(),
            artifact_layout=self.get_artifact_layout(),
            op_configs=self.get_op_configs(),
        )

    def resolve_completed_source_keys(
        self,
        *,
        input_rows: list[dict[str, Any]],
    ) -> set[str]:
        return self.build_completion_tracker().completed_source_keys(
            input_rows=input_rows,
            artifact_layout=self.get_artifact_layout(),
            allow_overwrite=self.get_run_bindings().allow_overwrite,
        )

    def resolve_transform_resources(
        self,
        *,
        op: Op,
        execution: RayConfig,
    ) -> RayTransformResources:
        return RayTransformResources()


class ObjectStorePipelineRuntimeAdapter(PipelineRuntimeAdapter):
    """Reusable runtime template for object-store backed pipelines.

    Public methods implement the executor-facing contract. Concrete pipelines
    should customize only the hook methods: source_root_key(),
    build_exporter(), build_completion_tracker(), tracking_extra_params(), and
    resolve_transform_resources() when an op needs resource overrides.
    """

    def __init__(
        self,
        *,
        config: Any,
        bindings: RuntimeRunBindings,
        object_store: Any,
        source_root_key: str = "",
        exporter_factory: Callable[[Any], Exporter] | None = None,
        completion_tracker_factory: Callable[[Any], OutputCompletionTracker] | None = None,
        tracking_extra_params: dict[str, int | float | str | bool] | None = None,
        transform_resources_resolver: Callable[[Op, RayConfig], RayTransformResources]
        | None = None,
    ) -> None:
        self.config = config
        self.bindings = bindings
        self.object_store = object_store
        self.resolved_source_root_key = source_root_key
        self.exporter_factory = exporter_factory
        self.completion_tracker_factory = completion_tracker_factory
        self.resolved_tracking_extra_params = tracking_extra_params
        self.transform_resources_resolver = transform_resources_resolver

    def get_run_bindings(self) -> RuntimeRunBindings:
        return self.bindings

    def get_execution_config(self) -> RayConfig:
        return self.config.ray

    def get_tracking_context(self) -> RuntimeTrackingContext:
        return RuntimeTrackingContext(
            enabled=self.config.mlflow.enabled,
            tracking_uri=self.config.mlflow.tracking_uri,
            experiment_name=self.config.mlflow.experiment_name,
            run_name=self.config.run_name,
            executor_type=self.config.ray.executor_type,
            batch_size=self.config.ray.batch_size,
            concurrency=self.config.ray.concurrency,
            target_num_blocks=self.config.ray.target_num_blocks,
            extra_params=self.tracking_extra_params(),
        )

    def get_op_configs(self) -> list[OpConfig]:
        return self.config.ops

    def get_artifact_layout(self) -> RunArtifactLayout:
        return RunArtifactLayout(
            source_root_key=self.source_root_key(),
            output_root_key=join_s3_key(self.config.r2.output_prefix, self.bindings.run_id),
        )

    def load_input_rows(self) -> list[dict[str, Any]]:
        return self.object_store.read_jsonl(self.bindings.input_manifest_key)

    def build_runtime_context(
        self,
        *,
        logger: ExecutionLogger,
        completed_source_keys: set[str],
    ) -> OpRuntimeContext:
        artifact_layout = self.get_artifact_layout()
        return OpRuntimeContext(
            config=self.config,
            run_id=self.bindings.run_id,
            object_store=self.object_store,
            output_root_key=artifact_layout.output_root_key,
            source_root_key=artifact_layout.source_root_key,
            completed_source_keys=completed_source_keys,
            allow_overwrite=self.bindings.allow_overwrite,
            logger=logger,
        )

    def build_op_registry(self, runtime_context: OpRuntimeContext) -> OpRegistry:
        return RegisteredOpRegistry(runtime_context=runtime_context)

    def source_root_key(self) -> str:
        if self.resolved_source_root_key:
            return self.resolved_source_root_key
        raise NotImplementedError

    def build_exporter(self) -> Exporter:
        if self.exporter_factory is None:
            raise NotImplementedError
        return self.exporter_factory(self.object_store)

    def build_completion_tracker(self) -> OutputCompletionTracker:
        if self.completion_tracker_factory is None:
            raise NotImplementedError
        return self.completion_tracker_factory(self.object_store)

    def tracking_extra_params(self) -> dict[str, int | float | str | bool]:
        if self.resolved_tracking_extra_params is not None:
            return dict(self.resolved_tracking_extra_params)
        return {}

    def resolve_transform_resources(
        self,
        *,
        op: Op,
        execution: RayConfig,
    ) -> RayTransformResources:
        if self.transform_resources_resolver is None:
            return super().resolve_transform_resources(op=op, execution=execution)
        return self.transform_resources_resolver(op, execution)


class StreamingRayExecutor(Executor):
    def __init__(self, pipeline: PipelineRuntimeAdapter) -> None:
        self.pipeline = pipeline

    def run(self) -> dict[str, object]:
        context = self._start_run_context()
        try:
            dataset = self._build_run_dataset(context)
            for op in context.resolved_pipeline.all_ops:
                dataset = self._apply_dataset_transform(context, dataset, op)
            return self._materialize_run_dataset(context, dataset)
        except Exception as exc:
            return self._fail_run(context, exc)

    def _start_run_context(self) -> ExecutorRunContext:
        run_spec = self.pipeline.run_spec()
        bindings = run_spec.bindings
        execution = run_spec.execution
        tracking = run_spec.tracking
        artifact_layout = run_spec.artifact_layout
        op_configs = run_spec.op_configs
        self._validate_contract(
            bindings=bindings,
            execution=execution,
            tracking=tracking,
            artifact_layout=artifact_layout,
            op_configs=op_configs,
        )

        logger = self._build_execution_logger(tracking)
        progress_tracker = MlflowProgressTracker(
            tracking=tracking,
            run_id=bindings.run_id,
            logger=logger,
        )
        input_rows = self.pipeline.load_input_rows()
        if not input_rows:
            raise PipelineContractError("Pipeline adapter returned zero input rows.")

        self._log_manifest_loaded(run_spec, input_rows, logger)
        completed_source_keys = self.pipeline.resolve_completed_source_keys(
            input_rows=input_rows,
        )
        self._log_resume_loaded(run_spec, completed_source_keys, logger)
        runtime_context = self.pipeline.build_runtime_context(
            logger=logger,
            completed_source_keys=completed_source_keys,
        )
        registry = self.pipeline.build_op_registry(runtime_context)
        resolved_pipeline = registry.resolve_pipeline(op_configs)
        exporter = self.pipeline.build_exporter()
        dataset_builder = self.pipeline.build_dataset_builder()

        total_items = len(input_rows)
        pending_items = max(total_items - len(completed_source_keys), 0)
        progress_reporter = self._build_progress_reporter(
            run_id=bindings.run_id,
            total_items=total_items,
            pending_items=pending_items,
            batch_size=execution.batch_size,
        )
        tracking_run_id = progress_tracker.log_run_started(
            total_items=total_items,
            uploaded_items=bindings.uploaded_items,
        )
        run_state = self._initialize_run_state(
            run_spec=run_spec,
            total_items=total_items,
            pending_items=pending_items,
            precompleted_count=len(completed_source_keys),
            artifact_layout=artifact_layout,
            tracking_run_id=tracking_run_id,
        )
        tracer = StructuredTracer(logger=logger, run_id=bindings.run_id)
        monitor = StructuredMonitor(logger=logger)
        monitor.start_run(run_state)
        progress_reporter.start_run()
        context = ExecutorRunContext(
            run_spec=run_spec,
            input_rows=input_rows,
            completed_source_keys=completed_source_keys,
            logger=logger,
            progress_tracker=progress_tracker,
            progress_reporter=progress_reporter,
            tracer=tracer,
            monitor=monitor,
            resolved_pipeline=resolved_pipeline,
            exporter=exporter,
            dataset_builder=dataset_builder,
            run_state=run_state,
        )
        self._transition_run_phase(
            context,
            phase="manifest_loaded",
            detail=f"input_manifest_key={bindings.input_manifest_key}",
        )
        self._transition_run_phase(
            context,
            phase="resume_state_loaded",
            detail=f"completed_source_keys={len(completed_source_keys)}",
        )
        return context

    def _log_manifest_loaded(
        self,
        run_spec: RunSpec,
        input_rows: list[dict[str, Any]],
        logger: ExecutionLogger,
    ) -> None:
        logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="executor.manifest.loaded",
                message="Loaded input manifest rows.",
                run_id=run_spec.bindings.run_id,
                details={
                    "input_manifest_key": run_spec.bindings.input_manifest_key,
                    "input_row_count": len(input_rows),
                },
            )
        )

    def _log_resume_loaded(
        self,
        run_spec: RunSpec,
        completed_source_keys: set[str],
        logger: ExecutionLogger,
    ) -> None:
        logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="executor.resume.loaded",
                message="Loaded output completion state.",
                run_id=run_spec.bindings.run_id,
                details={"completed_source_keys": len(completed_source_keys)},
            )
        )

    def _initialize_run_state(
        self,
        *,
        run_spec: RunSpec,
        total_items: int,
        pending_items: int,
        precompleted_count: int,
        artifact_layout: RunArtifactLayout,
        tracking_run_id: str,
    ) -> RunState:
        timestamp = utc_isoformat()
        return RunState(
            run_id=run_spec.bindings.run_id,
            status="running",
            total_items=total_items,
            pending_items=pending_items,
            success_count=0,
            failed_count=0,
            skipped_count=precompleted_count,
            last_committed_batch=0,
            started_at=timestamp,
            updated_at=timestamp,
            source_root_key=artifact_layout.source_root_key,
            output_root_key=artifact_layout.output_root_key,
            tracking_run_id=tracking_run_id,
            current_phase="run_initialized",
            last_phase_at=timestamp,
        )

    def _build_run_dataset(self, context: ExecutorRunContext):
        self._transition_run_phase(context, phase="dataset_build_start")
        dataset = context.dataset_builder.build_for_run(context.input_rows)
        self._transition_run_phase(context, phase="dataset_build_complete")
        return dataset

    def _apply_dataset_transform(self, context: ExecutorRunContext, dataset, op: Op):
        execution = context.run_spec.execution
        resources = self.pipeline.resolve_transform_resources(op=op, execution=execution)
        context.tracer.trace_before_op(op)
        context.logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="executor.dataset.transform",
                message=f"Scheduled dataset transform for op '{op.name}'.",
                run_id=context.run_spec.bindings.run_id,
                op_name=op.name,
                details={
                    "batch_size": execution.batch_size,
                    **resources.to_dict(),
                },
            )
        )
        transformed = context.dataset_builder.apply_op_transform(
            dataset,
            op=op,
            batch_size=execution.batch_size,
            concurrency=resources.concurrency,
            num_gpus=resources.num_gpus,
            num_cpus=resources.num_cpus,
        )
        context.tracer.trace_after_op(op)
        return transformed

    def _materialize_run_dataset(self, context: ExecutorRunContext, dataset) -> dict[str, object]:
        self._transition_run_phase(context, phase="iter_batches_start")
        for batch_offset, current_batch in enumerate(
            context.dataset_builder.iter_batches(
                dataset,
                batch_size=context.run_spec.execution.batch_size,
            ),
            start=1,
        ):
            batch_index = batch_offset
            batch_id = f"batch-{batch_index:05d}"
            self._export_and_record_batch(
                context,
                batch_id,
                batch_index,
                batch_offset,
                current_batch,
            )
        return self._finish_run(context)

    def _export_and_record_batch(
        self,
        context: ExecutorRunContext,
        batch_id: str,
        batch_index: int,
        batch_offset: int,
        rows: Batch,
    ) -> None:
        if batch_offset == 1:
            self._transition_run_phase(
                context,
                phase="first_batch_materialized",
                detail=f"batch_id={batch_id} input_rows={len(rows)}",
            )
        context.progress_reporter.start_batch(batch_id, batch_index, len(rows))
        export_result = context.exporter.export_batch(batch_id=batch_id, rows=rows)
        self._validate_export_result(
            batch_id=batch_id,
            input_row_count=len(rows),
            rows=rows,
            output_keys=export_result.output_keys,
            reported_batch_id=export_result.batch_id,
            reported_row_count=export_result.row_count,
        )
        batch_progress = self._build_batch_progress(
            batch_id=batch_id,
            input_row_count=len(rows),
            rows=rows,
        )
        self._validate_batch_progress(
            batch_id=batch_id,
            input_row_count=len(rows),
            output_row_count=len(rows),
            reported_batch_id=batch_progress.batch_id,
            reported_input_row_count=batch_progress.input_row_count,
            reported_output_row_count=batch_progress.output_row_count,
        )
        context.run_state = self._advance_run_state(
            run_state=context.run_state,
            batch_index=batch_index,
            input_row_count=len(rows),
            batch_progress=batch_progress,
        )
        context.progress_tracker.log_batch_progress(batch_progress, context.run_state)
        context.progress_reporter.finish_batch(batch_progress, context.run_state)
        context.exported_batches += 1
        context.output_keys.extend(export_result.output_keys)

    def _build_batch_progress(
        self,
        *,
        batch_id: str,
        input_row_count: int,
        rows: Batch,
    ) -> BatchProgress:
        success_count = 0
        failed_count = 0
        skipped_count = 0
        duration_sec = 0.0
        for row in rows:
            status = str(row.get("status", "success"))
            if status == "failed":
                failed_count += 1
            elif status == "skipped_existing":
                skipped_count += 1
            else:
                success_count += 1
            duration_sec += float(row.get("duration_sec", 0.0) or 0.0)
        return BatchProgress(
            batch_id=batch_id,
            input_row_count=input_row_count,
            output_row_count=len(rows),
            success_count=success_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            duration_sec=duration_sec,
        )

    def _advance_run_state(
        self,
        *,
        run_state: RunState,
        batch_index: int,
        input_row_count: int,
        batch_progress: BatchProgress,
    ) -> RunState:
        return RunState(
            **{
                **run_state.to_dict(),
                "pending_items": max(run_state.pending_items - input_row_count, 0),
                "success_count": run_state.success_count + batch_progress.success_count,
                "failed_count": run_state.failed_count + batch_progress.failed_count,
                "skipped_count": run_state.skipped_count + batch_progress.skipped_count,
                "last_committed_batch": batch_index,
                "updated_at": utc_isoformat(),
            }
        )

    def _finish_run(self, context: ExecutorRunContext) -> dict[str, object]:
        context.run_state = self._mark_run_finished(context.run_state)
        context.exporter.finalize_run(context.run_state)
        context.monitor.finish_run(context.run_state)
        context.progress_tracker.log_run_finished(context.run_state.status)
        context.progress_reporter.finish_run(context.run_state.status)
        return self._build_run_summary(context, status=context.run_state.status)

    def _fail_run(self, context: ExecutorRunContext, exc: Exception) -> dict[str, object]:
        message = str(exc)
        context.run_state = self._mark_run_failed(context.run_state, message)
        context.monitor.fail_run(context.run_state)
        context.progress_tracker.log_run_failed(message)
        context.progress_reporter.fail_run(message)
        return self._build_run_summary(context, status="failed", error_message=message)

    def _mark_run_finished(self, run_state: RunState) -> RunState:
        if (
            run_state.failed_count > 0
            and run_state.success_count == 0
            and run_state.skipped_count == 0
        ):
            status = "failed"
        elif run_state.failed_count > 0:
            status = "partial"
        else:
            status = "success"
        return RunState(
            **{
                **run_state.to_dict(),
                "status": status,
                "pending_items": 0,
                "updated_at": utc_isoformat(),
            }
        )

    def _mark_run_failed(self, run_state: RunState, message: str) -> RunState:
        return RunState(
            **{
                **run_state.to_dict(),
                "status": "failed",
                "error_message": message,
                "updated_at": utc_isoformat(),
            }
        )

    def _build_run_summary(
        self,
        context: ExecutorRunContext,
        *,
        status: str,
        error_message: str = "",
    ) -> dict[str, object]:
        return ExecutorRunSummary(
            run_id=context.run_spec.bindings.run_id,
            status=status,
            input_manifest_key=context.run_spec.bindings.input_manifest_key,
            resolved_op_names=context.resolved_pipeline.names,
            exported_batches=context.exported_batches,
            output_keys=context.output_keys,
            error_message=error_message,
        ).to_dict()

    def _transition_run_phase(
        self,
        context: ExecutorRunContext,
        phase: str,
        detail: str = "",
    ) -> None:
        timestamp = utc_isoformat()
        updated_state = RunState(
            **{
                **context.run_state.to_dict(),
                "current_phase": phase,
                "last_phase_at": timestamp,
                "updated_at": timestamp,
            }
        )
        context.run_state = updated_state
        context.progress_reporter.report_phase(phase, detail)
        context.logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="executor.phase",
                message=f"Run phase changed to '{phase}'.",
                run_id=updated_state.run_id,
                details={"phase": phase, "detail": detail, "run_state": updated_state.to_dict()},
            )
        )

    def _build_execution_logger(self, tracking: RuntimeTrackingContext) -> ExecutionLogger:
        if tracking.enabled:
            return MlflowExecutionLogger(
                tracking=tracking,
                logger_name="training_signal_processing",
            )
        return StructuredExecutionLogger("training_signal_processing")

    def _build_progress_reporter(
        self,
        *,
        run_id: str,
        total_items: int,
        pending_items: int,
        batch_size: int,
    ) -> ProgressReporter:
        if total_items <= 0:
            return NullProgressReporter()
        return TqdmProgressReporter(
            run_id=run_id,
            total_items=total_items,
            pending_items=pending_items,
            batch_size=batch_size,
        )

    def _validate_contract(
        self,
        *,
        bindings: RuntimeRunBindings,
        execution: RayConfig,
        tracking: RuntimeTrackingContext,
        artifact_layout: RunArtifactLayout,
        op_configs: list[OpConfig],
    ) -> None:
        if not bindings.run_id.strip():
            raise PipelineContractError("RuntimeRunBindings.run_id must be non-empty.")
        if not bindings.input_manifest_key.strip():
            raise PipelineContractError("RuntimeRunBindings.input_manifest_key must be non-empty.")
        if execution.batch_size <= 0:
            raise PipelineContractError("RayConfig.batch_size must be positive.")
        if execution.concurrency <= 0:
            raise PipelineContractError("RayConfig.concurrency must be positive.")
        if not tracking.run_name.strip():
            raise PipelineContractError("RuntimeTrackingContext.run_name must be non-empty.")
        if not artifact_layout.source_root_key.strip():
            raise PipelineContractError("RunArtifactLayout.source_root_key must be non-empty.")
        if not artifact_layout.output_root_key.strip():
            raise PipelineContractError("RunArtifactLayout.output_root_key must be non-empty.")
        if not op_configs:
            raise PipelineContractError("Pipeline adapter must declare at least one op config.")

    def _validate_export_result(
        self,
        *,
        batch_id: str,
        input_row_count: int,
        rows: Batch,
        output_keys: list[str],
        reported_batch_id: str,
        reported_row_count: int,
    ) -> None:
        if not batch_id.strip():
            raise PipelineContractError("Executor batch_id must be non-empty.")
        if input_row_count <= 0:
            raise PipelineContractError("Executor input_row_count must be positive.")
        if reported_batch_id != batch_id:
            raise PipelineContractError(
                f"Exporter returned batch_id '{reported_batch_id}' for '{batch_id}'."
            )
        if reported_row_count != len(rows):
            raise PipelineContractError(
                f"Exporter reported row_count={reported_row_count} for {len(rows)} rows."
            )
        if any(not key.strip() for key in output_keys):
            raise PipelineContractError("Exporter output_keys must not contain empty values.")

    def _validate_batch_progress(
        self,
        *,
        batch_id: str,
        input_row_count: int,
        output_row_count: int,
        reported_batch_id: str,
        reported_input_row_count: int,
        reported_output_row_count: int,
    ) -> None:
        if reported_batch_id != batch_id:
            raise PipelineContractError(
                f"Batch progress returned batch_id '{reported_batch_id}' for '{batch_id}'."
            )
        if reported_input_row_count != input_row_count:
            raise PipelineContractError(
                "Batch progress returned a mismatched input_row_count."
            )
        if reported_output_row_count != output_row_count:
            raise PipelineContractError(
                "Batch progress returned a mismatched output_row_count."
            )


__all__ = [
    "Executor",
    "ExecutorRunContext",
    "Exporter",
    "ObjectStorePipelineRuntimeAdapter",
    "OutputCompletionTracker",
    "PipelineContractError",
    "PipelineRuntimeAdapter",
    "RayExporter",
    "RunSpec",
    "StreamingRayExecutor",
]
