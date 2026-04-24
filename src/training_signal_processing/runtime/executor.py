from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from ..core.models import (
    ExecutionLogEvent,
    ExecutorRunSummary,
    OpConfig,
    OpRuntimeContext,
    RayConfig,
    RayTransformResources,
    RunArtifactLayout,
    RunState,
    RuntimeRunBindings,
    RuntimeTrackingContext,
)
from ..core.utils import utc_isoformat
from ..ops.base import Batch, Op
from ..ops.registry import OpRegistry
from ..storage.object_store import R2ObjectStore
from .async_upload_coordinator import AsyncUploadCoordinator
from .dataset import ConfiguredRayDatasetBuilder, DatasetBuilder
from .exporter import Exporter, RayExporter
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
from .resume import ResumeLedger

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


class PipelineContractError(ValueError):
    """Raised when a pipeline adapter violates the protected executor contract."""


class Executor(ABC):
    @abstractmethod
    def run(self) -> dict[str, object]:
        raise NotImplementedError


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
        completed_item_keys: set[str],
    ) -> OpRuntimeContext:
        raise NotImplementedError

    @abstractmethod
    def build_op_registry(self, runtime_context: OpRuntimeContext) -> OpRegistry:
        raise NotImplementedError

    @abstractmethod
    def build_exporter(self) -> Exporter:
        raise NotImplementedError

    @abstractmethod
    def build_resume_ledger(self) -> ResumeLedger:
        raise NotImplementedError

    def build_dataset_builder(self) -> DatasetBuilder:
        return ConfiguredRayDatasetBuilder(self.get_execution_config())

    def resolve_completed_item_keys(self, completed_item_keys: set[str]) -> set[str]:
        return completed_item_keys

    def resolve_transform_resources(
        self,
        *,
        op: Op,
        execution: RayConfig,
    ) -> RayTransformResources:
        return RayTransformResources()


class StreamingRayExecutor(Executor):
    def __init__(self, pipeline: PipelineRuntimeAdapter) -> None:
        self.pipeline = pipeline

    def run(self) -> dict[str, object]:
        bindings = self.pipeline.get_run_bindings()
        execution = self.pipeline.get_execution_config()
        tracking = self.pipeline.get_tracking_context()
        artifact_layout = self.pipeline.get_artifact_layout()
        op_configs = self.pipeline.get_op_configs()
        self.validate_contract(
            bindings=bindings,
            execution=execution,
            tracking=tracking,
            artifact_layout=artifact_layout,
            op_configs=op_configs,
        )

        logger = self.build_execution_logger(tracking)
        progress_tracker = MlflowProgressTracker(
            tracking=tracking,
            run_id=bindings.run_id,
            logger=logger,
        )
        resume_ledger = self.pipeline.build_resume_ledger()
        raw_completed_item_keys = resume_ledger.load_completed_item_keys(bindings.run_id)
        logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="executor.manifest.loaded",
                message="Loaded input manifest rows.",
                run_id=bindings.run_id,
                details={"input_manifest_key": bindings.input_manifest_key},
            )
        )
        completed_item_keys = self.pipeline.resolve_completed_item_keys(raw_completed_item_keys)
        logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="executor.resume.loaded",
                message="Loaded resume state.",
                run_id=bindings.run_id,
                details={"completed_item_keys": len(completed_item_keys)},
            )
        )
        runtime_context = self.pipeline.build_runtime_context(
            logger=logger,
            completed_item_keys=completed_item_keys,
        )
        registry = self.pipeline.build_op_registry(runtime_context)
        resolved_pipeline = registry.resolve_pipeline(op_configs)
        exporter = self.pipeline.build_exporter()
        coordinator = self._build_upload_coordinator(
            execution=execution,
            runtime_context=runtime_context,
        )
        if coordinator is not None and isinstance(exporter, RayExporter):
            exporter.upload_coordinator = coordinator
        dataset_builder = self.pipeline.build_dataset_builder()
        input_rows = self.pipeline.load_input_rows()
        if not input_rows:
            raise PipelineContractError("Pipeline adapter returned zero input rows.")

        total_items = len(input_rows)
        pending_items = max(total_items - len(completed_item_keys), 0)
        progress_reporter = self.build_progress_reporter(
            run_id=bindings.run_id,
            total_items=total_items,
            pending_items=pending_items,
            batch_size=execution.batch_size,
        )
        tracking_run_id = progress_tracker.log_run_started(
            total_items=total_items,
            uploaded_items=bindings.uploaded_items,
        )
        run_state = resume_ledger.initialize_run_state(
            run_id=bindings.run_id,
            total_items=total_items,
            pending_items=pending_items,
            precompleted_count=len(completed_item_keys),
            artifact_layout=artifact_layout,
            tracking_run_id=tracking_run_id,
        )
        tracer = StructuredTracer(logger=logger, run_id=bindings.run_id)
        monitor = StructuredMonitor(logger=logger)
        monitor.start_run(run_state)
        progress_reporter.start_run()
        run_state = self.transition_run_phase(
            run_state=run_state,
            resume_ledger=resume_ledger,
            logger=logger,
            progress_reporter=progress_reporter,
            phase="manifest_loaded",
            detail=f"input_manifest_key={bindings.input_manifest_key}",
        )
        run_state = self.transition_run_phase(
            run_state=run_state,
            resume_ledger=resume_ledger,
            logger=logger,
            progress_reporter=progress_reporter,
            phase="resume_state_loaded",
            detail=f"completed_item_keys={len(completed_item_keys)}",
        )
        exported_batches = 0
        output_keys: list[str] = []

        try:
            run_state = self.transition_run_phase(
                run_state=run_state,
                resume_ledger=resume_ledger,
                logger=logger,
                progress_reporter=progress_reporter,
                phase="dataset_build_start",
            )
            dataset = dataset_builder.build_for_run(input_rows)
            run_state = self.transition_run_phase(
                run_state=run_state,
                resume_ledger=resume_ledger,
                logger=logger,
                progress_reporter=progress_reporter,
                phase="dataset_build_complete",
            )
            dataset = self.apply_pipeline_transforms(
                dataset_builder=dataset_builder,
                dataset=dataset,
                resolved_pipeline=resolved_pipeline,
                execution=execution,
                tracer=tracer,
                logger=logger,
                run_id=bindings.run_id,
            )
            run_state = self.transition_run_phase(
                run_state=run_state,
                resume_ledger=resume_ledger,
                logger=logger,
                progress_reporter=progress_reporter,
                phase="iter_batches_start",
            )
            base_batch_index = run_state.last_committed_batch
            for batch_offset, current_batch in enumerate(
                dataset_builder.iter_batches(dataset, batch_size=execution.batch_size),
                start=1,
            ):
                batch_index = base_batch_index + batch_offset
                batch_id = f"batch-{batch_index:05d}"
                if batch_offset == 1:
                    run_state = self.transition_run_phase(
                        run_state=run_state,
                        resume_ledger=resume_ledger,
                        logger=logger,
                        progress_reporter=progress_reporter,
                        phase="first_batch_materialized",
                        detail=f"batch_id={batch_id} input_rows={len(current_batch)}",
                    )
                progress_reporter.start_batch(batch_id, batch_index, len(current_batch))
                export_result = exporter.export_batch(batch_id=batch_id, rows=current_batch)
                self.validate_export_result(
                    batch_id=batch_id,
                    input_row_count=len(current_batch),
                    rows=current_batch,
                    output_keys=export_result.output_keys,
                    reported_batch_id=export_result.batch_id,
                    reported_row_count=export_result.row_count,
                )
                if coordinator is not None:
                    drain_start = time.monotonic()
                    pre_drain_depth = coordinator.depth()
                    coordinator.drain()
                    logger.log_event(
                        ExecutionLogEvent(
                            level="INFO",
                            code="executor.upload.drain",
                            message=f"Drained upload coordinator for '{batch_id}'.",
                            run_id=bindings.run_id,
                            batch_id=batch_id,
                            details={
                                "depth_before_drain": pre_drain_depth,
                                "drain_seconds": time.monotonic() - drain_start,
                            },
                        )
                    )
                batch_commit, run_state = resume_ledger.commit_batch(
                    run_state=run_state,
                    batch_index=batch_index,
                    input_row_count=len(current_batch),
                    rows=current_batch,
                )
                self.validate_batch_commit(
                    batch_id=batch_id,
                    input_row_count=len(current_batch),
                    output_row_count=len(current_batch),
                    reported_batch_id=batch_commit.batch_id,
                    reported_input_row_count=batch_commit.input_row_count,
                    reported_output_row_count=batch_commit.output_row_count,
                )
                progress_tracker.log_batch_commit(batch_commit, run_state)
                progress_reporter.commit_batch(batch_commit, run_state)
                exported_batches += 1
                output_keys.extend(export_result.output_keys)
            run_state = resume_ledger.mark_run_finished(run_state)
            exporter.finalize_run(run_state)
            if coordinator is not None:
                coordinator.drain()
            monitor.finish_run(run_state)
            progress_tracker.log_run_finished(run_state.status)
            progress_reporter.finish_run(run_state.status)
            return ExecutorRunSummary(
                run_id=bindings.run_id,
                status=run_state.status,
                input_manifest_key=bindings.input_manifest_key,
                resolved_op_names=resolved_pipeline.names,
                exported_batches=exported_batches,
                output_keys=output_keys,
            ).to_dict()
        except Exception as exc:
            message = str(exc)
            if coordinator is not None:
                cancelled = coordinator.abort()
                logger.log_event(
                    ExecutionLogEvent(
                        level="WARNING",
                        code="executor.upload.aborted",
                        message=f"Aborted upload coordinator: {cancelled} cancelled.",
                        run_id=bindings.run_id,
                        details={"cancelled_count": cancelled, "error": message},
                    )
                )
            run_state = resume_ledger.mark_run_failed(run_state, message)
            monitor.fail_run(run_state)
            progress_tracker.log_run_failed(message)
            progress_reporter.fail_run(message)
            return ExecutorRunSummary(
                run_id=bindings.run_id,
                status="failed",
                input_manifest_key=bindings.input_manifest_key,
                resolved_op_names=resolved_pipeline.names,
                exported_batches=exported_batches,
                output_keys=output_keys,
                error_message=message,
            ).to_dict()
        finally:
            if coordinator is not None:
                try:
                    coordinator.close()
                except Exception as close_exc:  # pragma: no cover - defensive
                    logger.log_event(
                        ExecutionLogEvent(
                            level="WARNING",
                            code="executor.upload.close_error",
                            message=f"AsyncUploadCoordinator.close() raised: {close_exc}",
                            run_id=bindings.run_id,
                        )
                    )

    def _build_upload_coordinator(
        self,
        *,
        execution: RayConfig,
        runtime_context: OpRuntimeContext,
    ) -> AsyncUploadCoordinator | None:
        async_cfg = execution.async_upload
        if async_cfg is None or not async_cfg.enabled:
            return None
        object_store = runtime_context.get_object_store()
        if not isinstance(object_store, R2ObjectStore):
            return None
        return AsyncUploadCoordinator(
            object_store=object_store,
            max_in_flight=async_cfg.max_in_flight,
            max_queued=async_cfg.max_queued,
        )

    def transition_run_phase(
        self,
        *,
        run_state: RunState,
        resume_ledger: ResumeLedger,
        logger: ExecutionLogger,
        progress_reporter: ProgressReporter,
        phase: str,
        detail: str = "",
    ) -> RunState:
        timestamp = utc_isoformat()
        updated_state = RunState(
            **{
                **run_state.to_dict(),
                "current_phase": phase,
                "last_phase_at": timestamp,
                "updated_at": timestamp,
            }
        )
        resume_ledger.write_run_state(updated_state)
        progress_reporter.report_phase(phase, detail)
        logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="executor.phase",
                message=f"Run phase changed to '{phase}'.",
                run_id=run_state.run_id,
                details={"phase": phase, "detail": detail, "run_state": updated_state.to_dict()},
            )
        )
        return updated_state

    def build_execution_logger(self, tracking: RuntimeTrackingContext) -> ExecutionLogger:
        if tracking.enabled:
            return MlflowExecutionLogger(
                tracking=tracking,
                logger_name="training_signal_processing",
            )
        return StructuredExecutionLogger("training_signal_processing")

    def build_progress_reporter(
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

    def apply_op(
        self,
        *,
        op: Op,
        batch_id: str,
        batch: Batch,
        tracer: StructuredTracer,
        logger: ExecutionLogger,
        progress_reporter: ProgressReporter,
    ) -> Batch:
        tracer.trace_before_op(op)
        progress_reporter.start_op(batch_id, op.name, len(batch))
        rendered_batch = op.process_batch(batch)
        tracer.trace_after_op(op)
        if not isinstance(rendered_batch, list):
            raise PipelineContractError(
                f"Op '{op.name}' must return a list batch, got {type(rendered_batch).__name__}."
            )
        progress_reporter.finish_op(batch_id, op.name, len(rendered_batch))
        logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="executor.op.complete",
                message=f"Completed op '{op.name}'.",
                run_id=self.pipeline.get_run_bindings().run_id,
                op_name=op.name,
                batch_id=batch_id,
                details={
                    "input_row_count": len(batch),
                    "output_row_count": len(rendered_batch),
                },
            )
        )
        return rendered_batch

    def apply_pipeline_transforms(
        self,
        *,
        dataset_builder: DatasetBuilder,
        dataset,
        resolved_pipeline,
        execution: RayConfig,
        tracer: StructuredTracer,
        logger: ExecutionLogger,
        run_id: str,
    ):
        current_dataset = dataset
        for op in resolved_pipeline.all_ops:
            resources = self.pipeline.resolve_transform_resources(op=op, execution=execution)
            tracer.trace_before_op(op)
            logger.log_event(
                ExecutionLogEvent(
                    level="INFO",
                    code="executor.dataset.transform",
                    message=f"Scheduled dataset transform for op '{op.name}'.",
                    run_id=run_id,
                    op_name=op.name,
                    details={
                        "batch_size": execution.batch_size,
                        **resources.to_dict(),
                    },
                )
            )
            current_dataset = dataset_builder.apply_op_transform(
                current_dataset,
                op=op,
                batch_size=execution.batch_size,
                concurrency=resources.concurrency,
                num_gpus=resources.num_gpus,
                num_cpus=resources.num_cpus,
            )
            tracer.trace_after_op(op)
        return current_dataset

    def validate_contract(
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

    def validate_export_result(
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

    def validate_batch_commit(
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
                f"Resume ledger returned batch_id '{reported_batch_id}' for '{batch_id}'."
            )
        if reported_input_row_count != input_row_count:
            raise PipelineContractError(
                "Resume ledger returned a mismatched input_row_count."
            )
        if reported_output_row_count != output_row_count:
            raise PipelineContractError(
                "Resume ledger returned a mismatched output_row_count."
            )
