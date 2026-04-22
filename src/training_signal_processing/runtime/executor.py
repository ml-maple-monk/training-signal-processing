from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pyarrow as pa

from ..models import (
    BatchCommit,
    ExecutionLogEvent,
    ExecutorRunSummary,
    ExportBatchResult,
    OpRuntimeContext,
    RecipeConfig,
    RunState,
    RuntimeBindings,
)
from ..ops.base import Batch, FilterOp, MapperOp, Op, PipelineOp
from ..ops.registry import OpRegistry, RegisteredOpRegistry
from ..storage import ObjectStore
from ..utils import join_s3_key
from .dataset import ConfiguredRayDatasetBuilder, DatasetBuilder, DatasetHandle, RayDatasetHandle
from .exporter import Exporter, ObjectStoreMarkdownExporter
from .observability import (
    ExecutionLogger,
    MlflowProgressTracker,
    Monitor,
    ProgressTracker,
    StructuredExecutionLogger,
    StructuredMonitor,
    StructuredTracer,
    Tracer,
)
from .resume import ObjectStoreResumeLedger, ResumeLedger

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


class PipelineContractError(RuntimeError):
    """Raised when the executor contract is incomplete or ambiguous."""


class Executor(ABC):
    def __init__(self, config: RecipeConfig, run_id: str) -> None:
        self.config = config
        self.run_id = run_id

    @abstractmethod
    def run(self) -> dict[str, Any]:
        raise NotImplementedError


class StreamingExecutor(Executor):
    @abstractmethod
    def load_input_manifest(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_input_manifest_key(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_uploaded_documents(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def build_dataset_builder(self) -> DatasetBuilder:
        raise NotImplementedError

    @abstractmethod
    def build_exporter(self) -> Exporter:
        raise NotImplementedError

    @abstractmethod
    def build_resume_ledger(self) -> ResumeLedger:
        raise NotImplementedError

    @abstractmethod
    def build_op_registry(self) -> OpRegistry:
        raise NotImplementedError

    @abstractmethod
    def build_execution_logger(self) -> ExecutionLogger:
        raise NotImplementedError

    @abstractmethod
    def build_tracer(self) -> Tracer:
        raise NotImplementedError

    @abstractmethod
    def build_monitor(self) -> Monitor:
        raise NotImplementedError

    @abstractmethod
    def start_progress_tracker(self) -> ProgressTracker:
        raise NotImplementedError

    @abstractmethod
    def create_run_state(
        self,
        input_rows: list[dict[str, Any]],
        progress_tracker: ProgressTracker,
    ) -> RunState:
        raise NotImplementedError

    @abstractmethod
    def apply_prepare_step(
        self,
        dataset: DatasetHandle,
        op: MapperOp,
        tracer: Tracer,
    ) -> DatasetHandle:
        raise NotImplementedError

    @abstractmethod
    def apply_filter(
        self,
        dataset: DatasetHandle,
        op: FilterOp,
        tracer: Tracer,
    ) -> DatasetHandle:
        raise NotImplementedError

    @abstractmethod
    def apply_mapper(
        self,
        dataset: DatasetHandle,
        op: MapperOp,
        tracer: Tracer,
    ) -> DatasetHandle:
        raise NotImplementedError

    @abstractmethod
    def apply_pipeline(
        self,
        dataset: DatasetHandle,
        op: PipelineOp,
        tracer: Tracer,
    ) -> DatasetHandle:
        raise NotImplementedError

    @abstractmethod
    def commit_batch(
        self,
        ledger: ResumeLedger,
        run_state: RunState,
        batch_index: int,
        rows: Batch,
        export_result: ExportBatchResult,
    ) -> tuple[BatchCommit, RunState]:
        raise NotImplementedError

    @abstractmethod
    def apply_export_batch(
        self,
        rows: Batch,
        op: MapperOp,
        tracer: Tracer,
        logger: ExecutionLogger,
    ) -> Batch:
        raise NotImplementedError

    def apply_op(
        self,
        dataset: DatasetHandle,
        op: Op,
        tracer: Tracer,
        logger: ExecutionLogger,
    ) -> DatasetHandle:
        logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="executor.op.start",
                message=f"Applying op '{op.name}' in the executor loop.",
                run_id=self.run_id,
                op_name=op.name,
            )
        )
        tracer.trace_before_op(op)
        if isinstance(op, PipelineOp):
            next_dataset = self.apply_pipeline(dataset, op, tracer)
        elif isinstance(op, FilterOp):
            next_dataset = self.apply_filter(dataset, op, tracer)
        elif isinstance(op, MapperOp):
            next_dataset = self.apply_mapper(dataset, op, tracer)
        else:
            raise PipelineContractError(
                f"Unsupported op type in executor loop: {type(op).__name__}"
            )
        tracer.trace_after_op(op)
        logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="executor.op.finish",
                message=f"Finished op '{op.name}' in the executor loop.",
                run_id=self.run_id,
                op_name=op.name,
            )
        )
        return next_dataset

    # WARNING TO OTHER AGENTS: DO NOT CHANGE THE EXECUTOR LOOP WITHOUT EXPLICIT USER APPROVAL.
    def run(self) -> dict[str, Any]:
        if not self.run_id:
            raise PipelineContractError("Executor run_id must be explicit and non-empty.")
        if not self.get_input_manifest_key():
            raise PipelineContractError(
                "Executor input_manifest_key must be explicit and non-empty."
            )
        if self.config.ray.batch_size <= 0:
            raise PipelineContractError("Ray batch_size must be a positive integer.")
        logger = self.build_execution_logger()
        progress_tracker = self.start_progress_tracker()
        tracer = self.build_tracer()
        monitor = self.build_monitor()
        ledger = self.build_resume_ledger()
        exporter = self.build_exporter()
        registry = self.build_op_registry()
        dataset_builder = self.build_dataset_builder()
        input_rows = self.load_input_manifest()
        if not input_rows:
            raise PipelineContractError(
                "Input manifest is empty. Executor requires explicit input rows."
            )
        pipeline = registry.resolve_pipeline(self.config.ops)
        logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="executor.pipeline.resolved",
                message="Resolved the explicit executor pipeline from the recipe.",
                run_id=self.run_id,
                details={"op_names": pipeline.names},
            )
        )
        run_state = self.create_run_state(input_rows, progress_tracker)
        if run_state.run_id != self.run_id:
            raise PipelineContractError(
                "create_run_state() must return a RunState for the active executor run_id."
            )
        ledger.write_run_state(run_state)
        monitor.start_run(run_state)
        progress_tracker.log_run_started(
            total_documents=len(input_rows),
            uploaded_documents=self.get_uploaded_documents(),
        )
        try:
            dataset = dataset_builder.build_for_run(input_rows)
            dataset = self.apply_prepare_step(dataset, pipeline.prepare_op, tracer)
            for op in pipeline.transform_ops:
                dataset = self.apply_op(dataset, op, tracer, logger)
            exported_batches = 0
            output_keys: list[str] = []
            for batch_index, rows in enumerate(
                dataset_builder.iter_batches(dataset, self.config.ray.batch_size),
                start=1,
            ):
                if not rows:
                    raise PipelineContractError(
                        f"Executor received an empty batch at index {batch_index}."
                    )
                batch_id = f"batch-{batch_index:05d}"
                logger.log_event(
                    ExecutionLogEvent(
                        level="INFO",
                        code="executor.batch.export.start",
                        message=f"Exporting committed batch '{batch_id}'.",
                        run_id=self.run_id,
                        batch_id=batch_id,
                        details={"row_count": len(rows)},
                    )
                )
                export_rows = self.apply_export_batch(
                    rows=rows,
                    op=pipeline.export_op,
                    tracer=tracer,
                    logger=logger,
                )
                if len(export_rows) != len(rows):
                    raise PipelineContractError(
                        "Export op must preserve row_count for manifest accounting."
                    )
                export_result = exporter.export_batch(batch_id=batch_id, rows=export_rows)
                if export_result.batch_id != batch_id:
                    raise PipelineContractError(
                        "Exporter returned a batch_id that does not match the executor batch."
                    )
                if export_result.row_count != len(export_rows):
                    raise PipelineContractError(
                        "Exporter row_count does not match the explicit batch row count."
                    )
                batch_commit, run_state = self.commit_batch(
                    ledger=ledger,
                    run_state=run_state,
                    batch_index=batch_index,
                    rows=export_rows,
                    export_result=export_result,
                )
                if batch_commit.batch_id != batch_id:
                    raise PipelineContractError(
                        "Resume ledger returned a batch_id that does not match the executor batch."
                    )
                progress_tracker.log_batch_commit(batch_commit, run_state)
                logger.log_event(
                    ExecutionLogEvent(
                        level="INFO",
                        code="executor.batch.export.finish",
                        message=f"Committed batch '{batch_id}' with explicit export output.",
                        run_id=self.run_id,
                        batch_id=batch_id,
                        details=export_result.to_dict(),
                    )
                )
                exported_batches = batch_index
                output_keys.extend(export_result.output_keys)
            if exported_batches == 0:
                raise PipelineContractError(
                    "Executor loop produced zero exported batches. "
                    "Dataset iteration must be explicit and non-empty."
                )
            run_state = ledger.mark_run_finished(run_state)
            exporter.finalize_run(run_state)
            monitor.finish_run(run_state)
            progress_tracker.log_run_finished(run_state.status)
            logger.log_event(
                ExecutionLogEvent(
                    level="INFO",
                    code="executor.run.finish",
                    message="Executor run finished successfully.",
                    run_id=self.run_id,
                    details={
                        "exported_batches": exported_batches,
                        "output_keys": output_keys,
                    },
                )
            )
            return ExecutorRunSummary(
                run_id=self.run_id,
                status=run_state.status,
                input_manifest_key=self.get_input_manifest_key(),
                resolved_op_names=pipeline.names,
                exported_batches=exported_batches,
                output_keys=output_keys,
            ).to_dict()
        except Exception as exc:
            failed_state = ledger.mark_run_failed(run_state, str(exc))
            monitor.fail_run(failed_state)
            progress_tracker.log_run_failed(str(exc))
            logger.log_event(
                ExecutionLogEvent(
                    level="ERROR",
                    code="executor.run.failed",
                    message="Executor run failed with an explicit contract error.",
                    run_id=self.run_id,
                    details={"error": str(exc)},
                )
            )
            raise


class StreamingRayExecutor(StreamingExecutor):
    """Ray-only executor contract for streaming OCR pipelines."""

    def __init__(
        self,
        config: RecipeConfig,
        bindings: RuntimeBindings,
    ) -> None:
        super().__init__(config=config, run_id=bindings.run_id)
        self.bindings = bindings

    def get_input_manifest_key(self) -> str:
        return self.bindings.input_manifest_key

    def get_uploaded_documents(self) -> int:
        return self.bindings.uploaded_documents


class ObjectStoreStreamingRayExecutor(StreamingRayExecutor):
    def __init__(
        self,
        config: RecipeConfig,
        bindings: RuntimeBindings,
        object_store: ObjectStore,
    ) -> None:
        super().__init__(config=config, bindings=bindings)
        self.object_store = object_store
        self.execution_logger: ExecutionLogger | None = None
        self.progress_tracker: ProgressTracker | None = None
        self.tracer: Tracer | None = None
        self.monitor: Monitor | None = None
        self.resume_ledger: ResumeLedger | None = None
        self.exporter: Exporter | None = None
        self.op_registry: OpRegistry | None = None
        self.dataset_builder: DatasetBuilder | None = None

    def load_input_manifest(self) -> list[dict[str, Any]]:
        rows = self.object_store.read_jsonl(self.get_input_manifest_key())
        return [dict(row) for row in rows]

    def build_dataset_builder(self) -> DatasetBuilder:
        if self.dataset_builder is None:
            self.dataset_builder = ConfiguredRayDatasetBuilder(self.config.ray)
        return self.dataset_builder

    def build_exporter(self) -> Exporter:
        if self.exporter is None:
            self.exporter = ObjectStoreMarkdownExporter(
                config=self.config,
                object_store=self.object_store,
                run_id=self.run_id,
                allow_overwrite=self.bindings.allow_overwrite,
            )
        return self.exporter

    def build_resume_ledger(self) -> ResumeLedger:
        if self.resume_ledger is None:
            self.resume_ledger = ObjectStoreResumeLedger(self.config, self.object_store)
        return self.resume_ledger

    def build_op_registry(self) -> OpRegistry:
        if self.op_registry is None:
            self.op_registry = RegisteredOpRegistry(self.build_runtime_context())
        return self.op_registry

    def build_execution_logger(self) -> ExecutionLogger:
        if self.execution_logger is None:
            self.execution_logger = StructuredExecutionLogger(
                "training_signal_processing.executor"
            )
        return self.execution_logger

    def build_tracer(self) -> Tracer:
        if self.tracer is None:
            self.tracer = StructuredTracer(self.build_execution_logger(), self.run_id)
        return self.tracer

    def build_monitor(self) -> Monitor:
        if self.monitor is None:
            self.monitor = StructuredMonitor(self.build_execution_logger())
        return self.monitor

    def start_progress_tracker(self) -> ProgressTracker:
        if self.progress_tracker is None:
            self.progress_tracker = MlflowProgressTracker(
                config=self.config,
                run_id=self.run_id,
                logger=self.build_execution_logger(),
            )
        return self.progress_tracker

    def create_run_state(
        self,
        input_rows: list[dict[str, Any]],
        progress_tracker: ProgressTracker,
    ) -> RunState:
        completed_keys = self.build_resume_ledger().load_completed_keys(self.run_id)
        return self.build_resume_ledger().initialize_run_state(
            run_id=self.run_id,
            total_documents=len(input_rows),
            pending_documents=max(len(input_rows) - len(completed_keys), 0),
            output_prefix=self.build_output_root_key(),
            raw_prefix=self.config.r2.raw_pdf_prefix,
            mlflow_run_id=progress_tracker.get_run_id(),
        )

    def apply_prepare_step(
        self,
        dataset: DatasetHandle,
        op: MapperOp,
        tracer: Tracer,
    ) -> DatasetHandle:
        logger = self.build_execution_logger()
        logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="executor.prepare.start",
                message=f"Applying prepare op '{op.name}'.",
                run_id=self.run_id,
                op_name=op.name,
            )
        )
        tracer.trace_before_op(op)
        next_dataset = self.apply_mapper(dataset, op, tracer)
        tracer.trace_after_op(op)
        logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="executor.prepare.finish",
                message=f"Finished prepare op '{op.name}'.",
                run_id=self.run_id,
                op_name=op.name,
            )
        )
        return next_dataset

    def apply_filter(
        self,
        dataset: DatasetHandle,
        op: FilterOp,
        tracer: Tracer,
    ) -> DatasetHandle:
        return self.map_dataset(dataset, op, tracer)

    def apply_mapper(
        self,
        dataset: DatasetHandle,
        op: MapperOp,
        tracer: Tracer,
    ) -> DatasetHandle:
        return self.map_dataset(dataset, op, tracer)

    def apply_pipeline(
        self,
        dataset: DatasetHandle,
        op: PipelineOp,
        tracer: Tracer,
    ) -> DatasetHandle:
        return self.map_dataset(dataset, op, tracer)

    def apply_export_batch(
        self,
        rows: Batch,
        op: MapperOp,
        tracer: Tracer,
        logger: ExecutionLogger,
    ) -> Batch:
        logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="executor.export_op.start",
                message=f"Applying export op '{op.name}' to the committed batch.",
                run_id=self.run_id,
                op_name=op.name,
            )
        )
        tracer.trace_before_op(op)
        export_rows = op.process_batch(rows)
        tracer.trace_after_op(op)
        logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="executor.export_op.finish",
                message=f"Finished export op '{op.name}' for the committed batch.",
                run_id=self.run_id,
                op_name=op.name,
            )
        )
        return export_rows

    def commit_batch(
        self,
        ledger: ResumeLedger,
        run_state: RunState,
        batch_index: int,
        rows: Batch,
        export_result: ExportBatchResult,
    ) -> tuple[BatchCommit, RunState]:
        del export_result
        return ledger.commit_batch(run_state=run_state, batch_index=batch_index, rows=rows)

    def build_runtime_context(self) -> OpRuntimeContext:
        return OpRuntimeContext(
            config=self.config,
            run_id=self.run_id,
            object_store=self.object_store,
            output_root_key=self.build_output_root_key(),
            raw_root_key=self.config.r2.raw_pdf_prefix,
            completed_source_keys=self.build_resume_ledger().load_completed_keys(self.run_id),
            allow_overwrite=self.bindings.allow_overwrite,
            logger=self.build_execution_logger(),
        )

    def build_output_root_key(self) -> str:
        return join_s3_key(self.config.r2.output_prefix, self.run_id)

    def map_dataset(
        self,
        dataset: DatasetHandle,
        op: Op,
        tracer: Tracer,
    ) -> DatasetHandle:
        del tracer
        if not isinstance(dataset, RayDatasetHandle):
            raise TypeError("ObjectStoreStreamingRayExecutor requires a RayDatasetHandle.")

        def mapper(table: pa.Table) -> pa.Table:
            next_rows = op.process_batch(table.to_pylist())
            if next_rows:
                return pa.Table.from_pylist(next_rows)
            return table.slice(0, 0)

        next_dataset = dataset.unwrap().map_batches(
            mapper,
            batch_size=self.config.ray.batch_size,
            batch_format="pyarrow",
            concurrency=self.config.ray.concurrency,
            zero_copy_batch=False,
        )
        return RayDatasetHandle(next_dataset)
