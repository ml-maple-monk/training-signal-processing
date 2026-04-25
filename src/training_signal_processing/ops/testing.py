from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.dataset import (
    DatasetBuilder,
    DatasetHandle,
    LocalRayDatasetBuilder,
    RayDatasetHandle,
)
from ..core.models import ExecutionLogEvent, OpTestResult
from ..core.observability import ExecutionLogger, StructuredExecutionLogger
from .base import Batch, Op
from .registry import ResolvedOpPipeline

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


class OpTestHarness(ABC):
    @abstractmethod
    def run_op(self, op: Op, rows: Batch, batch_size: int) -> OpTestResult:
        raise NotImplementedError

    @abstractmethod
    def run_pipeline(
        self,
        pipeline: ResolvedOpPipeline,
        rows: Batch,
        batch_size: int,
    ) -> OpTestResult:
        raise NotImplementedError


class RayOpTestHarness(OpTestHarness):
    """Runs single ops or explicit op sequences inside the Ray environment only."""

    def __init__(
        self,
        dataset_builder: DatasetBuilder,
        logger: ExecutionLogger,
    ) -> None:
        self.dataset_builder = dataset_builder
        self.logger = logger

    def run_op(self, op: Op, rows: Batch, batch_size: int) -> OpTestResult:
        self._validate_test_request(op.name, rows, batch_size)
        run_id = f"op-test:{op.name}"
        self.logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="op_test.start",
                message=f"Starting Ray op test for '{op.name}'.",
                run_id=run_id,
                op_name=op.name,
                details={"input_row_count": len(rows), "batch_size": batch_size},
            )
        )
        dataset = self.dataset_builder.build_for_op_test(rows)
        result_rows = self._collect_rows(self._apply_op(dataset, op, batch_size), batch_size)
        self.logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="op_test.finish",
                message=f"Finished Ray op test for '{op.name}'.",
                run_id=run_id,
                op_name=op.name,
                details={"output_row_count": len(result_rows), "batch_size": batch_size},
            )
        )
        return OpTestResult(
            op_name=op.name,
            batch_size=batch_size,
            input_row_count=len(rows),
            output_row_count=len(result_rows),
            rows=result_rows,
        )

    def run_pipeline(
        self,
        pipeline: ResolvedOpPipeline,
        rows: Batch,
        batch_size: int,
    ) -> OpTestResult:
        self._validate_test_request("pipeline", rows, batch_size)
        run_id = "op-test:pipeline"
        self.logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="op_test.pipeline.start",
                message="Starting Ray pipeline-op test.",
                run_id=run_id,
                details={"op_names": pipeline.names, "input_row_count": len(rows)},
            )
        )
        dataset = self.dataset_builder.build_for_op_test(rows)
        dataset = self._apply_op(dataset, pipeline.prepare_op, batch_size)
        for op in pipeline.transform_ops:
            dataset = self._apply_op(dataset, op, batch_size)
        if pipeline.export_op is not None:
            dataset = self._apply_op(dataset, pipeline.export_op, batch_size)
        result_rows = self._collect_rows(dataset, batch_size)
        self.logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code="op_test.pipeline.finish",
                message="Finished Ray pipeline-op test.",
                run_id=run_id,
                details={"output_row_count": len(result_rows), "op_names": pipeline.names},
            )
        )
        return OpTestResult(
            op_name="pipeline",
            batch_size=batch_size,
            input_row_count=len(rows),
            output_row_count=len(result_rows),
            rows=result_rows,
        )

    def _apply_op(self, dataset: DatasetHandle, op: Op, batch_size: int) -> DatasetHandle:
        if not isinstance(dataset, RayDatasetHandle):
            raise TypeError("RayOpTestHarness requires a RayDatasetHandle dataset.")
        return self.dataset_builder.apply_op_transform(
            dataset,
            op=op,
            batch_size=batch_size,
        )

    def _collect_rows(self, dataset: DatasetHandle, batch_size: int) -> Batch:
        rows: Batch = []
        for batch in self.dataset_builder.iter_batches(dataset, batch_size):
            rows.extend(batch)
        return rows

    def _validate_test_request(self, op_name: str, rows: Batch, batch_size: int) -> None:
        if not op_name:
            raise ValueError("Op test requires a non-empty op name.")
        if not rows:
            raise ValueError("Op test requires at least one input row.")
        if batch_size <= 0:
            raise ValueError("Op test batch_size must be a positive integer.")


def build_default_ray_op_test_harness() -> RayOpTestHarness:
    return RayOpTestHarness(
        dataset_builder=LocalRayDatasetBuilder(),
        logger=StructuredExecutionLogger("training_signal_processing.op_test"),
    )
