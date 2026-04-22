from __future__ import annotations

from pathlib import Path
from typing import Any

from training_signal_processing.models import (
    BatchCommit,
    OpConfig,
    OpRuntimeContext,
    RayConfig,
    RunArtifactLayout,
    RunState,
    RuntimeRunBindings,
    RuntimeTrackingContext,
)
from training_signal_processing.ops.base import Batch, MapperOp
from training_signal_processing.ops.registry import RegisteredOpRegistry
from training_signal_processing.runtime.dataset import DatasetBuilder, DatasetHandle
from training_signal_processing.runtime.executor import PipelineRuntimeAdapter, StreamingRayExecutor
from training_signal_processing.runtime.exporter import Exporter
from training_signal_processing.runtime.observability import ExecutionLogger
from training_signal_processing.runtime.resume import ResumeLedger


class SimpleDatasetHandle(DatasetHandle):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def unwrap(self) -> Any:
        return self.rows


class SimpleDatasetBuilder(DatasetBuilder):
    def build_for_run(self, input_rows: list[dict[str, Any]]) -> DatasetHandle:
        return SimpleDatasetHandle(input_rows)

    def build_for_op_test(self, rows: Batch) -> DatasetHandle:
        return SimpleDatasetHandle(rows)

    def iter_batches(self, dataset: DatasetHandle, batch_size: int):
        rows = dataset.unwrap()
        for index in range(0, len(rows), batch_size):
            yield rows[index : index + batch_size]


class PrepareGenericOp(MapperOp):
    op_name = "test_prepare_generic"
    op_stage = "prepare"

    def process_batch(self, batch: Batch) -> Batch:
        return [{**row, "prepared": True} for row in batch]


class TransformGenericOp(MapperOp):
    op_name = "test_transform_generic"
    op_stage = "transform"

    def process_batch(self, batch: Batch) -> Batch:
        runtime = self.require_runtime()
        return [{**row, "run_id": runtime.run_id, "transformed": True} for row in batch]


class ExportGenericOp(MapperOp):
    op_name = "test_export_generic"
    op_stage = "export"

    def process_batch(self, batch: Batch) -> Batch:
        return [{**row, "ready_for_export": True} for row in batch]


class FakeExporter(Exporter):
    def __init__(self) -> None:
        self.finalized_run_state: RunState | None = None

    def export_batch(self, batch_id: str, rows: Batch):
        from training_signal_processing.models import ExportBatchResult

        return ExportBatchResult(
            batch_id=batch_id,
            row_count=len(rows),
            output_keys=[f"memory://{row['id']}" for row in rows],
        )

    def finalize_run(self, run_state: RunState) -> None:
        self.finalized_run_state = run_state


class FakeResumeLedger(ResumeLedger):
    def __init__(self) -> None:
        self.run_state: RunState | None = None

    def find_latest_partial_run(self) -> str | None:
        return None

    def load_run_state(self, run_id: str) -> RunState | None:
        if self.run_state is not None and self.run_state.run_id == run_id:
            return self.run_state
        return None

    def load_completed_item_keys(self, run_id: str) -> set[str]:
        return set()

    def initialize_run_state(
        self,
        *,
        run_id: str,
        total_items: int,
        pending_items: int,
        precompleted_count: int,
        artifact_layout: RunArtifactLayout,
        tracking_run_id: str,
    ) -> RunState:
        self.run_state = RunState(
            run_id=run_id,
            status="running",
            total_items=total_items,
            pending_items=pending_items,
            success_count=0,
            failed_count=0,
            skipped_count=precompleted_count,
            last_committed_batch=0,
            started_at="2026-04-22T00:00:00Z",
            updated_at="2026-04-22T00:00:00Z",
            source_root_key=artifact_layout.source_root_key,
            output_root_key=artifact_layout.output_root_key,
            tracking_run_id=tracking_run_id,
        )
        return self.run_state

    def commit_batch(
        self,
        *,
        run_state: RunState,
        batch_index: int,
        input_row_count: int,
        rows: list[dict[str, Any]],
    ) -> tuple[BatchCommit, RunState]:
        self.run_state = RunState(
            run_id=run_state.run_id,
            status="running",
            total_items=run_state.total_items,
            pending_items=max(run_state.pending_items - input_row_count, 0),
            success_count=run_state.success_count + len(rows),
            failed_count=run_state.failed_count,
            skipped_count=run_state.skipped_count,
            last_committed_batch=batch_index,
            started_at=run_state.started_at,
            updated_at="2026-04-22T00:00:01Z",
            source_root_key=run_state.source_root_key,
            output_root_key=run_state.output_root_key,
            tracking_run_id=run_state.tracking_run_id,
        )
        return (
            BatchCommit(
                batch_id=f"batch-{batch_index:05d}",
                input_row_count=input_row_count,
                output_row_count=len(rows),
                success_count=len(rows),
                failed_count=0,
                skipped_count=0,
                duration_sec=0.1,
                manifest_key=f"memory://manifest/{batch_index}",
                event_key=f"memory://event/{batch_index}",
            ),
            self.run_state,
        )

    def write_run_state(self, run_state: RunState) -> None:
        self.run_state = run_state

    def mark_run_finished(self, run_state: RunState) -> RunState:
        self.run_state = RunState(
            **{**run_state.to_dict(), "status": "success", "pending_items": 0}
        )
        return self.run_state

    def mark_run_failed(self, run_state: RunState, message: str) -> RunState:
        self.run_state = RunState(
            **{**run_state.to_dict(), "status": "failed", "error_message": message}
        )
        return self.run_state


class FakePipelineAdapter(PipelineRuntimeAdapter):
    def __init__(self) -> None:
        self.exporter = FakeExporter()
        self.resume_ledger = FakeResumeLedger()
        self.dataset_builder = SimpleDatasetBuilder()
        self.bindings = RuntimeRunBindings(
            run_id="fake-run-001",
            input_manifest_key="memory://manifest/input.jsonl",
            uploaded_items=2,
        )

    def get_run_bindings(self) -> RuntimeRunBindings:
        return self.bindings

    def get_execution_config(self) -> RayConfig:
        return RayConfig(
            executor_type="ray",
            batch_size=1,
            concurrency=1,
            target_num_blocks=1,
        )

    def get_tracking_context(self) -> RuntimeTrackingContext:
        return RuntimeTrackingContext(
            enabled=False,
            tracking_uri="",
            experiment_name="fake",
            run_name="generic-runtime-test",
            executor_type="ray",
            batch_size=1,
            concurrency=1,
        )

    def get_op_configs(self) -> list[OpConfig]:
        return [
            OpConfig(name="test_prepare_generic", type="mapper"),
            OpConfig(name="test_transform_generic", type="mapper"),
            OpConfig(name="test_export_generic", type="mapper"),
        ]

    def get_artifact_layout(self) -> RunArtifactLayout:
        return RunArtifactLayout(
            source_root_key="source/items",
            output_root_key="output/items/fake-run-001",
        )

    def load_input_rows(self) -> list[dict[str, object]]:
        return [{"id": "row-a"}, {"id": "row-b"}]

    def build_runtime_context(
        self,
        *,
        logger: ExecutionLogger,
        completed_item_keys: set[str],
    ) -> OpRuntimeContext:
        return OpRuntimeContext(
            config={"name": "fake"},
            run_id=self.bindings.run_id,
            object_store=object(),
            output_root_key="output/items/fake-run-001",
            source_root_key="source/items",
            completed_item_keys=completed_item_keys,
            logger=logger,
        )

    def build_op_registry(self, runtime_context: OpRuntimeContext):
        return RegisteredOpRegistry(runtime_context=runtime_context)

    def build_exporter(self) -> Exporter:
        return self.exporter

    def build_resume_ledger(self) -> ResumeLedger:
        return self.resume_ledger

    def build_dataset_builder(self) -> DatasetBuilder:
        return self.dataset_builder


def test_runtime_modules_do_not_import_ocr_pipeline() -> None:
    runtime_dir = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "training_signal_processing"
        / "runtime"
    )
    runtime_files = sorted(runtime_dir.glob("*.py"))
    assert runtime_files
    for runtime_file in runtime_files:
        contents = runtime_file.read_text(encoding="utf-8")
        assert "pipelines.ocr" not in contents, runtime_file


def test_streaming_executor_runs_with_fake_pipeline_adapter(capsys) -> None:
    adapter = FakePipelineAdapter()
    summary = StreamingRayExecutor(adapter).run()
    captured = capsys.readouterr()

    assert summary["status"] == "success"
    assert summary["run_id"] == "fake-run-001"
    assert summary["resolved_op_names"] == [
        "test_prepare_generic",
        "test_transform_generic",
        "test_export_generic",
    ]
    assert summary["exported_batches"] == 2
    assert summary["output_keys"] == ["memory://row-a", "memory://row-b"]
    assert adapter.exporter.finalized_run_state is not None
    assert adapter.exporter.finalized_run_state.status == "running"
    assert adapter.resume_ledger.run_state is not None
    assert adapter.resume_ledger.run_state.status == "success"
    assert "[run] run_id=fake-run-001" in captured.out
    assert "[batch:start] batch_id=batch-00001" in captured.out
    assert "[op:start] batch_id=batch-00001 op=test_prepare_generic" in captured.out
    assert "[batch:commit] batch_id=batch-00002" in captured.out
    assert "[run:finish] run_id=fake-run-001 status=success" in captured.out
    assert "run fake-run-001:" in captured.err
