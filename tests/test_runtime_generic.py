from __future__ import annotations

from pathlib import Path
from typing import Any

from training_signal_processing.core.dataset import (
    ConfiguredRayDatasetBuilder,
    DatasetBuilder,
    DatasetHandle,
)
from training_signal_processing.core.execution import (
    Exporter,
    OutputCompletionTracker,
    PipelineRuntimeAdapter,
    StreamingRayExecutor,
)
from training_signal_processing.core.models import (
    OpConfig,
    OpRuntimeContext,
    RayConfig,
    RayTransformResources,
    RunArtifactLayout,
    RunState,
    RuntimeRunBindings,
    RuntimeTrackingContext,
)
from training_signal_processing.core.observability import ExecutionLogger
from training_signal_processing.core.storage import ObjectStore
from training_signal_processing.core.utils import join_s3_key
from training_signal_processing.ops.base import Batch, MapperOp
from training_signal_processing.ops.registry import RegisteredOpRegistry


class MemoryObjectStore(ObjectStore):
    bucket = "memory"

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads

    def exists(self, key: str) -> bool:
        return key in self.payloads

    def list_keys(self, prefix: str) -> list[str]:
        return [key for key in self.payloads if key.startswith(prefix)]

    def read_bytes(self, key: str) -> bytes:
        return self.payloads[key]

    def write_bytes(self, key: str, body: bytes) -> None:
        self.payloads[key] = body

    def upload_file(self, path: Path, key: str) -> None:
        self.payloads[key] = path.read_bytes()

    def make_url(self, key: str) -> str:
        return f"memory://{key}"

    def build_pyarrow_filesystem(self):
        raise NotImplementedError


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

    def apply_op_transform(
        self,
        dataset: DatasetHandle,
        *,
        op,
        batch_size: int,
        concurrency: int | None = None,
        num_gpus: float | None = None,
        num_cpus: float | None = None,
    ) -> DatasetHandle:
        rows = dataset.unwrap()
        rendered_rows: list[dict[str, object]] = []
        for index in range(0, len(rows), batch_size):
            rendered_rows.extend(op.process_batch(rows[index : index + batch_size]))
        return SimpleDatasetHandle(rendered_rows)


class RecordingDatasetBuilder(SimpleDatasetBuilder):
    def __init__(self) -> None:
        self.transform_calls: list[dict[str, object]] = []

    def apply_op_transform(
        self,
        dataset: DatasetHandle,
        *,
        op,
        batch_size: int,
        concurrency: int | None = None,
        num_gpus: float | None = None,
        num_cpus: float | None = None,
    ) -> DatasetHandle:
        self.transform_calls.append(
            {
                "op_name": op.name,
                "batch_size": batch_size,
                "concurrency": concurrency,
                "num_gpus": num_gpus,
                "num_cpus": num_cpus,
            }
        )
        return super().apply_op_transform(
            dataset,
            op=op,
            batch_size=batch_size,
            concurrency=concurrency,
            num_gpus=num_gpus,
            num_cpus=num_cpus,
        )


class PrepareGenericOp(MapperOp):
    op_name = "test_prepare_generic"
    op_stage = "prepare"

    def process_batch(self, batch: Batch) -> Batch:
        runtime = self.require_runtime()
        return [
            {**row, "prepared": True}
            for row in batch
            if str(row["id"]) not in runtime.completed_source_keys
        ]


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
        return [
            {
                **row,
                "ready_for_export": True,
                "status": "success",
                "duration_sec": 0.1,
            }
            for row in batch
        ]


class FakeExporter(Exporter):
    def __init__(self, object_store: MemoryObjectStore) -> None:
        self.object_store = object_store
        self.finalized_run_state: RunState | None = None
        self.fail_on_batch_id = ""

    def export_batch(self, batch_id: str, rows: Batch):
        from training_signal_processing.core.models import ExportBatchResult

        if batch_id == self.fail_on_batch_id:
            raise RuntimeError(f"injected export failure on {batch_id}")
        output_keys = [
            join_s3_key(
                "output/items/fake-run-001",
                f"outputs/{row.get('id') or row.get('relative_path') or index}.json",
            )
            for index, row in enumerate(rows, start=1)
        ]
        for output_key in output_keys:
            self.object_store.write_bytes(output_key, b"{}")
        return ExportBatchResult(
            batch_id=batch_id,
            row_count=len(rows),
            output_keys=output_keys,
        )

    def finalize_run(self, run_state: RunState) -> None:
        self.finalized_run_state = run_state


class FakeCompletionTracker(OutputCompletionTracker):
    def source_key_for_input(self, row: dict[str, Any]) -> str:
        return str(row["id"])

    def output_key_for_input(
        self,
        row: dict[str, Any],
        artifact_layout: RunArtifactLayout,
    ) -> str:
        return join_s3_key(artifact_layout.output_root_key, f"outputs/{row['id']}.json")

    def output_listing_prefix(self, artifact_layout: RunArtifactLayout) -> str:
        return join_s3_key(artifact_layout.output_root_key, "outputs")


class FakePipelineAdapter(PipelineRuntimeAdapter):
    def __init__(self) -> None:
        self.object_store = MemoryObjectStore({})
        self.exporter = FakeExporter(self.object_store)
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
        completed_source_keys: set[str],
    ) -> OpRuntimeContext:
        return OpRuntimeContext(
            config={"name": "fake"},
            run_id=self.bindings.run_id,
            object_store=self.object_store,
            output_root_key="output/items/fake-run-001",
            source_root_key="source/items",
            completed_source_keys=completed_source_keys,
            logger=logger,
        )

    def build_op_registry(self, runtime_context: OpRuntimeContext):
        return RegisteredOpRegistry(runtime_context=runtime_context)

    def build_exporter(self) -> Exporter:
        return self.exporter

    def build_completion_tracker(self) -> OutputCompletionTracker:
        return FakeCompletionTracker(self.object_store)

    def build_dataset_builder(self) -> DatasetBuilder:
        return self.dataset_builder


def test_runtime_modules_do_not_import_pipeline_packages(capsys) -> None:
    """Enforce via import-linter that shared layers never import pipelines.

    The contract is declared in pyproject.toml under [tool.importlinter]. This test
    invokes the linter in-process via its public API and asserts exit code 0. When
    a contract breaks, import-linter prints the violation details to stdout; pytest
    captures that output and surfaces it on failure.
    """
    from importlinter.cli import lint_imports

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "pyproject.toml"
    exit_code = lint_imports(config_filename=str(config_path))

    captured = capsys.readouterr()
    assert exit_code == 0, (
        f"import-linter contract violated:\n"
        f"stdout:\n{captured.out}\n"
        f"stderr:\n{captured.err}"
    )


def test_root_package_exposes_only_main_entrypoint() -> None:
    package_dir = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "training_signal_processing"
    )

    root_python_files = {path.name for path in package_dir.glob("*.py")}

    assert root_python_files == {"main.py"}
    assert not (package_dir / "__main__.py").exists()
    assert not (package_dir / "__init__.py").exists()
    assert not (package_dir / "recipe.py").exists()
    assert not (package_dir / "models.py").exists()
    assert not (package_dir / "storage.py").exists()
    assert not (package_dir / "utils.py").exists()


def test_object_store_read_jsonl_parses_memory_body_and_ignores_blank_lines() -> None:
    store = MemoryObjectStore(
        {"input.jsonl": b'{"id": "a"}\n\n  {"id": "b", "value": 2}\n'}
    )

    assert store.read_jsonl("input.jsonl") == [
        {"id": "a"},
        {"id": "b", "value": 2},
    ]


def test_object_store_read_jsonl_returns_empty_list_for_blank_body() -> None:
    store = MemoryObjectStore({"input.jsonl": b"\n  \n"})

    assert store.read_jsonl("input.jsonl") == []


def test_object_store_read_jsonl_rejects_non_object_entries() -> None:
    store = MemoryObjectStore({"input.jsonl": b'{"id": "a"}\n42\n'})

    try:
        store.read_jsonl("input.jsonl")
    except ValueError as exc:
        assert "item 2 must be a JSON object" in str(exc)
    else:
        raise AssertionError("expected non-object JSONL entry to fail")


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
    assert summary["output_keys"] == [
        "output/items/fake-run-001/outputs/row-a.json",
        "output/items/fake-run-001/outputs/row-b.json",
    ]
    assert adapter.exporter.finalized_run_state is not None
    assert adapter.exporter.finalized_run_state.status == "success"
    assert "[run] run_id=fake-run-001" in captured.out
    assert "[phase] run_id=fake-run-001 phase=dataset_build_start" in captured.out
    assert "[phase] run_id=fake-run-001 phase=iter_batches_start" in captured.out
    assert "[batch:start] batch_id=batch-00001" in captured.out
    assert "[batch:finish] batch_id=batch-00002" in captured.out
    assert "[run:finish] run_id=fake-run-001 status=success" in captured.out
    assert "run fake-run-001:" in captured.err
    assert "[phase] run_id=fake-run-001 phase=manifest_loaded" in captured.out
    assert "[phase] run_id=fake-run-001 phase=resume_state_loaded" in captured.out
    assert "[phase] run_id=fake-run-001 phase=dataset_build_complete" in captured.out
    assert "[phase] run_id=fake-run-001 phase=first_batch_materialized" in captured.out


def test_output_completion_tracker_maps_existing_outputs_to_source_keys() -> None:
    adapter = FakePipelineAdapter()
    adapter.object_store.write_bytes(
        "output/items/fake-run-001/outputs/row-a.json",
        b"{}",
    )

    assert adapter.resolve_completed_source_keys(
        input_rows=adapter.load_input_rows(),
    ) == {"row-a"}


def test_output_completion_tracker_ignores_outputs_when_overwrite_allowed() -> None:
    adapter = FakePipelineAdapter()
    adapter.bindings.allow_overwrite = True
    adapter.object_store.write_bytes(
        "output/items/fake-run-001/outputs/row-a.json",
        b"{}",
    )

    assert adapter.resolve_completed_source_keys(
        input_rows=adapter.load_input_rows(),
    ) == set()


def test_streaming_executor_writes_outputs_without_checkpoint_artifacts() -> None:
    adapter = FakePipelineAdapter()

    summary = StreamingRayExecutor(adapter).run()

    assert summary["status"] == "success"
    assert sorted(adapter.object_store.payloads) == [
        "output/items/fake-run-001/outputs/row-a.json",
        "output/items/fake-run-001/outputs/row-b.json",
    ]
    forbidden_fragments = ("manifests/", "events/", "run_state.json", "run.json")
    assert not any(
        fragment in key
        for key in adapter.object_store.payloads
        for fragment in forbidden_fragments
    )


def test_streaming_executor_does_not_mark_missing_output_complete_when_export_fails() -> None:
    adapter = FakePipelineAdapter()
    adapter.exporter.fail_on_batch_id = "batch-00002"

    summary = StreamingRayExecutor(adapter).run()

    assert summary["status"] == "failed"
    assert "injected export failure on batch-00002" in summary["error_message"]
    assert sorted(adapter.object_store.payloads) == [
        "output/items/fake-run-001/outputs/row-a.json",
    ]
    assert adapter.resolve_completed_source_keys(
        input_rows=adapter.load_input_rows(),
    ) == {"row-a"}


def test_configured_ray_dataset_builder_clamps_target_num_blocks(monkeypatch) -> None:
    calls: list[int] = []

    class FakeDataset:
        def repartition(self, num_blocks: int):
            calls.append(num_blocks)
            return self

    monkeypatch.setattr(
        "training_signal_processing.core.dataset.ray.data.from_items",
        lambda rows: FakeDataset(),
    )
    builder = ConfiguredRayDatasetBuilder(
        RayConfig(
            executor_type="ray",
            batch_size=1,
            concurrency=1,
            target_num_blocks=32,
        )
    )

    builder.build_ray_dataset([{"id": "a"}, {"id": "b"}])

    assert calls == [2]


def test_configured_ray_dataset_builder_skips_repartition_for_single_row(monkeypatch) -> None:
    calls: list[int] = []

    class FakeDataset:
        def repartition(self, num_blocks: int):
            calls.append(num_blocks)
            return self

    monkeypatch.setattr(
        "training_signal_processing.core.dataset.ray.data.from_items",
        lambda rows: FakeDataset(),
    )
    builder = ConfiguredRayDatasetBuilder(
        RayConfig(
            executor_type="ray",
            batch_size=1,
            concurrency=1,
            target_num_blocks=32,
        )
    )

    builder.build_ray_dataset([{"id": "a"}])

    assert calls == []


class ResourceOverridePipelineAdapter(FakePipelineAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.dataset_builder = RecordingDatasetBuilder()

    def get_execution_config(self) -> RayConfig:
        return RayConfig(
            executor_type="ray",
            batch_size=1,
            concurrency=2,
            target_num_blocks=1,
        )

    def resolve_transform_resources(
        self,
        *,
        op,
        execution: RayConfig,
    ) -> RayTransformResources:
        if op.name != "test_transform_generic":
            return super().resolve_transform_resources(op=op, execution=execution)
        return RayTransformResources(
            concurrency=execution.concurrency,
            num_gpus=0.5,
            num_cpus=4.0,
        )

    def get_op_configs(self) -> list[OpConfig]:
        return [
            OpConfig(name="test_prepare_generic", type="mapper"),
            OpConfig(name="test_transform_generic", type="mapper"),
            OpConfig(name="test_export_generic", type="mapper"),
        ]

    def load_input_rows(self) -> list[dict[str, object]]:
        return [{"id": "resource-row"}]


def test_streaming_executor_uses_adapter_transform_resources() -> None:
    adapter = ResourceOverridePipelineAdapter()

    summary = StreamingRayExecutor(adapter).run()

    assert summary["status"] == "success"
    assert isinstance(adapter.dataset_builder, RecordingDatasetBuilder)
    assert adapter.dataset_builder.transform_calls == [
        {
            "op_name": "test_prepare_generic",
            "batch_size": 1,
            "concurrency": None,
            "num_gpus": None,
            "num_cpus": None,
        },
        {
            "op_name": "test_transform_generic",
            "batch_size": 1,
            "concurrency": 2,
            "num_gpus": 0.5,
            "num_cpus": 4.0,
        },
        {
            "op_name": "test_export_generic",
            "batch_size": 1,
            "concurrency": None,
            "num_gpus": None,
            "num_cpus": None,
        },
    ]
