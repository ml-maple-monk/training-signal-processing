from __future__ import annotations

import json
from pathlib import Path

from training_signal_processing.core.models import (
    MlflowConfig,
    ObservabilityConfig,
    OpConfig,
    OpRuntimeContext,
    R2Config,
    RayTransformResources,
    RemoteRuntimeConfig,
    RunArtifactLayout,
    RunState,
    RuntimeRunBindings,
    SshConfig,
)
from training_signal_processing.core.storage import ObjectStore
from training_signal_processing.core.utils import join_s3_key
from training_signal_processing.pipelines.ocr import ops as ocr_ops
from training_signal_processing.pipelines.ocr.models import (
    InputConfig,
    OcrRayConfig,
    RecipeConfig,
    ResumeConfig,
    build_markdown_r2_key,
)
from training_signal_processing.pipelines.ocr.resume import OcrCheckpointStore
from training_signal_processing.pipelines.ocr.runtime import OcrPipelineRuntimeAdapter


class FakeObjectStore(ObjectStore):
    def __init__(self) -> None:
        self.bucket = "test-bucket"
        self.payloads: dict[str, bytes] = {}

    def exists(self, key: str) -> bool:
        return key in self.payloads

    def list_keys(self, prefix: str) -> list[str]:
        return sorted(key for key in self.payloads if key.startswith(prefix))

    def read_bytes(self, key: str) -> bytes:
        return self.payloads[key]

    def write_bytes(self, key: str, body: bytes) -> None:
        self.payloads[key] = body

    def upload_file(self, path: Path, key: str) -> None:
        self.write_bytes(key, path.read_bytes())

    def make_url(self, key: str) -> str:
        return f"memory://{key}"

    def build_pyarrow_filesystem(self):  # type: ignore[override]
        return None


def build_recipe_config() -> RecipeConfig:
    return RecipeConfig(
        run_name="test-ocr",
        config_version=1,
        ssh=SshConfig(
            host="localhost",
            port=22,
            user="root",
            identity_file="~/.ssh/id_ed25519",
        ),
        remote=RemoteRuntimeConfig(
            root_dir="/tmp/ocr",
            python_version="3.12",
        ),
        ray=OcrRayConfig(
            executor_type="ray",
            batch_size=1,
            concurrency=1,
            target_num_blocks=1,
            marker_ocr_resources=RayTransformResources(
                num_gpus=1.0,
                num_cpus=4.0,
            ),
        ),
        r2=R2Config(
            config_file="r2",
            bucket="test-bucket",
            output_prefix="dataset/processed/pdf_ocr",
        ),
        input=InputConfig(
            local_pdf_root="/tmp/pdfs",
            include_glob="**/*.pdf",
            raw_pdf_prefix="dataset/raw/pdf",
        ),
        mlflow=MlflowConfig(
            enabled=False,
            experiment_name="test",
            tracking_uri="",
        ),
        observability=ObservabilityConfig(
            flush_interval_sec=5,
            log_per_file_events=True,
            heartbeat_interval_sec=10,
        ),
        resumability=ResumeConfig(
            strategy="batch_manifest",
            commit_every_batches=1,
            resume_mode="latest",
        ),
        ops=[OpConfig(name="prepare_pdf_document", type="mapper")],
    )


def build_artifact_layout(run_id: str) -> RunArtifactLayout:
    return RunArtifactLayout(
        source_root_key="dataset/raw/pdf",
        output_root_key=f"dataset/processed/pdf_ocr/{run_id}",
    )


def test_prepare_pdf_document_and_resume_recovery_share_markdown_key() -> None:
    config = build_recipe_config()
    bindings = RuntimeRunBindings(
        run_id="resume-run",
        input_manifest_key="memory://input.jsonl",
    )
    object_store = FakeObjectStore()
    adapter = OcrPipelineRuntimeAdapter(
        config=config,
        bindings=bindings,
        object_store=object_store,  # type: ignore[arg-type]
    )
    artifact_layout = adapter.get_artifact_layout()
    runtime = OpRuntimeContext(
        config=config,
        run_id=bindings.run_id,
        object_store=object_store,
        output_root_key=artifact_layout.output_root_key,
        source_root_key=artifact_layout.source_root_key,
    )
    row = {
        "source_r2_key": "dataset/raw/pdf/books/example.pdf",
        "relative_path": "books/example.pdf",
        "source_size_bytes": 123,
        "source_page_count": 5,
        "source_sha256": "abc123",
    }

    prepared = ocr_ops.PreparePdfDocumentOp().bind_runtime(runtime).process_row(row)

    assert prepared is not None
    assert prepared["markdown_r2_key"] == build_markdown_r2_key(
        artifact_layout.output_root_key,
        "books/example.pdf",
    )

    object_store.write_bytes(str(prepared["markdown_r2_key"]), b"markdown")

    recovered = adapter.resolve_completed_item_keys(
        input_rows=[row],
        completed_item_keys=set(),
    )

    assert recovered == {"dataset/raw/pdf/books/example.pdf"}


def test_ocr_resume_recovery_respects_allow_overwrite() -> None:
    config = build_recipe_config()
    bindings = RuntimeRunBindings(
        run_id="resume-run",
        input_manifest_key="memory://input.jsonl",
        allow_overwrite=True,
    )
    object_store = FakeObjectStore()
    adapter = OcrPipelineRuntimeAdapter(
        config=config,
        bindings=bindings,
        object_store=object_store,  # type: ignore[arg-type]
    )
    output_root_key = adapter.get_artifact_layout().output_root_key
    row = {
        "source_r2_key": "dataset/raw/pdf/books/example.pdf",
        "relative_path": "books/example.pdf",
        "source_size_bytes": 123,
        "source_page_count": 5,
        "source_sha256": "abc123",
    }

    object_store.write_bytes(
        build_markdown_r2_key(output_root_key, "books/example.pdf"),
        b"markdown",
    )

    recovered = adapter.resolve_completed_item_keys(
        input_rows=[row],
        completed_item_keys={"dataset/raw/pdf/from-manifest.pdf"},
    )

    assert recovered == set()


def test_ocr_resume_ledger_repairs_existing_state_from_durable_outputs() -> None:
    config = build_recipe_config()
    object_store = FakeObjectStore()
    ledger = OcrCheckpointStore(config=config, object_store=object_store)
    run_id = "resume-run"
    artifact_layout = build_artifact_layout(run_id)
    existing_state = RunState(
        run_id=run_id,
        status="failed",
        total_items=10,
        pending_items=0,
        success_count=10,
        failed_count=2,
        skipped_count=1,
        last_committed_batch=7,
        started_at="2026-04-22T00:00:00Z",
        updated_at="2026-04-22T00:00:00Z",
        source_root_key=artifact_layout.source_root_key,
        output_root_key=artifact_layout.output_root_key,
        tracking_run_id="old-tracking",
        error_message="stale failure",
        current_phase="resume_state_loaded",
        last_phase_at="2026-04-22T00:00:00Z",
    )
    object_store.write_bytes(
        ledger.build_run_state_key(run_id),
        json.dumps(existing_state.to_dict()).encode("utf-8"),
    )
    object_store.write_bytes(
        join_s3_key(artifact_layout.output_root_key, "manifests/batch-00012.jsonl"),
        b"",
    )

    repaired = ledger.initialize_run_state(
        run_id=run_id,
        total_items=10,
        pending_items=10,
        precompleted_count=8,
        artifact_layout=artifact_layout,
        tracking_run_id="new-tracking",
    )

    assert repaired.status == "running"
    assert repaired.last_committed_batch == 12
    assert repaired.pending_items == 2
    assert repaired.success_count == 8
    assert repaired.skipped_count == 0
    assert repaired.failed_count == 2
    assert repaired.tracking_run_id == "old-tracking"


def test_ocr_resume_ledger_repairs_skipped_count_for_recovered_outputs() -> None:
    config = build_recipe_config()
    object_store = FakeObjectStore()
    ledger = OcrCheckpointStore(config=config, object_store=object_store)
    run_id = "resume-run"
    artifact_layout = build_artifact_layout(run_id)
    existing_state = RunState(
        run_id=run_id,
        status="partial",
        total_items=10,
        pending_items=5,
        success_count=5,
        failed_count=1,
        skipped_count=0,
        last_committed_batch=4,
        started_at="2026-04-22T00:00:00Z",
        updated_at="2026-04-22T00:00:00Z",
        source_root_key=artifact_layout.source_root_key,
        output_root_key=artifact_layout.output_root_key,
        tracking_run_id="old-tracking",
        current_phase="resume_state_loaded",
        last_phase_at="2026-04-22T00:00:00Z",
    )
    object_store.write_bytes(
        ledger.build_run_state_key(run_id),
        json.dumps(existing_state.to_dict()).encode("utf-8"),
    )
    object_store.write_bytes(
        join_s3_key(artifact_layout.output_root_key, "manifests/batch-00009.jsonl"),
        b"",
    )

    repaired = ledger.initialize_run_state(
        run_id=run_id,
        total_items=10,
        pending_items=10,
        precompleted_count=8,
        artifact_layout=artifact_layout,
        tracking_run_id="new-tracking",
    )

    assert repaired.last_committed_batch == 9
    assert repaired.pending_items == 2
    assert repaired.success_count == 5
    assert repaired.skipped_count == 3
    assert repaired.failed_count == 1
