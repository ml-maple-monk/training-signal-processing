from __future__ import annotations

from pathlib import Path

from training_signal_processing.core.models import (
    MlflowConfig,
    ObservabilityConfig,
    OpConfig,
    OpRuntimeContext,
    R2Config,
    RayTransformResources,
    RemoteRuntimeConfig,
    RuntimeRunBindings,
    SshConfig,
)
from training_signal_processing.core.storage import ObjectStore
from training_signal_processing.pipelines.ocr import ops as ocr_ops
from training_signal_processing.pipelines.ocr.models import (
    InputConfig,
    OcrRayConfig,
    RecipeConfig,
    ResumeConfig,
)
from training_signal_processing.pipelines.ocr.runtime import build_adapter


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
            remote_jobs_root="/root/ocr-jobs",
            pgid_wait_attempts=20,
            pgid_wait_sleep_seconds=0.25,
            sync_paths=("pyproject.toml", "uv.lock", "src", "config"),
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
            upload_transfers=1,
            upload_checkers=1,
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


def test_prepare_pdf_document_and_resume_recovery_share_markdown_key() -> None:
    config = build_recipe_config()
    bindings = RuntimeRunBindings(
        run_id="resume-run",
        input_manifest_key="memory://input.jsonl",
    )
    object_store = FakeObjectStore()
    adapter = build_adapter(config, bindings, object_store)  # type: ignore[arg-type]
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
        "source_sha256": "abc123",
    }

    prepared = ocr_ops.PreparePdfDocumentOp().bind_runtime(runtime).process_row(row)

    assert prepared is not None
    assert prepared["markdown_r2_key"] == ocr_ops.build_markdown_r2_key(
        artifact_layout.output_root_key,
        "books/example.pdf",
    )

    object_store.write_bytes(str(prepared["markdown_r2_key"]), b"markdown")

    recovered = adapter.resolve_completed_source_keys(
        input_rows=[row],
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
    adapter = build_adapter(config, bindings, object_store)  # type: ignore[arg-type]
    output_root_key = adapter.get_artifact_layout().output_root_key
    row = {
        "source_r2_key": "dataset/raw/pdf/books/example.pdf",
        "relative_path": "books/example.pdf",
        "source_size_bytes": 123,
        "source_sha256": "abc123",
    }

    object_store.write_bytes(
        ocr_ops.build_markdown_r2_key(output_root_key, "books/example.pdf"),
        b"markdown",
    )

    recovered = adapter.resolve_completed_source_keys(
        input_rows=[row],
    )

    assert recovered == set()


def test_ocr_completion_tracker_requires_expected_markdown_output() -> None:
    config = build_recipe_config()
    bindings = RuntimeRunBindings(
        run_id="resume-run",
        input_manifest_key="memory://input.jsonl",
    )
    object_store = FakeObjectStore()
    adapter = build_adapter(config, bindings, object_store)  # type: ignore[arg-type]
    row = {
        "source_r2_key": "dataset/raw/pdf/books/example.pdf",
        "relative_path": "books/example.pdf",
        "source_size_bytes": 123,
        "source_sha256": "abc123",
    }
    object_store.write_bytes(
        ocr_ops.build_markdown_r2_key(
            adapter.get_artifact_layout().output_root_key,
            "other.pdf",
        ),
        b"markdown",
    )

    assert adapter.resolve_completed_source_keys(input_rows=[row]) == set()
