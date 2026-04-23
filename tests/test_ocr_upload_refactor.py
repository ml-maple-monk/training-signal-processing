from __future__ import annotations

from pathlib import Path

import pytest

from training_signal_processing.core.models import OpRuntimeContext
from training_signal_processing.custom_ops import user_ops
from training_signal_processing.pipelines.ocr import config as ocr_config
from training_signal_processing.pipelines.ocr.config import load_recipe_config
from training_signal_processing.pipelines.ocr.exporter import OcrMarkdownExporter
from training_signal_processing.pipelines.ocr.models import PdfTask
from training_signal_processing.pipelines.ocr.submission import OcrSubmissionAdapter
from training_signal_processing.runtime.submission import (
    ArtifactStore,
    AsyncCommandHandle,
    AsyncCommandRunner,
    BootstrapSpec,
    CommandOutput,
    LocalAsyncUploadSpec,
    PreparedRun,
    RemoteInvocationSpec,
    RemoteTransport,
    SubmissionAdapter,
    SubmissionCoordinator,
)


class FakeArtifactStore(ArtifactStore):
    def __init__(self) -> None:
        self.bucket = "test-bucket"
        self.written_json: dict[str, dict[str, object]] = {}
        self.written_jsonl: dict[str, list[dict[str, object]]] = {}
        self.uploaded_files: list[tuple[Path, str]] = []

    def exists(self, key: str) -> bool:
        return key in self.written_json or key in self.written_jsonl

    def read_json(self, key: str) -> dict[str, object]:
        return self.written_json[key]

    def read_jsonl(self, key: str) -> list[dict[str, object]]:
        return self.written_jsonl[key]

    def write_json(self, key: str, value: dict[str, object]) -> None:
        self.written_json[key] = value

    def write_jsonl(self, key: str, rows: list[dict[str, object]]) -> None:
        self.written_jsonl[key] = rows

    def upload_file(self, path: Path, key: str) -> None:
        self.uploaded_files.append((path, key))

    def build_remote_env(self) -> dict[str, str]:
        return {"R2_BUCKET": self.bucket}


class FakeObjectStore:
    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}

    def write_bytes(self, key: str, body: bytes) -> None:
        self.payloads[key] = body


class FakeAsyncHandle(AsyncCommandHandle):
    def __init__(self) -> None:
        self.wait_called = False
        self.terminate_called = False
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self) -> CommandOutput:
        self.wait_called = True
        self.returncode = 0 if self.returncode is None else self.returncode
        return CommandOutput(stdout="", stderr="")

    def terminate(self) -> None:
        self.terminate_called = True
        self.returncode = -15


class FakeAsyncRunner(AsyncCommandRunner):
    def __init__(self) -> None:
        self.started_commands: list[list[str]] = []
        self.started_envs: list[dict[str, str] | None] = []
        self.handle = FakeAsyncHandle()

    def start(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> AsyncCommandHandle:
        self.started_commands.append(command)
        self.started_envs.append(env)
        return self.handle


class FakeRemoteTransport(RemoteTransport):
    def __init__(self, *, fail_execute: bool = False) -> None:
        self.fail_execute = fail_execute
        self.events: list[str] = []

    def describe(self) -> dict[str, object]:
        return {"transport": "fake"}

    def sync(self, *, local_paths: tuple[str, ...], remote_root: str) -> None:
        self.events.append("sync")

    def bootstrap(self, *, remote_root: str, spec: BootstrapSpec) -> CommandOutput:
        self.events.append("bootstrap")
        return CommandOutput(stdout="", stderr="")

    def execute(self, *, remote_root: str, spec: RemoteInvocationSpec) -> CommandOutput:
        self.events.append("execute")
        if self.fail_execute:
            raise RuntimeError("remote execute failed")
        return CommandOutput(stdout='{"status":"success"}', stderr="")


class FakeSubmissionAdapter(SubmissionAdapter):
    def __init__(self, prepared_run: PreparedRun) -> None:
        self.prepared_run = prepared_run

    def prepare_new_run(self, artifact_store: ArtifactStore, *, dry_run: bool) -> PreparedRun:
        return self.prepared_run

    def prepare_resume_run(self, artifact_store: ArtifactStore, run_id: str) -> PreparedRun:
        return self.prepared_run

    def parse_remote_summary(self, stdout: str) -> dict[str, object]:
        return {"status": "success"}


@pytest.fixture()
def ocr_upload_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pdf_root = tmp_path / "pdfs"
    (pdf_root / "nested").mkdir(parents=True)
    (pdf_root / "alpha.pdf").write_bytes(b"%PDF-alpha")
    (pdf_root / "nested" / "beta.pdf").write_bytes(b"%PDF-beta")
    r2_config = tmp_path / "r2.env"
    r2_config.write_text(
        "\n".join(
            [
                "AWS_ACCESS_KEY_ID=test-access",
                "AWS_SECRET_ACCESS_KEY=test-secret",
                "AWS_DEFAULT_REGION=auto",
                "MLFLOW_S3_ENDPOINT_URL=https://example.r2.cloudflarestorage.com",
                "R2_BUCKET_NAME=test-bucket",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "ocr.yaml"
    config_path.write_text(
        f"""
run:
  name: test-ocr
  config_version: 1
ssh:
  host: localhost
  port: 22
  user: root
  identity_file: ~/.ssh/id_ed25519
remote:
  root_dir: /tmp/ocr
  python_version: "3.12"
ray:
  executor_type: ray
  batch_size: 1
  concurrency: 1
  target_num_blocks: 1
  ocr_worker_num_gpus: 1.0
  ocr_worker_num_cpus: 4
r2:
  config_file: {r2_config}
  bucket: test-bucket
  raw_pdf_prefix: dataset/raw/pdf
  output_prefix: dataset/processed/pdf_ocr
input:
  local_pdf_root: {pdf_root}
  include_glob: "**/*.pdf"
  max_files: 2
mlflow:
  enabled: false
  local_tracking_uri: http://127.0.0.1:5000
  remote_tunnel_port: 15000
  experiment_name: x
observability:
  flush_interval_sec: 5
  log_per_file_events: true
  heartbeat_interval_sec: 10
resumability:
  strategy: batch_manifest
  commit_every_batches: 1
  resume_mode: latest
ops:
  - name: prepare_pdf_document
    type: mapper
  - name: skip_existing
    type: filter
  - name: marker_ocr
    type: mapper
    force_ocr: true
  - name: export_markdown
    type: mapper
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(ocr_config, "CURRENT_MACHINE_PATH", tmp_path / "missing-machine")
    return config_path


def test_ocr_prepare_new_run_builds_async_upload_spec(
    monkeypatch: pytest.MonkeyPatch,
    ocr_upload_config: Path,
) -> None:
    config = load_recipe_config(ocr_upload_config)
    artifact_store = FakeArtifactStore()
    adapter = OcrSubmissionAdapter(
        config=config,
        config_path=ocr_upload_config,
        overrides=[],
    )
    monkeypatch.setattr(
        "training_signal_processing.pipelines.ocr.submission.shutil.which",
        lambda name: "/usr/bin/rclone" if name == "rclone" else None,
    )
    prepared = adapter.prepare_new_run(artifact_store, dry_run=False)

    assert prepared.uploaded_items == 0
    assert artifact_store.uploaded_files == []
    assert prepared.async_upload is not None
    assert prepared.async_upload.command[:4] == (
        "/usr/bin/rclone",
        "copy",
        config.input.local_pdf_root,
        "ocrinput:test-bucket/dataset/raw/pdf",
    )
    file_list_path = Path(prepared.async_upload.command[5])
    assert file_list_path.is_file()
    assert file_list_path.read_text(encoding="utf-8").splitlines() == [
        "alpha.pdf",
        "nested/beta.pdf",
    ]
    manifest_key = f"{config.r2.output_prefix}/{prepared.run_id}/control/input_manifest.jsonl"
    assert manifest_key in artifact_store.written_jsonl
    assert prepared.async_upload.env["RCLONE_CONFIG_OCRINPUT_PROVIDER"] == "Cloudflare"


def test_ocr_prepare_new_run_dry_run_skips_async_upload(ocr_upload_config: Path) -> None:
    config = load_recipe_config(ocr_upload_config)
    artifact_store = FakeArtifactStore()
    prepared = OcrSubmissionAdapter(
        config=config,
        config_path=ocr_upload_config,
        overrides=[],
    ).prepare_new_run(artifact_store, dry_run=True)

    assert prepared.async_upload is None
    assert artifact_store.written_json == {}
    assert artifact_store.written_jsonl == {}


def test_ocr_prepare_new_run_requires_rclone(
    monkeypatch: pytest.MonkeyPatch,
    ocr_upload_config: Path,
) -> None:
    config = load_recipe_config(ocr_upload_config)
    artifact_store = FakeArtifactStore()
    adapter = OcrSubmissionAdapter(
        config=config,
        config_path=ocr_upload_config,
        overrides=[],
    )
    monkeypatch.setattr(
        "training_signal_processing.pipelines.ocr.submission.shutil.which",
        lambda name: None,
    )

    with pytest.raises(RuntimeError, match="rclone is required"):
        adapter.prepare_new_run(artifact_store, dry_run=False)

    assert artifact_store.uploaded_files == []


def test_submission_coordinator_waits_for_async_upload_success(tmp_path: Path) -> None:
    cleanup_path = tmp_path / "upload-files.txt"
    cleanup_path.write_text("alpha.pdf\n", encoding="utf-8")
    prepared_run = PreparedRun(
        run_id="run-001",
        remote_root="/tmp/ocr",
        sync_paths=("src",),
        bootstrap=BootstrapSpec(command="echo bootstrap"),
        invocation=RemoteInvocationSpec(command="echo remote"),
        async_upload=LocalAsyncUploadSpec(
            command=("rclone", "copy"),
            cleanup_paths=(str(cleanup_path),),
        ),
    )
    async_runner = FakeAsyncRunner()
    transport = FakeRemoteTransport()
    result = SubmissionCoordinator(
        adapter=FakeSubmissionAdapter(prepared_run),
        artifact_store=FakeArtifactStore(),
        remote_transport=transport,
        async_command_runner=async_runner,
    ).submit(dry_run=False)

    assert result.mode == "executed"
    assert transport.events == ["sync", "bootstrap", "execute"]
    assert async_runner.started_commands == [["rclone", "copy"]]
    assert async_runner.handle.wait_called is True
    assert cleanup_path.exists() is False


def test_submission_coordinator_terminates_async_upload_on_remote_failure(tmp_path: Path) -> None:
    cleanup_path = tmp_path / "upload-files.txt"
    cleanup_path.write_text("alpha.pdf\n", encoding="utf-8")
    prepared_run = PreparedRun(
        run_id="run-001",
        remote_root="/tmp/ocr",
        sync_paths=("src",),
        bootstrap=BootstrapSpec(command="echo bootstrap"),
        invocation=RemoteInvocationSpec(command="echo remote"),
        async_upload=LocalAsyncUploadSpec(
            command=("rclone", "copy"),
            cleanup_paths=(str(cleanup_path),),
        ),
    )
    async_runner = FakeAsyncRunner()
    transport = FakeRemoteTransport(fail_execute=True)

    with pytest.raises(RuntimeError, match="remote execute failed"):
        SubmissionCoordinator(
            adapter=FakeSubmissionAdapter(prepared_run),
            artifact_store=FakeArtifactStore(),
            remote_transport=transport,
            async_command_runner=async_runner,
        ).submit(dry_run=False)

    assert async_runner.handle.terminate_called is True
    assert cleanup_path.exists() is False


@pytest.fixture
def ocr_runtime_context() -> OpRuntimeContext:
    class RuntimeObjectStore:
        def exists(self, key: str) -> bool:
            return True

        def read_bytes(self, key: str) -> bytes:
            return b"%PDF-fake"

    return OpRuntimeContext(
        config={"name": "ocr-test"},
        run_id="ocr-test-run",
        object_store=RuntimeObjectStore(),
        output_root_key="output/ocr-test",
        source_root_key="source/pdf",
        logger=None,
    )


def build_prepared_ocr_row(runtime_context: OpRuntimeContext) -> dict[str, object]:
    task = PdfTask(
        source_r2_key="dataset/raw/pdf/example.pdf",
        relative_path="example.pdf",
        source_size_bytes=8,
        source_sha256="abc123",
    )
    return user_ops.PreparePdfDocumentOp().bind_runtime(runtime_context).process_row(task.to_dict())


def test_marker_ocr_process_row_success(
    monkeypatch: pytest.MonkeyPatch,
    ocr_runtime_context: OpRuntimeContext,
) -> None:
    row = build_prepared_ocr_row(ocr_runtime_context)
    op = user_ops.MarkerOcrDocumentOp(force_ocr=True).bind_runtime(ocr_runtime_context)
    monkeypatch.setattr(
        op,
        "convert_pdf_file",
        lambda pdf_path, timeout_sec: ("hello markdown", {"torch_cuda_available": False}),
    )

    result = op.process_row(row)

    assert result["status"] == "success"
    assert result["markdown_text"] == "hello markdown"
    assert result["marker_exit_code"] == 0
    assert isinstance(result["diagnostics"], dict)


def test_marker_ocr_process_row_failure(
    monkeypatch: pytest.MonkeyPatch,
    ocr_runtime_context: OpRuntimeContext,
) -> None:
    row = build_prepared_ocr_row(ocr_runtime_context)
    op = user_ops.MarkerOcrDocumentOp(force_ocr=True).bind_runtime(ocr_runtime_context)

    def raise_error(pdf_path: Path, timeout_sec: int) -> tuple[str, dict[str, object]]:
        raise RuntimeError("converter exploded")

    monkeypatch.setattr(op, "convert_pdf_file", raise_error)

    result = op.process_row(row)

    assert result["status"] == "failed"
    assert result["marker_exit_code"] == 1
    assert "converter exploded" in result["error_message"]


def test_marker_ocr_process_row_timeout(
    monkeypatch: pytest.MonkeyPatch,
    ocr_runtime_context: OpRuntimeContext,
) -> None:
    row = build_prepared_ocr_row(ocr_runtime_context)
    op = user_ops.MarkerOcrDocumentOp(force_ocr=True).bind_runtime(ocr_runtime_context)

    def raise_timeout(pdf_path: Path, timeout_sec: int) -> tuple[str, dict[str, object]]:
        raise TimeoutError("Marker OCR conversion timed out after 300 seconds.")

    monkeypatch.setattr(op, "convert_pdf_file", raise_timeout)

    result = op.process_row(row)

    assert result["status"] == "failed"
    assert "timed out" in result["error_message"]


def test_convert_pdf_bytes_with_timeout_uses_spawn_context(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    class FakeReceiver:
        def __init__(self) -> None:
            self.closed = False

        def poll(self, timeout: int) -> bool:
            created["events"].append(("poll", timeout))
            return "payload" in created

        def recv(self) -> dict[str, object]:
            created["events"].append("recv")
            created["payload_received"] = True
            return created["payload"]

        def close(self) -> None:
            self.closed = True

    class FakeSender:
        def __init__(self) -> None:
            self.closed = False

        def send(self, payload: dict[str, object]) -> None:
            created["payload"] = payload

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self, *, target, args) -> None:
            created["target"] = target
            created["args"] = args
            created["started"] = False
            created["terminated"] = False
            created["join_calls"] = []
            self.alive = True

        def start(self) -> None:
            created["started"] = True
            sender = created["args"][2]
            sender.send(
                {
                    "status": "success",
                    "markdown_text": "spawned markdown" * 1024,
                    "diagnostics": {"mp_start_method": "spawn"},
                }
            )

        def join(self, timeout=None) -> None:
            created["join_calls"].append(timeout)
            if created.get("payload_received"):
                self.alive = False

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            created["terminated"] = True
            self.alive = False

    class FakeContext:
        def Pipe(self, duplex: bool = False) -> tuple[FakeReceiver, FakeSender]:
            assert duplex is False
            receiver = FakeReceiver()
            sender = FakeSender()
            created["receiver"] = receiver
            created["sender"] = sender
            created["events"] = []
            return receiver, sender

        def Process(self, *, target, args) -> FakeProcess:
            return FakeProcess(target=target, args=args)

    monkeypatch.setattr(user_ops, "get_marker_mp_context", lambda: FakeContext())

    markdown_text, diagnostics = user_ops.convert_pdf_bytes_with_timeout(
        b"%PDF-fake",
        {"force_ocr": True, "timeout_sec": 5},
    )

    assert markdown_text == "spawned markdown" * 1024
    assert diagnostics["mp_start_method"] == "spawn"
    assert created["started"] is True
    assert created["events"] == [("poll", 5), "recv"]
    assert created["join_calls"] == [5]
    assert created["target"].__name__ == "_run_marker_conversion"
    assert created["receiver"].closed is True
    assert created["sender"].closed is True


def test_wait_for_source_object_polls_until_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}
    sleep_calls: list[float] = []

    class FakeObjectStore:
        def exists(self, key: str) -> bool:
            calls["count"] += 1
            return calls["count"] >= 3

    monkeypatch.setattr(user_ops, "sleep", lambda seconds: sleep_calls.append(seconds))

    user_ops.wait_for_source_object(
        FakeObjectStore(),
        key="dataset/raw/pdf/example.pdf",
        timeout_sec=5,
    )

    assert calls["count"] == 3
    assert len(sleep_calls) == 2


def test_wait_for_source_object_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeObjectStore:
        def exists(self, key: str) -> bool:
            return False

    times = iter([0.0, 0.0, 0.5, 0.5, 1.1, 1.1])
    monkeypatch.setattr(user_ops, "perf_counter", lambda: next(times))
    monkeypatch.setattr(user_ops, "sleep", lambda seconds: None)

    with pytest.raises(TimeoutError, match="did not appear"):
        user_ops.wait_for_source_object(
            FakeObjectStore(),
            key="dataset/raw/pdf/example.pdf",
            timeout_sec=1,
        )


def test_exporter_cleans_up_staged_pdfs_after_result_materialization(tmp_path: Path) -> None:
    success_pdf = tmp_path / "success.pdf"
    success_pdf.write_bytes(b"%PDF-success")
    failed_pdf = tmp_path / "failed.pdf"
    failed_pdf.write_bytes(b"%PDF-failed")
    object_store = FakeObjectStore()
    exporter = OcrMarkdownExporter(object_store)  # type: ignore[arg-type]

    result = exporter.export_batch(
        batch_id="batch-00001",
        rows=[
            {
                "run_id": "run-001",
                "source_r2_key": "dataset/raw/pdf/alpha.pdf",
                "relative_path": "alpha.pdf",
                "markdown_r2_key": "dataset/processed/pdf_ocr/run-001/markdown/alpha.md",
                "status": "success",
                "error_message": "",
                "source_sha256": "sha-alpha",
                "source_size_bytes": 10,
                "started_at": "",
                "finished_at": "",
                "duration_sec": 1.0,
                "marker_exit_code": 0,
                "markdown_text": "# alpha",
                "staged_pdf_path": str(success_pdf),
            },
            {
                "run_id": "run-001",
                "source_r2_key": "dataset/raw/pdf/beta.pdf",
                "relative_path": "beta.pdf",
                "markdown_r2_key": "dataset/processed/pdf_ocr/run-001/markdown/beta.md",
                "status": "failed",
                "error_message": "boom",
                "source_sha256": "sha-beta",
                "source_size_bytes": 20,
                "started_at": "",
                "finished_at": "",
                "duration_sec": 1.0,
                "marker_exit_code": 1,
                "markdown_text": "",
                "staged_pdf_path": str(failed_pdf),
            },
        ],
    )

    assert result.output_keys == ["dataset/processed/pdf_ocr/run-001/markdown/alpha.md"]
    assert (
        object_store.payloads["dataset/processed/pdf_ocr/run-001/markdown/alpha.md"]
        == b"# alpha"
    )
    assert success_pdf.exists() is False
    assert failed_pdf.exists() is False
