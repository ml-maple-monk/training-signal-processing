from __future__ import annotations

import json
import shlex
from pathlib import Path
from urllib.parse import urlparse

from ...runtime.submission import (
    ArtifactRef,
    ArtifactStore,
    BootstrapSpec,
    PreparedRun,
    RemoteInvocationSpec,
    SubmissionAdapter,
)
from ...utils import compute_sha256_file, join_s3_key, utc_timestamp
from .config import load_resolved_recipe_mapping
from .models import PdfTask, RecipeConfig


class OcrSubmissionAdapter(SubmissionAdapter):
    def __init__(
        self,
        *,
        config: RecipeConfig,
        config_path: Path,
        overrides: list[str] | None = None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.overrides = overrides or []

    def prepare_new_run(self, artifact_store: ArtifactStore, *, dry_run: bool) -> PreparedRun:
        run_id = utc_timestamp()
        pdf_root = Path(self.config.input.local_pdf_root).expanduser()
        pdf_paths = self.discover_pdf_paths(pdf_root)
        input_manifest_key = self.build_control_key(run_id, "input_manifest.jsonl")
        config_object_key = self.build_control_key(run_id, "recipe.json")
        uploaded_documents = 0
        if not dry_run:
            manifest_rows = self.build_pdf_tasks(pdf_root, pdf_paths)
            uploaded_documents = self.upload_pdf_tasks(artifact_store, pdf_paths, manifest_rows)
            artifact_store.write_jsonl(
                input_manifest_key,
                [task.to_dict() for task in manifest_rows],
            )
            artifact_store.write_json(
                config_object_key,
                load_resolved_recipe_mapping(self.config_path, self.overrides),
            )
        return self.build_prepared_run(
            artifact_store=artifact_store,
            run_id=run_id,
            input_manifest_key=input_manifest_key,
            config_object_key=config_object_key,
            discovered_documents=len(pdf_paths),
            uploaded_documents=uploaded_documents,
            is_resume=False,
        )

    def prepare_resume_run(self, artifact_store: ArtifactStore, run_id: str) -> PreparedRun:
        input_manifest_key = self.build_control_key(run_id, "input_manifest.jsonl")
        config_object_key = self.build_control_key(run_id, "recipe.json")
        if not artifact_store.exists(input_manifest_key):
            raise ValueError(f"Resume manifest not found in R2: {input_manifest_key}")
        if not artifact_store.exists(config_object_key):
            raise ValueError(f"Resume recipe object not found in R2: {config_object_key}")
        manifest_rows = artifact_store.read_jsonl(input_manifest_key)
        return self.build_prepared_run(
            artifact_store=artifact_store,
            run_id=run_id,
            input_manifest_key=input_manifest_key,
            config_object_key=config_object_key,
            discovered_documents=len(manifest_rows),
            uploaded_documents=0,
            is_resume=True,
        )

    def parse_remote_summary(self, stdout: str) -> dict[str, object]:
        stripped = stdout.strip()
        if not stripped:
            raise ValueError("Remote job returned no JSON summary on stdout.")
        return json.loads(stripped)

    def build_prepared_run(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        input_manifest_key: str,
        config_object_key: str,
        discovered_documents: int,
        uploaded_documents: int,
        is_resume: bool,
    ) -> PreparedRun:
        return PreparedRun(
            run_id=run_id,
            remote_root=self.config.remote.root_dir,
            sync_paths=("pyproject.toml", "uv.lock", "src", "config"),
            bootstrap=self.build_bootstrap_spec(),
            invocation=self.build_invocation_spec(
                artifact_store=artifact_store,
                run_id=run_id,
                config_object_key=config_object_key,
                input_manifest_key=input_manifest_key,
                uploaded_documents=uploaded_documents,
            ),
            artifacts=(
                ArtifactRef(name="input_manifest", key=input_manifest_key, kind="jsonl"),
                ArtifactRef(name="config_object", key=config_object_key, kind="json"),
            ),
            discovered_documents=discovered_documents,
            uploaded_documents=uploaded_documents,
            is_resume=is_resume,
            metadata={
                "pipeline_family": "ocr",
                "input_manifest_key": input_manifest_key,
                "config_object_key": config_object_key,
            },
        )

    def build_bootstrap_spec(self) -> BootstrapSpec:
        command = " && ".join(
            [
                "command -v uv >/dev/null",
                f"uv python install {shlex.quote(self.config.remote.python_version)}",
                "uv sync --group remote_ocr --no-dev",
            ]
        )
        return BootstrapSpec(command=command)

    def build_invocation_spec(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        config_object_key: str,
        input_manifest_key: str,
        uploaded_documents: int,
    ) -> RemoteInvocationSpec:
        command = shlex.join(
            [
                "uv",
                "run",
                "--group",
                "remote_ocr",
                "python",
                "-m",
                "training_signal_processing.pipelines.ocr.remote_job",
                "--run-id",
                run_id,
                "--config-object-key",
                config_object_key,
                "--input-manifest-key",
                input_manifest_key,
                "--uploaded-documents",
                str(uploaded_documents),
            ]
        )
        env = artifact_store.build_remote_env()
        reverse_tunnels: tuple[str, ...] = ()
        tracking_uri = self.resolve_remote_tracking_uri()
        if tracking_uri:
            env["MLFLOW_TRACKING_URI"] = tracking_uri
            reverse_tunnels = (self.build_reverse_tunnel_spec(),)
        return RemoteInvocationSpec(
            command=command,
            env=env,
            reverse_tunnels=reverse_tunnels,
        )

    def resolve_remote_tracking_uri(self) -> str:
        if not self.config.mlflow.enabled:
            return ""
        return f"http://127.0.0.1:{self.config.mlflow.remote_tunnel_port}"

    def build_reverse_tunnel_spec(self) -> str:
        parsed = urlparse(self.config.mlflow.local_tracking_uri)
        if not parsed.hostname or not parsed.port:
            raise ValueError("mlflow.local_tracking_uri must include an explicit host and port.")
        return f"{self.config.mlflow.remote_tunnel_port}:{parsed.hostname}:{parsed.port}"

    def discover_pdf_paths(self, pdf_root: Path) -> list[Path]:
        if not pdf_root.is_dir():
            raise ValueError(f"Input PDF root not found: {pdf_root}")
        pdf_paths = sorted(
            path
            for path in pdf_root.glob(self.config.input.include_glob)
            if path.is_file()
        )
        if self.config.input.max_files is not None:
            pdf_paths = pdf_paths[: self.config.input.max_files]
        if not pdf_paths:
            raise ValueError(f"No PDF files matched under {pdf_root}")
        return pdf_paths

    def build_pdf_tasks(self, pdf_root: Path, pdf_paths: list[Path]) -> list[PdfTask]:
        tasks: list[PdfTask] = []
        for pdf_path in pdf_paths:
            relative_path = pdf_path.relative_to(pdf_root).as_posix()
            tasks.append(
                PdfTask(
                    source_r2_key=join_s3_key(self.config.r2.raw_pdf_prefix, relative_path),
                    relative_path=relative_path,
                    source_size_bytes=pdf_path.stat().st_size,
                    source_sha256=compute_sha256_file(pdf_path),
                )
            )
        return tasks

    def upload_pdf_tasks(
        self,
        artifact_store: ArtifactStore,
        pdf_paths: list[Path],
        tasks: list[PdfTask],
    ) -> int:
        uploaded = 0
        for pdf_path, task in zip(pdf_paths, tasks, strict=True):
            artifact_store.upload_file(pdf_path, task.source_r2_key)
            uploaded += 1
        return uploaded

    def build_control_key(self, run_id: str, name: str) -> str:
        return join_s3_key(self.build_run_root(run_id), f"control/{name}")

    def build_run_root(self, run_id: str) -> str:
        return join_s3_key(self.config.r2.output_prefix, run_id)
