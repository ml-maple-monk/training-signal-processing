from __future__ import annotations

import json
import shlex
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from ...core.utils import compute_sha256_file, join_s3_key, parse_env_file, utc_timestamp
from ...runtime.submission import (
    ArtifactRef,
    ArtifactStore,
    BootstrapSpec,
    LocalAsyncUploadSpec,
    PreparedRun,
    RemoteInvocationSpec,
    SubmissionAdapter,
)
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
        manifest_rows = self.build_pdf_tasks(pdf_root, pdf_paths)
        input_manifest_key = self.build_control_key(run_id, "input_manifest.jsonl")
        config_object_key = self.build_control_key(run_id, "recipe.json")
        async_upload: LocalAsyncUploadSpec | None = None
        if not dry_run:
            async_upload = self.build_async_upload_spec(
                artifact_store=artifact_store,
                pdf_root=pdf_root,
                pdf_paths=pdf_paths,
                run_id=run_id,
            )
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
            discovered_items=len(pdf_paths),
            uploaded_items=0,
            is_resume=False,
            async_upload=async_upload,
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
            discovered_items=len(manifest_rows),
            uploaded_items=0,
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
        discovered_items: int,
        uploaded_items: int,
        is_resume: bool,
        async_upload: LocalAsyncUploadSpec | None = None,
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
                uploaded_items=uploaded_items,
            ),
            artifacts=(
                ArtifactRef(name="input_manifest", key=input_manifest_key, kind="jsonl"),
                ArtifactRef(name="config_object", key=config_object_key, kind="json"),
            ),
            discovered_items=discovered_items,
            uploaded_items=uploaded_items,
            is_resume=is_resume,
            async_upload=async_upload,
            metadata={
                "pipeline_family": "ocr",
                "input_manifest_key": input_manifest_key,
                "config_object_key": config_object_key,
            },
        )

    def build_bootstrap_spec(self) -> BootstrapSpec:
        command = " && ".join(
            [
                "command -v rsync >/dev/null",
                "command -v uv >/dev/null || python3 -m pip install --break-system-packages uv",
                f"uv python install {shlex.quote(self.config.remote.python_version)}",
                (
                    "uv sync "
                    f"--python {shlex.quote(self.config.remote.python_version)} "
                    "--group remote_ocr --group model --no-dev --frozen"
                ),
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
        uploaded_items: int,
    ) -> RemoteInvocationSpec:
        command = shlex.join(
            [
                "uv",
                "run",
                "--python",
                self.config.remote.python_version,
                "--group",
                "remote_ocr",
                "--group",
                "model",
                "python",
                "-m",
                "training_signal_processing.main",
                "ocr-remote-job",
                "--run-id",
                run_id,
                "--config-object-key",
                config_object_key,
                "--input-manifest-key",
                input_manifest_key,
                "--uploaded-items",
                str(uploaded_items),
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

    def build_async_upload_spec(
        self,
        *,
        artifact_store: ArtifactStore,
        pdf_root: Path,
        pdf_paths: list[Path],
        run_id: str,
    ) -> LocalAsyncUploadSpec:
        rclone_path = shutil.which("rclone")
        if rclone_path is None:
            raise RuntimeError(
                "Local rclone is required for OCR input uploads. Install rclone before "
                "submitting a non-dry-run OCR job."
            )
        file_list_path = self.write_upload_file_list(
            run_id=run_id,
            pdf_root=pdf_root,
            pdf_paths=pdf_paths,
        )
        remote_name = "ocrinput"
        destination = self.build_rclone_destination(
            remote_name=remote_name,
            bucket=artifact_store.bucket,
        )
        return LocalAsyncUploadSpec(
            command=(
                rclone_path,
                "copy",
                str(pdf_root),
                destination,
                "--files-from-raw",
                str(file_list_path),
                "--transfers",
                "1",
                "--checkers",
                "1",
            ),
            cwd=str(pdf_root),
            env=self.build_rclone_env(remote_name=remote_name),
            cleanup_paths=(str(file_list_path),),
        )

    def write_upload_file_list(
        self,
        *,
        run_id: str,
        pdf_root: Path,
        pdf_paths: list[Path],
    ) -> Path:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f"ocr-upload-{run_id}-",
            suffix=".txt",
            delete=False,
        )
        with handle:
            for pdf_path in pdf_paths:
                handle.write(f"{pdf_path.relative_to(pdf_root).as_posix()}\n")
        return Path(handle.name)

    def build_rclone_destination(self, *, remote_name: str, bucket: str) -> str:
        prefix = self.config.r2.raw_pdf_prefix.strip("/")
        if prefix:
            return f"{remote_name}:{bucket}/{prefix}"
        return f"{remote_name}:{bucket}"

    def build_rclone_env(self, *, remote_name: str) -> dict[str, str]:
        config_values = parse_env_file(Path(self.config.r2.config_file).expanduser())
        required_keys = (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_DEFAULT_REGION",
            "MLFLOW_S3_ENDPOINT_URL",
        )
        missing_keys = [key for key in required_keys if not config_values.get(key)]
        if missing_keys:
            raise ValueError(
                "R2 config file is missing required rclone upload values: "
                + ", ".join(missing_keys)
            )
        prefix = f"RCLONE_CONFIG_{remote_name.upper()}_"
        return {
            f"{prefix}TYPE": "s3",
            f"{prefix}PROVIDER": "Cloudflare",
            f"{prefix}ACCESS_KEY_ID": config_values["AWS_ACCESS_KEY_ID"],
            f"{prefix}SECRET_ACCESS_KEY": config_values["AWS_SECRET_ACCESS_KEY"],
            f"{prefix}REGION": config_values["AWS_DEFAULT_REGION"],
            f"{prefix}ENDPOINT": config_values["MLFLOW_S3_ENDPOINT_URL"],
            f"{prefix}NO_CHECK_BUCKET": "true",
        }

    def build_control_key(self, run_id: str, name: str) -> str:
        return join_s3_key(self.build_run_root(run_id), f"control/{name}")

    def build_run_root(self, run_id: str) -> str:
        return join_s3_key(self.config.r2.output_prefix, run_id)
