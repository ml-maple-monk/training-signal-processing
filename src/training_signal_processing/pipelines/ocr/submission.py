from __future__ import annotations

import shlex
import shutil
import tempfile
from pathlib import Path

from ...core.submission import (
    ArtifactStore,
    BootstrapSpec,
    LocalAsyncUploadSpec,
    RemoteInvocationSpec,
    SubmissionAdapter,
    SubmissionManifest,
)
from ...core.utils import compute_sha256_file, join_s3_key, parse_env_file
from .config import load_resolved_recipe_mapping
from .models import PdfTask, RecipeConfig


class OcrSubmissionAdapter(SubmissionAdapter):
    config: RecipeConfig

    def pipeline_family(self) -> str:
        return "ocr"

    def build_new_run_manifest(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        dry_run: bool,
    ) -> SubmissionManifest:
        pdf_root = Path(self.config.input.local_pdf_root).expanduser()
        discovered_pdf_paths = self.discover_pdf_paths(pdf_root)
        manifest_rows = self.build_pdf_tasks(pdf_root, discovered_pdf_paths)
        pdf_paths = [pdf_root / task.relative_path for task in manifest_rows]
        async_upload: LocalAsyncUploadSpec | None = None
        if not dry_run:
            async_upload = self.build_async_upload_spec(
                artifact_store=artifact_store,
                pdf_root=pdf_root,
                pdf_paths=pdf_paths,
                run_id=run_id,
            )
        return SubmissionManifest(
            rows=[task.to_dict() for task in manifest_rows],
            discovered_items=len(manifest_rows),
            async_upload=async_upload,
        )

    def load_resolved_recipe_mapping(self) -> dict[str, object]:
        return load_resolved_recipe_mapping(
            self.config_path,
            self.overrides,
            overlay_paths=self.overlay_paths,
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
        return RemoteInvocationSpec(command=command, env=env)

    def discover_pdf_paths(self, pdf_root: Path) -> list[Path]:
        if not pdf_root.is_dir():
            raise ValueError(f"Input PDF root not found: {pdf_root}")
        pdf_paths = sorted(
            path
            for path in pdf_root.glob(self.config.input.include_glob)
            if path.is_file()
        )
        if not pdf_paths:
            raise ValueError(f"No PDF files matched under {pdf_root}")
        return pdf_paths

    def build_pdf_tasks(self, pdf_root: Path, pdf_paths: list[Path]) -> list[PdfTask]:
        tasks: list[PdfTask] = []
        for pdf_path in pdf_paths:
            relative_path = pdf_path.relative_to(pdf_root).as_posix()
            tasks.append(
                PdfTask(
                    source_r2_key=join_s3_key(self.config.input.raw_pdf_prefix, relative_path),
                    relative_path=relative_path,
                    source_size_bytes=pdf_path.stat().st_size,
                    source_sha256=compute_sha256_file(pdf_path),
                )
            )
        tasks.sort(
            key=lambda task: (
                task.source_size_bytes,
                task.relative_path,
            )
        )
        if self.config.input.max_files is not None:
            tasks = tasks[: self.config.input.max_files]
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
                str(self.config.input.upload_transfers),
                "--checkers",
                str(self.config.input.upload_checkers),
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
        prefix = self.config.input.raw_pdf_prefix.strip("/")
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
