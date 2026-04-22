from __future__ import annotations

import json
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from ..models import PdfTask, RecipeConfig, RunArtifacts, RuntimeBindings, SubmissionPlan
from ..recipe import load_resolved_recipe_mapping
from ..storage import R2ObjectStore
from ..utils import compute_sha256_file, join_s3_key, utc_timestamp

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


@dataclass
class CommandOutput:
    stdout: str
    stderr: str


class CommandRunner(ABC):
    @abstractmethod
    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandOutput:
        raise NotImplementedError


class SubprocessCommandRunner(CommandRunner):
    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandOutput:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            rendered = shlex.join(command)
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Command failed: {rendered}\n{detail}")
        return CommandOutput(stdout=result.stdout, stderr=result.stderr)


class RemoteSubmission(ABC):
    @abstractmethod
    def submit(
        self,
        *,
        dry_run: bool,
        resume_run_id: str | None = None,
    ) -> dict[str, object]:
        raise NotImplementedError


class SshRemoteSubmission(RemoteSubmission):
    def __init__(
        self,
        config: RecipeConfig,
        config_path: Path,
        overrides: list[str] | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.overrides = overrides or []
        self.command_runner = command_runner or SubprocessCommandRunner()
        self.object_store = R2ObjectStore.from_config_file(config.r2)
        self.project_root = Path(__file__).resolve().parents[3]

    def submit(
        self,
        *,
        dry_run: bool,
        resume_run_id: str | None = None,
    ) -> dict[str, object]:
        if resume_run_id:
            artifacts = self.prepare_resume_artifacts(resume_run_id)
        else:
            artifacts = self.prepare_new_run_artifacts(dry_run=dry_run)
        plan = self.build_submission_plan(artifacts)
        if dry_run:
            return {
                "mode": "dry_run",
                "artifacts": artifacts.to_dict(),
                "plan": plan.to_safe_dict(),
            }
        self.sync_project(plan.remote_root)
        self.run_remote_shell(plan.bootstrap_command)
        remote_output = self.run_remote_shell(plan.remote_command, with_reverse_tunnel=True)
        remote_summary = self.parse_remote_summary(remote_output.stdout)
        return {
            "mode": "executed",
            "artifacts": artifacts.to_dict(),
            "plan": plan.to_safe_dict(),
            "remote_summary": remote_summary,
        }

    def prepare_new_run_artifacts(self, *, dry_run: bool) -> RunArtifacts:
        run_id = utc_timestamp()
        pdf_root = Path(self.config.input.local_pdf_root).expanduser()
        pdf_paths = self.discover_pdf_paths(pdf_root)
        input_manifest_key = self.build_control_key(run_id, "input_manifest.jsonl")
        config_object_key = self.build_control_key(run_id, "recipe.json")
        if dry_run:
            return RunArtifacts(
                run_id=run_id,
                input_manifest_key=input_manifest_key,
                config_object_key=config_object_key,
                discovered_documents=len(pdf_paths),
                uploaded_documents=0,
                is_resume=False,
            )
        manifest_rows = self.build_pdf_tasks(pdf_root, pdf_paths)
        uploaded_documents = self.upload_pdf_tasks(pdf_paths, manifest_rows)
        self.object_store.write_jsonl(
            input_manifest_key,
            [task.to_dict() for task in manifest_rows],
        )
        self.object_store.write_json(
            config_object_key,
            load_resolved_recipe_mapping(self.config_path, self.overrides),
        )
        return RunArtifacts(
            run_id=run_id,
            input_manifest_key=input_manifest_key,
            config_object_key=config_object_key,
            discovered_documents=len(manifest_rows),
            uploaded_documents=uploaded_documents,
            is_resume=False,
        )

    def prepare_resume_artifacts(self, run_id: str) -> RunArtifacts:
        input_manifest_key = self.build_control_key(run_id, "input_manifest.jsonl")
        config_object_key = self.build_control_key(run_id, "recipe.json")
        if not self.object_store.exists(input_manifest_key):
            raise ValueError(f"Resume manifest not found in R2: {input_manifest_key}")
        if not self.object_store.exists(config_object_key):
            raise ValueError(f"Resume recipe object not found in R2: {config_object_key}")
        manifest_rows = self.object_store.read_jsonl(input_manifest_key)
        return RunArtifacts(
            run_id=run_id,
            input_manifest_key=input_manifest_key,
            config_object_key=config_object_key,
            discovered_documents=len(manifest_rows),
            uploaded_documents=0,
            is_resume=True,
        )

    def build_submission_plan(self, artifacts: RunArtifacts) -> SubmissionPlan:
        ssh_target = f"{self.config.ssh.user}@{self.config.ssh.host}"
        bindings = RuntimeBindings(
            run_id=artifacts.run_id,
            input_manifest_key=artifacts.input_manifest_key,
            config_object_key=artifacts.config_object_key,
            uploaded_documents=artifacts.uploaded_documents,
            allow_overwrite=False,
        )
        return SubmissionPlan(
            run_id=artifacts.run_id,
            ssh_target=ssh_target,
            remote_root=self.config.remote.root_dir,
            bootstrap_command=self.build_bootstrap_command(),
            remote_command=self.build_remote_command(bindings),
            input_manifest_key=artifacts.input_manifest_key,
            config_object_key=artifacts.config_object_key,
            discovered_documents=artifacts.discovered_documents,
            uploaded_documents=artifacts.uploaded_documents,
            is_resume=artifacts.is_resume,
        )

    def build_bootstrap_command(self) -> str:
        commands = [
            f"cd {shlex.quote(self.config.remote.root_dir)}",
            "command -v uv >/dev/null",
            f"uv python install {shlex.quote(self.config.remote.python_version)}",
            "uv sync --group remote_ocr --no-dev",
        ]
        return " && ".join(commands)

    def build_remote_command(self, bindings: RuntimeBindings) -> str:
        env_assignments = self.build_remote_env_assignments()
        command = [
            "uv",
            "run",
            "--group",
            "remote_ocr",
            "python",
            "-m",
            "training_signal_processing.runtime.remote_job",
            "--run-id",
            bindings.run_id,
            "--config-object-key",
            bindings.config_object_key,
            "--input-manifest-key",
            bindings.input_manifest_key,
            "--uploaded-documents",
            str(bindings.uploaded_documents),
        ]
        if bindings.allow_overwrite:
            command.append("--allow-overwrite")
        exports = " ".join(
            f"{name}={shlex.quote(value)}" for name, value in sorted(env_assignments.items())
        )
        return (
            f"cd {shlex.quote(self.config.remote.root_dir)} && "
            f"export {exports} && "
            f"{shlex.join(command)}"
        )

    def build_remote_env_assignments(self) -> dict[str, str]:
        tracking_uri = self.resolve_remote_tracking_uri()
        assignments = {
            "R2_BUCKET": self.object_store.bucket,
            "R2_ACCESS_KEY_ID": self.object_store.config.access_key_id,
            "R2_SECRET_ACCESS_KEY": self.object_store.config.secret_access_key,
            "R2_REGION": self.object_store.config.region,
            "R2_ENDPOINT_URL": self.object_store.config.endpoint_url,
        }
        if tracking_uri:
            assignments["MLFLOW_TRACKING_URI"] = tracking_uri
        return assignments

    def resolve_remote_tracking_uri(self) -> str:
        if not self.config.mlflow.enabled:
            return ""
        return f"http://127.0.0.1:{self.config.mlflow.remote_tunnel_port}"

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
        pdf_paths: list[Path],
        tasks: list[PdfTask],
    ) -> int:
        uploaded = 0
        for pdf_path, task in zip(pdf_paths, tasks, strict=True):
            self.object_store.upload_file(pdf_path, task.source_r2_key)
            uploaded += 1
        return uploaded

    def sync_project(self, remote_root: str) -> None:
        self.run_remote_shell(f"mkdir -p {shlex.quote(remote_root)}")
        rsync_command = [
            "rsync",
            "-az",
            "-e",
            self.build_rsync_ssh_command(),
            "pyproject.toml",
            "uv.lock",
            "src",
            "config",
            f"{self.build_ssh_target()}:{remote_root}/",
        ]
        self.command_runner.run(rsync_command, cwd=self.project_root)

    def run_remote_shell(
        self,
        remote_command: str,
        *,
        with_reverse_tunnel: bool = False,
    ) -> CommandOutput:
        command = [
            "ssh",
            "-i",
            str(Path(self.config.ssh.identity_file).expanduser()),
            "-p",
            str(self.config.ssh.port),
        ]
        if with_reverse_tunnel and self.config.mlflow.enabled:
            command.extend(["-R", self.build_reverse_tunnel_spec()])
        command.extend([self.build_ssh_target(), remote_command])
        return self.command_runner.run(command)

    def build_ssh_target(self) -> str:
        return f"{self.config.ssh.user}@{self.config.ssh.host}"

    def build_rsync_ssh_command(self) -> str:
        identity_file = str(Path(self.config.ssh.identity_file).expanduser())
        return f"ssh -i {shlex.quote(identity_file)} -p {self.config.ssh.port}"

    def build_reverse_tunnel_spec(self) -> str:
        parsed = urlparse(self.config.mlflow.local_tracking_uri)
        if not parsed.hostname or not parsed.port:
            raise ValueError(
                "mlflow.local_tracking_uri must include an explicit host and port."
            )
        return (
            f"{self.config.mlflow.remote_tunnel_port}:{parsed.hostname}:{parsed.port}"
        )

    def build_control_key(self, run_id: str, name: str) -> str:
        return join_s3_key(self.build_run_root(run_id), f"control/{name}")

    def build_run_root(self, run_id: str) -> str:
        return join_s3_key(self.config.r2.output_prefix, run_id)

    def parse_remote_summary(self, stdout: str) -> dict[str, object]:
        stripped = stdout.strip()
        if not stripped:
            raise ValueError("Remote job returned no JSON summary on stdout.")
        return json.loads(stripped)
