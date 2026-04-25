from __future__ import annotations

import os
import re
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import R2Config, RemoteRuntimeConfig, SshConfig
from .storage import build_r2_env
from .utils import join_s3_key, utc_timestamp

if TYPE_CHECKING:
    from .storage import R2ObjectStore

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.

# Allowlist for run_id values used in remote filesystem paths. Conservative: only
# alphanumeric, underscore, hyphen, and period. The canonical generator
# `core.utils.utc_timestamp()` produces values like "20260423T195035Z" which
# match this pattern; anything that doesn't should be rejected loudly rather
# than quietly quoted, because the run_id is used as a path segment.
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


@dataclass(frozen=True)
class CommandOutput:
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    key: str
    kind: str = "artifact"
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapSpec:
    command: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RemoteInvocationSpec:
    command: str
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "command": self.command,
            "env": dict(self.env),
        }

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "command": "<redacted remote command>",
            "env_keys": sorted(self.env),
        }


@dataclass(frozen=True)
class LocalAsyncUploadSpec:
    command: tuple[str, ...]
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    cleanup_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "command": list(self.command),
            "cwd": self.cwd,
            "env_keys": sorted(self.env),
            "cleanup_paths": list(self.cleanup_paths),
        }


@dataclass(frozen=True)
class SubmissionManifest:
    rows: list[dict[str, object]]
    discovered_items: int
    uploaded_items: int = 0
    async_upload: LocalAsyncUploadSpec | None = None


@dataclass(frozen=True)
class LaunchHandle:
    """Identifiers for a detached remote job so it can later be found, tailed, or killed.

    The pgid is written by the `setsid` session leader itself (`echo $$`) so it
    always matches the process-group leader — killing via `kill -TERM -<pgid>`
    signals the whole group (driver + workers + uploader).
    """

    run_id: str
    remote_jobs_root: str
    log_path: str
    pgid_path: str
    launcher_script_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "remote_jobs_root": self.remote_jobs_root,
            "log_path": self.log_path,
            "pgid_path": self.pgid_path,
            "launcher_script_path": self.launcher_script_path,
        }


@dataclass(frozen=True)
class PreparedRun:
    run_id: str
    remote_root: str
    sync_paths: tuple[str, ...]
    bootstrap: BootstrapSpec
    invocation: RemoteInvocationSpec
    artifacts: tuple[ArtifactRef, ...] = ()
    discovered_items: int = 0
    uploaded_items: int = 0
    is_resume: bool = False
    async_upload: LocalAsyncUploadSpec | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "remote_root": self.remote_root,
            "sync_paths": list(self.sync_paths),
            "bootstrap": self.bootstrap.to_dict(),
            "invocation": self.invocation.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "discovered_items": self.discovered_items,
            "uploaded_items": self.uploaded_items,
            "is_resume": self.is_resume,
            "async_upload": self.async_upload.to_dict() if self.async_upload else None,
            "metadata": dict(self.metadata),
        }

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "remote_root": self.remote_root,
            "sync_paths": list(self.sync_paths),
            "bootstrap": self.bootstrap.to_dict(),
            "invocation": self.invocation.to_safe_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "discovered_items": self.discovered_items,
            "uploaded_items": self.uploaded_items,
            "is_resume": self.is_resume,
            "has_async_upload": self.async_upload is not None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SubmissionResult:
    mode: str
    prepared_run: PreparedRun
    transport_details: dict[str, object] = field(default_factory=dict)
    remote_summary: dict[str, object] = field(default_factory=dict)
    launch: LaunchHandle | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "mode": self.mode,
            "prepared_run": self.prepared_run.to_dict(),
            "transport": dict(self.transport_details),
        }
        if self.remote_summary:
            payload["remote_summary"] = dict(self.remote_summary)
        if self.launch is not None:
            payload["launch"] = self.launch.to_dict()
        return payload

    def to_safe_dict(self) -> dict[str, object]:
        artifacts_payload = {
            "run_id": self.prepared_run.run_id,
            "discovered_items": self.prepared_run.discovered_items,
            "uploaded_items": self.prepared_run.uploaded_items,
            "is_resume": self.prepared_run.is_resume,
        }
        for artifact in self.prepared_run.artifacts:
            artifacts_payload[f"{artifact.name}_key"] = artifact.key
        plan_payload = {
            "run_id": self.prepared_run.run_id,
            "remote_root": self.prepared_run.remote_root,
            "sync_paths": list(self.prepared_run.sync_paths),
            "bootstrap_command": self.prepared_run.bootstrap.command,
            "remote_command": "<redacted remote command>",
            "discovered_items": self.prepared_run.discovered_items,
            "uploaded_items": self.prepared_run.uploaded_items,
            "is_resume": self.prepared_run.is_resume,
            **self.transport_details,
        }
        base: dict[str, object] = {
            "mode": self.mode,
            "artifacts": artifacts_payload,
            "plan": plan_payload,
        }
        if self.remote_summary:
            base["remote_summary"] = dict(self.remote_summary)
        if self.launch is not None:
            base["launch"] = self.launch.to_dict()
        return base


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


class AsyncCommandHandle(ABC):
    @abstractmethod
    def poll(self) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def wait(self) -> CommandOutput:
        raise NotImplementedError

    @abstractmethod
    def terminate(self) -> None:
        raise NotImplementedError


class AsyncCommandRunner(ABC):
    @abstractmethod
    def start(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> AsyncCommandHandle:
        raise NotImplementedError


class SubprocessAsyncCommandHandle(AsyncCommandHandle):
    def __init__(self, process: subprocess.Popen[str], command: list[str]) -> None:
        self.process = process
        self.command = command

    def poll(self) -> int | None:
        return self.process.poll()

    def wait(self) -> CommandOutput:
        stdout, stderr = self.process.communicate()
        if self.process.returncode != 0:
            rendered = shlex.join(self.command)
            detail = stderr.strip() or stdout.strip()
            raise RuntimeError(f"Command failed: {rendered}\n{detail}")
        return CommandOutput(stdout=stdout, stderr=stderr)

    def terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()


class SubprocessAsyncCommandRunner(AsyncCommandRunner):
    def start(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> AsyncCommandHandle:
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd) if cwd is not None else None,
                env={**os.environ, **(env or {})},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            rendered = shlex.join(command)
            raise RuntimeError(f"Async command executable not found: {rendered}") from exc
        return SubprocessAsyncCommandHandle(process, command)


class ArtifactStore(ABC):
    bucket: str

    @abstractmethod
    def exists(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def read_json(self, key: str) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def read_jsonl(self, key: str) -> list[dict[str, object]]:
        raise NotImplementedError

    @abstractmethod
    def write_json(self, key: str, value: dict[str, object]) -> None:
        raise NotImplementedError

    @abstractmethod
    def write_jsonl(self, key: str, rows: list[dict[str, object]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def upload_file(self, path: Path, key: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def build_remote_env(self) -> dict[str, str]:
        raise NotImplementedError


class R2ArtifactStore(ArtifactStore):
    def __init__(self, object_store: "R2ObjectStore") -> None:
        self.object_store = object_store
        self.bucket = object_store.bucket

    @classmethod
    def from_config_file(cls, config: R2Config) -> "R2ArtifactStore":
        from .storage import R2ObjectStore

        return cls(R2ObjectStore.from_config_file(config))

    @classmethod
    def from_environment(cls, config: R2Config) -> "R2ArtifactStore":
        from .storage import R2ObjectStore

        return cls(R2ObjectStore.from_environment(config))

    def as_object_store(self) -> "R2ObjectStore":
        return self.object_store

    def exists(self, key: str) -> bool:
        return self.object_store.exists(key)

    def read_json(self, key: str) -> dict[str, object]:
        return self.object_store.read_json(key)

    def read_jsonl(self, key: str) -> list[dict[str, object]]:
        return self.object_store.read_jsonl(key)

    def write_json(self, key: str, value: dict[str, object]) -> None:
        self.object_store.write_json(key, value)

    def write_jsonl(self, key: str, rows: list[dict[str, object]]) -> None:
        self.object_store.write_jsonl(key, rows)

    def upload_file(self, path: Path, key: str) -> None:
        self.object_store.upload_file(path, key)

    def build_remote_env(self) -> dict[str, str]:
        return build_r2_env(self.object_store.config)


class RemoteTransport(ABC):
    @abstractmethod
    def describe(self) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def sync(self, *, local_paths: tuple[str, ...], remote_root: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def bootstrap(self, *, remote_root: str, spec: BootstrapSpec) -> CommandOutput:
        raise NotImplementedError

    @abstractmethod
    def execute(self, *, remote_root: str, spec: RemoteInvocationSpec) -> CommandOutput:
        raise NotImplementedError

    @abstractmethod
    def launch_detached(
        self,
        *,
        remote_root: str,
        spec: RemoteInvocationSpec,
        run_id: str,
    ) -> LaunchHandle:
        raise NotImplementedError


class SshRemoteTransport(RemoteTransport):
    def __init__(
        self,
        ssh_config: SshConfig,
        remote_config: RemoteRuntimeConfig,
        command_runner: CommandRunner | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.ssh_config = ssh_config
        self.remote_config = remote_config
        self.command_runner = command_runner or SubprocessCommandRunner()
        self.project_root = project_root or Path(__file__).resolve().parents[3]

    def describe(self) -> dict[str, object]:
        return {
            "transport": "ssh",
            "ssh_target": self.build_ssh_target(),
        }

    def sync(self, *, local_paths: tuple[str, ...], remote_root: str) -> None:
        self._run_remote_shell(f"mkdir -p {shlex.quote(remote_root)}")
        rsync_command = [
            "rsync",
            "-az",
            "-e",
            self.build_rsync_ssh_command(),
            *local_paths,
            f"{self.build_ssh_target()}:{remote_root}/",
        ]
        self.command_runner.run(rsync_command, cwd=self.project_root)

    def bootstrap(self, *, remote_root: str, spec: BootstrapSpec) -> CommandOutput:
        return self._run_remote_shell(
            self.render_remote_shell_command(
                remote_root=remote_root,
                command=spec.command,
                env={},
            )
        )

    def execute(self, *, remote_root: str, spec: RemoteInvocationSpec) -> CommandOutput:
        return self._run_remote_shell(
            self.render_remote_shell_command(
                remote_root=remote_root,
                command=spec.command,
                env=spec.env,
            )
        )

    HEREDOC_TERMINATOR = "__OCR_LAUNCHER_EOF__"

    def launch_detached(
        self,
        *,
        remote_root: str,
        spec: RemoteInvocationSpec,
        run_id: str,
    ) -> LaunchHandle:
        """Launch `spec.command` in its own process group on the remote host.

        Survival of SSH disconnect depends on two things being right:
            - The child must not inherit a controlling TTY (done by `setsid`).
            - We must record the new *process group* ID (not `$!`), so a later
              `kill -TERM -<pgid>` can signal the whole tree (Ray driver,
              workers, uploader). The session leader writes its own `$$`.

        To avoid quote-in-quote hazards, `spec.command` is not inlined into a
        `bash -c '...'` — it is written verbatim into a heredoc'd launcher
        script, which `setsid` then exec's.
        """
        if not _RUN_ID_PATTERN.fullmatch(run_id or ""):
            raise ValueError(
                f"run_id must match {_RUN_ID_PATTERN.pattern}, got: {run_id!r}"
            )
        if not spec.command.strip():
            raise ValueError("Remote command must be explicit and non-empty.")

        jobs_root = f"{self.remote_config.remote_jobs_root.rstrip('/')}/{run_id}"
        log_path = f"{jobs_root}/job.log"
        pgid_path = f"{jobs_root}/job.pgid"
        launcher_path = f"{jobs_root}/launch.sh"

        self._write_launcher_script(
            launcher_path=launcher_path,
            jobs_root=jobs_root,
            remote_root=remote_root,
            env=spec.env,
            command=spec.command,
        )
        self._start_launcher_detached(
            launcher_path=launcher_path,
            log_path=log_path,
            pgid_path=pgid_path,
        )

        return LaunchHandle(
            run_id=run_id,
            remote_jobs_root=jobs_root,
            log_path=log_path,
            pgid_path=pgid_path,
            launcher_script_path=launcher_path,
        )

    def _write_launcher_script(
        self,
        *,
        launcher_path: str,
        jobs_root: str,
        remote_root: str,
        env: dict[str, str],
        command: str,
    ) -> None:
        env_exports = "".join(
            f"export {name}={shlex.quote(value)}\n"
            for name, value in sorted(env.items())
        )
        script_body = (
            "#!/bin/sh\n"
            "set -e\n"
            f"cd {shlex.quote(remote_root)}\n"
            f"{env_exports}"
            f"exec {command}\n"
        )
        terminator = self.HEREDOC_TERMINATOR
        if terminator in script_body:
            raise RuntimeError(
                f"Heredoc terminator {terminator!r} appeared in launcher script body; "
                "refusing to write to avoid heredoc injection."
            )
        write_command = (
            f"mkdir -p {shlex.quote(jobs_root)} && "
            f"cat > {shlex.quote(launcher_path)} <<'{terminator}'\n"
            f"{script_body}"
            f"{terminator}\n"
            f"chmod +x {shlex.quote(launcher_path)}"
        )
        self._run_remote_shell(write_command)

    def _start_launcher_detached(
        self,
        *,
        launcher_path: str,
        log_path: str,
        pgid_path: str,
    ) -> None:
        inner = (
            f"exec >{shlex.quote(log_path)} 2>&1 </dev/null; "
            f"echo $$ > {shlex.quote(pgid_path)}; "
            f"exec sh {shlex.quote(launcher_path)}"
        )
        start_command = (
            f"setsid sh -c {shlex.quote(inner)} </dev/null >/dev/null 2>&1 &"
        )
        wait_command = (
            f"i=0; while [ $i -lt {self.remote_config.pgid_wait_attempts} ]; do "
            f"  if [ -s {shlex.quote(pgid_path)} ]; then exit 0; fi; "
            f"  sleep {self.remote_config.pgid_wait_sleep_seconds}; i=$((i+1)); "
            f"done; "
            f"echo 'pgid file not written within "
            f"{self.remote_config.pgid_wait_attempts} attempts' >&2; "
            f"exit 1"
        )
        self._run_remote_shell(f"{start_command} {wait_command}")

    def build_ssh_target(self) -> str:
        return f"{self.ssh_config.user}@{self.ssh_config.host}"

    def build_rsync_ssh_command(self) -> str:
        identity_file = str(Path(self.ssh_config.identity_file).expanduser())
        return f"ssh -i {shlex.quote(identity_file)} -p {self.ssh_config.port}"

    def render_remote_shell_command(
        self,
        *,
        remote_root: str,
        command: str,
        env: dict[str, str],
    ) -> str:
        if not command.strip():
            raise ValueError("Remote command must be explicit and non-empty.")
        prefixes = [f"cd {shlex.quote(remote_root)}"]
        if env:
            exported = " ".join(
                f"{name}={shlex.quote(value)}" for name, value in sorted(env.items())
            )
            prefixes.append(f"export {exported}")
        prefixes.append(command)
        return " && ".join(prefixes)

    def _run_remote_shell(
        self,
        remote_command: str,
    ) -> CommandOutput:
        command = [
            "ssh",
            "-i",
            str(Path(self.ssh_config.identity_file).expanduser()),
            "-p",
            str(self.ssh_config.port),
        ]
        command.extend([self.build_ssh_target(), remote_command])
        return self.command_runner.run(command)


class SubmissionAdapter(ABC):
    """Template base for remote job submission.

    Public methods prepare runs and are called by the coordinator. Concrete
    pipelines customize only the hook methods below; internal helpers own the
    stable control artifact layout and prepared-run shape.
    """

    def __init__(
        self,
        *,
        config: Any,
        config_path: Path,
        overrides: list[str] | None = None,
        overlay_paths: tuple[Path, ...] = (),
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.overrides = overrides or []
        self.overlay_paths = tuple(overlay_paths)

    def prepare_new_run(self, artifact_store: ArtifactStore, *, dry_run: bool) -> PreparedRun:
        run_id = utc_timestamp()
        input_manifest_key = self._build_control_key(run_id, "input_manifest.jsonl")
        config_object_key = self._build_control_key(run_id, "recipe.json")
        manifest = self.build_new_run_manifest(
            artifact_store=artifact_store,
            run_id=run_id,
            dry_run=dry_run,
        )
        if not dry_run:
            artifact_store.write_jsonl(input_manifest_key, manifest.rows)
            artifact_store.write_json(config_object_key, self.load_resolved_recipe_mapping())
        return self._build_prepared_run(
            artifact_store=artifact_store,
            run_id=run_id,
            input_manifest_key=input_manifest_key,
            config_object_key=config_object_key,
            discovered_items=manifest.discovered_items,
            uploaded_items=manifest.uploaded_items,
            is_resume=False,
            async_upload=manifest.async_upload,
        )

    def prepare_resume_run(self, artifact_store: ArtifactStore, run_id: str) -> PreparedRun:
        input_manifest_key = self._build_control_key(run_id, "input_manifest.jsonl")
        config_object_key = self._build_control_key(run_id, "recipe.json")
        if not artifact_store.exists(input_manifest_key):
            raise ValueError(f"Resume manifest not found in R2: {input_manifest_key}")
        if not artifact_store.exists(config_object_key):
            raise ValueError(f"Resume recipe object not found in R2: {config_object_key}")
        manifest_rows = artifact_store.read_jsonl(input_manifest_key)
        return self._build_prepared_run(
            artifact_store=artifact_store,
            run_id=run_id,
            input_manifest_key=input_manifest_key,
            config_object_key=config_object_key,
            discovered_items=len(manifest_rows),
            uploaded_items=0,
            is_resume=True,
        )

    def _build_prepared_run(
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
            sync_paths=self._sync_paths(),
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
                "pipeline_family": self.pipeline_family(),
                "input_manifest_key": input_manifest_key,
                "config_object_key": config_object_key,
            },
        )

    def _sync_paths(self) -> tuple[str, ...]:
        return tuple(self.config.remote.sync_paths)

    def _build_control_key(self, run_id: str, name: str) -> str:
        return join_s3_key(self._build_run_root(run_id), f"control/{name}")

    def _build_run_root(self, run_id: str) -> str:
        return join_s3_key(self.config.r2.output_prefix, run_id)

    @abstractmethod
    def pipeline_family(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_new_run_manifest(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        dry_run: bool,
    ) -> SubmissionManifest:
        raise NotImplementedError

    @abstractmethod
    def load_resolved_recipe_mapping(self) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def build_bootstrap_spec(self) -> BootstrapSpec:
        raise NotImplementedError

    @abstractmethod
    def build_invocation_spec(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        config_object_key: str,
        input_manifest_key: str,
        uploaded_items: int,
    ) -> RemoteInvocationSpec:
        raise NotImplementedError

class SubmissionCoordinator:
    def __init__(
        self,
        *,
        adapter: SubmissionAdapter,
        artifact_store: ArtifactStore,
        remote_transport: RemoteTransport,
        async_command_runner: AsyncCommandRunner | None = None,
    ) -> None:
        self.adapter = adapter
        self.artifact_store = artifact_store
        self.remote_transport = remote_transport
        self.async_command_runner = async_command_runner or SubprocessAsyncCommandRunner()

    def submit(
        self,
        *,
        dry_run: bool,
        resume_run_id: str | None = None,
    ) -> SubmissionResult:
        if resume_run_id:
            prepared_run = self.adapter.prepare_resume_run(self.artifact_store, resume_run_id)
        else:
            prepared_run = self.adapter.prepare_new_run(self.artifact_store, dry_run=dry_run)
        transport_details = self.remote_transport.describe()
        if dry_run:
            return SubmissionResult(
                mode="dry_run",
                prepared_run=prepared_run,
                transport_details=transport_details,
            )
        self.remote_transport.sync(
            local_paths=prepared_run.sync_paths,
            remote_root=prepared_run.remote_root,
        )
        self.remote_transport.bootstrap(
            remote_root=prepared_run.remote_root,
            spec=prepared_run.bootstrap,
        )
        upload_handle: AsyncCommandHandle | None = None
        try:
            if prepared_run.async_upload is not None:
                upload_handle = self.async_command_runner.start(
                    list(prepared_run.async_upload.command),
                    cwd=(
                        Path(prepared_run.async_upload.cwd)
                        if prepared_run.async_upload.cwd
                        else None
                    ),
                    env=prepared_run.async_upload.env,
                )
            # Wait for the local input upload before launching the remote job, so
            # the detached remote never races ahead of its inputs. This still
            # blocks the local CLI for the upload duration, but that's a local
            # process — it is not subject to the SSH-disconnect kill that the
            # detached remote launcher fixes.
            if upload_handle is not None:
                upload_handle.wait()
            launch_handle = self.remote_transport.launch_detached(
                remote_root=prepared_run.remote_root,
                spec=prepared_run.invocation,
                run_id=prepared_run.run_id,
            )
            return SubmissionResult(
                mode="launched",
                prepared_run=prepared_run,
                transport_details=transport_details,
                launch=launch_handle,
            )
        except Exception:
            if upload_handle is not None and upload_handle.poll() is None:
                upload_handle.terminate()
                try:
                    upload_handle.wait()
                except Exception:
                    pass
            raise
        finally:
            if prepared_run.async_upload is not None:
                self._cleanup_local_paths(prepared_run.async_upload.cleanup_paths)

    def _cleanup_local_paths(self, paths: tuple[str, ...]) -> None:
        for raw_path in paths:
            try:
                Path(raw_path).unlink(missing_ok=True)
            except OSError:
                continue


__all__ = [
    "ArtifactRef",
    "ArtifactStore",
    "AsyncCommandHandle",
    "AsyncCommandRunner",
    "BootstrapSpec",
    "CommandOutput",
    "CommandRunner",
    "LaunchHandle",
    "LocalAsyncUploadSpec",
    "PreparedRun",
    "R2ArtifactStore",
    "RemoteInvocationSpec",
    "RemoteTransport",
    "SshRemoteTransport",
    "SubmissionAdapter",
    "SubmissionCoordinator",
    "SubmissionManifest",
    "SubmissionResult",
    "SubprocessAsyncCommandHandle",
    "SubprocessAsyncCommandRunner",
    "SubprocessCommandRunner",
]
