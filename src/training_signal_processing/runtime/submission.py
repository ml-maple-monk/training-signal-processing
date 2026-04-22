from __future__ import annotations

import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..models import R2Config, SshConfig

if TYPE_CHECKING:
    from ..storage import R2ObjectStore

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


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
    reverse_tunnels: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "command": self.command,
            "env": dict(self.env),
            "reverse_tunnels": list(self.reverse_tunnels),
        }

    def to_safe_dict(self) -> dict[str, object]:
        return {
            "command": "<redacted remote command>",
            "env_keys": sorted(self.env),
            "reverse_tunnels": list(self.reverse_tunnels),
        }


@dataclass(frozen=True)
class PreparedRun:
    run_id: str
    remote_root: str
    sync_paths: tuple[str, ...]
    bootstrap: BootstrapSpec
    invocation: RemoteInvocationSpec
    artifacts: tuple[ArtifactRef, ...] = ()
    discovered_documents: int = 0
    uploaded_documents: int = 0
    is_resume: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "remote_root": self.remote_root,
            "sync_paths": list(self.sync_paths),
            "bootstrap": self.bootstrap.to_dict(),
            "invocation": self.invocation.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "discovered_documents": self.discovered_documents,
            "uploaded_documents": self.uploaded_documents,
            "is_resume": self.is_resume,
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
            "discovered_documents": self.discovered_documents,
            "uploaded_documents": self.uploaded_documents,
            "is_resume": self.is_resume,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SubmissionResult:
    mode: str
    prepared_run: PreparedRun
    transport_details: dict[str, object] = field(default_factory=dict)
    remote_summary: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "mode": self.mode,
            "prepared_run": self.prepared_run.to_dict(),
            "transport": dict(self.transport_details),
        }
        if self.remote_summary:
            payload["remote_summary"] = dict(self.remote_summary)
        return payload

    def to_safe_dict(self) -> dict[str, object]:
        artifacts_payload = {
            "run_id": self.prepared_run.run_id,
            "discovered_documents": self.prepared_run.discovered_documents,
            "uploaded_documents": self.prepared_run.uploaded_documents,
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
            "discovered_documents": self.prepared_run.discovered_documents,
            "uploaded_documents": self.prepared_run.uploaded_documents,
            "is_resume": self.prepared_run.is_resume,
            **self.transport_details,
        }
        if self.prepared_run.invocation.reverse_tunnels:
            plan_payload["reverse_tunnels"] = list(self.prepared_run.invocation.reverse_tunnels)
        if self.remote_summary:
            return {
                "mode": self.mode,
                "artifacts": artifacts_payload,
                "plan": plan_payload,
                "remote_summary": dict(self.remote_summary),
            }
        return {
            "mode": self.mode,
            "artifacts": artifacts_payload,
            "plan": plan_payload,
        }


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
        from ..storage import R2ObjectStore

        return cls(R2ObjectStore.from_config_file(config))

    @classmethod
    def from_environment(cls, config: R2Config) -> "R2ArtifactStore":
        from ..storage import R2ObjectStore

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
        return {
            "R2_BUCKET": self.object_store.bucket,
            "R2_ACCESS_KEY_ID": self.object_store.config.access_key_id,
            "R2_SECRET_ACCESS_KEY": self.object_store.config.secret_access_key,
            "R2_REGION": self.object_store.config.region,
            "R2_ENDPOINT_URL": self.object_store.config.endpoint_url,
        }


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


class SshRemoteTransport(RemoteTransport):
    def __init__(
        self,
        ssh_config: SshConfig,
        command_runner: CommandRunner | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.ssh_config = ssh_config
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
            ),
            reverse_tunnels=spec.reverse_tunnels,
        )

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
        *,
        reverse_tunnels: tuple[str, ...] = (),
    ) -> CommandOutput:
        command = [
            "ssh",
            "-i",
            str(Path(self.ssh_config.identity_file).expanduser()),
            "-p",
            str(self.ssh_config.port),
        ]
        for tunnel in reverse_tunnels:
            command.extend(["-R", tunnel])
        command.extend([self.build_ssh_target(), remote_command])
        return self.command_runner.run(command)


class SubmissionAdapter(ABC):
    @abstractmethod
    def prepare_new_run(self, artifact_store: ArtifactStore, *, dry_run: bool) -> PreparedRun:
        raise NotImplementedError

    @abstractmethod
    def prepare_resume_run(self, artifact_store: ArtifactStore, run_id: str) -> PreparedRun:
        raise NotImplementedError

    @abstractmethod
    def parse_remote_summary(self, stdout: str) -> dict[str, object]:
        raise NotImplementedError


class SubmissionCoordinator:
    def __init__(
        self,
        *,
        adapter: SubmissionAdapter,
        artifact_store: ArtifactStore,
        remote_transport: RemoteTransport,
    ) -> None:
        self.adapter = adapter
        self.artifact_store = artifact_store
        self.remote_transport = remote_transport

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
        remote_output = self.remote_transport.execute(
            remote_root=prepared_run.remote_root,
            spec=prepared_run.invocation,
        )
        return SubmissionResult(
            mode="executed",
            prepared_run=prepared_run,
            transport_details=transport_details,
            remote_summary=self.adapter.parse_remote_summary(remote_output.stdout),
        )
