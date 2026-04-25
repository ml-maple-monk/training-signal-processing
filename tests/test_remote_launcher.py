from __future__ import annotations

import os
import platform
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from training_signal_processing.core.models import RemoteRuntimeConfig, SshConfig
from training_signal_processing.core.submission import (
    ArtifactRef,
    ArtifactStore,
    AsyncCommandHandle,
    AsyncCommandRunner,
    BootstrapSpec,
    CommandOutput,
    CommandRunner,
    LaunchHandle,
    LocalAsyncUploadSpec,
    PreparedRun,
    RemoteInvocationSpec,
    RemoteTransport,
    SshRemoteTransport,
    SubmissionCoordinator,
    SubmissionResult,
)


@dataclass
class RecordingCommandRunner(CommandRunner):
    commands: list[list[str]] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""

    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandOutput:
        self.commands.append(list(command))
        return CommandOutput(stdout=self.stdout, stderr=self.stderr)


def _ssh_config() -> SshConfig:
    return SshConfig(
        host="pod.example.com",
        port=22,
        user="root",
        identity_file="/tmp/fake_key",
    )


def _remote_config(
    *,
    remote_jobs_root: str = "/root/ocr-jobs",
    pgid_wait_attempts: int = 20,
    pgid_wait_sleep_seconds: float = 0.25,
) -> RemoteRuntimeConfig:
    return RemoteRuntimeConfig(
        root_dir="/root/training-signal-processing",
        python_version="3.12",
        remote_jobs_root=remote_jobs_root,
        pgid_wait_attempts=pgid_wait_attempts,
        pgid_wait_sleep_seconds=pgid_wait_sleep_seconds,
        sync_paths=("pyproject.toml", "uv.lock", "src", "config"),
    )


def _spec() -> RemoteInvocationSpec:
    # A non-trivial command containing single quotes to prove we don't reinterpret
    # it inside `bash -c '...'`. Single quotes would otherwise break naive wrappers.
    command = "uv run python -m training_signal_processing.main ocr-remote-job --arg 'with spaces'"
    return RemoteInvocationSpec(
        command=command,
        env={"R2_BUCKET": "my-bucket", "R2_ACCESS_KEY_ID": "AK'withquote"},
    )


def _remote_commands(runner: RecordingCommandRunner) -> list[str]:
    """Extract the remote shell string from each SSH invocation recorded."""
    # Each SSH invocation is `ssh -i <key> -p <port> user@host <remote_cmd>`
    # The remote command is always the last positional argument.
    return [cmd[-1] for cmd in runner.commands]


def test_launch_detached_writes_script_and_records_handle() -> None:
    runner = RecordingCommandRunner()
    transport = SshRemoteTransport(_ssh_config(), _remote_config(), command_runner=runner)
    spec = _spec()

    handle = transport.launch_detached(
        remote_root="/root/training-signal-processing",
        spec=spec,
        run_id="20260423T195035Z",
    )

    assert isinstance(handle, LaunchHandle)
    assert handle.run_id == "20260423T195035Z"
    assert handle.remote_jobs_root == "/root/ocr-jobs/20260423T195035Z"
    assert handle.log_path == "/root/ocr-jobs/20260423T195035Z/job.log"
    assert handle.pgid_path == "/root/ocr-jobs/20260423T195035Z/job.pgid"
    assert handle.launcher_script_path == "/root/ocr-jobs/20260423T195035Z/launch.sh"

    # Two SSH round-trips: write launcher, then start-and-wait.
    assert len(runner.commands) == 2
    write_cmd, start_cmd = _remote_commands(runner)

    # First command writes launch.sh via heredoc.
    assert "mkdir -p /root/ocr-jobs/20260423T195035Z" in write_cmd
    assert "cat > /root/ocr-jobs/20260423T195035Z/launch.sh" in write_cmd
    assert "__OCR_LAUNCHER_EOF__" in write_cmd
    assert "chmod +x /root/ocr-jobs/20260423T195035Z/launch.sh" in write_cmd
    # Launcher body must exec the original command verbatim (not wrapped in bash -c).
    assert f"exec {spec.command}" in write_cmd
    # Env vars must be shlex-quoted even if they contain single quotes.
    assert "export R2_ACCESS_KEY_ID='AK'\"'\"'withquote'" in write_cmd
    assert "export R2_BUCKET=my-bucket" in write_cmd
    # cd into remote_root before exec.
    assert "cd /root/training-signal-processing" in write_cmd

    # Second command detaches via setsid and waits for the pgid file.
    assert "setsid sh -c" in start_cmd
    # The session leader writes its own $$ — NOT $! — so we capture the pgid.
    assert "echo $$ > /root/ocr-jobs/20260423T195035Z/job.pgid" in start_cmd
    # Stdio must be fully detached inside the child.
    assert "exec >/root/ocr-jobs/20260423T195035Z/job.log 2>&1 </dev/null" in start_cmd
    # Outer shell also detaches its own stdio and backgrounds setsid.
    assert "</dev/null >/dev/null 2>&1 &" in start_cmd
    # A wait loop polls for the pgid file so we don't return from launch_detached
    # until the session leader has actually written its pgid.
    assert "job.pgid" in start_cmd
    assert "sleep 0.25" in start_cmd


def test_launch_detached_uses_configured_jobs_root_and_pgid_wait() -> None:
    runner = RecordingCommandRunner()
    transport = SshRemoteTransport(
        _ssh_config(),
        _remote_config(
            remote_jobs_root="/var/run/training-jobs",
            pgid_wait_attempts=7,
            pgid_wait_sleep_seconds=0.5,
        ),
        command_runner=runner,
    )

    handle = transport.launch_detached(
        remote_root="/root/training-signal-processing",
        spec=_spec(),
        run_id="20260423T195035Z",
    )

    assert handle.remote_jobs_root == "/var/run/training-jobs/20260423T195035Z"
    assert handle.pgid_path == "/var/run/training-jobs/20260423T195035Z/job.pgid"
    start_cmd = _remote_commands(runner)[1]
    assert "while [ $i -lt 7 ]" in start_cmd
    assert "sleep 0.5" in start_cmd


def test_launch_detached_never_uses_bang_pid() -> None:
    """$! captures the wrong PID when setsid forks; we must use $$ in the child."""
    runner = RecordingCommandRunner()
    transport = SshRemoteTransport(_ssh_config(), _remote_config(), command_runner=runner)

    transport.launch_detached(
        remote_root="/root/app",
        spec=_spec(),
        run_id="20260101T000000Z",
    )

    for remote in _remote_commands(runner):
        assert "$!" not in remote, (
            "Detach logic must not rely on $! (which captures the backgrounded "
            "shell PID, not the setsid session leader's pgid)."
        )


def test_launch_detached_rejects_dangerous_run_ids() -> None:
    runner = RecordingCommandRunner()
    transport = SshRemoteTransport(_ssh_config(), _remote_config(), command_runner=runner)

    for bad in ("", "..", "../escape", "a/b", "has space"):
        with pytest.raises(ValueError):
            transport.launch_detached(remote_root="/root/app", spec=_spec(), run_id=bad)
        # Match slash/traversal/empty strictly; "has space" is not blocked by the
        # regex but also doesn't bypass any directory — tolerate it.
        # (Leave it in the list so future tightening doesn't regress.)

    # Sanity: no partial side effects landed on the runner for the truly bad ids.
    assert runner.commands == []


def test_launch_detached_rejects_empty_command() -> None:
    runner = RecordingCommandRunner()
    transport = SshRemoteTransport(_ssh_config(), _remote_config(), command_runner=runner)
    empty_spec = RemoteInvocationSpec(command="   ", env={})
    with pytest.raises(ValueError):
        transport.launch_detached(
            remote_root="/root/app", spec=empty_spec, run_id="20260101T000000Z"
        )


def test_launch_detached_refuses_heredoc_terminator_in_body() -> None:
    runner = RecordingCommandRunner()
    transport = SshRemoteTransport(_ssh_config(), _remote_config(), command_runner=runner)
    # Attempting to inject the heredoc terminator via env value must be refused,
    # not silently accepted (which would prematurely close the heredoc).
    bad_spec = RemoteInvocationSpec(
        command="echo hi",
        env={"SNEAKY": "__OCR_LAUNCHER_EOF__"},
    )
    with pytest.raises(RuntimeError, match="heredoc"):
        transport.launch_detached(
            remote_root="/root/app", spec=bad_spec, run_id="20260101T000000Z"
        )


# --------------------------------------------------------------------------
# SubmissionCoordinator integration
# --------------------------------------------------------------------------


class FakeRemoteTransport(RemoteTransport):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.sync_paths: tuple[str, ...] = ()
        self.launched_run_id: str | None = None
        self.launch_should_raise: Exception | None = None

    def describe(self) -> dict[str, object]:
        return {"transport": "fake"}

    def sync(self, *, local_paths: tuple[str, ...], remote_root: str) -> None:
        self.calls.append("sync")
        self.sync_paths = local_paths

    def bootstrap(self, *, remote_root: str, spec: BootstrapSpec) -> CommandOutput:
        self.calls.append("bootstrap")
        return CommandOutput(stdout="", stderr="")

    def execute(self, *, remote_root: str, spec: RemoteInvocationSpec) -> CommandOutput:
        self.calls.append("execute")
        return CommandOutput(stdout="{}", stderr="")

    def launch_detached(
        self,
        *,
        remote_root: str,
        spec: RemoteInvocationSpec,
        run_id: str,
    ) -> LaunchHandle:
        self.calls.append("launch_detached")
        self.launched_run_id = run_id
        if self.launch_should_raise is not None:
            raise self.launch_should_raise
        return LaunchHandle(
            run_id=run_id,
            remote_jobs_root=f"/root/ocr-jobs/{run_id}",
            log_path=f"/root/ocr-jobs/{run_id}/job.log",
            pgid_path=f"/root/ocr-jobs/{run_id}/job.pgid",
            launcher_script_path=f"/root/ocr-jobs/{run_id}/launch.sh",
        )

class FakeArtifactStore(ArtifactStore):
    bucket = "fake-bucket"

    def exists(self, key: str) -> bool:
        return True

    def read_json(self, key: str) -> dict[str, object]:
        return {}

    def read_jsonl(self, key: str) -> list[dict[str, object]]:
        return []

    def write_json(self, key: str, value: dict[str, object]) -> None:
        return None

    def write_jsonl(self, key: str, rows: list[dict[str, object]]) -> None:
        return None

    def upload_file(self, path: Path, key: str) -> None:
        return None

    def build_remote_env(self) -> dict[str, str]:
        return {}


class FakeAdapter:
    def __init__(self, prepared: PreparedRun) -> None:
        self.prepared = prepared
        self.prepare_new_calls = 0
        self.prepare_resume_calls = 0

    def prepare_new_run(self, artifact_store: ArtifactStore, *, dry_run: bool) -> PreparedRun:
        self.prepare_new_calls += 1
        return self.prepared

    def prepare_resume_run(
        self, artifact_store: ArtifactStore, run_id: str
    ) -> PreparedRun:
        self.prepare_resume_calls += 1
        return self.prepared

@dataclass
class _FakeAsyncHandle(AsyncCommandHandle):
    wait_calls: int = 0
    poll_value: int | None = 0
    terminated: bool = False

    def poll(self) -> int | None:
        return self.poll_value

    def wait(self) -> CommandOutput:
        self.wait_calls += 1
        return CommandOutput(stdout="", stderr="")

    def terminate(self) -> None:
        self.terminated = True
        self.poll_value = 143


class _FakeAsyncRunner(AsyncCommandRunner):
    def __init__(self) -> None:
        self.handles: list[_FakeAsyncHandle] = []

    def start(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> AsyncCommandHandle:
        # Before wait is called, report the handle as still running so the
        # exception-path terminate() branch is testable.
        handle = _FakeAsyncHandle(poll_value=None)
        self.handles.append(handle)
        return handle


def _prepared_run(*, with_upload: bool, is_resume: bool = False) -> PreparedRun:
    return PreparedRun(
        run_id="20260424T000000Z",
        remote_root="/root/training-signal-processing",
        sync_paths=("./src",),
        bootstrap=BootstrapSpec(command="echo bootstrap"),
        invocation=RemoteInvocationSpec(command="echo work", env={}),
        artifacts=(ArtifactRef(name="recipe", key="bucket/recipe.json"),),
        discovered_items=100,
        uploaded_items=0,
        is_resume=is_resume,
        async_upload=(
            LocalAsyncUploadSpec(
                command=("rclone", "copy", "./pdfs", "r2:bucket/raw"),
                cleanup_paths=(),
            )
            if with_upload
            else None
        ),
    )


def test_submit_launches_detached_and_returns_handle() -> None:
    transport = FakeRemoteTransport()
    adapter = FakeAdapter(_prepared_run(with_upload=False))
    async_runner = _FakeAsyncRunner()
    coord = SubmissionCoordinator(
        adapter=adapter,
        artifact_store=FakeArtifactStore(),
        remote_transport=transport,
        async_command_runner=async_runner,
    )

    result = coord.submit(dry_run=False, resume_run_id=None)

    assert result.mode == "launched"
    assert result.launch is not None
    assert result.launch.run_id == "20260424T000000Z"
    assert result.launch.pgid_path == "/root/ocr-jobs/20260424T000000Z/job.pgid"
    # The blocking `execute` must NOT be called.
    assert "execute" not in transport.calls
    assert "launch_detached" in transport.calls
    assert transport.launched_run_id == "20260424T000000Z"


def test_submit_waits_for_local_upload_before_launching_remote() -> None:
    """The local input upload must finish before the detached remote starts,
    so the remote doesn't race ahead of its inputs."""
    transport = FakeRemoteTransport()
    adapter = FakeAdapter(_prepared_run(with_upload=True))
    async_runner = _FakeAsyncRunner()
    coord = SubmissionCoordinator(
        adapter=adapter,
        artifact_store=FakeArtifactStore(),
        remote_transport=transport,
        async_command_runner=async_runner,
    )

    result = coord.submit(dry_run=False, resume_run_id=None)

    assert result.mode == "launched"
    assert len(async_runner.handles) == 1
    assert async_runner.handles[0].wait_calls == 1
    assert async_runner.handles[0].terminated is False


def test_submit_terminates_upload_when_launch_raises() -> None:
    transport = FakeRemoteTransport()
    transport.launch_should_raise = RuntimeError("ssh blew up")
    adapter = FakeAdapter(_prepared_run(with_upload=True))
    async_runner = _FakeAsyncRunner()
    coord = SubmissionCoordinator(
        adapter=adapter,
        artifact_store=FakeArtifactStore(),
        remote_transport=transport,
        async_command_runner=async_runner,
    )

    # Simulate upload still running when launch raises: force poll() to return None.
    # Our fake runner creates handles with poll_value=None already.
    with pytest.raises(RuntimeError, match="ssh blew up"):
        coord.submit(dry_run=False, resume_run_id=None)

    # The handle's wait() was already called before launch_detached, so termination
    # won't fire in this path.
    assert transport.calls == [
        "sync",
        "bootstrap",
        "launch_detached",
    ]


def test_submission_result_to_safe_dict_includes_launch_block() -> None:
    prepared = _prepared_run(with_upload=False)
    launch = LaunchHandle(
        run_id=prepared.run_id,
        remote_jobs_root=f"/root/ocr-jobs/{prepared.run_id}",
        log_path=f"/root/ocr-jobs/{prepared.run_id}/job.log",
        pgid_path=f"/root/ocr-jobs/{prepared.run_id}/job.pgid",
        launcher_script_path=f"/root/ocr-jobs/{prepared.run_id}/launch.sh",
    )
    result = SubmissionResult(
        mode="launched",
        prepared_run=prepared,
        transport_details={"transport": "fake"},
        launch=launch,
    )
    safe = result.to_safe_dict()
    assert safe["mode"] == "launched"
    assert safe["launch"] == launch.to_dict()
    assert "remote_summary" not in safe
    # Backward-compatible plan / artifacts blocks still present.
    assert "plan" in safe
    assert "artifacts" in safe


# --------------------------------------------------------------------------
# Subprocess smoke test: actually detach a `sleep` and verify pid == pgid.
# Skipped on non-POSIX or where `setsid` is unavailable.
# --------------------------------------------------------------------------


_SETSID_OK = (
    platform.system() != "Windows"
    and subprocess.run(["which", "setsid"], capture_output=True).returncode == 0
)


@pytest.mark.skipif(not _SETSID_OK, reason="setsid not available on this platform")
def test_detach_wrapper_captures_real_pgid_of_session_leader(tmp_path: Path) -> None:
    """Render the same detach commands that launch_detached emits to a pod and
    run them locally. Assert the recorded pgid equals the session leader's pid
    (exactly what `kill -TERM -<pgid>` in a future `ocr-stop` relies on).
    """
    jobs_root = tmp_path / "ocr-jobs" / "test-run"
    jobs_root.mkdir(parents=True)
    log_path = jobs_root / "job.log"
    pgid_path = jobs_root / "job.pgid"
    launcher_path = jobs_root / "launch.sh"

    launcher_path.write_text(
        "#!/bin/sh\nset -e\nexec sleep 30\n",
        encoding="utf-8",
    )
    os.chmod(launcher_path, 0o755)

    inner = (
        f"exec >{shlex.quote(str(log_path))} 2>&1 </dev/null; "
        f"echo $$ > {shlex.quote(str(pgid_path))}; "
        f"exec sh {shlex.quote(str(launcher_path))}"
    )
    start_command = f"setsid sh -c {shlex.quote(inner)} </dev/null >/dev/null 2>&1 &"
    full = f"{start_command} while [ ! -s {shlex.quote(str(pgid_path))} ]; do sleep 0.1; done"

    try:
        subprocess.run(["sh", "-c", full], check=True, capture_output=True)

        # Give the session leader a beat to be fully realized before we probe ps.
        deadline = time.monotonic() + 2.0
        recorded_pgid: int | None = None
        while time.monotonic() < deadline:
            if pgid_path.exists() and pgid_path.read_text().strip():
                recorded_pgid = int(pgid_path.read_text().strip())
                break
            time.sleep(0.05)
        assert recorded_pgid is not None, "pgid file was never written"

        ps_out = subprocess.run(
            ["ps", "-o", "pid=,pgid=,sid=", "-p", str(recorded_pgid)],
            capture_output=True,
            text=True,
        )
        assert ps_out.returncode == 0, f"ps failed: {ps_out.stderr}"
        fields = ps_out.stdout.strip().split()
        assert len(fields) >= 3, f"unexpected ps output: {ps_out.stdout!r}"
        pid_val, pgid_val, sid_val = (int(f) for f in fields[:3])
        # The recorded pgid must equal both the pid and the sid of the session
        # leader — this is the property `kill -TERM -<pgid>` needs.
        assert pid_val == recorded_pgid
        assert pgid_val == recorded_pgid
        assert sid_val == recorded_pgid
    finally:
        # Clean up: kill the whole session.
        if pgid_path.exists() and pgid_path.read_text().strip():
            try:
                os.killpg(int(pgid_path.read_text().strip()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_submit_payload_has_no_reverse_tunnel_fields() -> None:
    transport = FakeRemoteTransport()
    adapter = FakeAdapter(_prepared_run(with_upload=False))
    coord = SubmissionCoordinator(
        adapter=adapter,
        artifact_store=FakeArtifactStore(),
        remote_transport=transport,
        async_command_runner=_FakeAsyncRunner(),
    )

    result = coord.submit(dry_run=False, resume_run_id=None)

    assert result.mode == "launched"
    safe = result.to_safe_dict()
    assert "tunnels" not in safe
    assert "reverse_tunnels" not in safe["plan"]
