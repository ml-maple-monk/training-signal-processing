from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from training_signal_processing.core.models import SshConfig
from training_signal_processing.runtime.reverse_tunnel import (
    TunnelHandle,
    _control_socket_path,
    close_reverse_tunnel,
    ensure_reverse_tunnels,
)


@dataclass
class RecordingRun:
    """Captures subprocess.run-compatible calls for assertion."""

    calls: list[list[str]] = field(default_factory=list)
    responses: list[subprocess.CompletedProcess] = field(default_factory=list)

    def __call__(self, args, **_kwargs) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        if self.responses:
            return self.responses.pop(0)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


def _ssh_config() -> SshConfig:
    return SshConfig(
        host="pod.example.com",
        port=45516,
        user="root",
        identity_file="/tmp/fake_key",
    )


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))


def test_empty_spec_returns_empty_tuple_without_subprocess_calls() -> None:
    run = RecordingRun()
    assert ensure_reverse_tunnels(_ssh_config(), (), run=run) == ()
    assert run.calls == []


def test_fresh_spawn_issues_ssh_fn_with_controlmaster() -> None:
    # Responses: one entry for the spawn call (the check is skipped because the
    # socket file does not exist yet — _is_alive short-circuits).
    run = RecordingRun(responses=[
        subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    ])
    handles = ensure_reverse_tunnels(
        _ssh_config(), ("15000:127.0.0.1:5000",), run=run
    )
    assert len(handles) == 1
    handle = handles[0]
    assert handle.spec == "15000:127.0.0.1:5000"
    assert handle.ssh_target == "root@pod.example.com"
    assert handle.ssh_port == 45516
    assert handle.started_fresh is True
    assert handle.control_socket.endswith(".sock")

    # Exactly one subprocess call (the spawn).
    assert len(run.calls) == 1
    spawn = run.calls[0]
    assert spawn[0] == "ssh"
    assert "-fN" in spawn
    assert "-R" in spawn
    assert "15000:127.0.0.1:5000" in spawn
    # ControlMaster + ControlPath wire the durable socket.
    assert "ControlMaster=yes" in spawn
    assert any(arg.startswith("ControlPath=") for arg in spawn)
    assert "BatchMode=yes" in spawn
    # Identity + port propagated.
    assert "/tmp/fake_key" in spawn
    assert "45516" in spawn


def test_reuse_when_socket_check_succeeds() -> None:
    sock = _control_socket_path(_ssh_config(), "15000:127.0.0.1:5000")
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.touch()
    run = RecordingRun(responses=[
        # ssh -S <sock> -O check → alive
        subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    ])
    handles = ensure_reverse_tunnels(
        _ssh_config(), ("15000:127.0.0.1:5000",), run=run
    )
    assert handles[0].started_fresh is False
    # Exactly one call — the -O check — no spawn.
    assert len(run.calls) == 1
    assert "-O" in run.calls[0] and "check" in run.calls[0]


def test_respawn_when_socket_is_stale() -> None:
    sock = _control_socket_path(_ssh_config(), "15000:127.0.0.1:5000")
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.touch()
    run = RecordingRun(responses=[
        # check → dead (1)
        subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="stale"),
        # spawn → 0
        subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    ])
    handles = ensure_reverse_tunnels(
        _ssh_config(), ("15000:127.0.0.1:5000",), run=run
    )
    assert handles[0].started_fresh is True
    assert len(run.calls) == 2
    assert "-O" in run.calls[0] and "check" in run.calls[0]
    assert "-fN" in run.calls[1]


def test_spawn_failure_raises_runtime_error() -> None:
    run = RecordingRun(responses=[
        subprocess.CompletedProcess(
            args=[], returncode=255, stdout="", stderr="Connection refused"
        ),
    ])
    with pytest.raises(RuntimeError, match="Failed to start reverse tunnel"):
        ensure_reverse_tunnels(
            _ssh_config(), ("15000:127.0.0.1:5000",), run=run
        )


def test_multiple_tunnels_handled_in_order() -> None:
    run = RecordingRun(responses=[
        subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    ])
    handles = ensure_reverse_tunnels(
        _ssh_config(),
        ("15000:127.0.0.1:5000", "18888:127.0.0.1:8888"),
        run=run,
    )
    assert [h.spec for h in handles] == [
        "15000:127.0.0.1:5000",
        "18888:127.0.0.1:8888",
    ]
    # Distinct control sockets — different hash per spec.
    assert handles[0].control_socket != handles[1].control_socket
    # Both spawns issued, no checks (no prior sockets).
    assert len(run.calls) == 2
    assert all("-fN" in call for call in run.calls)


def test_control_socket_path_is_deterministic_per_host_port_spec() -> None:
    cfg = _ssh_config()
    same = _control_socket_path(cfg, "15000:127.0.0.1:5000")
    again = _control_socket_path(cfg, "15000:127.0.0.1:5000")
    assert same == again
    # Different spec → different socket.
    other_spec = _control_socket_path(cfg, "18888:127.0.0.1:8888")
    assert other_spec != same
    # Different host → different socket for the same spec.
    other_host = _control_socket_path(
        SshConfig(host="other.example.com", port=45516, user="root", identity_file="/tmp/k"),
        "15000:127.0.0.1:5000",
    )
    assert other_host != same


def test_close_tunnel_issues_exit_and_unlinks_socket() -> None:
    sock = _control_socket_path(_ssh_config(), "15000:127.0.0.1:5000")
    sock.parent.mkdir(parents=True, exist_ok=True)
    sock.touch()
    handle = TunnelHandle(
        spec="15000:127.0.0.1:5000",
        ssh_target="root@pod.example.com",
        ssh_port=45516,
        identity_file="/tmp/fake_key",
        control_socket=str(sock),
        started_fresh=False,
    )
    run = RecordingRun(responses=[
        subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    ])
    assert close_reverse_tunnel(handle, run=run) is True
    assert len(run.calls) == 1
    assert "-O" in run.calls[0] and "exit" in run.calls[0]
    assert not sock.exists()


def test_close_tunnel_with_missing_socket_is_noop() -> None:
    handle = TunnelHandle(
        spec="15000:127.0.0.1:5000",
        ssh_target="root@pod.example.com",
        ssh_port=45516,
        identity_file="/tmp/fake_key",
        control_socket="/does/not/exist.sock",
        started_fresh=False,
    )
    run = RecordingRun()
    assert close_reverse_tunnel(handle, run=run) is False
    assert run.calls == []
