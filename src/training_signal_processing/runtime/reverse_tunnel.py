"""Persistent SSH reverse tunnel management.

Background: the detached OCR launcher (PR #1) closes its SSH connection as soon
as it spawns the remote `setsid` session. Any `-R` reverse tunnels (e.g. the
MLflow `15000:127.0.0.1:5000` tunnel) die with that SSH session — but the
detached remote job may still try to reach them, and will wedge on a poll()
waiting for `127.0.0.1:15000` on the pod that nothing is listening on.

This module runs a persistent `ssh -fN` (background, no command) with
`ControlMaster=yes` and a per-tunnel control socket, so the tunnel survives the
launcher's exit. It is idempotent: if a live tunnel already exists for the
same spec, it is reused.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..core.models import SshConfig

RunFn = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class TunnelHandle:
    spec: str
    ssh_target: str
    ssh_port: int
    identity_file: str
    control_socket: str
    started_fresh: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "spec": self.spec,
            "ssh_target": self.ssh_target,
            "ssh_port": self.ssh_port,
            "control_socket": self.control_socket,
            "started_fresh": self.started_fresh,
        }


def _tunnel_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    path = base / "ocr-remote-launcher" / "tunnels"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _control_socket_path(ssh_config: SshConfig, tunnel_spec: str) -> Path:
    # Unix socket paths are capped at ~104 bytes on many systems — keep the
    # filename short and scoped to (host, port, spec) so distinct tunnels never
    # collide and the same tunnel is always deterministically addressable.
    digest = hashlib.sha1(
        f"{ssh_config.host}:{ssh_config.port}:{tunnel_spec}".encode()
    ).hexdigest()[:12]
    return _tunnel_cache_dir() / f"t-{digest}.sock"


def _ssh_target(ssh_config: SshConfig) -> str:
    return f"{ssh_config.user}@{ssh_config.host}"


def _is_alive(
    ssh_config: SshConfig,
    socket_path: Path,
    *,
    run: RunFn,
) -> bool:
    if not socket_path.exists():
        return False
    result = run(
        ["ssh", "-S", str(socket_path), "-O", "check", _ssh_target(ssh_config)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.returncode == 0


def _spawn(
    ssh_config: SshConfig,
    tunnel_spec: str,
    socket_path: Path,
    *,
    run: RunFn,
) -> None:
    identity = str(Path(ssh_config.identity_file).expanduser())
    target = _ssh_target(ssh_config)
    cmd = [
        "ssh",
        "-fN",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-o", "ControlMaster=yes",
        "-o", f"ControlPath={socket_path}",
        "-i", identity,
        "-p", str(ssh_config.port),
        "-R", tunnel_spec,
        target,
    ]
    result = run(cmd, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"Failed to start reverse tunnel {tunnel_spec!r} to "
            f"{target}:{ssh_config.port}: {stderr or 'no stderr'}"
        )


def ensure_reverse_tunnels(
    ssh_config: SshConfig,
    reverse_tunnels: tuple[str, ...],
    *,
    run: RunFn = subprocess.run,
) -> tuple[TunnelHandle, ...]:
    """Guarantee a persistent `ssh -fN -R` tunnel exists for each spec.

    Idempotent — reuses a live ControlMaster socket when `ssh -O check` answers
    OK, respawns after removing a stale socket otherwise. Returns one
    ``TunnelHandle`` per spec in the same order.
    """
    if not reverse_tunnels:
        return ()
    target = _ssh_target(ssh_config)
    identity = str(Path(ssh_config.identity_file).expanduser())
    handles: list[TunnelHandle] = []
    for spec in reverse_tunnels:
        socket_path = _control_socket_path(ssh_config, spec)
        if _is_alive(ssh_config, socket_path, run=run):
            handles.append(TunnelHandle(
                spec=spec,
                ssh_target=target,
                ssh_port=ssh_config.port,
                identity_file=identity,
                control_socket=str(socket_path),
                started_fresh=False,
            ))
            continue
        if socket_path.exists():
            socket_path.unlink()
        _spawn(ssh_config, spec, socket_path, run=run)
        handles.append(TunnelHandle(
            spec=spec,
            ssh_target=target,
            ssh_port=ssh_config.port,
            identity_file=identity,
            control_socket=str(socket_path),
            started_fresh=True,
        ))
    return tuple(handles)


def close_reverse_tunnel(
    handle: TunnelHandle,
    *,
    run: RunFn = subprocess.run,
) -> bool:
    """Tear down a tunnel via `ssh -S <sock> -O exit`.

    Returns True if a socket was found and a close was attempted, False if the
    socket was already gone. Silent on transport errors — the caller just cares
    whether the socket is now gone.
    """
    sock = Path(handle.control_socket)
    if not sock.exists():
        return False
    run(
        ["ssh", "-S", str(sock), "-O", "exit", handle.ssh_target],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if sock.exists():
        try:
            sock.unlink()
        except FileNotFoundError:
            pass
    return True
