from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from .models import MlflowConfig, OpConfig

# WARNING TO OTHER AGENTS: DO NOT CHANGE THIS ABSTRACTION WITHOUT EXPLICIT USER APPROVAL.

DEFAULT_CURRENT_MACHINE_PATH = (
    Path(__file__).resolve().parents[3] / "infra" / "current-machine"
)
REMOVED_RAY_ASYNC_UPLOAD_MESSAGE = (
    "ray.async_upload was removed; remote output uploads are synchronous "
    "before output progress is recorded."
)
REMOVED_MLFLOW_TUNNEL_MESSAGE = (
    "Reverse-tunnel MLflow was removed; mlflow.local_tracking_uri and "
    "mlflow.remote_tunnel_port are no longer supported. Use mlflow.tracking_uri "
    "reachable from the logging process, or set mlflow.enabled=false and use "
    "R2 event logs."
)


class AbstractRecipeConfigLoader(ABC):
    """Template for YAML-first pipeline config loaders.

    Public methods load resolved YAML and build typed recipe dataclasses.
    Concrete loaders customize only section names, current-machine behavior,
    and typed recipe construction.
    """

    required_sections: tuple[str, ...] = ()
    current_machine_path: Path | None = None

    def load_recipe_config(
        self,
        config_path: Path,
        overrides: list[str] | None = None,
        *,
        overlay_paths: Sequence[Path] = (),
    ) -> Any:
        raw = self.load_resolved_recipe_mapping(
            config_path,
            overrides,
            overlay_paths=overlay_paths,
        )
        return self.build_recipe_config(raw, config_path)

    def load_resolved_recipe_mapping(
        self,
        config_path: Path,
        overrides: list[str] | None = None,
        *,
        overlay_paths: Sequence[Path] = (),
    ) -> dict[str, Any]:
        return load_recipe_mapping(
            config_path,
            overrides,
            current_machine_path=self.current_machine_path,
            overlay_paths=overlay_paths,
        )

    def build_recipe_config(self, raw: dict[str, Any], config_path: Path) -> Any:
        require_sections(raw, config_path, list(self.required_sections))
        return self.recipe_from_mapping(raw, config_path)

    def build_op_configs(self, raw: dict[str, Any]) -> list[OpConfig]:
        return [build_op_config(item) for item in raw["ops"]]

    def build_ray_mapping(self, raw: dict[str, Any]) -> dict[str, Any]:
        ray_raw = dict(raw["ray"])
        reject_removed_ray_async_upload(ray_raw)
        return ray_raw

    @abstractmethod
    def recipe_from_mapping(self, raw: dict[str, Any], config_path: Path) -> Any:
        raise NotImplementedError


def load_recipe_mapping(
    config_path: Path,
    overrides: list[str] | None = None,
    *,
    current_machine_path: Path | None = None,
    overlay_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    resolved_overrides = list(overrides or [])
    raw_config = read_recipe_file(config_path)
    for overlay_path in overlay_paths:
        raw_config = deep_merge_mapping(raw_config, read_recipe_file(overlay_path))
    merged_config = apply_overrides(raw_config, resolved_overrides)
    if current_machine_path is not None:
        merged_config = apply_current_machine_target(
            merged_config,
            override_keys=extract_override_keys(resolved_overrides),
            current_machine_path=current_machine_path,
        )
    return expand_recipe_values(merged_config)


def deep_merge_mapping(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = clone_mapping(base)
    for key, overlay_value in overlay.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(overlay_value, dict):
            result[key] = deep_merge_mapping(existing, overlay_value)
        else:
            result[key] = clone_mapping(overlay_value)
    return result


def read_recipe_file(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise ValueError(f"Recipe file not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Recipe must be a mapping: {config_path}")
    return loaded


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = clone_mapping(config)
    for override in overrides:
        key_path, value = split_override(override)
        set_override_value(updated, key_path.split("."), parse_override_value(value))
    return updated


def extract_override_keys(overrides: list[str]) -> set[str]:
    return {split_override(override)[0] for override in overrides}


def clone_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clone_mapping(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_mapping(item) for item in value]
    return value


def split_override(override: str) -> tuple[str, str]:
    if "=" not in override:
        raise ValueError(f"Override must use key=value: {override}")
    key_path, value = override.split("=", 1)
    if not key_path.strip():
        raise ValueError(f"Override key is empty: {override}")
    return key_path.strip(), value.strip()


def set_override_value(config: dict[str, Any], path_parts: list[str], value: Any) -> None:
    current = config
    for part in path_parts[:-1]:
        next_value = current.get(part)
        if next_value is None:
            next_value = {}
            current[part] = next_value
        if not isinstance(next_value, dict):
            raise ValueError(f"Override path is not a mapping: {'.'.join(path_parts)}")
        current = next_value
    current[path_parts[-1]] = value


def parse_override_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if value.isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def apply_current_machine_target(
    config: dict[str, Any],
    *,
    override_keys: set[str],
    current_machine_path: Path,
) -> dict[str, Any]:
    if not isinstance(config.get("ssh"), dict):
        return config
    needs_host = "ssh.host" not in override_keys
    needs_port = "ssh.port" not in override_keys
    if not (needs_host or needs_port):
        return config
    if not current_machine_path.is_file():
        return config
    host, port = parse_current_machine_ssh_target(current_machine_path)
    updated = clone_mapping(config)
    ssh_config = dict(updated["ssh"])
    if needs_host:
        ssh_config["host"] = host
    if needs_port:
        ssh_config["port"] = port
    updated["ssh"] = ssh_config
    return updated


def parse_current_machine_ssh_target(path: Path) -> tuple[str, int]:
    command_text = path.read_text(encoding="utf-8").strip()
    if not command_text:
        raise ValueError(f"{path} is empty; expected an ssh command.")
    tokens = shlex.split(command_text)
    if not tokens or tokens[0] != "ssh":
        raise ValueError(f"{path} must start with an ssh command: {command_text}")
    port: int | None = None
    host: str | None = None
    for index, token in enumerate(tokens):
        if token == "-p":
            if index + 1 >= len(tokens):
                raise ValueError(f"{path} is missing a port value after -p: {command_text}")
            try:
                port = int(tokens[index + 1])
            except ValueError as exc:
                raise ValueError(f"{path} contains a non-integer ssh port: {command_text}") from exc
        elif "@" in token and not token.startswith("-"):
            host = token.rsplit("@", 1)[1]
    if not host or port is None:
        raise ValueError(f"{path} must contain both ssh host and -p <port>: {command_text}")
    return host, port


def expand_recipe_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_recipe_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_recipe_values(item) for item in value]
    if isinstance(value, str):
        return str(Path(value).expanduser()) if value.startswith("~") else value
    return value


def require_sections(raw: dict[str, Any], config_path: Path, required: list[str]) -> None:
    missing = [name for name in required if name not in raw]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Recipe missing required sections in {config_path}: {joined}")


def reject_removed_ray_async_upload(ray_raw: dict[str, Any]) -> None:
    if "async_upload" in ray_raw:
        raise ValueError(REMOVED_RAY_ASYNC_UPLOAD_MESSAGE)


def build_mlflow_config(raw: dict[str, Any]) -> MlflowConfig:
    removed_keys = sorted(
        key
        for key in ("local_tracking_uri", "remote_tunnel_port")
        if key in raw
    )
    if removed_keys:
        raise ValueError(REMOVED_MLFLOW_TUNNEL_MESSAGE)
    enabled = bool(raw.get("enabled", False))
    tracking_uri = str(raw.get("tracking_uri", "")).strip()
    if enabled and not tracking_uri:
        raise ValueError(
            "mlflow.tracking_uri is required when mlflow.enabled=true; "
            "reverse-tunnel MLflow was removed."
        )
    experiment_name = str(raw.get("experiment_name", "")).strip()
    if not experiment_name:
        raise ValueError("mlflow.experiment_name must be non-empty")
    return MlflowConfig(
        enabled=enabled,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )


def build_op_config(raw: dict[str, Any]) -> OpConfig:
    if not isinstance(raw, dict):
        raise ValueError("Each op must be a mapping")
    name = raw.get("name")
    op_type = str(raw.get("type", "")).strip()
    if not name:
        raise ValueError("Each op requires name")
    options = {key: value for key, value in raw.items() if key not in {"name", "type"}}
    return OpConfig(name=name, type=op_type, options=options)


__all__ = [
    "AbstractRecipeConfigLoader",
    "DEFAULT_CURRENT_MACHINE_PATH",
    "REMOVED_RAY_ASYNC_UPLOAD_MESSAGE",
    "load_recipe_mapping",
    "deep_merge_mapping",
    "read_recipe_file",
    "apply_overrides",
    "extract_override_keys",
    "clone_mapping",
    "split_override",
    "set_override_value",
    "parse_override_value",
    "apply_current_machine_target",
    "parse_current_machine_ssh_target",
    "expand_recipe_values",
    "require_sections",
    "reject_removed_ray_async_upload",
    "REMOVED_MLFLOW_TUNNEL_MESSAGE",
    "build_mlflow_config",
    "build_op_config",
]
