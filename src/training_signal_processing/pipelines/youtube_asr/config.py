from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

import yaml

from ...core.models import (
    MlflowConfig,
    ObservabilityConfig,
    OpConfig,
    R2Config,
    RayConfig,
    RemoteRuntimeConfig,
    SshConfig,
)
from .models import AsrConfig, DownloadConfig, InputConfig, RecipeConfig, ResumeConfig

CURRENT_MACHINE_PATH = Path(__file__).resolve().parents[4] / "infra" / "current-machine"


def load_recipe_config(config_path: Path, overrides: list[str] | None = None) -> RecipeConfig:
    expanded_config = load_resolved_recipe_mapping(config_path, overrides)
    return build_recipe_config(expanded_config, config_path)


def load_resolved_recipe_mapping(
    config_path: Path,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    resolved_overrides = list(overrides or [])
    raw_config = read_recipe_file(config_path)
    merged_config = apply_overrides(raw_config, resolved_overrides)
    merged_config = apply_current_machine_target(
        merged_config,
        override_keys=extract_override_keys(resolved_overrides),
    )
    return expand_recipe_values(merged_config)


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
) -> dict[str, Any]:
    if not isinstance(config.get("ssh"), dict):
        return config
    needs_host = "ssh.host" not in override_keys
    needs_port = "ssh.port" not in override_keys
    if not (needs_host or needs_port):
        return config
    if not CURRENT_MACHINE_PATH.is_file():
        return config
    host, port = parse_current_machine_ssh_target(CURRENT_MACHINE_PATH)
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


def build_recipe_config(raw: dict[str, Any], config_path: Path) -> RecipeConfig:
    require_sections(raw, config_path)
    validate_recipe_constraints(raw)
    ops = [build_op_config(item) for item in raw["ops"]]
    return RecipeConfig(
        run_name=raw["run"]["name"],
        config_version=int(raw["run"]["config_version"]),
        ssh=SshConfig(**raw["ssh"]),
        remote=RemoteRuntimeConfig(**raw["remote"]),
        ray=RayConfig(**raw["ray"]),
        r2=R2Config(**raw["r2"]),
        input=InputConfig(**raw["input"]),
        download=DownloadConfig(**raw["download"]),
        asr=AsrConfig(**raw["asr"]),
        mlflow=MlflowConfig(**raw["mlflow"]),
        observability=ObservabilityConfig(**raw["observability"]),
        resumability=ResumeConfig(**raw["resumability"]),
        ops=ops,
    )


def require_sections(raw: dict[str, Any], config_path: Path) -> None:
    required = [
        "run",
        "ssh",
        "remote",
        "ray",
        "r2",
        "input",
        "download",
        "asr",
        "mlflow",
        "observability",
        "resumability",
        "ops",
    ]
    missing = [name for name in required if name not in raw]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Recipe missing required sections in {config_path}: {joined}")


def build_op_config(raw: dict[str, Any]) -> OpConfig:
    if not isinstance(raw, dict):
        raise ValueError("Each op must be a mapping")
    name = raw.get("name")
    op_type = str(raw.get("type", "")).strip()
    if not name:
        raise ValueError("Each op requires name")
    options = {key: value for key, value in raw.items() if key not in {"name", "type"}}
    return OpConfig(name=name, type=op_type, options=options)


def validate_recipe_constraints(raw: dict[str, Any]) -> None:
    if raw["ray"]["executor_type"] != "ray":
        raise ValueError("Only ray executor_type is supported")
    if raw["resumability"]["strategy"] != "batch_manifest":
        raise ValueError("Only batch_manifest resumability is supported")
    if int(raw["input"]["videos_per_channel"]) <= 0:
        raise ValueError("input.videos_per_channel must be positive")
    if not str(raw["input"]["media_r2_prefix"]).strip():
        raise ValueError("input.media_r2_prefix must be non-empty")
    if not str(raw["download"]["local_staging_dir"]).strip():
        raise ValueError("download.local_staging_dir must be non-empty")
    if not str(raw["download"]["format_selector"]).strip():
        raise ValueError("download.format_selector must be non-empty")
    if not str(raw["asr"]["model_name"]).strip():
        raise ValueError("asr.model_name must be non-empty")
    if float(raw["asr"]["gpu_memory_utilization"]) <= 0:
        raise ValueError("asr.gpu_memory_utilization must be positive")
    if int(raw["asr"]["max_inference_batch_size"]) <= 0:
        raise ValueError("asr.max_inference_batch_size must be positive")
    if int(raw["asr"]["max_new_tokens"]) <= 0:
        raise ValueError("asr.max_new_tokens must be positive")
    if int(raw["asr"]["max_media_file_size_mb"]) <= 0:
        raise ValueError("asr.max_media_file_size_mb must be positive")
