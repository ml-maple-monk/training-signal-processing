from __future__ import annotations

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
from .models import InputConfig, ParquetFamilySpec, RecipeConfig, ResumeConfig, TokenizerConfig


def load_recipe_config(config_path: Path, overrides: list[str] | None = None) -> RecipeConfig:
    expanded_config = load_resolved_recipe_mapping(config_path, overrides)
    return build_recipe_config(expanded_config, config_path)


def load_resolved_recipe_mapping(
    config_path: Path,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    raw_config = read_recipe_file(config_path)
    merged_config = apply_overrides(raw_config, overrides or [])
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
    family_specs = [ParquetFamilySpec.from_dict(item) for item in raw["input"]["family_specs"]]
    return RecipeConfig(
        run_name=raw["run"]["name"],
        config_version=int(raw["run"]["config_version"]),
        ssh=SshConfig(**raw["ssh"]),
        remote=RemoteRuntimeConfig(**raw["remote"]),
        ray=RayConfig(**raw["ray"]),
        r2=R2Config(**raw["r2"]),
        input=InputConfig(
            source_prefix=str(raw["input"]["source_prefix"]),
            family_specs=family_specs,
        ),
        tokenizer=TokenizerConfig(**raw["tokenizer"]),
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
        "tokenizer",
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
    tokenizer = raw["tokenizer"]
    if str(tokenizer.get("output_compression", "gzip")) != "gzip":
        raise ValueError("Only gzip output_compression is supported")
    family_specs = raw["input"].get("family_specs")
    if not isinstance(family_specs, list) or not family_specs:
        raise ValueError("input.family_specs must declare at least one family spec")
    seen_names: set[str] = set()
    for index, item in enumerate(family_specs, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"input.family_specs[{index}] must be a mapping")
        name = str(item.get("name", "")).strip()
        glob_value = str(item.get("glob", "")).strip()
        text_column = str(item.get("text_column", "")).strip()
        if not name or not glob_value or not text_column:
            raise ValueError(
                f"input.family_specs[{index}] must include name, glob, and text_column"
            )
        if name in seen_names:
            raise ValueError(f"Duplicate family spec name: {name}")
        seen_names.add(name)
