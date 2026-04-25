from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ...core import config_loading
from ...core.models import (
    ObservabilityConfig,
    R2Config,
    RayConfig,
    RemoteRuntimeConfig,
    SshConfig,
)
from .models import InputConfig, ParquetFamilySpec, RecipeConfig, ResumeConfig, TokenizerConfig

REQUIRED_SECTIONS = [
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


def load_recipe_config(
    config_path: Path,
    overrides: list[str] | None = None,
    *,
    overlay_paths: Sequence[Path] = (),
) -> RecipeConfig:
    raw = load_resolved_recipe_mapping(config_path, overrides, overlay_paths=overlay_paths)
    return build_recipe_config(raw, config_path)


def load_resolved_recipe_mapping(
    config_path: Path,
    overrides: list[str] | None = None,
    *,
    overlay_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    return config_loading.load_recipe_mapping(
        config_path,
        overrides,
        overlay_paths=overlay_paths,
    )


def build_recipe_config(raw: dict[str, Any], config_path: Path) -> RecipeConfig:
    config_loading.require_sections(raw, config_path, REQUIRED_SECTIONS)
    validate_recipe_constraints(raw)
    ops = [config_loading.build_op_config(item) for item in raw["ops"]]
    family_specs = [ParquetFamilySpec.from_dict(item) for item in raw["input"]["family_specs"]]
    ray_raw = dict(raw["ray"])
    config_loading.reject_removed_ray_async_upload(ray_raw)
    return RecipeConfig(
        run_name=raw["run"]["name"],
        config_version=int(raw["run"]["config_version"]),
        ssh=SshConfig(**raw["ssh"]),
        remote=RemoteRuntimeConfig(**raw["remote"]),
        ray=RayConfig(**ray_raw),
        r2=R2Config(**raw["r2"]),
        input=InputConfig(
            source_prefix=str(raw["input"]["source_prefix"]),
            family_specs=family_specs,
        ),
        tokenizer=TokenizerConfig(**raw["tokenizer"]),
        mlflow=config_loading.build_mlflow_config(raw["mlflow"]),
        observability=ObservabilityConfig(**raw["observability"]),
        resumability=ResumeConfig(**raw["resumability"]),
        ops=ops,
    )


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
