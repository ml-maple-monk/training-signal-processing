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
from .models import InputConfig, RecipeConfig, ResumeConfig

CURRENT_MACHINE_PATH = config_loading.DEFAULT_CURRENT_MACHINE_PATH

REQUIRED_SECTIONS = [
    "run",
    "ssh",
    "remote",
    "ray",
    "r2",
    "input",
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
        current_machine_path=CURRENT_MACHINE_PATH,
        overlay_paths=overlay_paths,
    )


def build_recipe_config(raw: dict[str, Any], config_path: Path) -> RecipeConfig:
    config_loading.require_sections(raw, config_path, REQUIRED_SECTIONS)
    validate_recipe_constraints(raw)
    ops = [config_loading.build_op_config(item) for item in raw["ops"]]
    ray_raw = dict(raw["ray"])
    config_loading.reject_removed_ray_async_upload(ray_raw)
    return RecipeConfig(
        run_name=raw["run"]["name"],
        config_version=int(raw["run"]["config_version"]),
        ssh=SshConfig(**raw["ssh"]),
        remote=RemoteRuntimeConfig(**raw["remote"]),
        ray=RayConfig(**ray_raw),
        r2=R2Config(**raw["r2"]),
        input=InputConfig(items=list(raw["input"].get("items", []))),
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
    items = raw["input"].get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("input.items must declare at least one item")
    seen_ids: set[str] = set()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"input.items[{index}] must be a mapping")
        source_id = str(item.get("source_id", "")).strip()
        message = str(item.get("message", "")).strip()
        if not source_id or not message:
            raise ValueError(
                f"input.items[{index}] must include non-empty source_id and message"
            )
        if source_id in seen_ids:
            raise ValueError(f"Duplicate source_id: {source_id}")
        seen_ids.add(source_id)
