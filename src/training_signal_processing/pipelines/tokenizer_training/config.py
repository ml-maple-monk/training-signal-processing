from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from ...core.config_loading import (
    load_recipe_mapping,
    require_sections,
)
from ...core.models import R2Config
from .models import (
    BudgetConfig,
    CheckpointConfig,
    InputConfig,
    OutputConfig,
    RecipeConfig,
    TrainingConfig,
)

REQUIRED_SECTIONS = (
    "run",
    "r2",
    "input",
    "training",
    "budget",
    "checkpoint",
    "output",
)


def load_recipe_config(
    config_path: Path,
    overrides: list[str] | None = None,
    *,
    overlay_paths: Sequence[Path] = (),
) -> RecipeConfig:
    raw = load_resolved_recipe_mapping(
        config_path,
        overrides,
        overlay_paths=overlay_paths,
    )
    return build_recipe_config(raw, config_path)


def load_resolved_recipe_mapping(
    config_path: Path,
    overrides: list[str] | None = None,
    *,
    overlay_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    return load_recipe_mapping(
        config_path,
        overrides,
        overlay_paths=overlay_paths,
    )


def build_recipe_config(raw: dict[str, Any], config_path: Path) -> RecipeConfig:
    require_sections(raw, config_path, list(REQUIRED_SECTIONS))
    run_raw = cast(dict[str, Any], raw["run"])
    return RecipeConfig(
        run_name=str(run_raw["name"]),
        config_version=int(run_raw["config_version"]),
        r2=R2Config(**cast(dict[str, Any], raw["r2"])),
        input=InputConfig(**cast(dict[str, Any], raw["input"])),
        training=TrainingConfig(**cast(dict[str, Any], raw["training"])),
        budget=BudgetConfig(**cast(dict[str, Any], raw["budget"])),
        checkpoint=CheckpointConfig(**cast(dict[str, Any], raw["checkpoint"])),
        output=OutputConfig(**cast(dict[str, Any], raw["output"])),
    )
