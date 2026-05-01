from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from ...core import config_loading
from ...core.models import (
    ObservabilityConfig,
    R2Config,
    RayConfig,
    RemoteRuntimeConfig,
    SshConfig,
)
from .models import ExportConfig, InputConfig, RecipeConfig, ResumeConfig

CURRENT_MACHINE_PATH = config_loading.DEFAULT_CURRENT_MACHINE_PATH

REQUIRED_SECTIONS = (
    "run",
    "ssh",
    "remote",
    "ray",
    "r2",
    "input",
    "export",
    "mlflow",
    "observability",
    "resumability",
    "ops",
)


class FineWebUnifiedRecipeConfigLoader(config_loading.AbstractRecipeConfigLoader):
    required_sections = REQUIRED_SECTIONS
    current_machine_path = CURRENT_MACHINE_PATH

    def recipe_from_mapping(self, raw: dict[str, Any], config_path: Path) -> RecipeConfig:
        del config_path
        run_raw = cast(dict[str, Any], raw["run"])
        ssh_raw = cast(dict[str, Any], raw["ssh"])
        remote_raw = cast(dict[str, Any], raw["remote"])
        r2_raw = cast(dict[str, Any], raw["r2"])
        input_raw = cast(dict[str, Any], raw["input"])
        export_raw = cast(dict[str, Any], raw["export"])
        mlflow_raw = cast(dict[str, Any], raw["mlflow"])
        observability_raw = cast(dict[str, Any], raw["observability"])
        resumability_raw = cast(dict[str, Any], raw["resumability"])
        return RecipeConfig(
            run_name=str(run_raw["name"]),
            config_version=int(run_raw["config_version"]),
            ssh=SshConfig(**ssh_raw),
            remote=RemoteRuntimeConfig(**remote_raw),
            ray=build_ray_config(self.build_ray_mapping(raw)),
            r2=R2Config(**r2_raw),
            input=InputConfig(**input_raw),
            export=ExportConfig(**export_raw),
            mlflow=config_loading.build_mlflow_config(mlflow_raw),
            observability=ObservabilityConfig(**observability_raw),
            resumability=ResumeConfig(**resumability_raw),
            ops=self.build_op_configs(raw),
        )


fineweb_unified_recipe_config_loader = FineWebUnifiedRecipeConfigLoader()


def load_recipe_config(
    config_path: Path,
    overrides: list[str] | None = None,
    *,
    overlay_paths: Sequence[Path] = (),
) -> RecipeConfig:
    fineweb_unified_recipe_config_loader.current_machine_path = CURRENT_MACHINE_PATH
    return fineweb_unified_recipe_config_loader.load_recipe_config(
        config_path,
        overrides,
        overlay_paths=overlay_paths,
    )


def load_resolved_recipe_mapping(
    config_path: Path,
    overrides: list[str] | None = None,
    *,
    overlay_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    fineweb_unified_recipe_config_loader.current_machine_path = CURRENT_MACHINE_PATH
    return fineweb_unified_recipe_config_loader.load_resolved_recipe_mapping(
        config_path,
        overrides,
        overlay_paths=overlay_paths,
    )


def build_recipe_config(raw: dict[str, Any], config_path: Path) -> RecipeConfig:
    return fineweb_unified_recipe_config_loader.build_recipe_config(raw, config_path)


def build_ray_config(ray_raw: dict[str, Any]) -> RayConfig:
    return RayConfig(
        executor_type=str(ray_raw["executor_type"]),
        batch_size=int(ray_raw["batch_size"]),
        concurrency=int(ray_raw["concurrency"]),
        target_num_blocks=int(ray_raw["target_num_blocks"]),
    )
