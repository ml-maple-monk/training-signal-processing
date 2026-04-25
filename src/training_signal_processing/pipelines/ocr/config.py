from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from ...core import config_loading
from ...core.models import (
    ObservabilityConfig,
    R2Config,
    RayTransformResources,
    RemoteRuntimeConfig,
    SshConfig,
)
from .models import InputConfig, OcrRayConfig, RecipeConfig, ResumeConfig

CURRENT_MACHINE_PATH = config_loading.DEFAULT_CURRENT_MACHINE_PATH

REQUIRED_SECTIONS = (
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
)


class OcrRecipeConfigLoader(config_loading.AbstractRecipeConfigLoader):
    required_sections = REQUIRED_SECTIONS
    current_machine_path = CURRENT_MACHINE_PATH

    def recipe_from_mapping(self, raw: dict[str, Any], config_path: Path) -> RecipeConfig:
        del config_path
        run_raw = cast(dict[str, Any], raw["run"])
        ssh_raw = cast(dict[str, Any], raw["ssh"])
        remote_raw = cast(dict[str, Any], raw["remote"])
        r2_raw = cast(dict[str, Any], raw["r2"])
        input_raw = cast(dict[str, Any], raw["input"])
        mlflow_raw = cast(dict[str, Any], raw["mlflow"])
        observability_raw = cast(dict[str, Any], raw["observability"])
        resumability_raw = cast(dict[str, Any], raw["resumability"])
        return RecipeConfig(
            run_name=str(run_raw["name"]),
            config_version=int(run_raw["config_version"]),
            ssh=SshConfig(**ssh_raw),
            remote=RemoteRuntimeConfig(**remote_raw),
            ray=build_ocr_ray_config(self.build_ray_mapping(raw)),
            r2=R2Config(**r2_raw),
            input=InputConfig(**input_raw),
            mlflow=config_loading.build_mlflow_config(mlflow_raw),
            observability=ObservabilityConfig(**observability_raw),
            resumability=ResumeConfig(**resumability_raw),
            ops=self.build_op_configs(raw),
        )


ocr_recipe_config_loader = OcrRecipeConfigLoader()


def load_recipe_config(
    config_path: Path,
    overrides: list[str] | None = None,
    *,
    overlay_paths: Sequence[Path] = (),
) -> RecipeConfig:
    ocr_recipe_config_loader.current_machine_path = CURRENT_MACHINE_PATH
    return ocr_recipe_config_loader.load_recipe_config(
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
    ocr_recipe_config_loader.current_machine_path = CURRENT_MACHINE_PATH
    return ocr_recipe_config_loader.load_resolved_recipe_mapping(
        config_path,
        overrides,
        overlay_paths=overlay_paths,
    )


def build_recipe_config(raw: dict[str, Any], config_path: Path) -> RecipeConfig:
    return ocr_recipe_config_loader.build_recipe_config(raw, config_path)


def build_ocr_ray_config(ray_raw: dict[str, Any]) -> OcrRayConfig:
    ray_values = dict(ray_raw)
    marker_values = cast(dict[str, Any], ray_values["marker_ocr_resources"])
    return OcrRayConfig(
        executor_type=str(ray_values["executor_type"]),
        batch_size=int(ray_values["batch_size"]),
        concurrency=int(ray_values["concurrency"]),
        target_num_blocks=int(ray_values["target_num_blocks"]),
        marker_ocr_resources=RayTransformResources(
            num_gpus=float(marker_values["num_gpus"]),
            num_cpus=float(marker_values["num_cpus"]),
        ),
    )
