from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    return RecipeConfig(
        run_name=raw["run"]["name"],
        config_version=int(raw["run"]["config_version"]),
        ssh=SshConfig(**raw["ssh"]),
        remote=RemoteRuntimeConfig(**raw["remote"]),
        ray=build_ocr_ray_config(raw["ray"]),
        r2=R2Config(**raw["r2"]),
        input=InputConfig(**raw["input"]),
        mlflow=config_loading.build_mlflow_config(raw["mlflow"]),
        observability=ObservabilityConfig(**raw["observability"]),
        resumability=ResumeConfig(**raw["resumability"]),
        ops=ops,
    )


def validate_recipe_constraints(raw: dict[str, Any]) -> None:
    validate_ray_config(raw["ray"])
    validate_resumability_config(raw["resumability"])


def build_ocr_ray_config(ray_raw: dict[str, Any]) -> OcrRayConfig:
    ray_values = dict(ray_raw)
    config_loading.reject_removed_ray_async_upload(ray_values)
    marker_values = dict(ray_values["marker_ocr_resources"])
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


def validate_ray_config(ray: dict[str, Any]) -> None:
    if ray["executor_type"] != "ray":
        raise ValueError("Only ray executor_type is supported")
    if int(ray.get("batch_size", 0)) <= 0:
        raise ValueError("ray.batch_size must be positive")
    if int(ray.get("concurrency", 0)) <= 0:
        raise ValueError("ray.concurrency must be positive")
    if int(ray.get("target_num_blocks", 0)) < 0:
        raise ValueError("ray.target_num_blocks must be zero or positive")
    marker = ray.get("marker_ocr_resources")
    if not isinstance(marker, dict):
        raise ValueError("ray.marker_ocr_resources must be a mapping with num_gpus and num_cpus")
    if float(marker.get("num_gpus", 0)) <= 0:
        raise ValueError("ray.marker_ocr_resources.num_gpus must be positive")
    if float(marker.get("num_cpus", 0)) <= 0:
        raise ValueError("ray.marker_ocr_resources.num_cpus must be positive")


def validate_resumability_config(resumability: dict[str, Any]) -> None:
    if resumability["strategy"] != "batch_manifest":
        raise ValueError("Only batch_manifest resumability is supported")
