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
from .models import AsrConfig, DownloadConfig, InputConfig, RecipeConfig, ResumeConfig

CURRENT_MACHINE_PATH = config_loading.DEFAULT_CURRENT_MACHINE_PATH

REQUIRED_SECTIONS = [
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
        input=InputConfig(**raw["input"]),
        download=DownloadConfig(**raw["download"]),
        asr=AsrConfig(**raw["asr"]),
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
