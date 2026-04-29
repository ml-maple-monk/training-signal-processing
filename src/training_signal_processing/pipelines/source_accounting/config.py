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
from .models import (
    RecipeConfig,
    ResumeConfig,
    SourceAccountingInputConfig,
    SourceSpec,
    TokenizerConfig,
    extract_required_sources_from_plan,
)

CURRENT_MACHINE_PATH = config_loading.DEFAULT_CURRENT_MACHINE_PATH

REQUIRED_SECTIONS = (
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
    "sources",
    "ops",
)


class SourceAccountingRecipeConfigLoader(config_loading.AbstractRecipeConfigLoader):
    required_sections = REQUIRED_SECTIONS
    current_machine_path = CURRENT_MACHINE_PATH

    def recipe_from_mapping(self, raw: dict[str, Any], config_path: Path) -> RecipeConfig:
        del config_path
        run_raw = cast(dict[str, Any], raw["run"])
        ssh_raw = cast(dict[str, Any], raw["ssh"])
        remote_raw = cast(dict[str, Any], raw["remote"])
        r2_raw = cast(dict[str, Any], raw["r2"])
        input_raw = cast(dict[str, Any], raw["input"])
        tokenizer_raw = cast(dict[str, Any], raw["tokenizer"])
        mlflow_raw = cast(dict[str, Any], raw["mlflow"])
        observability_raw = cast(dict[str, Any], raw["observability"])
        resumability_raw = cast(dict[str, Any], raw["resumability"])
        sources = build_source_specs(cast(list[dict[str, Any]], raw["sources"]))
        input_config = SourceAccountingInputConfig(**input_raw)
        validate_source_specs_cover_plan(
            plan_path=Path(input_config.plan_path),
            sources=sources,
        )
        return RecipeConfig(
            run_name=str(run_raw["name"]),
            config_version=int(run_raw["config_version"]),
            ssh=SshConfig(**ssh_raw),
            remote=RemoteRuntimeConfig(**remote_raw),
            ray=build_ray_config(self.build_ray_mapping(raw)),
            r2=R2Config(**r2_raw),
            input=input_config,
            tokenizer=TokenizerConfig(**tokenizer_raw),
            mlflow=config_loading.build_mlflow_config(mlflow_raw),
            observability=ObservabilityConfig(**observability_raw),
            resumability=ResumeConfig(**resumability_raw),
            sources=sources,
            ops=self.build_op_configs(raw),
        )


source_accounting_recipe_config_loader = SourceAccountingRecipeConfigLoader()


def load_recipe_config(
    config_path: Path,
    overrides: list[str] | None = None,
    *,
    overlay_paths: Sequence[Path] = (),
) -> RecipeConfig:
    source_accounting_recipe_config_loader.current_machine_path = CURRENT_MACHINE_PATH
    return source_accounting_recipe_config_loader.load_recipe_config(
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
    source_accounting_recipe_config_loader.current_machine_path = CURRENT_MACHINE_PATH
    return source_accounting_recipe_config_loader.load_resolved_recipe_mapping(
        config_path,
        overrides,
        overlay_paths=overlay_paths,
    )


def build_recipe_config(raw: dict[str, Any], config_path: Path) -> RecipeConfig:
    return source_accounting_recipe_config_loader.build_recipe_config(raw, config_path)


def build_ray_config(ray_raw: dict[str, Any]) -> RayConfig:
    return RayConfig(
        executor_type=str(ray_raw["executor_type"]),
        batch_size=int(ray_raw["batch_size"]),
        concurrency=int(ray_raw["concurrency"]),
        target_num_blocks=int(ray_raw["target_num_blocks"]),
    )


def build_source_specs(raw_sources: list[dict[str, Any]]) -> list[SourceSpec]:
    if not raw_sources:
        raise ValueError("sources must declare at least one source.")
    sources = [SourceSpec(**raw) for raw in raw_sources]
    names = [source.name for source in sources]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("sources must not contain duplicate names: " + ", ".join(duplicates))
    return sources


def validate_source_specs_cover_plan(
    *,
    plan_path: Path,
    sources: list[SourceSpec],
) -> None:
    required_sources = extract_required_sources_from_plan(plan_path)
    if not required_sources:
        raise ValueError(f"No source names were found in source accounting plan: {plan_path}")
    configured = {source.name for source in sources if source.r2_relative_glob_path.strip()}
    missing = [source for source in required_sources if source not in configured]
    if missing:
        raise ValueError(
            "Source accounting config is missing explicit R2 mappings for: "
            + ", ".join(missing)
        )
