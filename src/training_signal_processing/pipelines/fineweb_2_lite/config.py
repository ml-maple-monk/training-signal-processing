from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatabaseConfig:
    dsn_env_var: str
    fallback_dsn: str = ""

    def __post_init__(self) -> None:
        if not self.dsn_env_var.strip():
            raise ValueError("database.dsn_env_var must be non-empty.")

    def resolve_dsn(self) -> str:
        dsn = os.environ.get(self.dsn_env_var.strip()) or self.fallback_dsn.strip()
        if not dsn:
            raise ValueError(
                f"Set ${self.dsn_env_var.strip()} or database.fallback_dsn in the config."
            )
        return dsn


@dataclass(frozen=True)
class InputConfig:
    batch_size: int = 1000
    min_text_length: int = 1
    include_unified: bool = True
    include_sea_pile_malay: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("input.batch_size must be positive.")
        if self.min_text_length < 0:
            raise ValueError("input.min_text_length must be non-negative.")
        if not self.include_unified and not self.include_sea_pile_malay:
            raise ValueError("At least one input source must be enabled.")


@dataclass(frozen=True)
class ProcessingConfig:
    write_batch_size: int = 1000
    worker_count: int = 1

    def __post_init__(self) -> None:
        if self.write_batch_size <= 0:
            raise ValueError("processing.write_batch_size must be positive.")
        if self.worker_count <= 0:
            raise ValueError("processing.worker_count must be positive.")


@dataclass(frozen=True)
class LanguageProfile:
    name: str
    language_score: float
    dup_line_frac: float
    top_n_grams: tuple[tuple[int, float], ...]
    dup_n_grams: tuple[tuple[int, float], ...]
    line_punct_thr: float
    new_line_ratio: float
    min_avg_word_length: int
    max_avg_word_length: int
    max_non_alpha_words_ratio: float
    stopwords: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("language profile names must be non-empty.")
        if not self.stopwords:
            raise ValueError(f"language profile {self.name!r} must define stopwords.")
        if not self.top_n_grams:
            raise ValueError(f"language profile {self.name!r} must define top_n_grams.")
        if not self.dup_n_grams:
            raise ValueError(f"language profile {self.name!r} must define dup_n_grams.")


@dataclass(frozen=True)
class RecipeConfig:
    run_name: str
    database: DatabaseConfig
    input: InputConfig
    processing: ProcessingConfig
    source_language_defaults: dict[str, str]
    language_profiles: dict[str, LanguageProfile]

    def __post_init__(self) -> None:
        if not self.run_name.strip():
            raise ValueError("run.name must be non-empty.")
        if not self.language_profiles:
            raise ValueError("language_profiles must define at least one profile.")
        unknown = sorted(
            {
                profile
                for profile in self.source_language_defaults.values()
                if profile not in self.language_profiles
            }
        )
        if unknown:
            raise ValueError(
                "language_resolution.source_defaults references unknown profiles: "
                + ", ".join(unknown)
            )


def load_recipe_config(config_path: Path) -> RecipeConfig:
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("FineWeb2-lite config must be a YAML mapping.")
    return build_recipe_config(raw)


def build_recipe_config(raw: dict[str, Any]) -> RecipeConfig:
    run_raw = _mapping(raw, "run")
    language_resolution_raw = _mapping(raw, "language_resolution")
    source_defaults = {
        str(source).strip().lower(): str(profile).strip()
        for source, profile in _mapping(language_resolution_raw, "source_defaults").items()
    }
    profiles = {
        name: _build_language_profile(name, profile_raw)
        for name, profile_raw in _mapping(raw, "language_profiles").items()
    }
    return RecipeConfig(
        run_name=str(run_raw["name"]),
        database=DatabaseConfig(**_mapping(raw, "database")),
        input=InputConfig(**_mapping(raw, "input")),
        processing=ProcessingConfig(**_mapping(raw, "processing")),
        source_language_defaults=source_defaults,
        language_profiles=profiles,
    )


def _build_language_profile(name: object, raw_profile: object) -> LanguageProfile:
    raw = dict(raw_profile) if isinstance(raw_profile, dict) else {}
    return LanguageProfile(
        name=str(name),
        language_score=float(raw["language_score"]),
        dup_line_frac=float(raw["dup_line_frac"]),
        top_n_grams=_threshold_pairs(raw["top_n_grams"]),
        dup_n_grams=_threshold_pairs(raw["dup_n_grams"]),
        line_punct_thr=float(raw["line_punct_thr"]),
        new_line_ratio=float(raw["new_line_ratio"]),
        min_avg_word_length=int(raw["min_avg_word_length"]),
        max_avg_word_length=int(raw["max_avg_word_length"]),
        max_non_alpha_words_ratio=float(raw["max_non_alpha_words_ratio"]),
        stopwords=tuple(str(item).strip().lower() for item in raw["stopwords"]),
    )


def _threshold_pairs(raw: object) -> tuple[tuple[int, float], ...]:
    if isinstance(raw, dict):
        items = raw.items()
    else:
        items = raw if isinstance(raw, list) else ()
    pairs: list[tuple[int, float]] = []
    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            key, value = item
        elif isinstance(item, list) and len(item) == 2:
            key, value = item
        else:
            raise ValueError(f"Invalid threshold pair: {item!r}")
        pairs.append((int(key), float(value)))
    return tuple(sorted(pairs))


def _mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a YAML mapping.")
    return value
