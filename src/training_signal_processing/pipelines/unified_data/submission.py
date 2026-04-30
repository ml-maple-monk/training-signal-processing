from __future__ import annotations

import shlex
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from ...core.storage import ObjectStore
from ...core.submission import (
    ArtifactStore,
    BootstrapSpec,
    RemoteInvocationSpec,
    SubmissionAdapter,
    SubmissionManifest,
)
from ...core.utils import join_s3_key
from .config import load_resolved_recipe_mapping
from .models import MergeSegment, RecipeConfig, UnifiedDataPartTask


@dataclass(frozen=True)
class LidShard:
    source_name: str
    source_object_key: str
    source_row_group_index: int
    output_shard_key: str

    @classmethod
    def from_manifest_row(cls, row: dict[str, Any]) -> "LidShard":
        return cls(
            source_name=str(row["source_name"]),
            source_object_key=str(row["source_object_key"]),
            source_row_group_index=int(row["source_row_group_index"]),
            output_shard_key=str(row["output_shard_key"]),
        )

    @property
    def join_key(self) -> tuple[str, int, str]:
        return (self.source_object_key, self.source_row_group_index, self.source_name)


@dataclass(frozen=True)
class CleaningShard:
    source_name: str
    cleaning_source: str
    source_object_key: str
    source_row_group_index: int
    source_row_group_num_rows: int | None
    filters: dict[str, str]
    unified_shard_key: str
    metrics_key: str
    done_key: str
    error_key: str

    @classmethod
    def from_manifest_row(cls, row: dict[str, Any]) -> "CleaningShard":
        return cls(
            source_name=str(row["source_name"]),
            cleaning_source=str(row["cleaning_source"]),
            source_object_key=str(row["source_object_key"]),
            source_row_group_index=int(row["source_row_group_index"]),
            source_row_group_num_rows=optional_int(row.get("source_row_group_num_rows")),
            filters=normalize_filter_values(row.get("filters", {})),
            unified_shard_key=str(row["unified_shard_key"]),
            metrics_key=str(row["metrics_key"]),
            done_key=str(row["done_key"]),
            error_key=str(row["error_key"]),
        )

    @property
    def join_key(self) -> tuple[str, int, str]:
        return (self.source_object_key, self.source_row_group_index, self.source_name)


@dataclass(frozen=True)
class ExternalRunAvailability:
    cleaning_done_keys: frozenset[str]
    cleaning_error_keys: frozenset[str]
    cleaning_unified_keys: frozenset[str]
    lid_output_keys: frozenset[str]
    lid_error_keys: frozenset[str]


class UnifiedDataSubmissionAdapter(SubmissionAdapter):
    config: RecipeConfig

    def pipeline_family(self) -> str:
        return "unified_data"

    def build_new_run_manifest(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        dry_run: bool,
    ) -> SubmissionManifest:
        del dry_run
        object_store = resolve_manifest_object_store(artifact_store)
        rows = build_unified_data_manifest_rows(
            config=self.config,
            object_store=object_store,
            run_id=run_id,
        )
        return SubmissionManifest(
            rows=[task.to_dict() for task in rows],
            discovered_items=len(rows),
            uploaded_items=0,
            async_upload=None,
        )

    def load_resolved_recipe_mapping(self) -> dict[str, object]:
        return load_resolved_recipe_mapping(
            self.config_path,
            self.overrides,
            overlay_paths=self.overlay_paths,
        )

    def build_bootstrap_spec(self) -> BootstrapSpec:
        command = " && ".join(
            [
                "command -v uv >/dev/null || python3 -m pip install --break-system-packages uv",
                f"uv python install {shlex.quote(self.config.remote.python_version)}",
                (
                    "uv sync "
                    f"--python {shlex.quote(self.config.remote.python_version)} "
                    "--group unified_data --no-dev --frozen"
                ),
            ]
        )
        return BootstrapSpec(command=command)

    def build_invocation_spec(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        config_object_key: str,
        input_manifest_key: str,
        uploaded_items: int,
    ) -> RemoteInvocationSpec:
        command = shlex.join(
            [
                "uv",
                "run",
                "--python",
                self.config.remote.python_version,
                "--group",
                "unified_data",
                "python",
                "-m",
                "training_signal_processing.main",
                "unified-data-remote-job",
                "--run-id",
                run_id,
                "--config-object-key",
                config_object_key,
                "--input-manifest-key",
                input_manifest_key,
                "--uploaded-items",
                str(uploaded_items),
            ]
        )
        return RemoteInvocationSpec(command=command, env=artifact_store.build_remote_env())


def resolve_manifest_object_store(artifact_store: ArtifactStore) -> ObjectStore:
    as_object_store = getattr(artifact_store, "as_object_store", None)
    if callable(as_object_store):
        return as_object_store()
    raise ValueError("unified_data submission requires an artifact store with as_object_store().")


def build_unified_data_manifest_rows(
    *,
    config: RecipeConfig,
    object_store: ObjectStore,
    run_id: str,
) -> list[UnifiedDataPartTask]:
    lid_rows, cleaning_rows = load_external_manifest_rows(config, object_store)
    lid_shards = [LidShard.from_manifest_row(row) for row in lid_rows]
    cleaning_shards = [CleaningShard.from_manifest_row(row) for row in cleaning_rows]
    availability = collect_external_run_availability(config=config, object_store=object_store)
    validate_external_runs(
        config=config,
        object_store=object_store,
        lid_shards=lid_shards,
        availability=availability,
    )
    pairs = pair_external_shards(
        config=config,
        object_store=object_store,
        lid_shards=lid_shards,
        cleaning_shards=cleaning_shards,
        availability=availability,
    )
    tasks = build_part_tasks(
        config=config,
        run_id=run_id,
        shard_pairs=pairs,
    )
    if not tasks:
        raise ValueError(
            "No mergeable upstream row groups found for unified data export. "
            "Check LID parquet shards and source-cleaning done sentinels."
        )
    return tasks


def load_external_manifest_rows(
    config: RecipeConfig,
    object_store: ObjectStore,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lid_manifest_key = external_manifest_key(
        config.input.lid_metadata_output_prefix,
        config.input.lid_run_id,
    )
    cleaning_manifest_key = external_manifest_key(
        config.input.source_cleaning_output_prefix,
        config.input.source_cleaning_run_id,
    )
    if not object_store.exists(lid_manifest_key):
        raise ValueError(f"LID metadata manifest not found: {lid_manifest_key}")
    if not object_store.exists(cleaning_manifest_key):
        raise ValueError(f"Source cleaning manifest not found: {cleaning_manifest_key}")
    return object_store.read_jsonl(lid_manifest_key), object_store.read_jsonl(
        cleaning_manifest_key
    )


def validate_external_runs(
    *,
    config: RecipeConfig,
    object_store: ObjectStore,
    lid_shards: list[LidShard],
    availability: ExternalRunAvailability,
) -> None:
    if config.input.allow_partial_upstream:
        return
    if availability.cleaning_error_keys:
        raise ValueError(
            f"Source cleaning run has {len(availability.cleaning_error_keys)} "
            f"error sentinels; first={sorted(availability.cleaning_error_keys)[0]}"
        )
    if availability.lid_error_keys:
        raise ValueError(
            f"LID metadata run has {len(availability.lid_error_keys)} error sidecars; "
            f"first={sorted(availability.lid_error_keys)[0]}"
        )

    lid_output_keys = {shard.output_shard_key for shard in lid_shards}
    missing_lid = sorted(lid_output_keys - availability.lid_output_keys)
    if missing_lid:
        raise ValueError(
            f"LID metadata run is incomplete: {len(missing_lid)} missing shards; "
            f"first={missing_lid[0]}"
        )

    # This intentionally stays separate from the LID output check because source
    # cleaning completion is controlled by done sentinels, not parquet files.
    cleaning_manifest_rows = object_store.read_jsonl(
        external_manifest_key(
            config.input.source_cleaning_output_prefix,
            config.input.source_cleaning_run_id,
        )
    )
    expected_cleaning_done = {str(row["done_key"]) for row in cleaning_manifest_rows}
    missing_done = sorted(expected_cleaning_done - availability.cleaning_done_keys)
    if missing_done:
        raise ValueError(
            f"Source cleaning run is incomplete: "
            f"{len(availability.cleaning_done_keys)}/{len(expected_cleaning_done)} "
            "done sentinels; "
            f"first_missing={missing_done[0]}"
        )


def collect_external_run_availability(
    *,
    config: RecipeConfig,
    object_store: ObjectStore,
) -> ExternalRunAvailability:
    cleaning_root = join_s3_key(
        config.input.source_cleaning_output_prefix,
        config.input.source_cleaning_run_id,
    )
    lid_root = join_s3_key(config.input.lid_metadata_output_prefix, config.input.lid_run_id)
    return ExternalRunAvailability(
        cleaning_done_keys=frozenset(object_store.list_keys(join_s3_key(cleaning_root, "done"))),
        cleaning_error_keys=frozenset(
            key
            for key in object_store.list_keys(join_s3_key(cleaning_root, "errors"))
            if key.endswith(".error.json")
        ),
        cleaning_unified_keys=frozenset(
            key
            for key in object_store.list_keys(join_s3_key(cleaning_root, "unified"))
            if key.endswith(".parquet")
        ),
        lid_output_keys=frozenset(
            key
            for key in object_store.list_keys(join_s3_key(lid_root, "shards"))
            if key.endswith(".parquet")
        ),
        lid_error_keys=frozenset(
            key
            for key in object_store.list_keys(join_s3_key(lid_root, "shards"))
            if key.endswith(".error.json")
        ),
    )


def pair_external_shards(
    *,
    config: RecipeConfig,
    object_store: ObjectStore,
    lid_shards: list[LidShard],
    cleaning_shards: list[CleaningShard],
    availability: ExternalRunAvailability,
) -> list[tuple[LidShard, CleaningShard, int]]:
    lid_by_key = {shard.join_key: shard for shard in lid_shards}
    included_sources = selected_cleaning_sources(config)
    ready_pairs: list[tuple[LidShard, CleaningShard]] = []
    for cleaning in cleaning_shards:
        if included_sources and cleaning.cleaning_source not in included_sources:
            continue
        try:
            lid = lid_by_key[cleaning.join_key]
        except KeyError as exc:
            raise ValueError(
                "No matching LID shard for source cleaning shard: "
                f"{cleaning.join_key!r}"
            ) from exc
        if config.input.allow_partial_upstream and not shard_pair_is_ready(
            lid=lid,
            cleaning=cleaning,
            availability=availability,
        ):
            continue
        ready_pairs.append((lid, cleaning))
    return read_shard_pair_row_counts(object_store=object_store, ready_pairs=ready_pairs)


def selected_cleaning_sources(config: RecipeConfig) -> frozenset[str]:
    raw_value = config.input.include_cleaning_sources
    if not raw_value:
        return frozenset()
    return frozenset(item.strip() for item in raw_value.split(",") if item.strip())


def read_shard_pair_row_counts(
    *,
    object_store: ObjectStore,
    ready_pairs: list[tuple[LidShard, CleaningShard]],
) -> list[tuple[LidShard, CleaningShard, int]]:
    if len(ready_pairs) <= 16:
        return [
            (lid, cleaning, resolve_cleaning_row_count(object_store, cleaning))
            for lid, cleaning in ready_pairs
        ]
    with ThreadPoolExecutor(max_workers=32) as executor:
        row_counts = list(
            executor.map(
                lambda pair: resolve_cleaning_row_count(object_store, pair[1]),
                ready_pairs,
            )
        )
    return [
        (lid, cleaning, row_count)
        for (lid, cleaning), row_count in zip(ready_pairs, row_counts, strict=True)
    ]


def resolve_cleaning_row_count(object_store: ObjectStore, cleaning: CleaningShard) -> int:
    if not cleaning.filters and cleaning.source_row_group_num_rows is not None:
        return cleaning.source_row_group_num_rows
    return read_cleaning_row_count(object_store, cleaning)


def read_cleaning_row_count(object_store: ObjectStore, cleaning: CleaningShard) -> int:
    metric = object_store.read_json(cleaning.metrics_key)
    row_count = int(metric.get("filtered_row_count") or metric.get("row_count") or 0)
    if row_count < 0:
        raise ValueError(f"Invalid row count in cleaning metric: {cleaning.metrics_key}")
    return row_count


def shard_pair_is_ready(
    *,
    lid: LidShard,
    cleaning: CleaningShard,
    availability: ExternalRunAvailability,
) -> bool:
    if lid.output_shard_key not in availability.lid_output_keys:
        return False
    if lid_error_key(lid) in availability.lid_error_keys:
        return False
    if cleaning.done_key not in availability.cleaning_done_keys:
        return False
    if cleaning.error_key in availability.cleaning_error_keys:
        return False
    if cleaning.unified_shard_key not in availability.cleaning_unified_keys:
        return False
    return True


def lid_error_key(lid: LidShard) -> str:
    return lid.output_shard_key.removesuffix(".parquet") + ".error.json"


def build_part_tasks(
    *,
    config: RecipeConfig,
    run_id: str,
    shard_pairs: list[tuple[LidShard, CleaningShard, int]],
) -> list[UnifiedDataPartTask]:
    output_root_key = join_s3_key(config.r2.output_prefix, run_id)
    part_tasks: list[UnifiedDataPartTask] = []
    current_segments: list[MergeSegment] = []
    current_rows = 0

    def flush_part() -> None:
        nonlocal current_segments, current_rows
        if not current_segments:
            return
        part_index = len(part_tasks)
        keys = build_part_output_keys(output_root_key=output_root_key, part_index=part_index)
        part_tasks.append(
            UnifiedDataPartTask(
                part_index=part_index,
                part_key=keys["part_key"],
                metrics_key=keys["metrics_key"],
                done_key=keys["done_key"],
                error_key=keys["error_key"],
                expected_row_count=current_rows,
                rows_per_row_group=config.export.rows_per_row_group,
                tokenizer_encoding=config.export.tokenizer_encoding,
                tokenizer_threads=config.export.tokenizer_threads,
                merge_engine=config.export.merge_engine,
                duckdb_threads=config.export.duckdb_threads,
                parquet_compression=config.export.parquet_compression,
                parquet_compression_level=config.export.parquet_compression_level,
                segments=tuple(current_segments),
            )
        )
        current_segments = []
        current_rows = 0

    for lid, cleaning, shard_row_count in shard_pairs:
        row_offset = 0
        remaining = shard_row_count
        while remaining > 0:
            capacity = config.export.rows_per_row_group - current_rows
            take = min(capacity, remaining)
            current_segments.append(
                MergeSegment(
                    source_name=cleaning.source_name,
                    cleaning_source=cleaning.cleaning_source,
                    source_object_key=cleaning.source_object_key,
                    source_row_group_index=cleaning.source_row_group_index,
                    row_offset=row_offset,
                    row_count=take,
                    lid_shard_key=lid.output_shard_key,
                    cleaning_unified_shard_key=cleaning.unified_shard_key,
                    cleaning_metrics_key=cleaning.metrics_key,
                )
            )
            current_rows += take
            row_offset += take
            remaining -= take
            if current_rows == config.export.rows_per_row_group:
                flush_part()
    flush_part()
    return part_tasks


def build_part_output_keys(*, output_root_key: str, part_index: int) -> dict[str, str]:
    part_name = f"part-{part_index:06d}"
    return {
        "part_key": join_s3_key(output_root_key, f"parts/{part_name}.parquet"),
        "metrics_key": join_s3_key(output_root_key, f"metrics/{part_name}.metrics.json"),
        "done_key": join_s3_key(output_root_key, f"done/{part_name}.done.json"),
        "error_key": join_s3_key(output_root_key, f"errors/{part_name}.error.json"),
    }


def external_manifest_key(output_prefix: str, run_id: str) -> str:
    if run_id:
        return join_s3_key(join_s3_key(output_prefix, run_id), "control/input_manifest.jsonl")
    return join_s3_key(output_prefix, "control/input_manifest.jsonl")


def summarize_part_manifest(tasks: list[UnifiedDataPartTask]) -> dict[str, Any]:
    source_counts: Counter[str] = Counter()
    rows_by_source: Counter[str] = Counter()
    for task in tasks:
        for segment in task.segments:
            source_counts[segment.cleaning_source] += 1
            rows_by_source[segment.cleaning_source] += segment.row_count
    return {
        "part_count": len(tasks),
        "row_count": sum(task.expected_row_count for task in tasks),
        "segment_count": sum(len(task.segments) for task in tasks),
        "segments_by_source": dict(sorted(source_counts.items())),
        "rows_by_source": dict(sorted(rows_by_source.items())),
    }


def optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def normalize_filter_values(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}
