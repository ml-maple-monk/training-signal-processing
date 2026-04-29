from __future__ import annotations

import shlex
from fnmatch import fnmatchcase

import pyarrow.parquet as pq

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
from .models import RecipeConfig, SourceCleaningRowGroupTask, SourceSpec, source_slug


class SourceCleaningSubmissionAdapter(SubmissionAdapter):
    config: RecipeConfig

    def pipeline_family(self) -> str:
        return "source_cleaning"

    def build_new_run_manifest(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        dry_run: bool,
    ) -> SubmissionManifest:
        del dry_run
        object_store = resolve_manifest_object_store(artifact_store)
        rows = build_row_group_manifest_rows(
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
                    "--group source_cleaning --no-dev --frozen"
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
                "source_cleaning",
                "python",
                "-m",
                "training_signal_processing.main",
                "source-cleaning-remote-job",
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
        env = artifact_store.build_remote_env()
        env["POLARS_MAX_THREADS"] = str(self.config.cleaning.polars_max_threads)
        return RemoteInvocationSpec(command=command, env=env)


def resolve_manifest_object_store(artifact_store: ArtifactStore) -> ObjectStore:
    as_object_store = getattr(artifact_store, "as_object_store", None)
    if callable(as_object_store):
        return as_object_store()
    raise ValueError(
        "source_cleaning submission requires an artifact store with as_object_store()."
    )


def build_row_group_manifest_rows(
    *,
    config: RecipeConfig,
    object_store: ObjectStore,
    run_id: str,
) -> list[SourceCleaningRowGroupTask]:
    task_groups: list[tuple[list[SourceCleaningRowGroupTask], int]] = []
    for source_order, source in enumerate(config.sources):
        source_keys = list_matching_keys(object_store, source.r2_relative_glob_path)
        if not source_keys:
            raise ValueError(f"No R2 objects matched: {source.r2_relative_glob_path}")
        source_tasks: list[SourceCleaningRowGroupTask] = []
        for source_key in source_keys:
            source_tasks.extend(
                build_source_row_group_tasks(
                    config=config,
                    object_store=object_store,
                    source=source,
                    source_order=source_order,
                    source_key=source_key,
                    run_id=run_id,
                )
            )
        task_groups.append((source_tasks, source.scheduling_weight))
    return weighted_interleave_row_group_tasks(task_groups)


def weighted_interleave_row_group_tasks(
    task_groups: list[tuple[list[SourceCleaningRowGroupTask], int]],
) -> list[SourceCleaningRowGroupTask]:
    interleaved: list[SourceCleaningRowGroupTask] = []
    pending_groups = [
        (list(group), max(1, weight)) for group, weight in task_groups if group
    ]
    while pending_groups:
        next_groups: list[tuple[list[SourceCleaningRowGroupTask], int]] = []
        for group, weight in pending_groups:
            for _ in range(min(weight, len(group))):
                interleaved.append(group.pop(0))
            if group:
                next_groups.append((group, weight))
        pending_groups = next_groups
    return interleaved


def build_source_row_group_tasks(
    *,
    config: RecipeConfig,
    object_store: ObjectStore,
    source: SourceSpec,
    source_order: int,
    source_key: str,
    run_id: str,
) -> list[SourceCleaningRowGroupTask]:
    parquet_file = pq.ParquetFile(
        f"{object_store.bucket}/{source_key}",
        filesystem=object_store.build_pyarrow_filesystem(),
    )
    validate_source_columns(
        source=source,
        source_key=source_key,
        schema_names=parquet_file.schema_arrow.names,
    )
    output_root_key = join_s3_key(config.r2.output_prefix, run_id)
    tasks: list[SourceCleaningRowGroupTask] = []
    start_index = 0
    for row_group_index in range(parquet_file.num_row_groups):
        row_count = parquet_file.metadata.row_group(row_group_index).num_rows
        keys = build_row_group_output_keys(
            output_root_key=output_root_key,
            source=source.name,
            source_key=source_key,
            row_group_index=row_group_index,
        )
        tasks.append(
            SourceCleaningRowGroupTask(
                source_order=source_order,
                source_name=source.name,
                source_scheduling_weight=source.scheduling_weight,
                cleaning_source=source.cleaning_source,
                source_bucket=object_store.bucket,
                source_object_key=source_key,
                source_parquet_url=object_store.make_url(source_key),
                source_row_group_index=row_group_index,
                source_row_group_start_index=start_index,
                source_row_group_num_rows=row_count,
                text_column=source.text_column,
                filters=source.filters,
                source_shard_key=keys["source_shard_key"],
                unified_shard_key=keys["unified_shard_key"],
                metrics_key=keys["metrics_key"],
                done_key=keys["done_key"],
                error_key=keys["error_key"],
            )
        )
        start_index += row_count
    return tasks


def validate_source_columns(
    *,
    source: SourceSpec,
    source_key: str,
    schema_names: list[str],
) -> None:
    schema = set(schema_names)
    if source.text_column not in schema:
        raise ValueError(
            f"Parquet source '{source_key}' is missing text column '{source.text_column}'."
        )
    missing_filters = sorted(set(source.filters) - schema)
    if missing_filters:
        raise ValueError(
            f"Parquet source '{source_key}' is missing filter columns: "
            + ", ".join(missing_filters)
        )


def build_row_group_output_keys(
    *,
    output_root_key: str,
    source: str,
    source_key: str,
    row_group_index: int,
) -> dict[str, str]:
    source_value = source_slug(source)
    source_file_slug = source_slug(source_key.rsplit("/", 1)[-1].removesuffix(".parquet"))
    shard_name = f"{source_file_slug}.rg{row_group_index:05d}"
    return {
        "source_shard_key": join_s3_key(
            output_root_key,
            f"source_shards/{source_value}/{shard_name}.parquet",
        ),
        "unified_shard_key": join_s3_key(
            output_root_key,
            f"unified/source={source_value}/{shard_name}.parquet",
        ),
        "metrics_key": join_s3_key(
            output_root_key,
            f"metrics/{source_value}/{shard_name}.metrics.json",
        ),
        "done_key": join_s3_key(
            output_root_key,
            f"done/{source_value}/{shard_name}.done.json",
        ),
        "error_key": join_s3_key(
            output_root_key,
            f"errors/{source_value}/{shard_name}.error.json",
        ),
    }


def list_matching_keys(object_store: ObjectStore, r2_relative_glob_path: str) -> list[str]:
    prefix = glob_listing_prefix(r2_relative_glob_path)
    keys = object_store.list_keys(prefix)
    return sorted(key for key in keys if fnmatchcase(key, r2_relative_glob_path))


def glob_listing_prefix(pattern: str) -> str:
    wildcard_positions = [
        position
        for token in ("*", "?", "[")
        if (position := pattern.find(token)) != -1
    ]
    if not wildcard_positions:
        slash_index = pattern.rfind("/")
        return pattern[: slash_index + 1] if slash_index >= 0 else pattern
    prefix = pattern[: min(wildcard_positions)]
    slash_index = prefix.rfind("/")
    return prefix[: slash_index + 1] if slash_index >= 0 else ""
