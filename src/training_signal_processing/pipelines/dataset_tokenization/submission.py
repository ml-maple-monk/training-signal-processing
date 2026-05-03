from __future__ import annotations

import hashlib
import shlex
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
from .models import DatasetTokenizationTask, RecipeConfig


@dataclass(frozen=True)
class FineWebAvailability:
    done_part_keys: tuple[str, ...]
    error_only_part_keys: tuple[str, ...]
    orphan_part_keys: tuple[str, ...]
    stale_error_keys: tuple[str, ...]


class DatasetTokenizationSubmissionAdapter(SubmissionAdapter):
    config: RecipeConfig

    def pipeline_family(self) -> str:
        return "dataset_tokenization"

    def build_new_run_manifest(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        dry_run: bool,
    ) -> SubmissionManifest:
        tokenizer_path = Path(self.config.tokenizer.json_path)
        tokenizer_sha = sha256_file(tokenizer_path)
        output_root_key = join_s3_key(self.config.r2.output_prefix, run_id)
        tokenizer_object_key = join_s3_key(output_root_key, "control/tokenizer.json")
        if not dry_run:
            artifact_store.upload_file(tokenizer_path, tokenizer_object_key)
        object_store = resolve_manifest_object_store(artifact_store)
        source_parts, fineweb_availability = discover_dataset_tokenization_inputs(
            config=self.config,
            object_store=object_store,
        )
        tasks = build_dataset_tokenization_manifest_rows(
            config=self.config,
            run_id=run_id,
            source_parts=source_parts,
            tokenizer_object_key=tokenizer_object_key,
            tokenizer_json_sha256=tokenizer_sha,
        )
        if not dry_run:
            artifact_store.write_json(
                join_s3_key(output_root_key, "control/manifest_summary.json"),
                summarize_dataset_tokenization_manifest(
                    tasks,
                    fineweb_availability=fineweb_availability,
                ),
            )
        return SubmissionManifest(
            rows=[task.to_dict() for task in tasks],
            discovered_items=len(tasks),
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
                    "--group dataset_tokenization --no-dev --frozen"
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
                "dataset_tokenization",
                "python",
                "-m",
                "training_signal_processing.main",
                "dataset-tokenization-remote-job",
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
    raise ValueError(
        "dataset_tokenization submission requires an artifact store with as_object_store()."
    )


def build_dataset_tokenization_manifest_rows(
    *,
    config: RecipeConfig,
    run_id: str,
    source_parts: list[tuple[str, str]] | None = None,
    tokenizer_object_key: str,
    tokenizer_json_sha256: str,
) -> list[DatasetTokenizationTask]:
    output_root_key = join_s3_key(config.r2.output_prefix, run_id)
    if source_parts is None:
        raise ValueError("source_parts must be provided by dataset tokenization discovery.")
    if not source_parts:
        raise ValueError("Dataset tokenization manifest discovered no input parquet parts.")

    tasks: list[DatasetTokenizationTask] = []
    for task_index, (source_group, source_part_key) in enumerate(source_parts):
        keys = build_part_output_keys(output_root_key=output_root_key, task_index=task_index)
        tasks.append(
            DatasetTokenizationTask(
                task_index=task_index,
                source_group=source_group,
                source_part_key=source_part_key,
                text_column=config.input.text_column,
                dropped_column=config.input.dropped_column,
                tokenizer_name=config.tokenizer.name,
                tokenizer_object_key=tokenizer_object_key,
                tokenizer_json_sha256=tokenizer_json_sha256,
                read_batch_rows=config.export.read_batch_rows,
                rows_per_row_group=config.export.rows_per_row_group,
                parquet_compression=config.export.parquet_compression,
                parquet_compression_level=config.export.parquet_compression_level,
                **keys,
            )
        )
    return tasks


def discover_dataset_tokenization_inputs(
    *,
    config: RecipeConfig,
    object_store: ObjectStore,
) -> tuple[list[tuple[str, str]], FineWebAvailability]:
    source_parts = [
        ("final", key)
        for key in list_parquet_keys(
            object_store,
            prefix=config.input.final_parts_prefix,
        )
    ]
    fineweb = collect_fineweb_availability(
        object_store=object_store,
        run_root=config.input.fineweb_run_root,
    )
    source_parts.extend(("fineweb", key) for key in fineweb.done_part_keys)
    return source_parts, fineweb


def collect_fineweb_availability(
    *,
    object_store: ObjectStore,
    run_root: str,
) -> FineWebAvailability:
    root = run_root.strip().strip("/")
    done_keys = object_store.list_keys(join_s3_key(root, "done"))
    error_keys = object_store.list_keys(join_s3_key(root, "errors"))
    parquet_keys = list_parquet_keys(object_store, prefix=join_s3_key(root, "parts"))

    done_part_keys: set[str] = set()
    done_stems: set[str] = set()
    done_payloads = read_json_objects_concurrently(object_store, done_keys)
    for done_key, done in done_payloads:
        part_key = str(done.get("part_key") or "").strip()
        if part_key:
            done_part_keys.add(part_key)
        done_stems.add(part_stem(done_key))

    error_part_keys: set[str] = set()
    stale_error_keys: list[str] = []
    error_payloads = read_json_objects_concurrently(object_store, error_keys)
    for error_key, error in error_payloads:
        part_key = str(error.get("part_key") or "").strip()
        if part_key:
            error_part_keys.add(part_key)
        if part_stem(error_key) in done_stems:
            stale_error_keys.append(error_key)

    error_only_part_keys = sorted(error_part_keys - done_part_keys)
    orphan_part_keys = sorted(
        key for key in parquet_keys if key not in done_part_keys and key not in error_part_keys
    )
    return FineWebAvailability(
        done_part_keys=tuple(sorted(done_part_keys)),
        error_only_part_keys=tuple(error_only_part_keys),
        orphan_part_keys=tuple(orphan_part_keys),
        stale_error_keys=tuple(sorted(stale_error_keys)),
    )


def read_json_objects_concurrently(
    object_store: ObjectStore,
    keys: list[str],
) -> list[tuple[str, dict[str, object]]]:
    if not keys:
        return []
    worker_count = min(32, len(keys))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(lambda key: (key, object_store.read_json(key)), keys))


def list_parquet_keys(object_store: ObjectStore, *, prefix: str) -> list[str]:
    return sorted(
        key
        for key in object_store.list_keys(prefix.strip().lstrip("/"))
        if key.endswith(".parquet")
    )


def build_part_output_keys(
    *,
    output_root_key: str,
    task_index: int,
) -> dict[str, str]:
    part_name = f"part-{task_index:06d}"
    return {
        "part_key": join_s3_key(output_root_key, f"parts/{part_name}.parquet"),
        "metrics_key": join_s3_key(output_root_key, f"metrics/{part_name}.metrics.json"),
        "done_key": join_s3_key(output_root_key, f"done/{part_name}.done.json"),
        "error_key": join_s3_key(output_root_key, f"errors/{part_name}.error.json"),
    }


def summarize_dataset_tokenization_manifest(
    tasks: list[DatasetTokenizationTask],
    *,
    fineweb_availability: FineWebAvailability | None = None,
) -> dict[str, Any]:
    source_groups = sorted({task.source_group for task in tasks})
    summary = {
        "part_count": len(tasks),
        "source_groups": source_groups,
        "parts_by_source_group": {
            source_group: sum(1 for task in tasks if task.source_group == source_group)
            for source_group in source_groups
        },
        "tokenizer_name": tasks[0].tokenizer_name if tasks else "",
        "tokenizer_json_sha256": tasks[0].tokenizer_json_sha256 if tasks else "",
    }
    if fineweb_availability is not None:
        summary["fineweb_excluded"] = {
            "error_only_part_count": len(fineweb_availability.error_only_part_keys),
            "orphan_part_count": len(fineweb_availability.orphan_part_keys),
            "stale_error_count": len(fineweb_availability.stale_error_keys),
            "error_only_part_keys": list(fineweb_availability.error_only_part_keys),
            "orphan_part_keys": list(fineweb_availability.orphan_part_keys),
            "stale_error_keys": list(fineweb_availability.stale_error_keys),
        }
    return summary


def part_stem(key: str) -> str:
    name = PurePosixPath(key).name
    for suffix in (".done.json", ".error.json", ".metrics.json", ".parquet"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"Tokenizer JSON file not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()
