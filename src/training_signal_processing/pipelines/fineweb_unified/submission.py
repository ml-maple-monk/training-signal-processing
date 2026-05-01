from __future__ import annotations

import os
import shlex
from typing import Any

from ...core.submission import (
    ArtifactStore,
    BootstrapSpec,
    RemoteInvocationSpec,
    SubmissionAdapter,
    SubmissionManifest,
)
from ...core.utils import join_s3_key
from .config import load_resolved_recipe_mapping
from .models import FineWebPartTask, RecipeConfig
from .ops import build_month_byte_quotas, discover_fineweb_months, split_month_quota


class FineWebUnifiedSubmissionAdapter(SubmissionAdapter):
    config: RecipeConfig

    def pipeline_family(self) -> str:
        return "fineweb_unified"

    def build_new_run_manifest(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        dry_run: bool,
    ) -> SubmissionManifest:
        del artifact_store, dry_run
        rows = build_fineweb_unified_manifest_rows(config=self.config, run_id=run_id)
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
                    "--group fineweb_unified --no-dev --frozen"
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
                "fineweb_unified",
                "python",
                "-m",
                "training_signal_processing.main",
                "fineweb-unified-remote-job",
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
        hf_token = os.environ.get(self.config.input.hf_token_env_var, "")
        if hf_token:
            env[self.config.input.hf_token_env_var] = hf_token
        hub_token = os.environ.get("HUGGINGFACE_HUB_TOKEN", "")
        if hub_token and "HUGGINGFACE_HUB_TOKEN" not in env:
            env["HUGGINGFACE_HUB_TOKEN"] = hub_token
        return RemoteInvocationSpec(command=command, env=env)


def build_fineweb_unified_manifest_rows(
    *,
    config: RecipeConfig,
    run_id: str,
) -> list[FineWebPartTask]:
    months = discover_fineweb_months(
        dataset_name=config.input.dataset_name,
        dataset_configs=config.input.dataset_configs,
        split=config.input.split,
        sample_rows_per_config=config.input.discovery_sample_rows_per_config,
        shuffle_seed=config.input.shuffle_seed,
        shuffle_buffer_size=config.input.shuffle_buffer_size,
        hf_token_env_var=config.input.hf_token_env_var,
    )
    month_quotas = build_month_byte_quotas(months=months, byte_cap=config.export.byte_cap)
    output_root_key = join_s3_key(config.r2.output_prefix, run_id)
    tasks: list[FineWebPartTask] = []
    for month in sorted(month_quotas):
        part_quotas = split_month_quota(
            quota=month_quotas[month],
            part_target_bytes=config.export.part_target_bytes,
        )
        for month_part_index, byte_quota in enumerate(part_quotas):
            part_index = len(tasks)
            keys = build_part_output_keys(
                output_root_key=output_root_key,
                month=month,
                part_index=part_index,
            )
            tasks.append(
                FineWebPartTask(
                    part_index=part_index,
                    month=month,
                    month_part_index=month_part_index,
                    month_part_count=len(part_quotas),
                    byte_quota=byte_quota,
                    dataset_name=config.input.dataset_name,
                    dataset_configs=tuple(config.input.dataset_configs),
                    split=config.input.split,
                    source_name=config.input.source_name,
                    cleaning_source=config.input.cleaning_source,
                    text_column=config.input.text_column,
                    hf_token_env_var=config.input.hf_token_env_var,
                    shuffle_seed=config.input.shuffle_seed,
                    shuffle_buffer_size=config.input.shuffle_buffer_size,
                    stream_shards_per_config=config.input.stream_shards_per_config,
                    enforce_month_filter=config.input.enforce_month_filter,
                    rows_per_row_group=config.export.rows_per_row_group,
                    write_batch_rows=config.export.write_batch_rows,
                    compute_exact_token_counts=config.export.compute_exact_token_counts,
                    tokenizer_encoding=config.export.tokenizer_encoding,
                    tokenizer_threads=config.export.tokenizer_threads,
                    parquet_compression=config.export.parquet_compression,
                    parquet_compression_level=config.export.parquet_compression_level,
                    **keys,
                )
            )
    if sum(task.byte_quota for task in tasks) > config.export.byte_cap:
        raise ValueError("FineWeb manifest byte quotas exceed export.byte_cap.")
    return tasks


def build_part_output_keys(
    *,
    output_root_key: str,
    month: str,
    part_index: int,
) -> dict[str, str]:
    part_name = f"part-{part_index:06d}"
    month_key = month.replace("-", "")
    return {
        "part_key": join_s3_key(output_root_key, f"parts/month={month_key}/{part_name}.parquet"),
        "metrics_key": join_s3_key(output_root_key, f"metrics/{part_name}.metrics.json"),
        "done_key": join_s3_key(output_root_key, f"done/{part_name}.done.json"),
        "error_key": join_s3_key(output_root_key, f"errors/{part_name}.error.json"),
    }


def summarize_fineweb_manifest(tasks: list[FineWebPartTask]) -> dict[str, Any]:
    return {
        "part_count": len(tasks),
        "byte_quota": sum(task.byte_quota for task in tasks),
        "months": sorted({task.month for task in tasks}),
        "parts_by_month": {
            month: sum(1 for task in tasks if task.month == month)
            for month in sorted({task.month for task in tasks})
        },
    }
