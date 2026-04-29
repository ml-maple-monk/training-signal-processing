from __future__ import annotations

from ...core.execution import (
    ObjectStorePipelineRuntimeAdapter,
    OutputCompletionTracker,
    RayExporter,
)
from ...core.models import (
    ExportBatchResult,
    RayConfig,
    RayTransformResources,
    RunArtifactLayout,
    RuntimeRunBindings,
)
from ...core.remote import build_remote_job_cli
from ...core.storage import R2ObjectStore
from ...core.utils import write_json_bytes
from ...ops.base import Op
from .config import build_recipe_config
from .models import (
    RecipeConfig,
    SourceCleaningRowGroupTask,
    SourceCleaningShardResult,
)
from .ops import row_group_source_key


class SourceCleaningExporter(RayExporter):
    def export_batch(self, batch_id: str, rows: list[dict[str, object]]) -> ExportBatchResult:
        output_keys: list[str] = []
        for row in rows:
            result = SourceCleaningShardResult.from_dict(row)
            if result.status == "success":
                metrics = result.metrics()
                self._put_bytes(result.metrics_key, write_json_bytes(metrics))
                self._put_bytes(
                    result.done_key,
                    write_json_bytes(
                        {
                            "status": "success",
                            "source_name": result.source_name,
                            "cleaning_source": result.cleaning_source,
                            "source_object_key": result.source_object_key,
                            "source_row_group_index": result.source_row_group_index,
                            "source_shard_key": result.source_shard_key,
                            "unified_shard_key": result.unified_shard_key,
                            "metrics_key": result.metrics_key,
                            "duration_sec": result.duration_sec,
                        }
                    ),
                )
                output_keys.extend(
                    [
                        result.source_shard_key,
                        result.unified_shard_key,
                        result.metrics_key,
                        result.done_key,
                    ]
                )
                continue
            self._put_bytes(
                result.error_key,
                write_json_bytes(
                    {
                        "status": result.status,
                        "source_name": result.source_name,
                        "cleaning_source": result.cleaning_source,
                        "source_object_key": result.source_object_key,
                        "source_row_group_index": result.source_row_group_index,
                        "error_message": result.error_message,
                        "duration_sec": result.duration_sec,
                    }
                ),
            )
            output_keys.append(result.error_key)
        return ExportBatchResult(
            batch_id=batch_id,
            row_count=len(rows),
            output_keys=output_keys,
        )


class SourceCleaningCompletionTracker(OutputCompletionTracker):
    def source_key_for_input(self, row: dict[str, object]) -> str:
        return row_group_source_key(SourceCleaningRowGroupTask.from_dict(row))

    def output_key_for_input(
        self,
        row: dict[str, object],
        artifact_layout: RunArtifactLayout,
    ) -> str:
        del artifact_layout
        return SourceCleaningRowGroupTask.from_dict(row).done_key

    def output_listing_prefix(self, artifact_layout: RunArtifactLayout) -> str:
        return f"{artifact_layout.output_root_key.rstrip('/')}/done"


def resolve_source_cleaning_transform_resources(
    config: RecipeConfig,
    op: Op,
    execution: RayConfig,
) -> RayTransformResources:
    if op.name != "clean_source_row_group":
        return RayTransformResources()
    return RayTransformResources(
        concurrency=execution.concurrency,
        num_cpus=config.cleaning.ray_num_cpus_per_worker,
    )


def build_adapter(
    config: RecipeConfig,
    bindings: RuntimeRunBindings,
    object_store: R2ObjectStore,
) -> ObjectStorePipelineRuntimeAdapter:
    return ObjectStorePipelineRuntimeAdapter(
        config=config,
        bindings=bindings,
        object_store=object_store,
        source_root_key=config.input.source_root_key,
        exporter_factory=SourceCleaningExporter,
        completion_tracker_factory=SourceCleaningCompletionTracker,
        transform_resources_resolver=lambda op, execution: (
            resolve_source_cleaning_transform_resources(config, op, execution)
        ),
    )


source_cleaning_remote_job_cli = build_remote_job_cli(
    recipe_loader=build_recipe_config,
    adapter_factory=build_adapter,
)
