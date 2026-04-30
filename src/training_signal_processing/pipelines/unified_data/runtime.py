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
from . import ops as _ops  # noqa: F401
from .config import build_recipe_config
from .models import RecipeConfig, UnifiedDataPartResult, UnifiedDataPartTask, part_source_key


class UnifiedDataExporter(RayExporter):
    def export_batch(self, batch_id: str, rows: list[dict[str, object]]) -> ExportBatchResult:
        output_keys: list[str] = []
        for row in rows:
            result = UnifiedDataPartResult.from_dict(row)
            if result.status == "success":
                metrics = result.metrics()
                self._put_bytes(result.metrics_key, write_json_bytes(metrics))
                self._put_bytes(
                    result.done_key,
                    write_json_bytes(
                        {
                            "status": "success",
                            "part_index": result.part_index,
                            "part_key": result.part_key,
                            "metrics_key": result.metrics_key,
                            "row_count": result.output_row_count,
                            "duration_sec": result.duration_sec,
                        }
                    ),
                )
                output_keys.extend([result.part_key, result.metrics_key, result.done_key])
                continue
            self._put_bytes(
                result.error_key,
                write_json_bytes(
                    {
                        "status": result.status,
                        "part_index": result.part_index,
                        "part_key": result.part_key,
                        "expected_row_count": result.expected_row_count,
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


class UnifiedDataCompletionTracker(OutputCompletionTracker):
    def source_key_for_input(self, row: dict[str, object]) -> str:
        return part_source_key(UnifiedDataPartTask.from_dict(row))

    def output_key_for_input(
        self,
        row: dict[str, object],
        artifact_layout: RunArtifactLayout,
    ) -> str:
        del artifact_layout
        return UnifiedDataPartTask.from_dict(row).done_key

    def output_listing_prefix(self, artifact_layout: RunArtifactLayout) -> str:
        return f"{artifact_layout.output_root_key.rstrip('/')}/done"


def resolve_unified_data_transform_resources(
    config: RecipeConfig,
    op: Op,
    execution: RayConfig,
) -> RayTransformResources:
    if op.name != "write_unified_data_part":
        return RayTransformResources()
    return RayTransformResources(
        concurrency=execution.concurrency,
        num_cpus=config.export.ray_num_cpus_per_worker,
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
        exporter_factory=UnifiedDataExporter,
        completion_tracker_factory=UnifiedDataCompletionTracker,
        transform_resources_resolver=lambda op, execution: (
            resolve_unified_data_transform_resources(config, op, execution)
        ),
    )


unified_data_remote_job_cli = build_remote_job_cli(
    recipe_loader=build_recipe_config,
    adapter_factory=build_adapter,
)
