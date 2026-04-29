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
from .models import RecipeConfig, SourceAccountingResult, SourceSpec, render_markdown_table
from .ops import build_source_row_r2_key


class SourceAccountingMarkdownExporter(RayExporter):
    def __init__(self, object_store) -> None:  # type: ignore[no-untyped-def]
        super().__init__(object_store)
        self.results: list[SourceAccountingResult] = []
        self.table_key = ""

    def export_batch(self, batch_id: str, rows: list[dict[str, object]]) -> ExportBatchResult:
        output_keys: list[str] = []
        for row in rows:
            result = SourceAccountingResult.from_dict(row)
            self.results.append(result)
            self.table_key = result.table_r2_key
            self._put_bytes(result.source_row_r2_key, write_json_bytes(result.to_dict()))
            output_keys.append(result.source_row_r2_key)
        if self.table_key:
            self.write_table()
            output_keys.append(self.table_key)
        return ExportBatchResult(
            batch_id=batch_id,
            row_count=len(rows),
            output_keys=output_keys,
        )

    def finalize_run(self, run_state) -> None:  # type: ignore[no-untyped-def]
        del run_state
        if self.table_key:
            self.write_table()

    def write_table(self) -> None:
        self._put_bytes(
            self.table_key,
            render_markdown_table(self.results).encode("utf-8"),
        )


class SourceAccountingCompletionTracker(OutputCompletionTracker):
    def source_key_for_input(self, row: dict[str, object]) -> str:
        return SourceSpec.from_dict(row).name

    def output_key_for_input(
        self,
        row: dict[str, object],
        artifact_layout: RunArtifactLayout,
    ) -> str:
        return build_source_row_r2_key(
            artifact_layout.output_root_key,
            SourceSpec.from_dict(row).name,
        )

    def output_listing_prefix(self, artifact_layout: RunArtifactLayout) -> str:
        return f"{artifact_layout.output_root_key.rstrip('/')}/sources"


def resolve_source_accounting_transform_resources(
    config: RecipeConfig,
    op: Op,
    execution: RayConfig,
) -> RayTransformResources:
    del config
    if op.name != "count_source_accounting_source":
        return RayTransformResources()
    return RayTransformResources(concurrency=execution.concurrency)


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
        exporter_factory=SourceAccountingMarkdownExporter,
        completion_tracker_factory=SourceAccountingCompletionTracker,
        transform_resources_resolver=lambda op, execution: (
            resolve_source_accounting_transform_resources(config, op, execution)
        ),
    )


source_accounting_remote_job_cli = build_remote_job_cli(
    recipe_loader=build_recipe_config,
    adapter_factory=build_adapter,
)
