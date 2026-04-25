from __future__ import annotations

from pathlib import Path

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
from ...core.utils import join_s3_key
from ...ops.base import Op
from .config import build_recipe_config
from .models import DocumentResult, PdfTask, RecipeConfig
from .ops import build_markdown_r2_key


class OcrMarkdownExporter(RayExporter):
    def export_batch(self, batch_id: str, rows: list[dict[str, object]]) -> ExportBatchResult:
        output_keys: list[str] = []
        for row in rows:
            result = DocumentResult.from_dict(row)
            try:
                if result.status != "success":
                    continue
                if not result.markdown_r2_key.strip():
                    raise ValueError("Successful OCR rows must include markdown_r2_key.")
                if not result.markdown_text:
                    raise ValueError("Successful OCR rows must include markdown_text.")
                self._put_bytes(
                    result.markdown_r2_key,
                    result.markdown_text.encode("utf-8"),
                )
                output_keys.append(result.markdown_r2_key)
            finally:
                self.cleanup_staged_pdf(result.staged_pdf_path)
        return ExportBatchResult(
            batch_id=batch_id,
            row_count=len(rows),
            output_keys=output_keys,
        )

    def cleanup_staged_pdf(self, staged_pdf_path: str) -> None:
        if not staged_pdf_path.strip():
            return
        try:
            Path(staged_pdf_path).unlink(missing_ok=True)
        except OSError:
            return


class OcrCompletionTracker(OutputCompletionTracker):
    def source_key_for_input(self, row: dict[str, object]) -> str:
        return PdfTask.from_dict(row).source_r2_key

    def output_key_for_input(
        self,
        row: dict[str, object],
        artifact_layout: RunArtifactLayout,
    ) -> str:
        task = PdfTask.from_dict(row)
        return build_markdown_r2_key(artifact_layout.output_root_key, task.relative_path)

    def output_listing_prefix(self, artifact_layout: RunArtifactLayout) -> str:
        return join_s3_key(artifact_layout.output_root_key, "markdown")


def build_ocr_tracking_extra_params(
    config: RecipeConfig,
) -> dict[str, int | float | str | bool]:
    marker = config.ray.marker_ocr_resources
    return {
        "marker_ocr_num_gpus": marker.num_gpus if marker.num_gpus is not None else 0.0,
        "marker_ocr_num_cpus": marker.num_cpus if marker.num_cpus is not None else 0.0,
    }


def resolve_ocr_transform_resources(
    config: RecipeConfig,
    op: Op,
    execution: RayConfig,
) -> RayTransformResources:
    if op.name != "marker_ocr":
        return RayTransformResources()
    marker = config.ray.marker_ocr_resources
    return RayTransformResources(
        concurrency=execution.concurrency,
        num_gpus=marker.num_gpus,
        num_cpus=marker.num_cpus,
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
        source_root_key=config.input.raw_pdf_prefix,
        exporter_factory=OcrMarkdownExporter,
        completion_tracker_factory=OcrCompletionTracker,
        tracking_extra_params=build_ocr_tracking_extra_params(config),
        transform_resources_resolver=lambda op, execution: resolve_ocr_transform_resources(
            config,
            op,
            execution,
        ),
    )


ocr_remote_job_cli = build_remote_job_cli(
    recipe_loader=build_recipe_config,
    adapter_factory=build_adapter,
)
