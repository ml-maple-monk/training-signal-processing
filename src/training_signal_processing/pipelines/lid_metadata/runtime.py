from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

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
from .models import LidMetadataShardResult, ParquetRowGroupTask, RecipeConfig
from .ops import row_group_source_key

LINGUA_SPAN_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("start_index", pa.int64()),
            pa.field("end_index", pa.int64()),
            pa.field("language_label", pa.string()),
        ]
    )
)
MALAYA_SCORE_TYPE = pa.list_(
    pa.struct([pa.field("label", pa.string()), pa.field("score", pa.float64())])
)
MALAYA_WORD_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("word_index", pa.int64()),
            pa.field("start_index", pa.int64()),
            pa.field("end_index", pa.int64()),
            pa.field("token", pa.string()),
            pa.field("label", pa.string()),
        ]
    )
)
MALAYA_COUNT_TYPE = pa.list_(
    pa.struct([pa.field("label", pa.string()), pa.field("count", pa.int64())])
)

LID_METADATA_SCHEMA = pa.schema(
    [
        ("sample_uid", pa.string()),
        ("sample_uid_hash", pa.string()),
        ("source_name", pa.string()),
        ("source_bucket", pa.string()),
        ("source_object_key", pa.string()),
        ("source_parquet_url", pa.string()),
        ("source_row_group_index", pa.int64()),
        ("source_row_index", pa.int64()),
        ("row_index_in_row_group", pa.int64()),
        ("text_column", pa.string()),
        ("original_text_sha256", pa.string()),
        ("original_char_count", pa.int64()),
        ("cleaned_char_count", pa.int64()),
        ("cleaned_token_count", pa.int64()),
        ("reference_removed", pa.bool_()),
        ("reference_removal_method", pa.string()),
        ("removed_reference_char_count", pa.int64()),
        ("lingua_primary_language", pa.string()),
        ("lingua_spans", LINGUA_SPAN_TYPE),
        ("malaya_document_label", pa.string()),
        ("malaya_document_scores", MALAYA_SCORE_TYPE),
        ("malaya_word_detections", MALAYA_WORD_TYPE),
        ("malaya_word_label_counts", MALAYA_COUNT_TYPE),
        ("document_id", pa.string()),
        ("id", pa.string()),
        ("url", pa.string()),
        ("permalink", pa.string()),
        ("thread_url", pa.string()),
        ("subreddit", pa.string()),
        ("language", pa.string()),
        ("row_language_code", pa.string()),
    ]
)


class LidMetadataParquetExporter(RayExporter):
    def export_batch(self, batch_id: str, rows: list[dict[str, object]]) -> ExportBatchResult:
        output_keys: list[str] = []
        for row in rows:
            result = LidMetadataShardResult.from_dict(row)
            if result.status == "success":
                table = pa.Table.from_pylist(
                    result.records,
                    schema=LID_METADATA_SCHEMA,
                )
                sink = pa.BufferOutputStream()
                pq.write_table(table, sink)
                self._put_bytes(result.output_shard_key, sink.getvalue().to_pybytes())
                metrics_key = result.output_shard_key.removesuffix(".parquet") + ".metrics.json"
                self._put_bytes(metrics_key, write_json_bytes(build_result_metrics(result)))
                output_keys.append(result.output_shard_key)
                output_keys.append(metrics_key)
            else:
                error_key = result.output_shard_key.removesuffix(".parquet") + ".error.json"
                self._put_bytes(
                    error_key,
                    write_json_bytes(
                        {
                            "source_name": result.source_name,
                            "source_object_key": result.source_object_key,
                            "source_row_group_index": result.source_row_group_index,
                            "status": result.status,
                            "error_message": result.error_message,
                            "metrics": build_result_metrics(result),
                        }
                    ),
                )
                output_keys.append(error_key)
        return ExportBatchResult(
            batch_id=batch_id,
            row_count=len(rows),
            output_keys=output_keys,
        )


def build_result_metrics(result: LidMetadataShardResult) -> dict[str, object]:
    return {
        "source_name": result.source_name,
        "source_object_key": result.source_object_key,
        "source_row_group_index": result.source_row_group_index,
        "output_shard_key": result.output_shard_key,
        "status": result.status,
        "row_count": result.row_count,
        "success_count": result.success_count,
        "failed_count": result.failed_count,
        "cleaned_token_count": result.cleaned_token_count,
        "duration_sec": result.duration_sec,
        "tokens_per_sec": result.tokens_per_sec,
        "rows_per_sec": result.rows_per_sec,
        "experiment_name": result.experiment_name,
        "variant_name": result.variant_name,
        "inner_parallelism": result.inner_parallelism,
        "inner_workers": result.inner_workers,
        "row_batch_size": result.row_batch_size,
        "checkpoint_key": result.checkpoint_key,
        "parallelism_fallback_reason": result.parallelism_fallback_reason,
    }


class LidMetadataCompletionTracker(OutputCompletionTracker):
    def source_key_for_input(self, row: dict[str, object]) -> str:
        return row_group_source_key(ParquetRowGroupTask.from_dict(row))

    def output_key_for_input(
        self,
        row: dict[str, object],
        artifact_layout: RunArtifactLayout,
    ) -> str:
        del artifact_layout
        return ParquetRowGroupTask.from_dict(row).output_shard_key

    def output_listing_prefix(self, artifact_layout: RunArtifactLayout) -> str:
        return f"{artifact_layout.output_root_key.rstrip('/')}/shards"


def resolve_lid_metadata_transform_resources(
    config: RecipeConfig,
    op: Op,
    execution: RayConfig,
) -> RayTransformResources:
    del config
    if op.name != "detect_lid_metadata_row_group":
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
        exporter_factory=LidMetadataParquetExporter,
        completion_tracker_factory=LidMetadataCompletionTracker,
        transform_resources_resolver=lambda op, execution: (
            resolve_lid_metadata_transform_resources(config, op, execution)
        ),
    )


lid_metadata_remote_job_cli = build_remote_job_cli(
    recipe_loader=build_recipe_config,
    adapter_factory=build_adapter,
)
