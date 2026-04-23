from __future__ import annotations

import os

from ...core.models import (
    OpRuntimeContext,
    RayConfig,
    RayTransformResources,
    RunArtifactLayout,
    RuntimeRunBindings,
    RuntimeTrackingContext,
)
from ...core.utils import join_s3_key
from ...ops.base import Op
from ...ops.registry import OpRegistry, RegisteredOpRegistry
from ...runtime.executor import PipelineRuntimeAdapter
from ...runtime.exporter import Exporter
from ...runtime.observability import ExecutionLogger
from ...runtime.resume import ResumeLedger
from ...storage.object_store import R2ObjectStore
from .exporter import OcrMarkdownExporter
from .models import RecipeConfig
from .resume import OcrResumeLedger


class OcrPipelineRuntimeAdapter(PipelineRuntimeAdapter):
    def __init__(
        self,
        *,
        config: RecipeConfig,
        bindings: RuntimeRunBindings,
        object_store: R2ObjectStore,
    ) -> None:
        self.config = config
        self.bindings = bindings
        self.object_store = object_store

    def get_run_bindings(self) -> RuntimeRunBindings:
        return self.bindings

    def get_execution_config(self):  # type: ignore[override]
        return self.config.ray

    def get_tracking_context(self) -> RuntimeTrackingContext:
        return RuntimeTrackingContext(
            enabled=self.config.mlflow.enabled,
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", ""),
            experiment_name=self.config.mlflow.experiment_name,
            run_name=self.config.run_name,
            executor_type=self.config.ray.executor_type,
            batch_size=self.config.ray.batch_size,
            concurrency=self.config.ray.concurrency,
            target_num_blocks=self.config.ray.target_num_blocks,
            extra_params={
                "ocr_worker_num_gpus": self.config.ray.ocr_worker_num_gpus,
                "ocr_worker_num_cpus": self.config.ray.ocr_worker_num_cpus,
            },
        )

    def get_op_configs(self):
        return self.config.ops

    def get_artifact_layout(self) -> RunArtifactLayout:
        return RunArtifactLayout(
            source_root_key=self.config.r2.raw_pdf_prefix,
            output_root_key=join_s3_key(self.config.r2.output_prefix, self.bindings.run_id),
        )

    def load_input_rows(self) -> list[dict[str, object]]:
        return self.object_store.read_jsonl(self.bindings.input_manifest_key)

    def build_runtime_context(
        self,
        *,
        logger: ExecutionLogger,
        completed_item_keys: set[str],
    ) -> OpRuntimeContext:
        artifact_layout = self.get_artifact_layout()
        return OpRuntimeContext(
            config=self.config,
            run_id=self.bindings.run_id,
            object_store=self.object_store,
            output_root_key=artifact_layout.output_root_key,
            source_root_key=artifact_layout.source_root_key,
            completed_item_keys=completed_item_keys,
            allow_overwrite=self.bindings.allow_overwrite,
            logger=logger,
        )

    def build_op_registry(self, runtime_context: OpRuntimeContext) -> OpRegistry:
        return RegisteredOpRegistry(runtime_context=runtime_context)

    def build_exporter(self) -> Exporter:
        return OcrMarkdownExporter(self.object_store)

    def build_resume_ledger(self) -> ResumeLedger:
        return OcrResumeLedger(config=self.config, object_store=self.object_store)

    def resolve_completed_item_keys(self, completed_item_keys: set[str]) -> set[str]:
        if self.bindings.allow_overwrite:
            return set()
        return completed_item_keys

    def resolve_transform_resources(
        self,
        *,
        op: Op,
        execution: RayConfig,
    ) -> RayTransformResources:
        if op.name != "marker_ocr":
            return super().resolve_transform_resources(op=op, execution=execution)
        return RayTransformResources(
            concurrency=execution.concurrency,
            num_gpus=self.config.ray.ocr_worker_num_gpus,
            num_cpus=float(self.config.ray.ocr_worker_num_cpus),
        )
