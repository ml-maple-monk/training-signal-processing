from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import DocumentResult, ExportBatchResult, RecipeConfig, RunState
from ..ops.base import Batch
from ..storage import ObjectStore
from ..utils import join_s3_key, write_json_bytes

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


class Exporter(ABC):
    @abstractmethod
    def export_batch(self, batch_id: str, rows: Batch) -> ExportBatchResult:
        raise NotImplementedError

    @abstractmethod
    def finalize_run(self, run_state: RunState) -> None:
        raise NotImplementedError


class RayExporter(Exporter):
    """Ray-only exporter contract for explicit batch materialization."""


class ObjectStoreMarkdownExporter(RayExporter):
    def __init__(
        self,
        config: RecipeConfig,
        object_store: ObjectStore,
        run_id: str,
        allow_overwrite: bool,
    ) -> None:
        self.config = config
        self.object_store = object_store
        self.run_id = run_id
        self.allow_overwrite = allow_overwrite

    def export_batch(self, batch_id: str, rows: Batch) -> ExportBatchResult:
        output_keys: list[str] = []
        for row in rows:
            result = DocumentResult.from_dict(row)
            if result.status != "success":
                continue
            if not result.markdown_r2_key:
                raise ValueError(
                    f"Result row for '{result.relative_path}' is missing markdown_r2_key."
                )
            if self.object_store.exists(result.markdown_r2_key) and not self.allow_overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing markdown object: {result.markdown_r2_key}"
                )
            self.object_store.write_bytes(
                result.markdown_r2_key,
                result.markdown_text.encode("utf-8"),
            )
            output_keys.append(result.markdown_r2_key)
        return ExportBatchResult(batch_id=batch_id, row_count=len(rows), output_keys=output_keys)

    def finalize_run(self, run_state: RunState) -> None:
        run_json_key = join_s3_key(run_state.output_prefix, "run.json")
        self.object_store.write_bytes(run_json_key, write_json_bytes(run_state.to_dict()))
