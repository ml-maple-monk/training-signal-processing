from __future__ import annotations

from ...core.models import ExportBatchResult, RunState
from ...core.utils import join_s3_key
from ...runtime.exporter import RayExporter
from ...storage.object_store import ObjectStore
from .models import TokenizedShardResult


class TokenJsonlExporter(RayExporter):
    def __init__(self, object_store: ObjectStore) -> None:
        self.object_store = object_store

    def export_batch(self, batch_id: str, rows: list[dict[str, object]]) -> ExportBatchResult:
        output_keys: list[str] = []
        for row in rows:
            result = TokenizedShardResult.from_dict(row)
            if result.status != "success" or not result.output_written:
                continue
            output_keys.append(result.output_r2_key)
        return ExportBatchResult(
            batch_id=batch_id,
            row_count=len(rows),
            output_keys=output_keys,
        )

    def finalize_run(self, run_state: RunState) -> None:
        self.object_store.write_json(
            join_s3_key(run_state.output_root_key, "run.json"),
            run_state.to_dict(),
        )
