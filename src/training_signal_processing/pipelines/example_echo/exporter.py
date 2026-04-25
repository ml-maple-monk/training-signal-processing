from __future__ import annotations

import json

from ...core.exporter import RayExporter
from ...core.models import ExportBatchResult, RunState
from ...core.storage import ObjectStore
from ...core.utils import join_s3_key
from .models import EchoResult


class EchoExporter(RayExporter):
    def __init__(self, object_store: ObjectStore) -> None:
        self.object_store = object_store

    def export_batch(self, batch_id: str, rows: list[dict[str, object]]) -> ExportBatchResult:
        output_keys: list[str] = []
        for row in rows:
            result = EchoResult.from_dict(row)
            if result.status != "success":
                continue
            payload = {
                "source_id": result.source_id,
                "message": result.message,
                "echoed_at": result.echoed_at,
            }
            self._put_bytes(
                result.output_r2_key,
                json.dumps(payload, sort_keys=True).encode("utf-8"),
            )
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
