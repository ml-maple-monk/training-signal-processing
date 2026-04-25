"""Example echo ops — minimal reference for EXTENDING.md.

Three ops cover the mandatory prepare → transform → export stages:

- `PrepareEchoOp`: takes an `EchoTask` input row and produces the canonical
  document row with all downstream fields initialized.
- `TimestampEchoOp`: annotates each row with `echoed_at = <utc>` and marks
  success. Pure in-memory work; no I/O.
- `ExportEchoOp`: gate op — validates rows and promotes status to `failed` if
  required fields are missing. The actual R2 write lives in
  `pipelines/example_echo/exporter.py` (run once per batch after Ray
  materializes).

Op registration is automatic via `Op.__init_subclass__` at `ops/base.py:22`.
The example pipeline package imports this module from its `__init__.py`, so
the concrete ops are registered before the recipe is validated.
"""

from __future__ import annotations

from typing import Any

from ...core.utils import join_s3_key, utc_isoformat
from ...ops.builtin import BatchTransformOp, RowWiseMapperOp, SourcePreparationOp
from .models import EchoTask


class PrepareEchoOp(SourcePreparationOp):
    op_name = "prepare_echo"

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        runtime = self.require_runtime()
        task = EchoTask.from_dict(row)
        output_r2_key = join_s3_key(
            join_s3_key(runtime.output_root_key, "outputs"),
            f"{task.source_id}.json",
        )
        return {
            "run_id": runtime.run_id,
            "source_id": task.source_id,
            "message": task.message,
            "echoed_at": "",
            "status": "pending",
            "error_message": "",
            "output_r2_key": output_r2_key,
            "started_at": "",
            "finished_at": "",
            "duration_sec": 0.0,
            "output_written": False,
        }


class TimestampEchoOp(BatchTransformOp):
    op_name = "timestamp_echo"

    def process_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for row in batch:
            started = utc_isoformat()
            annotated = dict(row)
            annotated["started_at"] = started
            try:
                now = utc_isoformat()
                annotated["echoed_at"] = now
                annotated["status"] = "success"
                annotated["finished_at"] = now
                annotated["duration_sec"] = 0.0
            except Exception as exc:  # pragma: no cover - trivial compute
                annotated["status"] = "failed"
                annotated["error_message"] = str(exc)
                annotated["finished_at"] = utc_isoformat()
            output.append(annotated)
        return output


class ExportEchoOp(RowWiseMapperOp):
    op_name = "export_echo"
    op_stage = "export"

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if row.get("status") != "success":
            return row
        if not str(row.get("output_r2_key", "")).strip():
            row["status"] = "failed"
            row["error_message"] = "missing output_r2_key"
        return row
