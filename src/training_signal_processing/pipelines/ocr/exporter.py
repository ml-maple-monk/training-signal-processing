from __future__ import annotations

from pathlib import Path

from ...core.exporter import RayExporter
from ...core.models import ExportBatchResult, RunState
from ...core.storage import ObjectStore
from ...core.utils import join_s3_key
from .models import DocumentResult


class OcrMarkdownExporter(RayExporter):
    def __init__(self, object_store: ObjectStore) -> None:
        self.object_store = object_store

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

    def finalize_run(self, run_state: RunState) -> None:
        self.object_store.write_json(
            join_s3_key(run_state.output_root_key, "run.json"),
            run_state.to_dict(),
        )

    def cleanup_staged_pdf(self, staged_pdf_path: str) -> None:
        if not staged_pdf_path.strip():
            return
        try:
            Path(staged_pdf_path).unlink(missing_ok=True)
        except OSError:
            return
