from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import perf_counter

from ...core.utils import join_s3_key, utc_isoformat
from ...ops.builtin import (
    MarkerOcrMapper,
    SkipExistingFilter,
    SourcePreparationOp,
)
from .marker_runtime import MarkerRuntime
from .models import DocumentResult, PdfTask

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.

"""
ADD NEW CONCRETE OCR OPS TO THIS FILE OR TO ANOTHER MODULE IN THIS PACKAGE.

Design contract:
- Define one concrete subclass per OCR pipeline op.
- Set `op_name` so the class auto-registers on import.
- Inherit from `SourcePreparationOp`, `BatchTransformOp`, or `SkipExistingFilter`
  so the executor can infer the stage automatically.
- The pipeline owner should only need to modify this package and the YAML recipe
  to add a new op.
"""

def build_flat_markdown_name(relative_path: str) -> str:
    source_name = Path(relative_path).with_suffix(".md").name
    path_digest = sha256(relative_path.encode("utf-8")).hexdigest()[:16]
    return f"{path_digest}-{source_name}"


def build_markdown_r2_key(output_root_key: str, relative_path: str) -> str:
    return join_s3_key(output_root_key, f"markdown/{build_flat_markdown_name(relative_path)}")


class PreparePdfDocumentOp(SourcePreparationOp):
    op_name = "prepare_pdf_document"

    def process_row(self, row: dict[str, object]) -> dict[str, object]:
        runtime = self.require_runtime()
        task = PdfTask.from_dict(row)
        return DocumentResult.pending_from_task(
            task=task,
            run_id=runtime.run_id,
            markdown_r2_key=build_markdown_r2_key(
                runtime.output_root_key,
                task.relative_path,
            ),
        ).to_dict()


class SkipExistingDocumentsOp(SkipExistingFilter):
    op_name = "skip_existing"

    def keep_row(self, row: dict[str, object]) -> bool:
        runtime = self.require_runtime()
        if runtime.allow_overwrite:
            return True
        return str(row["source_r2_key"]) not in runtime.completed_source_keys


class MarkerOcrDocumentOp(MarkerOcrMapper):
    op_name = "marker_ocr"

    def process_row(self, row: dict[str, object]) -> dict[str, object]:
        runtime = self.require_runtime()
        marker_runtime = MarkerRuntime(op_name=self.op_name, options=dict(self.options))
        started_at = utc_isoformat()
        started_clock = perf_counter()
        timeout_sec, poll_interval_sec = marker_runtime.read_options()
        diagnostics = marker_runtime.build_diagnostics()
        source_key = str(row["source_r2_key"])
        pdf_bytes = marker_runtime.read_source_pdf(
            runtime=runtime,
            source_key=source_key,
            diagnostics=diagnostics,
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
        )
        markdown_text, conversion_diagnostics = marker_runtime.convert_source_pdf(
            runtime=runtime,
            pdf_bytes=pdf_bytes,
            diagnostics=diagnostics,
            timeout_sec=timeout_sec,
            started_clock=started_clock,
        )
        diagnostics.update(conversion_diagnostics)
        return DocumentResult.success_from_row(
            row=row,
            run_id=runtime.run_id,
            started_at=started_at,
            finished_at=utc_isoformat(),
            duration_sec=perf_counter() - started_clock,
            markdown_text=markdown_text,
            diagnostics=diagnostics,
        ).to_dict()
