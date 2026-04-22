from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

from ..models import PdfTask
from ..utils import join_s3_key, utc_isoformat
from .base import Batch
from .builtin import (
    BatchTransformOp,
    ExportMarkdownMapper,
    MarkerOcrMapper,
    SkipExistingFilter,
    SourcePreparationOp,
)

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.

"""
ADD NEW CONCRETE USER OPS TO THIS FILE.

Design contract:
- Define one concrete subclass per user-customized op.
- Set `op_name` so the class auto-registers on import.
- Inherit from `SourcePreparationOp`, `BatchTransformOp`, `SkipExistingFilter`,
  or `ExportMarkdownMapper` so the executor can infer the stage automatically.
- The user should only need to modify this file and the YAML recipe to add a new op.
"""


class IdentityPreviewOp(BatchTransformOp):
    """
    Minimal working example for local Ray op testing.

    Copy this class, rename `op_name`, and replace `process_batch` with your own logic.
    """

    op_name = "identity_preview"

    def process_batch(self, batch: Batch) -> Batch:
        return list(batch)


class PreparePdfDocumentOp(SourcePreparationOp):
    op_name = "prepare_pdf_document"

    def process_row(self, row: dict[str, object]) -> dict[str, object]:
        runtime = self.require_runtime()
        task = PdfTask.from_dict(row)
        markdown_name = Path(task.relative_path).with_suffix(".md").as_posix()
        return {
            "run_id": runtime.run_id,
            "source_r2_key": task.source_r2_key,
            "relative_path": task.relative_path,
            "source_size_bytes": task.source_size_bytes,
            "source_sha256": task.source_sha256,
            "markdown_r2_key": join_s3_key(runtime.output_root_key, f"markdown/{markdown_name}"),
            "status": "pending",
            "error_message": "",
            "started_at": "",
            "finished_at": "",
            "duration_sec": 0.0,
            "marker_exit_code": 0,
            "markdown_text": "",
        }


class SkipExistingDocumentsOp(SkipExistingFilter):
    op_name = "skip_existing"

    def keep_row(self, row: dict[str, object]) -> bool:
        runtime = self.require_runtime()
        if runtime.allow_overwrite:
            return True
        return str(row["source_r2_key"]) not in runtime.completed_source_keys


class MarkerOcrDocumentOp(MarkerOcrMapper):
    op_name = "marker_ocr"

    def __init__(self, name: str | None = None, **options: object) -> None:
        super().__init__(name=name, **options)
        self.converter = None

    def process_row(self, row: dict[str, object]) -> dict[str, object]:
        runtime = self.require_runtime()
        started_at = utc_isoformat()
        started_clock = perf_counter()
        try:
            pdf_bytes = runtime.get_object_store().read_bytes(str(row["source_r2_key"]))
            markdown_text = self.convert_pdf_bytes(pdf_bytes)
            status = "success"
            error_message = ""
            marker_exit_code = 0
        except Exception as exc:
            markdown_text = ""
            status = "failed"
            error_message = str(exc)
            marker_exit_code = 1
        finished_at = utc_isoformat()
        return {
            **row,
            "run_id": runtime.run_id,
            "status": status,
            "error_message": error_message,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": perf_counter() - started_clock,
            "marker_exit_code": marker_exit_code,
            "markdown_text": markdown_text,
        }

    def convert_pdf_bytes(self, pdf_bytes: bytes) -> str:
        converter = self.get_converter()
        with NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            handle.write(pdf_bytes)
            temp_path = Path(handle.name)
        try:
            rendered = converter(str(temp_path))
            from marker.output import text_from_rendered

            markdown_text, _, _ = text_from_rendered(rendered)
            return markdown_text
        finally:
            temp_path.unlink(missing_ok=True)

    def get_converter(self):  # type: ignore[no-untyped-def]
        if self.converter is not None:
            return self.converter
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        marker_options = dict(self.options)
        renderer = marker_options.pop("renderer", None)
        processor_list = marker_options.pop("processor_list", None)
        device = marker_options.pop("device", None)
        dtype = marker_options.pop("dtype", None)
        attention_implementation = marker_options.pop("attention_implementation", None)
        self.converter = PdfConverter(
            artifact_dict=create_model_dict(
                device=device,
                dtype=dtype,
                attention_implementation=attention_implementation,
            ),
            processor_list=processor_list,
            renderer=renderer,
            config=marker_options,
        )
        return self.converter


class ExportMarkdownResultOp(ExportMarkdownMapper):
    op_name = "export_markdown"

    def process_row(self, row: dict[str, object]) -> dict[str, object]:
        status = str(row["status"])
        if status != "success":
            return {**row, "markdown_text": ""}
        markdown_text = str(row.get("markdown_text", ""))
        if not markdown_text:
            raise ValueError("Successful OCR rows must include non-empty markdown_text.")
        if not str(row.get("markdown_r2_key", "")):
            raise ValueError("Successful OCR rows must include markdown_r2_key.")
        return dict(row)
