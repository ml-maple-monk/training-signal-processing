from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter, sleep

from ..core.models import ExecutionLogEvent
from ..core.utils import join_s3_key, utc_isoformat
from ..ops.base import Batch
from ..ops.builtin import (
    BatchTransformOp,
    ExportMarkdownMapper,
    MarkerOcrMapper,
    SkipExistingFilter,
    SourcePreparationOp,
)
from ..pipelines.ocr.models import PdfTask

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.

"""
ADD NEW CONCRETE USER OPS TO THIS FILE OR TO ANOTHER MODULE IN THIS DIRECTORY.

Design contract:
- Define one concrete subclass per user-customized op.
- Set `op_name` so the class auto-registers on import.
- Inherit from `SourcePreparationOp`, `BatchTransformOp`, `SkipExistingFilter`,
  or `ExportMarkdownMapper` so the executor can infer the stage automatically.
- The user should only need to modify files in `custom_ops/` and the YAML recipe
  to add a new op.
"""

OCR_CONVERSION_TIMEOUT_SEC = 1800
SOURCE_OBJECT_POLL_INTERVAL_SEC = 2.0


class MarkerConversionError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def build_marker_diagnostics(options: dict[str, object]) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "device_option": options.get("device"),
        "dtype_option": options.get("dtype"),
        "attention_implementation_option": options.get("attention_implementation"),
        "force_ocr": options.get("force_ocr"),
        "mp_start_method": "spawn",
    }
    try:
        import torch

        diagnostics["torch_cuda_available"] = torch.cuda.is_available()
        diagnostics["torch_cuda_device_count"] = torch.cuda.device_count()
    except Exception as exc:
        diagnostics["torch_cuda_error"] = str(exc)
    return diagnostics


def get_marker_mp_context() -> mp.context.BaseContext:
    return mp.get_context("spawn")


def _run_marker_conversion(
    pdf_path: str,
    options: dict[str, object],
    result_sender: object,
) -> None:
    diagnostics = build_marker_diagnostics(options)
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        converter = PdfConverter(
            artifact_dict=create_model_dict(
                device=options.get("device"),
                dtype=options.get("dtype"),
                attention_implementation=options.get("attention_implementation"),
            ),
            processor_list=options.get("processor_list"),
            renderer=options.get("renderer"),
            config=options,
        )
        rendered = converter(pdf_path)
        markdown_text, _, _ = text_from_rendered(rendered)
        result_sender.send(
            {
                "status": "success",
                "markdown_text": markdown_text,
                "diagnostics": diagnostics,
            }
        )
    except Exception as exc:
        result_sender.send(
            {
                "status": "failed",
                "error_message": str(exc),
                "diagnostics": diagnostics,
            }
        )
    finally:
        close_sender = getattr(result_sender, "close", None)
        if callable(close_sender):
            close_sender()


def convert_pdf_path_with_timeout(
    pdf_path: Path,
    options: dict[str, object],
) -> tuple[str, dict[str, object]]:
    mp_context = get_marker_mp_context()
    result_receiver, result_sender = mp_context.Pipe(duplex=False)
    timeout_sec = int(options.get("timeout_sec", OCR_CONVERSION_TIMEOUT_SEC))
    process = mp_context.Process(
        target=_run_marker_conversion,
        args=(str(pdf_path), options, result_sender),
    )
    try:
        process.start()
        if not result_receiver.poll(timeout_sec):
            process.terminate()
            process.join(timeout=5)
            raise MarkerConversionError(
                f"Marker OCR conversion timed out after {timeout_sec} seconds.",
                diagnostics={"timeout_sec": timeout_sec},
            )
        try:
            payload = result_receiver.recv()
        except EOFError as exc:
            raise MarkerConversionError(
                "Marker OCR conversion exited without returning a result.",
            ) from exc
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        diagnostics = (
            dict(payload["diagnostics"])
            if isinstance(payload.get("diagnostics"), dict)
            else {}
        )
        if payload.get("status") != "success":
            raise MarkerConversionError(
                str(payload.get("error_message", "Marker OCR conversion failed.")),
                diagnostics=diagnostics,
            )
        return str(payload.get("markdown_text", "")), diagnostics
    finally:
        close_receiver = getattr(result_receiver, "close", None)
        if callable(close_receiver):
            close_receiver()
        close_sender = getattr(result_sender, "close", None)
        if callable(close_sender):
            close_sender()


def stage_pdf_bytes_for_ocr(pdf_bytes: bytes) -> Path:
    with NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(pdf_bytes)
        return Path(handle.name)


def convert_pdf_bytes_with_timeout(
    pdf_bytes: bytes,
    options: dict[str, object],
) -> tuple[str, dict[str, object]]:
    temp_path = stage_pdf_bytes_for_ocr(pdf_bytes)
    try:
        return convert_pdf_path_with_timeout(temp_path, options)
    finally:
        temp_path.unlink(missing_ok=True)


def wait_for_source_object(
    object_store: object,
    *,
    key: str,
    timeout_sec: int,
) -> None:
    deadline = perf_counter() + timeout_sec
    while perf_counter() < deadline:
        if object_store.exists(key):
            return
        remaining = deadline - perf_counter()
        if remaining <= 0:
            break
        sleep(min(SOURCE_OBJECT_POLL_INTERVAL_SEC, remaining))
    raise TimeoutError(f"OCR source object did not appear within {timeout_sec} seconds: {key}")


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
        return str(row["source_r2_key"]) not in runtime.completed_item_keys


class MarkerOcrDocumentOp(MarkerOcrMapper):
    op_name = "marker_ocr"

    def process_row(self, row: dict[str, object]) -> dict[str, object]:
        runtime = self.require_runtime()
        started_at = utc_isoformat()
        started_clock = perf_counter()
        diagnostics = build_marker_diagnostics(dict(self.options))
        timeout_sec = int(self.options.get("timeout_sec", OCR_CONVERSION_TIMEOUT_SEC))
        staged_pdf_path = ""
        try:
            object_store = runtime.get_object_store()
            self.log_runtime_event(
                runtime,
                code="ocr.pdf.read.start",
                message="Starting PDF read for OCR.",
                details={"source_r2_key": str(row["source_r2_key"]), "diagnostics": diagnostics},
            )
            wait_for_source_object(
                object_store,
                key=str(row["source_r2_key"]),
                timeout_sec=timeout_sec,
            )
            diagnostics["source_object_ready"] = True
            pdf_bytes = object_store.read_bytes(str(row["source_r2_key"]))
            diagnostics["pdf_bytes_loaded"] = len(pdf_bytes)
            staged_pdf_path = str(stage_pdf_bytes_for_ocr(pdf_bytes))
            diagnostics["staged_pdf_path"] = staged_pdf_path
            self.log_runtime_event(
                runtime,
                code="ocr.converter.init.start",
                message="Starting OCR converter process.",
                details={"diagnostics": diagnostics},
            )
            elapsed_sec = perf_counter() - started_clock
            remaining_timeout_sec = max(int(timeout_sec - elapsed_sec), 1)
            markdown_text, conversion_diagnostics = self.convert_pdf_file(
                Path(staged_pdf_path),
                timeout_sec=remaining_timeout_sec,
            )
            diagnostics.update(conversion_diagnostics)
            status = "success"
            error_message = ""
            marker_exit_code = 0
            self.log_runtime_event(
                runtime,
                code="ocr.converter.init.finish",
                message="OCR converter process completed successfully.",
                details={"diagnostics": diagnostics},
            )
        except Exception as exc:
            markdown_text = ""
            status = "failed"
            error_message = str(exc)
            marker_exit_code = 1
            diagnostics.update(getattr(exc, "diagnostics", {}))
            diagnostics["failure"] = error_message
            self.log_runtime_event(
                runtime,
                code="ocr.converter.failed",
                message="OCR converter process failed.",
                details={"diagnostics": diagnostics, "error": error_message},
            )
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
            "staged_pdf_path": staged_pdf_path,
            "diagnostics": diagnostics,
        }

    def convert_pdf_bytes(self, pdf_bytes: bytes) -> tuple[str, dict[str, object]]:
        return convert_pdf_bytes_with_timeout(pdf_bytes, dict(self.options))

    def convert_pdf_file(
        self,
        pdf_path: Path,
        *,
        timeout_sec: int,
    ) -> tuple[str, dict[str, object]]:
        return convert_pdf_path_with_timeout(
            pdf_path,
            {**dict(self.options), "timeout_sec": timeout_sec},
        )

    def log_runtime_event(
        self,
        runtime: object,
        *,
        code: str,
        message: str,
        details: dict[str, object],
    ) -> None:
        logger = getattr(runtime, "logger", None)
        run_id = getattr(runtime, "run_id", "")
        if logger is None or not run_id:
            return
        logger.log_event(
            ExecutionLogEvent(
                level="INFO",
                code=code,
                message=message,
                run_id=run_id,
                op_name=self.op_name,
                details=details,
            )
        )


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
