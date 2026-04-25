from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter, sleep

from ...core.models import ExecutionLogEvent
from ...core.storage import resolve_runtime_object_store


class MarkerConversionError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def require_positive_int_option(options: dict[str, object], name: str) -> int:
    if name not in options:
        raise ValueError(f"marker_ocr option '{name}' is required.")
    value = int(options[name])
    if value <= 0:
        raise ValueError(f"marker_ocr option '{name}' must be positive.")
    return value


def require_positive_float_option(options: dict[str, object], name: str) -> float:
    if name not in options:
        raise ValueError(f"marker_ocr option '{name}' is required.")
    value = float(options[name])
    if value <= 0:
        raise ValueError(f"marker_ocr option '{name}' must be positive.")
    return value


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
    timeout_sec = require_positive_int_option(options, "timeout_sec")
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
    poll_interval_sec: float,
) -> None:
    deadline = perf_counter() + timeout_sec
    while perf_counter() < deadline:
        if object_store.exists(key):
            return
        remaining = deadline - perf_counter()
        if remaining <= 0:
            break
        sleep(min(poll_interval_sec, remaining))
    raise TimeoutError(f"OCR source object did not appear within {timeout_sec} seconds: {key}")


class MarkerRuntime:
    def __init__(self, *, op_name: str, options: dict[str, object]) -> None:
        self.op_name = op_name
        self.options = options

    def read_options(self) -> tuple[int, float]:
        return (
            require_positive_int_option(self.options, "timeout_sec"),
            require_positive_float_option(self.options, "source_object_poll_interval_sec"),
        )

    def build_diagnostics(self) -> dict[str, object]:
        return build_marker_diagnostics(self.options)

    def read_source_pdf(
        self,
        *,
        runtime: object,
        source_key: str,
        diagnostics: dict[str, object],
        timeout_sec: int,
        poll_interval_sec: float,
    ) -> bytes:
        object_store = resolve_runtime_object_store(runtime)
        self.log_event(
            runtime,
            code="ocr.pdf.read.start",
            message="Starting PDF read for OCR.",
            details={"source_r2_key": source_key, "diagnostics": diagnostics},
        )
        wait_for_source_object(
            object_store,
            key=source_key,
            timeout_sec=timeout_sec,
            poll_interval_sec=poll_interval_sec,
        )
        diagnostics["source_object_ready"] = True
        pdf_bytes = object_store.read_bytes(source_key)
        diagnostics["pdf_bytes_loaded"] = len(pdf_bytes)
        return pdf_bytes

    def convert_source_pdf(
        self,
        *,
        runtime: object,
        pdf_bytes: bytes,
        diagnostics: dict[str, object],
        timeout_sec: int,
        started_clock: float,
    ) -> tuple[str, dict[str, object]]:
        staged_pdf_path = stage_pdf_bytes_for_ocr(pdf_bytes)
        diagnostics["staged_pdf_path"] = str(staged_pdf_path)
        try:
            self.log_event(
                runtime,
                code="ocr.converter.init.start",
                message="Starting OCR converter process.",
                details={"diagnostics": diagnostics},
            )
            markdown_text, conversion_diagnostics = self.convert_pdf_file(
                staged_pdf_path,
                timeout_sec=self.remaining_timeout_sec(timeout_sec, started_clock),
            )
            self.log_event(
                runtime,
                code="ocr.converter.init.finish",
                message="OCR converter process completed successfully.",
                details={"diagnostics": {**diagnostics, **conversion_diagnostics}},
            )
            return markdown_text, conversion_diagnostics
        finally:
            staged_pdf_path.unlink(missing_ok=True)

    def remaining_timeout_sec(self, timeout_sec: int, started_clock: float) -> int:
        elapsed_sec = perf_counter() - started_clock
        return max(int(timeout_sec - elapsed_sec), 1)

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

    def log_event(
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
