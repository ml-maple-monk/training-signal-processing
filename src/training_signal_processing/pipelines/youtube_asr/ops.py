from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from ...core.utils import utc_isoformat
from ...ops.base import Batch
from ...ops.builtin import (
    BatchTransformOp,
    ExportMarkdownMapper,
    SkipExistingFilter,
    SourcePreparationOp,
)
from .models import YoutubeMediaTask, YoutubeTranscriptResult


class PrepareYoutubeMediaOp(SourcePreparationOp):
    op_name = "prepare_youtube_media"

    def process_row(self, row: dict[str, object]) -> dict[str, object]:
        runtime = self.require_runtime()
        task = YoutubeMediaTask.from_dict(row)
        return {
            "run_id": runtime.run_id,
            "channel_id": task.channel_id,
            "channel_title": task.channel_title,
            "channel_url": task.channel_url,
            "video_id": task.video_id,
            "video_url": task.video_url,
            "video_title": task.video_title,
            "upload_date": task.upload_date,
            "source_media_r2_key": task.source_media_r2_key,
            "source_media_ext": task.source_media_ext,
            "source_media_size_bytes": task.source_media_size_bytes,
            "transcript_r2_key": task.transcript_r2_key,
            "status": "pending",
            "error_message": "",
            "transcript_text": "",
            "detected_language": "",
            "started_at": "",
            "finished_at": "",
            "duration_sec": 0.0,
            "output_written": False,
        }


class SkipExistingYoutubeMediaOp(SkipExistingFilter):
    op_name = "skip_existing_youtube_media"

    def keep_row(self, row: dict[str, object]) -> bool:
        runtime = self.require_runtime()
        if runtime.allow_overwrite:
            return True
        return str(row["source_media_r2_key"]) not in runtime.completed_item_keys


class Qwen3AsrVllmOp(BatchTransformOp):
    op_name = "qwen3_asr_vllm"
    _model = None

    def process_batch(self, batch: Batch) -> Batch:
        runtime = self.require_runtime()
        started_at = utc_isoformat()
        started_clock = perf_counter()
        try:
            processed = self.transcribe_batch(batch)
        except Exception as exc:
            finished_at = utc_isoformat()
            return [
                {
                    **row,
                    "status": "failed",
                    "error_message": str(exc),
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_sec": perf_counter() - started_clock,
                    "transcript_text": "",
                    "detected_language": "",
                }
                for row in batch
            ]
        finished_at = utc_isoformat()
        duration_sec = perf_counter() - started_clock
        rows: Batch = []
        for row, result in zip(batch, processed, strict=True):
            rows.append(
                {
                    **row,
                    "run_id": runtime.run_id,
                    "status": "success",
                    "error_message": "",
                    "transcript_text": str(getattr(result, "text", "")),
                    "detected_language": str(getattr(result, "language", "")),
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_sec": duration_sec,
                }
            )
        return rows

    def transcribe_batch(self, batch: Batch):  # type: ignore[no-untyped-def]
        runtime = self.require_runtime()
        object_store = runtime.get_object_store()
        model = self.get_model()
        max_size_bytes = int(runtime.config.asr.max_media_file_size_mb) * 1024 * 1024
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            media_paths: list[str] = []
            for row in batch:
                media_bytes = object_store.read_bytes(str(row["source_media_r2_key"]))
                if len(media_bytes) > max_size_bytes:
                    raise ValueError(
                        f"Media file exceeds configured size limit of "
                        f"{runtime.config.asr.max_media_file_size_mb} MB: "
                        f"{row['source_media_r2_key']}"
                    )
                extension = str(row.get("source_media_ext", "")).strip() or "bin"
                media_path = temp_dir / f"{row['video_id']}.{extension}"
                media_path.write_bytes(media_bytes)
                media_paths.append(str(media_path))
            language = runtime.config.asr.language or None
            return model.transcribe(audio=media_paths, language=language)

    def get_model(self):  # type: ignore[no-untyped-def]
        if self._model is not None:
            return self._model
        runtime = self.require_runtime()
        from qwen_asr import Qwen3ASRModel

        self._model = Qwen3ASRModel.LLM(
            model=runtime.config.asr.model_name,
            gpu_memory_utilization=runtime.config.asr.gpu_memory_utilization,
            max_inference_batch_size=runtime.config.asr.max_inference_batch_size,
            max_new_tokens=runtime.config.asr.max_new_tokens,
        )
        return self._model


class ExportYoutubeTranscriptOp(ExportMarkdownMapper):
    op_name = "export_youtube_transcript"

    def process_row(self, row: dict[str, object]) -> dict[str, object]:
        status = str(row["status"])
        if status != "success":
            return {**row, "output_written": False}
        transcript_text = str(row.get("transcript_text", ""))
        if not transcript_text:
            raise ValueError("Successful transcript rows must include non-empty transcript_text.")
        transcript_key = str(row.get("transcript_r2_key", ""))
        if not transcript_key:
            raise ValueError("Successful transcript rows must include transcript_r2_key.")
        runtime = self.require_runtime()
        payload = YoutubeTranscriptResult.from_dict(
            {
                **row,
                "output_written": True,
            }
        ).to_dict()
        runtime.get_object_store().write_bytes(
            transcript_key,
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
        )
        return {
            **row,
            "output_written": True,
        }
