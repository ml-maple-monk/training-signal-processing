from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ...core.models import (
    MlflowConfig,
    ObservabilityConfig,
    OpConfig,
    R2Config,
    RayConfig,
    RemoteRuntimeConfig,
    SshConfig,
)


@dataclass
class InputConfig:
    channel_urls_path: str
    videos_per_channel: int
    media_r2_prefix: str


@dataclass
class DownloadConfig:
    local_staging_dir: str
    format_selector: str
    cookies_file: str = ""
    cookies_from_browser: str = ""


@dataclass
class AsrConfig:
    model_name: str
    gpu_memory_utilization: float
    max_inference_batch_size: int
    max_new_tokens: int
    max_media_file_size_mb: int
    language: str = ""


@dataclass
class ResumeConfig:
    strategy: str
    commit_every_batches: int
    resume_mode: str


@dataclass
class RecipeConfig:
    run_name: str
    config_version: int
    ssh: SshConfig
    remote: RemoteRuntimeConfig
    ray: RayConfig
    r2: R2Config
    input: InputConfig
    download: DownloadConfig
    asr: AsrConfig
    mlflow: MlflowConfig
    observability: ObservabilityConfig
    resumability: ResumeConfig
    ops: list[OpConfig]


@dataclass
class YoutubeMediaTask:
    channel_url: str
    channel_id: str
    channel_title: str
    video_id: str
    video_url: str
    video_title: str
    upload_date: str
    source_media_r2_key: str
    source_media_url: str
    source_media_ext: str
    source_media_size_bytes: int
    transcript_r2_key: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "YoutubeMediaTask":
        return cls(
            channel_url=str(row["channel_url"]),
            channel_id=str(row["channel_id"]),
            channel_title=str(row["channel_title"]),
            video_id=str(row["video_id"]),
            video_url=str(row["video_url"]),
            video_title=str(row["video_title"]),
            upload_date=str(row.get("upload_date", "")),
            source_media_r2_key=str(row["source_media_r2_key"]),
            source_media_url=str(row["source_media_url"]),
            source_media_ext=str(row["source_media_ext"]),
            source_media_size_bytes=int(row["source_media_size_bytes"]),
            transcript_r2_key=str(row["transcript_r2_key"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class YoutubeTranscriptResult:
    run_id: str
    channel_id: str
    channel_title: str
    channel_url: str
    video_id: str
    video_url: str
    video_title: str
    upload_date: str
    source_media_r2_key: str
    source_media_ext: str
    source_media_size_bytes: int
    transcript_r2_key: str
    status: str
    error_message: str
    transcript_text: str
    detected_language: str
    started_at: str
    finished_at: str
    duration_sec: float
    output_written: bool = False

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "YoutubeTranscriptResult":
        return cls(
            run_id=str(row["run_id"]),
            channel_id=str(row["channel_id"]),
            channel_title=str(row["channel_title"]),
            channel_url=str(row["channel_url"]),
            video_id=str(row["video_id"]),
            video_url=str(row["video_url"]),
            video_title=str(row["video_title"]),
            upload_date=str(row.get("upload_date", "")),
            source_media_r2_key=str(row["source_media_r2_key"]),
            source_media_ext=str(row["source_media_ext"]),
            source_media_size_bytes=int(row["source_media_size_bytes"]),
            transcript_r2_key=str(row["transcript_r2_key"]),
            status=str(row["status"]),
            error_message=str(row.get("error_message", "")),
            transcript_text=str(row.get("transcript_text", "")),
            detected_language=str(row.get("detected_language", "")),
            started_at=str(row.get("started_at", "")),
            finished_at=str(row.get("finished_at", "")),
            duration_sec=float(row.get("duration_sec", 0.0)),
            output_written=bool(row.get("output_written", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
