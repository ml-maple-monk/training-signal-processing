from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from ...core.utils import join_s3_key, make_s3_url, utc_timestamp
from ...runtime.submission import (
    ArtifactRef,
    ArtifactStore,
    BootstrapSpec,
    PreparedRun,
    RemoteInvocationSpec,
    SubmissionAdapter,
)
from .config import load_resolved_recipe_mapping
from .models import RecipeConfig, YoutubeMediaTask


@dataclass(frozen=True)
class ChannelVideoEntry:
    channel_url: str
    channel_id: str
    channel_title: str
    video_id: str
    video_url: str
    video_title: str
    upload_date: str


class YoutubeAsrSubmissionAdapter(SubmissionAdapter):
    def __init__(
        self,
        *,
        config: RecipeConfig,
        config_path: Path,
        overrides: list[str] | None = None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.overrides = overrides or []

    def prepare_new_run(self, artifact_store: ArtifactStore, *, dry_run: bool) -> PreparedRun:
        run_id = utc_timestamp()
        channel_urls = self.read_channel_urls(Path(self.config.input.channel_urls_path))
        input_manifest_key = self.build_control_key(run_id, "input_manifest.jsonl")
        config_object_key = self.build_control_key(run_id, "recipe.json")
        if dry_run:
            discovered_items = len(channel_urls)
            uploaded_items = 0
        else:
            tasks = self.build_media_tasks(artifact_store=artifact_store, run_id=run_id)
            artifact_store.write_jsonl(input_manifest_key, [task.to_dict() for task in tasks])
            artifact_store.write_json(
                config_object_key,
                load_resolved_recipe_mapping(self.config_path, self.overrides),
            )
            discovered_items = len(tasks)
            uploaded_items = len(tasks)
        return self.build_prepared_run(
            artifact_store=artifact_store,
            run_id=run_id,
            input_manifest_key=input_manifest_key,
            config_object_key=config_object_key,
            discovered_items=discovered_items,
            uploaded_items=uploaded_items,
            is_resume=False,
        )

    def prepare_resume_run(self, artifact_store: ArtifactStore, run_id: str) -> PreparedRun:
        input_manifest_key = self.build_control_key(run_id, "input_manifest.jsonl")
        config_object_key = self.build_control_key(run_id, "recipe.json")
        if not artifact_store.exists(input_manifest_key):
            raise ValueError(f"Resume manifest not found in R2: {input_manifest_key}")
        if not artifact_store.exists(config_object_key):
            raise ValueError(f"Resume recipe object not found in R2: {config_object_key}")
        manifest_rows = artifact_store.read_jsonl(input_manifest_key)
        return self.build_prepared_run(
            artifact_store=artifact_store,
            run_id=run_id,
            input_manifest_key=input_manifest_key,
            config_object_key=config_object_key,
            discovered_items=len(manifest_rows),
            uploaded_items=0,
            is_resume=True,
        )

    def parse_remote_summary(self, stdout: str) -> dict[str, object]:
        stripped = stdout.strip()
        if not stripped:
            raise ValueError("Remote job returned no JSON summary on stdout.")
        return json.loads(stripped)

    def build_prepared_run(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        input_manifest_key: str,
        config_object_key: str,
        discovered_items: int,
        uploaded_items: int,
        is_resume: bool,
    ) -> PreparedRun:
        return PreparedRun(
            run_id=run_id,
            remote_root=self.config.remote.root_dir,
            sync_paths=("pyproject.toml", "uv.lock", "src", "config"),
            bootstrap=self.build_bootstrap_spec(),
            invocation=self.build_invocation_spec(
                artifact_store=artifact_store,
                run_id=run_id,
                config_object_key=config_object_key,
                input_manifest_key=input_manifest_key,
            ),
            artifacts=(
                ArtifactRef(name="input_manifest", key=input_manifest_key, kind="jsonl"),
                ArtifactRef(name="config_object", key=config_object_key, kind="json"),
            ),
            discovered_items=discovered_items,
            uploaded_items=uploaded_items,
            is_resume=is_resume,
            metadata={
                "pipeline_family": "youtube_asr",
                "input_manifest_key": input_manifest_key,
                "config_object_key": config_object_key,
            },
        )

    def build_bootstrap_spec(self) -> BootstrapSpec:
        command = " && ".join(
            [
                "command -v uv >/dev/null",
                f"uv python install {shlex.quote(self.config.remote.python_version)}",
                "uv sync --group remote_ocr --group model --no-dev",
                "uv pip install -U 'qwen-asr[vllm]'",
            ]
        )
        return BootstrapSpec(command=command)

    def build_invocation_spec(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
        config_object_key: str,
        input_manifest_key: str,
    ) -> RemoteInvocationSpec:
        command = shlex.join(
            [
                "uv",
                "run",
                "--group",
                "remote_ocr",
                "--group",
                "model",
                "python",
                "-m",
                "training_signal_processing.pipelines.youtube_asr.remote_job",
                "--run-id",
                run_id,
                "--config-object-key",
                config_object_key,
                "--input-manifest-key",
                input_manifest_key,
            ]
        )
        env = artifact_store.build_remote_env()
        reverse_tunnels: tuple[str, ...] = ()
        tracking_uri = self.resolve_remote_tracking_uri()
        if tracking_uri:
            env["MLFLOW_TRACKING_URI"] = tracking_uri
            reverse_tunnels = (self.build_reverse_tunnel_spec(),)
        return RemoteInvocationSpec(
            command=command,
            env=env,
            reverse_tunnels=reverse_tunnels,
        )

    def resolve_remote_tracking_uri(self) -> str:
        if not self.config.mlflow.enabled:
            return ""
        return f"http://127.0.0.1:{self.config.mlflow.remote_tunnel_port}"

    def build_reverse_tunnel_spec(self) -> str:
        parsed = urlparse(self.config.mlflow.local_tracking_uri)
        if not parsed.hostname or not parsed.port:
            raise ValueError("mlflow.local_tracking_uri must include an explicit host and port.")
        return f"{self.config.mlflow.remote_tunnel_port}:{parsed.hostname}:{parsed.port}"

    def read_channel_urls(self, path: Path) -> list[str]:
        if not path.is_file():
            raise ValueError(f"Channel URL list not found: {path}")
        urls = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if not urls:
            raise ValueError(f"No channel URLs found in {path}")
        return urls

    def build_media_tasks(
        self,
        *,
        artifact_store: ArtifactStore,
        run_id: str,
    ) -> list[YoutubeMediaTask]:
        tasks: list[YoutubeMediaTask] = []
        staging_root = Path(self.config.download.local_staging_dir).expanduser() / run_id
        staging_root.mkdir(parents=True, exist_ok=True)
        for channel_url in self.read_channel_urls(Path(self.config.input.channel_urls_path)):
            entries = self.list_channel_entries(channel_url)
            for entry in entries:
                media_path = self.download_video_media(
                    channel_entry=entry,
                    staging_root=staging_root,
                )
                source_key = self.build_source_media_key(entry=entry, media_path=media_path)
                artifact_store.upload_file(media_path, source_key)
                tasks.append(
                    YoutubeMediaTask(
                        channel_url=entry.channel_url,
                        channel_id=entry.channel_id,
                        channel_title=entry.channel_title,
                        video_id=entry.video_id,
                        video_url=entry.video_url,
                        video_title=entry.video_title,
                        upload_date=entry.upload_date,
                        source_media_r2_key=source_key,
                        source_media_url=make_s3_url(artifact_store.bucket, source_key),
                        source_media_ext=media_path.suffix.lstrip("."),
                        source_media_size_bytes=media_path.stat().st_size,
                        transcript_r2_key=self.build_transcript_key(run_id=run_id, entry=entry),
                    )
                )
                media_path.unlink(missing_ok=True)
        if not tasks:
            raise ValueError(
                "No downloadable YouTube videos were discovered from the provided list."
            )
        return tasks

    def list_channel_entries(self, channel_url: str) -> list[ChannelVideoEntry]:
        command = [
            *self.resolve_yt_dlp_command(),
            "--flat-playlist",
            "--dump-single-json",
            "--playlist-end",
            str(self.config.input.videos_per_channel),
            "--ignore-errors",
            "--no-warnings",
            channel_url,
        ]
        payload = self.run_command(command)
        listing = json.loads(payload)
        entries = listing.get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"yt-dlp did not return entries for channel URL: {channel_url}")
        channel_title = self.clean_value(
            listing.get("title") or listing.get("channel") or listing.get("uploader")
        )
        channel_id = self.clean_value(
            listing.get("channel_id") or listing.get("uploader_id") or channel_title
        )
        resolved: list[ChannelVideoEntry] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            video_id = self.clean_value(item.get("id"))
            if not video_id:
                continue
            video_url = self.clean_value(item.get("webpage_url")) or (
                f"https://www.youtube.com/watch?v={video_id}"
            )
            resolved.append(
                ChannelVideoEntry(
                    channel_url=channel_url,
                    channel_id=channel_id or "unknown-channel",
                    channel_title=channel_title or "unknown-channel",
                    video_id=video_id,
                    video_url=video_url,
                    video_title=self.clean_value(item.get("title")) or video_id,
                    upload_date=self.clean_value(item.get("upload_date")),
                )
            )
        return resolved

    def download_video_media(
        self,
        *,
        channel_entry: ChannelVideoEntry,
        staging_root: Path,
    ) -> Path:
        channel_name = channel_entry.channel_id or channel_entry.channel_title
        channel_dir = staging_root / self.slugify(channel_name)
        channel_dir.mkdir(parents=True, exist_ok=True)
        output_template = channel_dir / f"{channel_entry.video_id}.%(ext)s"
        command = [
            *self.resolve_yt_dlp_command(),
            "--ignore-errors",
            "--no-warnings",
            "--no-progress",
            "--restrict-filenames",
            "-f",
            self.config.download.format_selector,
            "-o",
            str(output_template),
        ]
        if self.config.download.cookies_file:
            command.extend(["--cookies", self.config.download.cookies_file])
        if self.config.download.cookies_from_browser:
            command.extend(["--cookies-from-browser", self.config.download.cookies_from_browser])
        command.append(channel_entry.video_url)
        self.run_command(command)
        candidates = sorted(
            path
            for path in channel_dir.glob(f"{channel_entry.video_id}.*")
            if path.is_file() and path.suffix not in {".part", ".ytdl"}
        )
        if not candidates:
            raise ValueError(f"yt-dlp did not produce a media file for {channel_entry.video_url}")
        return candidates[0]

    def resolve_yt_dlp_command(self) -> list[str]:
        if shutil.which("yt-dlp"):
            return ["yt-dlp"]
        if shutil.which("uvx"):
            return ["uvx", "--from", "yt-dlp", "yt-dlp"]
        raise ValueError("yt-dlp is not available on PATH and uvx is not installed.")

    def run_command(self, command: list[str]) -> str:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            rendered = shlex.join(command)
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"Command failed: {rendered}\n{detail}")
        return result.stdout

    def build_source_media_key(self, *, entry: ChannelVideoEntry, media_path: Path) -> str:
        relative_name = f"{self.slugify(entry.channel_id)}/{entry.video_id}{media_path.suffix}"
        return join_s3_key(self.config.input.media_r2_prefix, relative_name)

    def build_transcript_key(self, *, run_id: str, entry: ChannelVideoEntry) -> str:
        return join_s3_key(
            self.build_run_root(run_id),
            f"transcripts/{self.slugify(entry.channel_id)}/{entry.video_id}.json",
        )

    def build_control_key(self, run_id: str, name: str) -> str:
        return join_s3_key(self.build_run_root(run_id), f"control/{name}")

    def build_run_root(self, run_id: str) -> str:
        return join_s3_key(self.config.r2.output_prefix, run_id)

    def clean_value(self, value: object) -> str:
        return str(value).strip() if value is not None else ""

    def slugify(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
        return cleaned.strip("-") or "item"
