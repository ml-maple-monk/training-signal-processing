from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from training_signal_processing.pipelines.youtube_asr.config import load_recipe_config
from training_signal_processing.pipelines.youtube_asr.models import (
    RecipeConfig,
    YoutubeMediaTask,
    YoutubeTranscriptResult,
)
from training_signal_processing.pipelines.youtube_asr.resume import YoutubeAsrResumeLedger
from training_signal_processing.pipelines.youtube_asr.submission import YoutubeAsrSubmissionAdapter
from training_signal_processing.storage.object_store import ObjectStore


class FakeR2Store(ObjectStore):
    def __init__(self) -> None:
        self.bucket = "test-bucket"
        self.payloads: dict[str, bytes] = {}
        self.keys: list[str] = []
        self.remote_env = {"R2_BUCKET": self.bucket}

    def exists(self, key: str) -> bool:
        return key in self.payloads

    def list_keys(self, prefix: str) -> list[str]:
        return [key for key in self.keys if key.startswith(prefix)]

    def read_bytes(self, key: str) -> bytes:
        return self.payloads[key]

    def write_bytes(self, key: str, body: bytes) -> None:
        self.payloads[key] = body
        if key not in self.keys:
            self.keys.append(key)

    def upload_file(self, path: Path, key: str) -> None:
        self.write_bytes(key, path.read_bytes())

    def make_url(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def build_pyarrow_filesystem(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def build_remote_env(self) -> dict[str, str]:
        return dict(self.remote_env)


class FakeArtifactStore:
    def __init__(self, object_store: FakeR2Store) -> None:
        self.object_store = object_store
        self.bucket = object_store.bucket
        self.written_json: dict[str, dict[str, object]] = {}
        self.written_jsonl: dict[str, list[dict[str, object]]] = {}

    def exists(self, key: str) -> bool:
        return self.object_store.exists(key)

    def read_json(self, key: str) -> dict[str, object]:
        return self.object_store.read_json(key)

    def read_jsonl(self, key: str) -> list[dict[str, object]]:
        return self.object_store.read_jsonl(key)

    def write_json(self, key: str, value: dict[str, object]) -> None:
        self.written_json[key] = value
        self.object_store.write_json(key, value)

    def write_jsonl(self, key: str, rows: list[dict[str, object]]) -> None:
        self.written_jsonl[key] = rows
        self.object_store.write_jsonl(key, rows)

    def upload_file(self, path: Path, key: str) -> None:
        self.object_store.upload_file(path, key)

    def build_remote_env(self) -> dict[str, str]:
        return self.object_store.build_remote_env()


@pytest.fixture()
def sample_config(tmp_path: Path) -> Path:
    channels_path = tmp_path / "channels.txt"
    channels_path.write_text("https://www.youtube.com/@sample\n", encoding="utf-8")
    config_path = tmp_path / "youtube_asr.yaml"
    config_path.write_text(
        f"""
run:
  name: test-youtube-asr
  config_version: 1
ssh:
  host: localhost
  port: 22
  user: root
  identity_file: ~/.ssh/id_ed25519
remote:
  root_dir: /tmp/youtube-asr
  python_version: "3.12"
ray:
  executor_type: ray
  batch_size: 1
  concurrency: 1
  target_num_blocks: 1
r2:
  config_file: r2
  bucket: gpu-poor
  output_prefix: dataset/processed/youtube_asr
input:
  channel_urls_path: {channels_path}
  videos_per_channel: 5
  media_r2_prefix: dataset/raw/youtube_media
download:
  local_staging_dir: {tmp_path / "staging"}
  format_selector: bestaudio/best
  cookies_file: ""
  cookies_from_browser: ""
asr:
  model_name: Qwen/Qwen3-ASR-0.6B
  gpu_memory_utilization: 0.7
  max_inference_batch_size: 2
  max_new_tokens: 256
  max_media_file_size_mb: 128
  language: ""
mlflow:
  enabled: false
  local_tracking_uri: http://127.0.0.1:5000
  remote_tunnel_port: 15000
  experiment_name: youtube-asr-test
observability:
  flush_interval_sec: 5
  log_per_file_events: true
  heartbeat_interval_sec: 10
resumability:
  strategy: batch_manifest
  commit_every_batches: 1
  resume_mode: latest
ops:
  - name: prepare_youtube_media
    type: mapper
  - name: skip_existing_youtube_media
    type: filter
  - name: qwen3_asr_vllm
    type: mapper
  - name: export_youtube_transcript
    type: mapper
""",
        encoding="utf-8",
    )
    return config_path


def test_load_recipe_config_parses_youtube_asr_sections(sample_config: Path) -> None:
    config = load_recipe_config(sample_config)
    assert isinstance(config, RecipeConfig)
    assert config.input.videos_per_channel == 5
    assert config.asr.model_name == "Qwen/Qwen3-ASR-0.6B"


def test_load_recipe_config_rejects_non_positive_video_limit(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
run: {name: bad, config_version: 1}
ssh: {host: localhost, port: 22, user: root, identity_file: ~/.ssh/id_ed25519}
remote: {root_dir: /tmp/youtube-asr, python_version: "3.12"}
ray: {executor_type: ray, batch_size: 1, concurrency: 1, target_num_blocks: 1}
r2: {config_file: r2, bucket: gpu-poor, output_prefix: dataset/processed/youtube_asr}
input:
  channel_urls_path: /tmp/channels.txt
  videos_per_channel: 0
  media_r2_prefix: dataset/raw/youtube_media
download:
  local_staging_dir: /tmp/staging
  format_selector: bestaudio/best
asr:
  model_name: Qwen/Qwen3-ASR-0.6B
  gpu_memory_utilization: 0.7
  max_inference_batch_size: 2
  max_new_tokens: 256
  max_media_file_size_mb: 128
mlflow:
  enabled: false
  local_tracking_uri: http://127.0.0.1:5000
  remote_tunnel_port: 15000
  experiment_name: x
observability: {flush_interval_sec: 5, log_per_file_events: true, heartbeat_interval_sec: 10}
resumability: {strategy: batch_manifest, commit_every_batches: 1, resume_mode: latest}
ops:
  - {name: prepare_youtube_media, type: mapper}
  - {name: skip_existing_youtube_media, type: filter}
  - {name: qwen3_asr_vllm, type: mapper}
  - {name: export_youtube_transcript, type: mapper}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="videos_per_channel"):
        load_recipe_config(config_path)


def test_prepare_new_run_lists_downloads_and_uploads_media(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_config: Path,
) -> None:
    config = load_recipe_config(sample_config)
    object_store = FakeR2Store()
    artifact_store = FakeArtifactStore(object_store)
    adapter = YoutubeAsrSubmissionAdapter(config=config, config_path=sample_config)
    media_bytes = b"audio-bytes"

    def fake_list(channel_url: str):  # type: ignore[no-untyped-def]
        return [
            type(
                "Entry",
                (),
                {
                    "channel_url": channel_url,
                    "channel_id": "sample-channel",
                    "channel_title": "Sample Channel",
                    "video_id": "vid-001",
                    "video_url": "https://www.youtube.com/watch?v=vid-001",
                    "video_title": "Video One",
                    "upload_date": "20260422",
                },
            )()
        ]

    def fake_download(*, channel_entry, staging_root):  # type: ignore[no-untyped-def]
        path = staging_root / "sample-channel" / f"{channel_entry.video_id}.m4a"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(media_bytes)
        return path

    monkeypatch.setattr(adapter, "list_channel_entries", fake_list)
    monkeypatch.setattr(adapter, "download_video_media", fake_download)

    prepared = adapter.prepare_new_run(artifact_store, dry_run=False)

    assert prepared.discovered_items == 1
    manifest_rows = artifact_store.written_jsonl[
        f"{config.r2.output_prefix}/{prepared.run_id}/control/input_manifest.jsonl"
    ]
    expected_key = "dataset/raw/youtube_media/sample-channel/vid-001.m4a"
    assert manifest_rows[0]["source_media_r2_key"] == expected_key
    assert object_store.read_bytes(expected_key) == media_bytes


def test_youtube_asr_ops_transcribe_and_export(
    monkeypatch: pytest.MonkeyPatch,
    sample_config: Path,
) -> None:
    class FakeResult:
        def __init__(self, text: str, language: str) -> None:
            self.text = text
            self.language = language

    class FakeModel:
        def transcribe(self, audio, language=None):  # type: ignore[no-untyped-def]
            assert len(audio) == 1
            assert language is None
            return [FakeResult("hello transcript", "en")]

    class FakeQwen3ASRModel:
        @staticmethod
        def LLM(**_: object) -> FakeModel:
            return FakeModel()

    monkeypatch.setitem(
        sys.modules,
        "qwen_asr",
        type("FakeQwenAsrModule", (), {"Qwen3ASRModel": FakeQwen3ASRModel})(),
    )

    from training_signal_processing.core.models import OpRuntimeContext
    from training_signal_processing.custom_ops.youtube_asr_ops import (
        ExportYoutubeTranscriptOp,
        PrepareYoutubeMediaOp,
        Qwen3AsrVllmOp,
    )

    object_store = FakeR2Store()
    config = load_recipe_config(sample_config)
    object_store.write_bytes("dataset/raw/youtube_media/sample-channel/vid-001.m4a", b"audio")
    runtime = OpRuntimeContext(
        config=config,
        run_id="run-001",
        object_store=object_store,
        output_root_key="dataset/processed/youtube_asr/run-001",
        source_root_key="dataset/raw/youtube_media",
    )
    task = YoutubeMediaTask(
        channel_url="https://www.youtube.com/@sample",
        channel_id="sample-channel",
        channel_title="Sample Channel",
        video_id="vid-001",
        video_url="https://www.youtube.com/watch?v=vid-001",
        video_title="Video One",
        upload_date="20260422",
        source_media_r2_key="dataset/raw/youtube_media/sample-channel/vid-001.m4a",
        source_media_url="s3://test-bucket/dataset/raw/youtube_media/sample-channel/vid-001.m4a",
        source_media_ext="m4a",
        source_media_size_bytes=5,
        transcript_r2_key="dataset/processed/youtube_asr/run-001/transcripts/sample-channel/vid-001.json",
    )

    prepared = PrepareYoutubeMediaOp().bind_runtime(runtime).process_batch([task.to_dict()])
    transcribed = Qwen3AsrVllmOp().bind_runtime(runtime).process_batch(prepared)
    exported = ExportYoutubeTranscriptOp().bind_runtime(runtime).process_batch(transcribed)

    assert exported[0]["transcript_text"] == "hello transcript"
    assert exported[0]["output_written"] is True
    stored = json.loads(
        object_store.read_bytes(
            "dataset/processed/youtube_asr/run-001/transcripts/sample-channel/vid-001.json"
        ).decode("utf-8")
    )
    assert stored["detected_language"] == "en"
    assert stored["transcript_text"] == "hello transcript"


def test_resume_ledger_tracks_completed_media(sample_config: Path) -> None:
    config = load_recipe_config(sample_config)
    object_store = FakeR2Store()
    ledger = YoutubeAsrResumeLedger(config=config, object_store=object_store)
    run_id = "run-123"
    manifest_key = f"{config.r2.output_prefix}/{run_id}/manifests/batch-00001.jsonl"
    object_store.write_jsonl(
        manifest_key,
        [
            YoutubeTranscriptResult(
                run_id=run_id,
                channel_id="sample-channel",
                channel_title="Sample Channel",
                channel_url="https://www.youtube.com/@sample",
                video_id="vid-001",
                video_url="https://www.youtube.com/watch?v=vid-001",
                video_title="Video One",
                upload_date="20260422",
                source_media_r2_key="dataset/raw/youtube_media/sample-channel/vid-001.m4a",
                source_media_ext="m4a",
                source_media_size_bytes=5,
                transcript_r2_key="dataset/processed/youtube_asr/run-123/transcripts/sample-channel/vid-001.json",
                status="success",
                error_message="",
                transcript_text="hello transcript",
                detected_language="en",
                started_at="2026-04-22T00:00:00Z",
                finished_at="2026-04-22T00:00:01Z",
                duration_sec=1.0,
                output_written=True,
            ).to_dict()
        ],
    )

    completed = ledger.load_completed_item_keys(run_id)

    assert completed == {"dataset/raw/youtube_media/sample-channel/vid-001.m4a"}
