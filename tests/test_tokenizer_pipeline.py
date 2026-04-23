from __future__ import annotations

import gzip
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from training_signal_processing.pipelines.tokenizer.config import load_recipe_config
from training_signal_processing.pipelines.tokenizer.models import (
    ParquetShardTask,
    RecipeConfig,
    TokenizedShardResult,
)
from training_signal_processing.pipelines.tokenizer.resume import TokenizerResumeLedger
from training_signal_processing.pipelines.tokenizer.submission import TokenizerSubmissionAdapter
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

    def read_json(self, key: str) -> dict[str, object]:
        return super().read_json(key)

    def read_jsonl(self, key: str) -> list[dict[str, object]]:
        return super().read_jsonl(key)

    def write_json(self, key: str, value: dict[str, object]) -> None:
        super().write_json(key, value)
        if key not in self.keys:
            self.keys.append(key)

    def write_jsonl(self, key: str, rows: list[dict[str, object]]) -> None:
        super().write_jsonl(key, rows)
        if key not in self.keys:
            self.keys.append(key)

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

    def as_object_store(self) -> FakeR2Store:
        return self.object_store


@pytest.fixture()
def sample_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "tokenizer.yaml"
    config_path.write_text(
        """
run:
  name: test-tokenizer
  config_version: 1
ssh:
  host: localhost
  port: 22
  user: root
  identity_file: ~/.ssh/id_ed25519
remote:
  root_dir: /tmp/tokenizer
  python_version: "3.12"
ray:
  executor_type: ray
  batch_size: 1
  concurrency: 1
  target_num_blocks: 1
r2:
  config_file: r2
  bucket: gpu-poor
  output_prefix: dataset/processed/tokenizer
input:
  source_prefix: dataset/raw
  family_specs:
    - name: books
      glob: books/*.parquet
      text_column: markdown_text
      id_columns: [book_id, markdown_sha256]
    - name: zhihu_qa
      glob: zhihu_qa/*.parquet
      text_column: answer_content
      id_columns: [answer_id, question_id]
tokenizer:
  model_name: Qwen/Qwen3-0.6B
  output_compression: gzip
mlflow:
  enabled: false
  local_tracking_uri: http://127.0.0.1:5000
  remote_tunnel_port: 15000
  experiment_name: tokenizer-test
observability:
  flush_interval_sec: 5
  log_per_file_events: true
  heartbeat_interval_sec: 10
resumability:
  strategy: batch_manifest
  commit_every_batches: 1
  resume_mode: latest
ops:
  - name: prepare_parquet_shard
    type: mapper
  - name: skip_existing_shards
    type: filter
  - name: tokenize_hf_token_ids
    type: mapper
  - name: export_token_jsonl
    type: mapper
""",
        encoding="utf-8",
    )
    return config_path


def test_load_recipe_config_parses_family_specs(sample_config: Path) -> None:
    config = load_recipe_config(sample_config)
    assert isinstance(config, RecipeConfig)
    assert config.tokenizer.model_name == "Qwen/Qwen3-0.6B"
    assert [spec.name for spec in config.input.family_specs] == ["books", "zhihu_qa"]


def test_load_recipe_config_rejects_duplicate_family_names(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
run: {name: bad, config_version: 1}
ssh: {host: localhost, port: 22, user: root, identity_file: ~/.ssh/id_ed25519}
remote: {root_dir: /tmp/tokenizer, python_version: "3.12"}
ray: {executor_type: ray, batch_size: 1, concurrency: 1, target_num_blocks: 1}
r2: {config_file: r2, bucket: gpu-poor, output_prefix: dataset/processed/tokenizer}
input:
  source_prefix: dataset/raw
  family_specs:
    - {name: books, glob: "books/*.parquet", text_column: markdown_text}
    - {name: books, glob: "books2/*.parquet", text_column: markdown_text}
tokenizer: {model_name: Qwen/Qwen3-0.6B, output_compression: gzip}
mlflow:
  enabled: false
  local_tracking_uri: http://127.0.0.1:5000
  remote_tunnel_port: 15000
  experiment_name: x
observability: {flush_interval_sec: 5, log_per_file_events: true, heartbeat_interval_sec: 10}
resumability: {strategy: batch_manifest, commit_every_batches: 1, resume_mode: latest}
ops:
  - {name: prepare_parquet_shard, type: mapper}
  - {name: skip_existing_shards, type: filter}
  - {name: tokenize_hf_token_ids, type: mapper}
  - {name: export_token_jsonl, type: mapper}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate family spec name"):
        load_recipe_config(config_path)


def test_submission_discovers_matching_shards_and_output_keys(sample_config: Path) -> None:
    config = load_recipe_config(sample_config)
    object_store = FakeR2Store()
    object_store.keys.extend(
        [
            "dataset/raw/books/part-00000.parquet",
            "dataset/raw/zhihu_qa/part-00001.parquet",
            "dataset/raw/unknown/part-00002.parquet",
            "dataset/raw/books/readme.txt",
        ]
    )
    artifact_store = FakeArtifactStore(object_store)
    adapter = TokenizerSubmissionAdapter(config=config, config_path=sample_config)

    tasks = adapter.discover_shard_tasks(artifact_store, run_id="run-123")

    assert [task.family_name for task in tasks] == ["books", "zhihu_qa"]
    assert tasks[0].output_r2_key == (
        "dataset/processed/tokenizer/run-123/tokenized/books/part-00000.jsonl.gz"
    )
    assert tasks[1].output_r2_key == (
        "dataset/processed/tokenizer/run-123/tokenized/zhihu_qa/part-00001.jsonl.gz"
    )


def test_prepare_new_run_writes_manifest(sample_config: Path) -> None:
    config = load_recipe_config(sample_config)
    object_store = FakeR2Store()
    object_store.keys.append("dataset/raw/books/part-00000.parquet")
    artifact_store = FakeArtifactStore(object_store)
    adapter = TokenizerSubmissionAdapter(config=config, config_path=sample_config)

    prepared = adapter.prepare_new_run(artifact_store, dry_run=False)

    assert prepared.discovered_items == 1
    manifest_key = f"{config.r2.output_prefix}/{prepared.run_id}/control/input_manifest.jsonl"
    assert manifest_key in artifact_store.written_jsonl


def test_tokenize_and_export_ops_use_primary_text_and_fallback_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_config: Path,
) -> None:
    parquet_path = tmp_path / "qa.parquet"
    table = pa.table(
        {
            "answer_id": ["a-1", None],
            "question_id": ["q-1", "q-2"],
            "answer_content": ["hello world", "  "],
            "question_title": ["ignored", "ignored 2"],
        }
    )
    pq.write_table(table, parquet_path)

    class FakeTokenizer:
        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            return {"input_ids": [len(part) for part in text.split()]}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str) -> FakeTokenizer:
            assert model_name == "Qwen/Qwen3-0.6B"
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        type("FakeTransformers", (), {"AutoTokenizer": FakeAutoTokenizer})(),
    )

    from training_signal_processing.core.models import OpRuntimeContext
    from training_signal_processing.custom_ops.tokenizer_ops import (
        ExportTokenJsonlOp,
        PrepareParquetShardOp,
        TokenizeHfTokenIdsOp,
    )

    class LocalObjectStore(FakeR2Store):
        def __init__(self, source_path: Path) -> None:
            super().__init__()
            self.source_path = source_path

        def build_pyarrow_filesystem(self):  # type: ignore[no-untyped-def]
            import pyarrow.fs as pafs

            return pafs.LocalFileSystem()

    object_store = LocalObjectStore(parquet_path)
    config = load_recipe_config(sample_config)
    runtime = OpRuntimeContext(
        config=config,
        run_id="run-001",
        object_store=object_store,
        output_root_key="dataset/processed/tokenizer/run-001",
        source_root_key="dataset/raw",
    )

    task = ParquetShardTask(
        family_name="zhihu_qa",
        source_r2_key=str(parquet_path),
        source_r2_url=f"file://{parquet_path}",
        source_rel_path="zhihu_qa/qa.parquet",
        text_column="answer_content",
        id_columns=["answer_id", "question_id"],
        output_r2_key="dataset/processed/tokenizer/run-001/tokenized/zhihu_qa/qa.jsonl.gz",
    )

    prepared = PrepareParquetShardOp().bind_runtime(runtime).process_batch([task.to_dict()])
    tokenized = TokenizeHfTokenIdsOp().bind_runtime(runtime).process_batch(prepared)
    exported = ExportTokenJsonlOp().bind_runtime(runtime).process_batch(tokenized)

    assert exported[0]["tokenized_row_count"] == 1
    assert exported[0]["output_written"] is True
    payload = gzip.decompress(
        object_store.read_bytes(
            "dataset/processed/tokenizer/run-001/tokenized/zhihu_qa/qa.jsonl.gz"
        )
    ).decode("utf-8")
    assert '"row_id": "a-1"' in payload
    assert "ignored" not in payload


def test_resume_ledger_tracks_completed_shards(sample_config: Path) -> None:
    config = load_recipe_config(sample_config)
    object_store = FakeR2Store()
    ledger = TokenizerResumeLedger(config=config, object_store=object_store)
    run_id = "run-123"
    manifest_key = f"{config.r2.output_prefix}/{run_id}/manifests/batch-00001.jsonl"
    object_store.write_jsonl(
        manifest_key,
        [
            TokenizedShardResult(
                run_id=run_id,
                family_name="books",
                source_r2_key="dataset/raw/books/part-00000.parquet",
                source_rel_path="books/part-00000.parquet",
                output_r2_key="dataset/processed/tokenizer/run-123/tokenized/books/part-00000.jsonl.gz",
                status="success",
                error_message="",
                text_column="markdown_text",
                tokenizer_model="Qwen/Qwen3-0.6B",
                row_count=10,
                tokenized_row_count=9,
                started_at="2026-04-22T00:00:00Z",
                finished_at="2026-04-22T00:00:01Z",
                duration_sec=1.0,
                output_written=True,
            ).to_dict()
        ],
    )

    completed = ledger.load_completed_item_keys(run_id)

    assert completed == {"dataset/raw/books/part-00000.parquet"}
