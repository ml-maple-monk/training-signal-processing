from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
import tiktoken

from training_signal_processing.core.models import RunArtifactLayout
from training_signal_processing.core.storage import ObjectStore
from training_signal_processing.core.submission import ArtifactStore
from training_signal_processing.main import cli
from training_signal_processing.pipelines.unified_data.config import load_recipe_config
from training_signal_processing.pipelines.unified_data.ops import write_unified_data_part
from training_signal_processing.pipelines.unified_data.runtime import UnifiedDataCompletionTracker
from training_signal_processing.pipelines.unified_data.submission import (
    UnifiedDataSubmissionAdapter,
    build_unified_data_manifest_rows,
)


class LocalObjectStore(ObjectStore):
    def __init__(self, root: Path, bucket: str = "gpu-poor") -> None:
        self.root = root
        self.storage_root = root / bucket
        self.bucket = bucket
        self.read_json_calls: list[str] = []

    def exists(self, key: str) -> bool:
        return (self.storage_root / key).exists()

    def list_keys(self, prefix: str) -> list[str]:
        return sorted(
            path.relative_to(self.storage_root).as_posix()
            for path in self.storage_root.rglob("*")
            if path.is_file() and path.relative_to(self.storage_root).as_posix().startswith(prefix)
        )

    def read_bytes(self, key: str) -> bytes:
        return (self.storage_root / key).read_bytes()

    def read_json(self, key: str) -> dict[str, object]:
        self.read_json_calls.append(key)
        return super().read_json(key)

    def write_bytes(self, key: str, body: bytes) -> None:
        path = self.storage_root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)

    def upload_file(self, path: Path, key: str) -> None:
        self.write_bytes(key, path.read_bytes())

    def make_url(self, key: str) -> str:
        return f"r2://{self.bucket}/{key}"

    def build_pyarrow_filesystem(self):
        return pafs.SubTreeFileSystem(self.root.as_posix(), pafs.LocalFileSystem())


class FakeArtifactStore(ArtifactStore):
    def __init__(self, object_store: LocalObjectStore) -> None:
        self.object_store = object_store
        self.bucket = object_store.bucket

    def as_object_store(self) -> LocalObjectStore:
        return self.object_store

    def exists(self, key: str) -> bool:
        return self.object_store.exists(key)

    def read_json(self, key: str) -> dict[str, object]:
        return self.object_store.read_json(key)

    def read_jsonl(self, key: str) -> list[dict[str, object]]:
        return self.object_store.read_jsonl(key)

    def write_json(self, key: str, value: dict[str, object]) -> None:
        self.object_store.write_json(key, value)

    def write_jsonl(self, key: str, rows: list[dict[str, object]]) -> None:
        self.object_store.write_jsonl(key, rows)

    def upload_file(self, path: Path, key: str) -> None:
        self.object_store.upload_file(path, key)

    def build_remote_env(self) -> dict[str, str]:
        return {"R2_BUCKET": self.bucket}


def test_unified_data_sample_config_loads_and_cli_is_registered() -> None:
    config = load_recipe_config(Path("config/unified_data.sample.yaml"))

    assert "unified-data-remote-job" in cli.commands
    assert config.r2.output_prefix == "dataset/processed/unified-data"
    assert config.input.lid_run_id == "20260429T015335Z"
    assert config.input.source_cleaning_run_id == "source-cleaning-full-20260429T171532Z"
    assert config.export.rows_per_row_group == 500_000
    assert config.export.tokenizer_encoding == "o200k_base"


def test_unified_data_manifest_splits_fixed_row_group_parts(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    config = load_recipe_config(write_config(tmp_path, rows_per_row_group=2))
    write_external_run_fixture(store, row_count=3)

    tasks = build_unified_data_manifest_rows(config=config, object_store=store, run_id="out-run")

    assert [task.expected_row_count for task in tasks] == [2, 1]
    assert tasks[0].segments[0].row_offset == 0
    assert tasks[0].segments[0].row_count == 2
    assert tasks[1].segments[0].row_offset == 2
    assert tasks[1].segments[0].row_count == 1
    assert tasks[0].part_key.endswith("parts/part-000000.parquet")
    assert tasks[1].done_key.endswith("done/part-000001.done.json")


def test_unified_data_manifest_uses_unfiltered_row_count_hint(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    config = load_recipe_config(write_config(tmp_path, rows_per_row_group=10))
    write_external_run_fixture(store, row_count=3)

    tasks = build_unified_data_manifest_rows(config=config, object_store=store, run_id="out-run")

    assert [task.expected_row_count for task in tasks] == [3]
    assert not any(key.endswith(".metrics.json") for key in store.read_json_calls)


def test_unified_data_manifest_reads_metrics_for_filtered_sources(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    config = load_recipe_config(write_config(tmp_path, rows_per_row_group=10))
    write_external_run_fixture(
        store,
        row_count=2,
        manifest_row_count=5,
        filters={"subreddit": "Bolehland"},
    )

    tasks = build_unified_data_manifest_rows(config=config, object_store=store, run_id="out-run")

    assert [task.expected_row_count for task in tasks] == [2]
    assert any(key.endswith(".metrics.json") for key in store.read_json_calls)


def test_unified_data_part_join_writes_tokens_and_single_row_group(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    config = load_recipe_config(write_config(tmp_path, rows_per_row_group=2))
    write_external_run_fixture(store, row_count=3)
    task = build_unified_data_manifest_rows(config=config, object_store=store, run_id="out-run")[0]

    metrics = write_unified_data_part(task=task, object_store=store)

    assert metrics["row_count"] == 2
    assert metrics["cleaned_o200k_token_count"] == sum(
        len(tiktoken.get_encoding("o200k_base").encode_ordinary(text))
        for text in ["hello world", ""]
    )
    output_path = tmp_path / store.bucket / task.part_key
    parquet_file = pq.ParquetFile(output_path)
    assert parquet_file.num_row_groups == 1
    assert parquet_file.metadata.row_group(0).num_rows == 2
    table = pq.read_table(output_path)
    assert table.column("cleaned_text").to_pylist() == ["hello world", ""]
    assert table.column("cleaning_is_dropped").to_pylist() == [False, True]
    assert table.column("cleaned_o200k_tokenizer").to_pylist() == ["o200k_base", "o200k_base"]
    assert table.column("lid_cleaned_token_count").to_pylist() == [10, 11]
    assert table.column("malaya_word_detections").to_pylist()[0] == [
        {
            "word_index": 0,
            "start_index": 0,
            "end_index": 5,
            "token": "hello",
            "label": "ENGLISH",
        }
    ]
    assert "malaya_word_label_counts" in table.schema.names
    assert "body" not in table.schema.names
    assert "text" not in table.schema.names
    assert "body_text" not in table.schema.names
    assert "markdown_text" not in table.schema.names


def test_unified_data_part_fails_on_join_row_count_mismatch(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    config = load_recipe_config(write_config(tmp_path, rows_per_row_group=2))
    write_external_run_fixture(store, row_count=2, lid_row_count=1)
    task = build_unified_data_manifest_rows(config=config, object_store=store, run_id="out-run")[0]

    with pytest.raises(ValueError, match="Join row count mismatch"):
        write_unified_data_part(task=task, object_store=store)


def test_unified_data_part_fails_on_original_hash_mismatch(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    config = load_recipe_config(write_config(tmp_path, rows_per_row_group=2))
    write_external_run_fixture(store, row_count=2, bad_lid_hash=True)
    task = build_unified_data_manifest_rows(config=config, object_store=store, run_id="out-run")[0]

    with pytest.raises(ValueError, match="Join validation failed"):
        write_unified_data_part(task=task, object_store=store)


def test_unified_data_part_fails_on_lid_row_order_mismatch(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    config = load_recipe_config(write_config(tmp_path, rows_per_row_group=3))
    write_external_run_fixture(store, row_count=3, reverse_lid_rows=True)
    task = build_unified_data_manifest_rows(config=config, object_store=store, run_id="out-run")[0]

    with pytest.raises(ValueError, match="positionally mismatched rows"):
        write_unified_data_part(task=task, object_store=store)


def test_unified_data_manifest_requires_source_cleaning_done_sentinels(
    tmp_path: Path,
) -> None:
    store = LocalObjectStore(tmp_path)
    config = load_recipe_config(write_config(tmp_path, rows_per_row_group=2))
    write_external_run_fixture(store, row_count=2, write_done=False)

    with pytest.raises(ValueError, match="Source cleaning run is incomplete"):
        build_unified_data_manifest_rows(config=config, object_store=store, run_id="out-run")


def test_unified_data_partial_manifest_uses_only_ready_pairs(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    config = load_recipe_config(
        write_config(tmp_path, rows_per_row_group=2, allow_partial_upstream=True)
    )
    write_external_run_fixture(store, row_count=2)

    assert build_unified_data_manifest_rows(config=config, object_store=store, run_id="out-run")
    done_key = (
        "dataset/processed/source-cleaning/cleaning-run/"
        "done/books-ocr/source.rg00000.done.json"
    )
    (store.storage_root / done_key).unlink()

    with pytest.raises(ValueError, match="No mergeable upstream row groups"):
        build_unified_data_manifest_rows(config=config, object_store=store, run_id="out-run")


def test_unified_data_manifest_can_filter_cleaning_sources(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    config = load_recipe_config(
        write_config(
            tmp_path,
            rows_per_row_group=2,
            allow_partial_upstream=True,
            include_cleaning_sources="reddit-bolehland",
        )
    )
    write_external_run_fixture(store, row_count=2)

    with pytest.raises(ValueError, match="No mergeable upstream row groups"):
        build_unified_data_manifest_rows(config=config, object_store=store, run_id="out-run")


def test_unified_data_completion_tracker_skips_only_done_sentinel(
    tmp_path: Path,
) -> None:
    store = LocalObjectStore(tmp_path)
    config = load_recipe_config(write_config(tmp_path, rows_per_row_group=2))
    write_external_run_fixture(store, row_count=2)
    task = build_unified_data_manifest_rows(config=config, object_store=store, run_id="out-run")[0]
    tracker = UnifiedDataCompletionTracker(store)
    layout = RunArtifactLayout(
        source_root_key="dataset",
        output_root_key="dataset/processed/unified-data/out-run",
    )

    assert tracker.completed_source_keys(
        input_rows=[task.to_dict()],
        artifact_layout=layout,
        allow_overwrite=False,
    ) == set()

    store.write_bytes(task.part_key, b"partial")
    assert tracker.completed_source_keys(
        input_rows=[task.to_dict()],
        artifact_layout=layout,
        allow_overwrite=False,
    ) == set()

    store.write_json(task.done_key, {"status": "success"})
    assert tracker.completed_source_keys(
        input_rows=[task.to_dict()],
        artifact_layout=layout,
        allow_overwrite=False,
    ) == {"part=000000"}


def test_unified_data_submission_uses_independent_remote_entrypoint(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    config_path = write_config(tmp_path, rows_per_row_group=2)
    config = load_recipe_config(config_path)
    write_external_run_fixture(store, row_count=2)

    prepared = UnifiedDataSubmissionAdapter(
        config=config,
        config_path=config_path,
    ).prepare_new_run(FakeArtifactStore(store), dry_run=True)

    assert prepared.discovered_items == 1
    assert "--group unified_data" in prepared.bootstrap.command
    assert "training_signal_processing.main unified-data-remote-job" in prepared.invocation.command


def write_config(
    tmp_path: Path,
    *,
    rows_per_row_group: int,
    allow_partial_upstream: bool = False,
    include_cleaning_sources: str = "",
) -> Path:
    config_path = tmp_path / "unified_data.yaml"
    config_path.write_text(
        f"""
run:
  name: unified-data
  config_version: 1
ssh:
  host: 127.0.0.1
  port: 22
  user: root
  identity_file: ~/.ssh/id_ed25519
remote:
  root_dir: /tmp/training-signal-processing
  python_version: "3.12"
  remote_jobs_root: /tmp/unified-data-jobs
  pgid_wait_attempts: 20
  pgid_wait_sleep_seconds: 0.25
  sync_paths: [pyproject.toml, uv.lock, src, config]
ray:
  executor_type: ray
  batch_size: 1
  concurrency: 2
  target_num_blocks: 2
r2:
  config_file: r2
  bucket: gpu-poor
  output_prefix: dataset/processed/unified-data
input:
  source_root_key: dataset
  lid_metadata_output_prefix: dataset/processed/lid-metadata
  lid_run_id: lid-run
  source_cleaning_output_prefix: dataset/processed/source-cleaning
  source_cleaning_run_id: cleaning-run
  allow_partial_upstream: {str(allow_partial_upstream).lower()}
  include_cleaning_sources: "{include_cleaning_sources}"
export:
  rows_per_row_group: {rows_per_row_group}
  tokenizer_encoding: o200k_base
  tokenizer_threads: 2
  ray_num_cpus_per_worker: 1
  merge_engine: duckdb
  duckdb_threads: 1
  parquet_compression: zstd
  parquet_compression_level: 1
mlflow:
  enabled: false
  tracking_uri: ""
  experiment_name: unified-data
observability:
  flush_interval_sec: 5
  log_per_file_events: true
  heartbeat_interval_sec: 10
resumability:
  strategy: batch_manifest
  commit_every_batches: 1
  resume_mode: latest
ops:
  - name: prepare_unified_data_part
    type: mapper
  - name: write_unified_data_part
    type: mapper
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


def write_external_run_fixture(
    store: LocalObjectStore,
    *,
    row_count: int,
    lid_row_count: int | None = None,
    manifest_row_count: int | None = None,
    bad_lid_hash: bool = False,
    reverse_lid_rows: bool = False,
    filters: dict[str, str] | None = None,
    write_done: bool = True,
) -> None:
    lid_row_count = row_count if lid_row_count is None else lid_row_count
    manifest_row_count = row_count if manifest_row_count is None else manifest_row_count
    filters = filters or {}
    source_key = "dataset/raw/source.parquet"
    lid_root = "dataset/processed/lid-metadata/lid-run"
    cleaning_root = "dataset/processed/source-cleaning/cleaning-run"
    lid_shard_key = f"{lid_root}/shards/books-ocr/source.rg00000.parquet"
    cleaning_unified_key = f"{cleaning_root}/unified/source=books-ocr/source.rg00000.parquet"
    cleaning_metrics_key = f"{cleaning_root}/metrics/books-ocr/source.rg00000.metrics.json"
    cleaning_done_key = f"{cleaning_root}/done/books-ocr/source.rg00000.done.json"
    cleaning_error_key = f"{cleaning_root}/errors/books-ocr/source.rg00000.error.json"

    store.write_jsonl(
        f"{lid_root}/control/input_manifest.jsonl",
        [
            {
                "source_name": "Books + OCR",
                "source_object_key": source_key,
                "source_row_group_index": 0,
                "output_shard_key": lid_shard_key,
            }
        ],
    )
    store.write_jsonl(
        f"{cleaning_root}/control/input_manifest.jsonl",
        [
            {
                "source_name": "Books + OCR",
                "cleaning_source": "books-ocr",
                "source_object_key": source_key,
                "source_row_group_index": 0,
                "source_row_group_num_rows": manifest_row_count,
                "filters": filters,
                "unified_shard_key": cleaning_unified_key,
                "metrics_key": cleaning_metrics_key,
                "done_key": cleaning_done_key,
                "error_key": cleaning_error_key,
            }
        ],
    )
    store.write_json(
        cleaning_metrics_key,
        {
            "filtered_row_count": row_count,
            "source_shard_key": f"{cleaning_root}/source_shards/books-ocr/source.rg00000.parquet",
            "unified_shard_key": cleaning_unified_key,
        },
    )
    if write_done:
        store.write_json(cleaning_done_key, {"status": "success"})

    cleaning_rows = build_cleaning_rows(source_key=source_key, row_count=row_count)
    lid_rows = build_lid_rows(
        source_key=source_key,
        row_count=lid_row_count,
        bad_lid_hash=bad_lid_hash,
    )
    if reverse_lid_rows:
        lid_rows = list(reversed(lid_rows))
    write_table(store, cleaning_unified_key, pa.Table.from_pylist(cleaning_rows))
    write_table(store, lid_shard_key, pa.Table.from_pylist(lid_rows))


def build_cleaning_rows(*, source_key: str, row_count: int) -> list[dict[str, Any]]:
    texts = ["hello world", "", "selamat pagi", "extra"]
    rows = []
    for index in range(row_count):
        original = f"original-{index}"
        cleaned = texts[index % len(texts)]
        rows.append(
            {
                "sample_uid": sample_uid(source_key, index),
                "sample_uid_hash": sha256(sample_uid(source_key, index)),
                "source_name": "Books + OCR",
                "cleaning_source": "books-ocr",
                "source_bucket": "gpu-poor",
                "source_object_key": source_key,
                "source_parquet_url": f"r2://gpu-poor/{source_key}",
                "source_row_group_index": 0,
                "source_row_index": index,
                "row_index_in_row_group": index,
                "text_column": "markdown_text",
                "cleaned_text": cleaned,
                "original_text_sha256": sha256(original),
                "cleaned_text_sha256": sha256(cleaned),
                "original_char_count": len(original),
                "cleaned_char_count": len(cleaned),
                "removed_char_count": max(0, len(original) - len(cleaned)),
                "approximate_original_token_count": 1,
                "approximate_cleaned_token_count": 1 if cleaned else 0,
                "approximate_removed_token_count": 0,
                "cleaning_is_dropped": cleaned == "",
                "cleaning_rules_triggered": ["tiny"] if cleaned == "" else [],
                "document_id": f"doc-{index}",
                "body": cleaned,
            }
        )
    return rows


def build_lid_rows(
    *,
    source_key: str,
    row_count: int,
    bad_lid_hash: bool,
) -> list[dict[str, Any]]:
    rows = []
    for index in range(row_count):
        original = f"original-{index}"
        rows.append(
            {
                "sample_uid": sample_uid(source_key, index),
                "sample_uid_hash": sha256(sample_uid(source_key, index)),
                "source_row_group_index": 0,
                "source_row_index": index,
                "row_index_in_row_group": index,
                "original_text_sha256": sha256("bad") if bad_lid_hash else sha256(original),
                "cleaned_token_count": 10 + index,
                "reference_removed": False,
                "reference_removal_method": "",
                "removed_reference_char_count": 0,
                "lingua_primary_language": "english",
                "lingua_spans": [
                    {"start_index": 0, "end_index": 5, "language_label": "ENGLISH"}
                ],
                "malaya_document_label": "ENGLISH",
                "malaya_document_scores": [{"label": "ENGLISH", "score": 0.98}],
                "malaya_word_detections": [
                    {
                        "word_index": 0,
                        "start_index": 0,
                        "end_index": 5,
                        "token": "hello",
                        "label": "ENGLISH",
                    }
                ],
                "malaya_word_label_counts": [{"label": "ENGLISH", "count": 1}],
            }
        )
    return rows


def write_table(store: LocalObjectStore, key: str, table: pa.Table) -> None:
    path = store.storage_root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def sample_uid(source_key: str, index: int) -> str:
    return f"r2://gpu-poor/{source_key}#row={index}"


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
