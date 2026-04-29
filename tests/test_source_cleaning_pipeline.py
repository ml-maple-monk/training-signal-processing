from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from training_signal_processing.core.models import RunArtifactLayout
from training_signal_processing.core.storage import ObjectStore
from training_signal_processing.core.submission import ArtifactStore
from training_signal_processing.main import cli
from training_signal_processing.pipelines.source_cleaning.config import load_recipe_config
from training_signal_processing.pipelines.source_cleaning.models import (
    SourceCleaningShardResult,
    row_group_source_key,
)
from training_signal_processing.pipelines.source_cleaning.ops import clean_source_row_group
from training_signal_processing.pipelines.source_cleaning.runtime import (
    SourceCleaningCompletionTracker,
    SourceCleaningExporter,
)
from training_signal_processing.pipelines.source_cleaning.submission import (
    SourceCleaningSubmissionAdapter,
    build_row_group_manifest_rows,
    build_row_group_output_keys,
)

BOOKS_OCR_PARQUET_KEY = "dataset/processed/pdf_ocr/20260423T195035Z/markdown.parquet"
REDDIT_PARQUET_KEY = "dataset/processed/malay/reddit.parquet"


class LocalObjectStore(ObjectStore):
    def __init__(self, root: Path, bucket: str = "gpu-poor") -> None:
        self.root = root
        self.storage_root = root / bucket
        self.bucket = bucket

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
        self.json_payloads: dict[str, dict[str, object]] = {}
        self.jsonl_payloads: dict[str, list[dict[str, object]]] = {}

    def as_object_store(self) -> LocalObjectStore:
        return self.object_store

    def exists(self, key: str) -> bool:
        return (
            key in self.json_payloads
            or key in self.jsonl_payloads
            or self.object_store.exists(key)
        )

    def read_json(self, key: str) -> dict[str, object]:
        return self.json_payloads[key]

    def read_jsonl(self, key: str) -> list[dict[str, object]]:
        return self.jsonl_payloads[key]

    def write_json(self, key: str, value: dict[str, object]) -> None:
        self.json_payloads[key] = value

    def write_jsonl(self, key: str, rows: list[dict[str, object]]) -> None:
        self.jsonl_payloads[key] = rows

    def upload_file(self, path: Path, key: str) -> None:
        self.object_store.upload_file(path, key)

    def build_remote_env(self) -> dict[str, str]:
        return {"R2_BUCKET": self.bucket}


def test_source_cleaning_sample_config_loads_recipe() -> None:
    config = load_recipe_config(Path("config/source_cleaning.sample.yaml"))

    assert config.r2.bucket == "gpu-poor"
    assert config.r2.output_prefix == "dataset/processed/source-cleaning"
    assert [source.name for source in config.sources] == [
        "Books + OCR",
        "Lowyat",
        "Reddit Bolehland",
        "Reddit Indonesia",
        "Cari",
        "HPLT Malay",
        "HPLT Indonesia",
    ]
    assert [source.cleaning_source for source in config.sources] == [
        "books-ocr",
        "lowyat",
        "reddit-bolehland",
        "reddit-indonesia",
        "cari",
        "hplt-malay",
        "hplt-indonesia",
    ]
    assert config.sources[0].text_column == "markdown_text"
    assert config.sources[2].filters == {"subreddit": "Bolehland"}
    assert [source.scheduling_weight for source in config.sources] == [1, 2, 1, 1, 1, 8, 12]
    assert config.ray.concurrency == 64
    assert config.ray.target_num_blocks == 8192
    assert config.cleaning.ray_num_cpus_per_worker == pytest.approx(1.0)
    assert config.cleaning.polars_max_threads == 1


def test_source_cleaning_config_accepts_alternate_canary_thread_settings() -> None:
    config = load_recipe_config(
        Path("config/source_cleaning.sample.yaml"),
        [
            "ray.concurrency=32",
            "cleaning.ray_num_cpus_per_worker=2",
            "cleaning.polars_max_threads=2",
        ],
    )

    assert config.ray.concurrency == 32
    assert config.cleaning.ray_num_cpus_per_worker == pytest.approx(2.0)
    assert config.cleaning.polars_max_threads == 2


def test_source_cleaning_config_rejects_unknown_cleaning_source(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        [
            source_block(
                "Books + OCR",
                BOOKS_OCR_PARQUET_KEY,
                "markdown_text",
                "not-a-source",
            )
        ],
    )

    with pytest.raises(ValueError, match="cleaning_source"):
        load_recipe_config(config_path)


def test_source_cleaning_manifest_expands_local_parquet_row_groups(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    parquet_path = tmp_path / "gpu-poor" / BOOKS_OCR_PARQUET_KEY
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"markdown_text": ["one", "two", "three"], "document_id": [1, 2, 3]}),
        parquet_path,
        row_group_size=2,
    )
    config = load_recipe_config(
        write_config(
            tmp_path,
            [
                source_block(
                    "Books + OCR",
                    BOOKS_OCR_PARQUET_KEY,
                    "markdown_text",
                    "books-ocr",
                )
            ],
        )
    )

    rows = build_row_group_manifest_rows(config=config, object_store=store, run_id="run-001")

    assert [(row.source_row_group_index, row.source_row_group_start_index) for row in rows] == [
        (0, 0),
        (1, 2),
    ]
    assert rows[0].source_shard_key.endswith(
        "source_shards/books-ocr/markdown.rg00000.parquet"
    )
    assert rows[0].unified_shard_key.endswith(
        "unified/source=books-ocr/markdown.rg00000.parquet"
    )
    assert rows[0].metrics_key.endswith("metrics/books-ocr/markdown.rg00000.metrics.json")
    assert rows[0].done_key.endswith("done/books-ocr/markdown.rg00000.done.json")
    assert rows[0].error_key.endswith("errors/books-ocr/markdown.rg00000.error.json")


def test_source_cleaning_manifest_uses_source_scheduling_weights(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    books_path = tmp_path / "gpu-poor" / BOOKS_OCR_PARQUET_KEY
    hplt_path = tmp_path / "gpu-poor/dataset/processed/malay/hplt/sample_malay.parquet"
    books_path.parent.mkdir(parents=True)
    hplt_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"markdown_text": ["b0", "b1", "b2", "b3"]}),
        books_path,
        row_group_size=1,
    )
    pq.write_table(
        pa.table({"text": ["h0", "h1", "h2", "h3"]}),
        hplt_path,
        row_group_size=1,
    )
    config = load_recipe_config(
        write_config(
            tmp_path,
            [
                source_block(
                    "Books + OCR",
                    BOOKS_OCR_PARQUET_KEY,
                    "markdown_text",
                    "books-ocr",
                    scheduling_weight=1,
                ),
                source_block(
                    "HPLT Malay",
                    "dataset/processed/malay/hplt/*_malay.parquet",
                    "text",
                    "hplt-malay",
                    scheduling_weight=3,
                ),
            ],
        )
    )

    rows = build_row_group_manifest_rows(config=config, object_store=store, run_id="run-001")

    assert [row.source_name for row in rows] == [
        "Books + OCR",
        "HPLT Malay",
        "HPLT Malay",
        "HPLT Malay",
        "Books + OCR",
        "HPLT Malay",
        "Books + OCR",
        "Books + OCR",
    ]
    assert {row.source_scheduling_weight for row in rows if row.source_name == "HPLT Malay"} == {3}


def test_source_cleaning_output_key_layout_is_stable() -> None:
    keys = build_row_group_output_keys(
        output_root_key="dataset/processed/source-cleaning/run-001",
        source="Reddit Bolehland",
        source_key=REDDIT_PARQUET_KEY,
        row_group_index=12,
    )

    assert keys == {
        "source_shard_key": (
            "dataset/processed/source-cleaning/run-001/"
            "source_shards/reddit-bolehland/reddit.rg00012.parquet"
        ),
        "unified_shard_key": (
            "dataset/processed/source-cleaning/run-001/"
            "unified/source=reddit-bolehland/reddit.rg00012.parquet"
        ),
        "metrics_key": (
            "dataset/processed/source-cleaning/run-001/"
            "metrics/reddit-bolehland/reddit.rg00012.metrics.json"
        ),
        "done_key": (
            "dataset/processed/source-cleaning/run-001/"
            "done/reddit-bolehland/reddit.rg00012.done.json"
        ),
        "error_key": (
            "dataset/processed/source-cleaning/run-001/"
            "errors/reddit-bolehland/reddit.rg00012.error.json"
        ),
    }


def test_source_cleaning_row_group_writes_traceable_shards(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    parquet_path = tmp_path / "gpu-poor" / REDDIT_PARQUET_KEY
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "body": ["[deleted]", "jangan pilih", "aku rasa ok"],
                "subreddit": ["Bolehland", "indonesia", "Bolehland"],
                "score": [1, 2, 3],
                "post_id": ["p1", "p2", "p3"],
            }
        ),
        parquet_path,
        row_group_size=3,
    )
    config = load_recipe_config(
        write_config(
            tmp_path,
            [
                source_block(
                    "Reddit Bolehland",
                    REDDIT_PARQUET_KEY,
                    "body",
                    "reddit-bolehland",
                    {"subreddit": "Bolehland"},
                )
            ],
        )
    )
    task = build_row_group_manifest_rows(config=config, object_store=store, run_id="run-001")[0]

    metrics = clean_source_row_group(task=task, object_store=store)

    assert metrics["filtered_row_count"] == 2
    assert metrics["dropped_row_count"] == 1
    source_shard = pq.read_table(tmp_path / "gpu-poor" / task.source_shard_key).to_pydict()
    assert source_shard["source_row_index"] == [0, 2]
    assert source_shard["row_index_in_row_group"] == [0, 2]
    assert source_shard["sample_uid"] == [
        f"r2://gpu-poor/{REDDIT_PARQUET_KEY}#row=0",
        f"r2://gpu-poor/{REDDIT_PARQUET_KEY}#row=2",
    ]
    assert source_shard["body"] == ["", "aku rasa ok"]
    assert source_shard["cleaned_text"] == ["", "aku rasa ok"]
    assert source_shard["score"] == [1, 3]
    assert source_shard["cleaning_is_dropped"] == [True, False]
    assert source_shard["cleaning_rules_triggered"][0] == [
        "reddit_bolehland.deleted_placeholder"
    ]
    assert source_shard["original_char_count"] == [9, 11]
    assert source_shard["cleaned_char_count"] == [0, 11]
    assert all(len(value) == 64 for value in source_shard["sample_uid_hash"])
    assert all(len(value) == 64 for value in source_shard["original_text_sha256"])

    unified_shard = pq.read_table(tmp_path / "gpu-poor" / task.unified_shard_key).to_pydict()
    assert unified_shard["cleaning_source"] == ["reddit-bolehland", "reddit-bolehland"]
    assert unified_shard["body"] == ["", "aku rasa ok"]
    assert unified_shard["score"] == ["1", "3"]
    assert unified_shard["text_column"] == ["body", "body"]


def test_source_cleaning_row_group_allows_empty_filtered_shards(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    parquet_path = tmp_path / "gpu-poor" / REDDIT_PARQUET_KEY
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"body": ["aku suka"], "subreddit": ["Bolehland"]}),
        parquet_path,
        row_group_size=1,
    )
    config = load_recipe_config(
        write_config(
            tmp_path,
            [
                source_block(
                    "Reddit Indonesia",
                    REDDIT_PARQUET_KEY,
                    "body",
                    "reddit-indonesia",
                    {"subreddit": "indonesia"},
                )
            ],
        )
    )
    task = build_row_group_manifest_rows(config=config, object_store=store, run_id="run-001")[0]

    metrics = clean_source_row_group(task=task, object_store=store)

    assert metrics["filtered_row_count"] == 0
    assert pq.ParquetFile(tmp_path / "gpu-poor" / task.source_shard_key).metadata.num_rows == 0
    assert pq.ParquetFile(tmp_path / "gpu-poor" / task.unified_shard_key).metadata.num_rows == 0


def test_source_cleaning_completion_uses_done_sentinel(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    task = build_sample_task(tmp_path, store)
    tracker = SourceCleaningCompletionTracker(store)
    layout = RunArtifactLayout(
        source_root_key="dataset",
        output_root_key="dataset/processed/source-cleaning/run-001",
    )

    store.write_bytes(task.source_shard_key, b"partial parquet payload")
    assert tracker.completed_source_keys(
        input_rows=[task.to_dict()],
        artifact_layout=layout,
        allow_overwrite=False,
    ) == set()

    result = SourceCleaningShardResult.success_from_task(
        task=task,
        metrics={"filtered_row_count": 1},
        duration_sec=0.1,
    )
    exporter = SourceCleaningExporter(store)
    export_result = exporter.export_batch("batch-00001", [result.to_dict()])

    assert task.metrics_key in export_result.output_keys
    assert task.done_key in export_result.output_keys
    assert tracker.completed_source_keys(
        input_rows=[task.to_dict()],
        artifact_layout=layout,
        allow_overwrite=False,
    ) == {row_group_source_key(task)}
    assert tracker.completed_source_keys(
        input_rows=[task.to_dict()],
        artifact_layout=layout,
        allow_overwrite=True,
    ) == set()
    assert json.loads(store.read_bytes(task.done_key))["status"] == "success"


def test_source_cleaning_cli_and_submission_use_source_cleaning_group(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    parquet_path = tmp_path / "gpu-poor" / BOOKS_OCR_PARQUET_KEY
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"markdown_text": ["hello"], "document_id": [1]}), parquet_path)
    config_path = write_config(
        tmp_path,
        [
            source_block(
                "Books + OCR",
                BOOKS_OCR_PARQUET_KEY,
                "markdown_text",
                "books-ocr",
            )
        ],
    )
    config = load_recipe_config(config_path)
    adapter = SourceCleaningSubmissionAdapter(config=config, config_path=config_path)

    prepared = adapter.prepare_new_run(FakeArtifactStore(store), dry_run=True)
    cli_result = CliRunner().invoke(cli, ["source-cleaning-validate", "--config", str(config_path)])

    assert cli_result.exit_code == 0
    assert "Validated source cleaning recipe" in cli_result.output
    assert "Declared sources: 1" in cli_result.output
    assert "source-cleaning-run" in cli.commands
    assert "source-cleaning-resume" in cli.commands
    assert "source-cleaning-remote-job" in cli.commands
    assert "--group source_cleaning --no-dev --frozen" in prepared.bootstrap.command
    assert " --group source_cleaning " in prepared.invocation.command
    assert prepared.invocation.env["POLARS_MAX_THREADS"] == "1"
    assert "remote_ocr" not in prepared.bootstrap.command


def build_sample_task(tmp_path: Path, store: LocalObjectStore):
    parquet_path = tmp_path / "gpu-poor" / BOOKS_OCR_PARQUET_KEY
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"markdown_text": ["hello"], "document_id": [1]}), parquet_path)
    config = load_recipe_config(
        write_config(
            tmp_path,
            [
                source_block(
                    "Books + OCR",
                    BOOKS_OCR_PARQUET_KEY,
                    "markdown_text",
                    "books-ocr",
                )
            ],
        )
    )
    return build_row_group_manifest_rows(config=config, object_store=store, run_id="run-001")[0]


def write_config(tmp_path: Path, source_blocks: list[str]) -> Path:
    config_path = tmp_path / "source_cleaning.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: source-cleaning-test",
                "  config_version: 1",
                "ssh:",
                "  host: 127.0.0.1",
                "  port: 22",
                "  user: root",
                "  identity_file: ~/.ssh/id_ed25519",
                "remote:",
                "  root_dir: /tmp/training-signal-processing",
                "  python_version: '3.12'",
                "  remote_jobs_root: /tmp/source-cleaning-jobs",
                "  pgid_wait_attempts: 1",
                "  pgid_wait_sleep_seconds: 0.1",
                "  sync_paths:",
                "    - pyproject.toml",
                "    - uv.lock",
                "    - src",
                "ray:",
                "  executor_type: ray",
                "  batch_size: 1",
                "  concurrency: 1",
                "  target_num_blocks: 1",
                "r2:",
                "  config_file: r2",
                "  bucket: gpu-poor",
                "  output_prefix: dataset/processed/source-cleaning",
                "input:",
                "  source_root_key: dataset",
                "cleaning:",
                "  ray_num_cpus_per_worker: 1",
                "  polars_max_threads: 1",
                "mlflow:",
                "  enabled: false",
                "  tracking_uri: ''",
                "  experiment_name: source-cleaning",
                "observability:",
                "  flush_interval_sec: 5",
                "  log_per_file_events: true",
                "  heartbeat_interval_sec: 10",
                "resumability:",
                "  strategy: batch_manifest",
                "  commit_every_batches: 1",
                "  resume_mode: latest",
                "ops:",
                "  - name: prepare_source_cleaning_row_group",
                "    type: mapper",
                "  - name: clean_source_row_group",
                "    type: mapper",
                "sources:",
                *source_blocks,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def source_block(
    name: str,
    r2_relative_glob_path: str,
    text_column: str,
    cleaning_source: str,
    filters: dict[str, str] | None = None,
    scheduling_weight: int = 1,
) -> str:
    block = (
        f"  - name: {name!r}\n"
        "    format: parquet\n"
        f"    r2_relative_glob_path: {r2_relative_glob_path!r}\n"
        f"    text_column: {text_column!r}\n"
        f"    cleaning_source: {cleaning_source!r}\n"
        f"    scheduling_weight: {scheduling_weight}"
    )
    if not filters:
        return block
    filter_lines = ["    filters:"]
    for key, value in filters.items():
        filter_lines.append(f"      {key}: {value!r}")
    return block + "\n" + "\n".join(filter_lines)
