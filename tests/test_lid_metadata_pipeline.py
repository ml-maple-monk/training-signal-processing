from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from training_signal_processing.core.models import RayConfig
from training_signal_processing.core.storage import ObjectStore
from training_signal_processing.core.submission import ArtifactStore
from training_signal_processing.main import cli
from training_signal_processing.pipelines.lid_metadata.config import load_recipe_config
from training_signal_processing.pipelines.lid_metadata.models import (
    LidConfig,
    LidMetadataShardResult,
    ParquetRowGroupTask,
    sample_uid,
    sample_uid_hash,
)
from training_signal_processing.pipelines.lid_metadata.ops import (
    MalayaRuntime,
    PrepareLidMetadataRowGroupOp,
    build_checkpoint_key,
    count_text_tokens,
    detect_lid_metadata_for_row_group,
    detect_lid_metadata_shard,
    maybe_run_process_pool_safety_check,
    row_group_source_key,
    trim_reference_section,
)
from training_signal_processing.pipelines.lid_metadata.runtime import (
    LidMetadataParquetExporter,
    resolve_lid_metadata_transform_resources,
)
from training_signal_processing.pipelines.lid_metadata.submission import (
    LidMetadataSubmissionAdapter,
    build_row_group_manifest_rows,
)


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


class CountingObjectStore(LocalObjectStore):
    def __init__(self, root: Path, bucket: str = "gpu-poor") -> None:
        super().__init__(root, bucket)
        self.written_keys: list[str] = []

    def write_bytes(self, key: str, body: bytes) -> None:
        self.written_keys.append(key)
        super().write_bytes(key, body)


class FakeArtifactStore(ArtifactStore):
    def __init__(self, object_store: LocalObjectStore) -> None:
        self.object_store = object_store
        self.bucket = object_store.bucket
        self.jsonl_payloads: dict[str, list[dict[str, object]]] = {}
        self.json_payloads: dict[str, dict[str, object]] = {}

    def as_object_store(self) -> LocalObjectStore:
        return self.object_store

    def exists(self, key: str) -> bool:
        return (
            key in self.jsonl_payloads
            or key in self.json_payloads
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


def test_lid_metadata_sample_config_loads_recipe() -> None:
    config = load_recipe_config(Path("config/lid_metadata.sample.yaml"))

    assert config.r2.bucket == "gpu-poor"
    assert config.r2.output_prefix == "dataset/processed/lid-metadata"
    assert [source.name for source in config.sources] == [
        "Books + OCR",
        "Lowyat",
        "Reddit Bolehland",
        "Reddit Indonesia",
        "Cari",
        "HPLT Malay",
        "HPLT Indonesia",
    ]
    assert config.sources[0].reference_removal.enabled is True
    assert config.sources[1].reference_removal.enabled is False
    assert config.lid.tokenizer_encoding == "o200k_base"
    assert config.lid.row_batch_size == 64
    assert config.lid.ray_num_cpus_per_worker == pytest.approx(1.0)
    assert config.lid.inner_parallelism == "none"
    assert config.lid.variant_name == "row-batch-64"


def test_lid_metadata_config_rejects_missing_text_column(tmp_path: Path) -> None:
    config_path = write_lid_config(tmp_path, text_column="")

    with pytest.raises(ValueError, match="text_column is required"):
        load_recipe_config(config_path)


def test_lid_metadata_transform_resources_use_configured_cpu_per_worker() -> None:
    config = load_recipe_config(Path("config/lid_metadata.sample.yaml"))
    config.lid.ray_num_cpus_per_worker = 2.0
    execution = RayConfig(
        executor_type="ray",
        batch_size=1,
        concurrency=64,
        target_num_blocks=8192,
    )

    resources = resolve_lid_metadata_transform_resources(
        config,
        SimpleNamespace(name="detect_lid_metadata_row_group"),
        execution,
    )

    assert resources.concurrency == 64
    assert resources.num_cpus == pytest.approx(2.0)


def test_parquet_row_group_task_drops_null_filter_values_from_arrow_structs() -> None:
    table = pa.Table.from_pylist(
        [
            {"filters": {}},
            {"filters": {"subreddit": "Bolehland"}},
        ]
    )
    inferred_empty_filters = table.to_pylist()[0]["filters"]
    row = {
        "source_order": 0,
        "source_name": "Books + OCR",
        "source_bucket": "gpu-poor",
        "source_object_key": "dataset/processed/pdf_ocr/markdown.parquet",
        "source_parquet_url": "r2://gpu-poor/dataset/processed/pdf_ocr/markdown.parquet",
        "source_row_group_index": 0,
        "source_row_group_start_index": 0,
        "source_row_group_num_rows": 1,
        "text_column": "markdown_text",
        "filters": inferred_empty_filters,
        "pass_through_columns": [],
        "output_shard_key": "dataset/processed/lid-metadata/run/shards/books/rg00000.parquet",
    }

    assert inferred_empty_filters == {"subreddit": None}
    assert ParquetRowGroupTask.from_dict(row).filters == {}


def test_lid_metadata_config_rejects_unsafe_experiment_values(tmp_path: Path) -> None:
    config_path = write_lid_config(tmp_path)

    with pytest.raises(ValueError, match="inner_parallelism"):
        load_recipe_config(config_path, ["lid.inner_parallelism=fork"])

    with pytest.raises(ValueError, match="inner_workers"):
        load_recipe_config(config_path, ["lid.inner_workers=0"])


def test_sample_uid_uses_parquet_url_row_index_shape() -> None:
    uid = sample_uid(
        bucket="gpu-poor",
        source_object_key="dataset/processed/example.parquet",
        source_row_index=42,
    )

    assert uid == "r2://gpu-poor/dataset/processed/example.parquet#row=42"
    assert len(sample_uid_hash(uid)) == 64


def test_trim_reference_section_preserves_article_body() -> None:
    text = (
        "# Title\n\n"
        "Main article body.\n\n"
        "### REFERENCES\n\n"
        "- Smith, J. 2020. Example. https://doi.org/10.1000/example\n"
    )

    assert trim_reference_section(text=text, heading_names=("REFERENCES",)) == (
        "# Title\n\nMain article body."
    )


def test_row_group_manifest_expands_parquet_row_groups(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    parquet_path = tmp_path / "gpu-poor/dataset/processed/example.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"body": ["one", "two", "three"], "subreddit": ["a", "b", "a"]}),
        parquet_path,
        row_group_size=2,
    )
    config = load_recipe_config(write_lid_config(tmp_path))

    rows = build_row_group_manifest_rows(
        config=config,
        object_store=store,
        run_id="run-001",
    )

    assert [(row.source_row_group_index, row.source_row_group_start_index) for row in rows] == [
        (0, 0),
        (1, 2),
    ]
    assert rows[0].output_shard_key.endswith("example.rg00000.parquet")


def test_row_group_manifest_interleaves_sources_for_early_parallelism(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    parquet_root = tmp_path / "gpu-poor/dataset/processed"
    parquet_root.mkdir(parents=True)
    pq.write_table(
        pa.table({"body": ["a1", "a2", "a3"]}),
        parquet_root / "a.parquet",
        row_group_size=1,
    )
    pq.write_table(pa.table({"body": ["b1", "b2"]}), parquet_root / "b.parquet", row_group_size=1)
    config_path = write_lid_config(tmp_path)
    text = config_path.read_text(encoding="utf-8")
    text = text.replace(
        "sources:\n"
        "  - name: Reddit Bolehland\n"
        "    format: parquet\n"
        "    r2_relative_glob_path: dataset/processed/example.parquet\n"
        "    text_column: 'body'\n"
        "    filters:\n"
        "      subreddit: Bolehland\n"
        "    reference_removal:\n"
        "      enabled: false\n",
        "sources:\n"
        "  - name: Source A\n"
        "    format: parquet\n"
        "    r2_relative_glob_path: dataset/processed/a.parquet\n"
        "    text_column: body\n"
        "  - name: Source B\n"
        "    format: parquet\n"
        "    r2_relative_glob_path: dataset/processed/b.parquet\n"
        "    text_column: body\n",
    )
    config_path.write_text(text, encoding="utf-8")
    config = load_recipe_config(config_path)

    rows = build_row_group_manifest_rows(config=config, object_store=store, run_id="run-001")

    assert [row.source_name for row in rows] == [
        "Source A",
        "Source B",
        "Source A",
        "Source B",
        "Source A",
    ]


def test_lid_detection_row_group_outputs_traceable_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_lid_models(monkeypatch)
    store = LocalObjectStore(tmp_path)
    parquet_path = tmp_path / "gpu-poor/dataset/processed/example.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "body": [
                    "Aku suka this model because dia boleh handle mixed bahasa Indonesia juga.",
                    "skip me",
                ],
                "subreddit": ["Bolehland", "indonesia"],
                "url": ["https://example.test/a", "https://example.test/b"],
            }
        ),
        parquet_path,
        row_group_size=2,
    )
    config = load_recipe_config(write_lid_config(tmp_path))
    task = build_row_group_manifest_rows(
        config=config,
        object_store=store,
        run_id="run-001",
    )[0]

    records = detect_lid_metadata_for_row_group(task=task, object_store=store)

    assert len(records) == 1
    row = records[0]
    assert row["sample_uid"] == "r2://gpu-poor/dataset/processed/example.parquet#row=0"
    assert row["source_row_index"] == 0
    assert row["row_index_in_row_group"] == 0
    assert row["subreddit"] == "Bolehland"
    assert {span["language_label"] for span in row["lingua_spans"]} <= {
        "ENGLISH",
        "MALAY",
        "INDONESIAN",
    }
    assert row["malaya_document_label"] == "standard-malay"
    assert row["malaya_word_label_counts"] == [
        {"label": "EN", "count": 1},
        {"label": "MS", "count": 2},
    ]
    assert row["cleaned_token_count"] > 0


def test_lid_detection_shard_writes_checkpoint_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_lid_models(monkeypatch)
    store = CountingObjectStore(tmp_path)
    parquet_path = tmp_path / "gpu-poor/dataset/processed/example.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "body": ["saya suka chicken", "saya suka ayam"],
                "subreddit": ["Bolehland", "Bolehland"],
            }
        ),
        parquet_path,
        row_group_size=2,
    )
    config = load_recipe_config(
        write_lid_config(tmp_path),
        [
            "lid.row_batch_size=1",
            "lid.checkpoint_every_rows=1",
            "lid.variant_name=batch-1",
        ],
    )
    task = build_row_group_manifest_rows(config=config, object_store=store, run_id="run-001")[0]

    records, metrics = detect_lid_metadata_shard(
        task=task,
        object_store=store,
        lid_config=config.lid,
    )

    checkpoint_key = build_checkpoint_key(task)
    assert len(records) == 2
    assert metrics["row_count"] == 2
    assert metrics["success_count"] == 2
    assert metrics["failed_count"] == 0
    assert metrics["cleaned_token_count"] == sum(
        row["cleaned_token_count"] for row in records
    )
    assert metrics["tokens_per_sec"] > 0
    assert metrics["variant_name"] == "batch-1"
    assert checkpoint_key in store.written_keys
    checkpoint = store.read_json(checkpoint_key)
    assert checkpoint["rows_completed"] == 2
    assert checkpoint["current_tokens_per_sec"] > 0


def test_lid_token_count_uses_configured_encoding() -> None:
    assert count_text_tokens("Aku suka this model", encoding_name="o200k_base") > 0


def test_process_pool_probe_uses_spawn_timeout_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from training_signal_processing.pipelines.lid_metadata import ops

    contexts: list[str] = []
    shutdown_calls: list[tuple[bool, bool]] = []

    class FakeProcessPoolExecutor:
        def __init__(self, *, max_workers: int, mp_context: object) -> None:
            self.max_workers = max_workers
            self.mp_context = mp_context

        def submit(self, *args: object) -> object:
            return object()

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            shutdown_calls.append((wait, cancel_futures))

    def fake_get_context(name: str) -> object:
        contexts.append(name)
        return object()

    def fake_as_completed(futures: list[object], timeout: float):
        del futures, timeout
        raise ops.TimeoutError()
        yield

    monkeypatch.setattr(ops.multiprocessing, "get_context", fake_get_context)
    monkeypatch.setattr(ops, "ProcessPoolExecutor", FakeProcessPoolExecutor)
    monkeypatch.setattr(ops, "as_completed", fake_as_completed)

    reason = maybe_run_process_pool_safety_check(
        samples=[{"cleaned_text": "saya suka ayam"}],
        lid_config=LidConfig(
            inner_parallelism="process_pool",
            inner_workers=2,
            multiprocessing_context="spawn",
            process_pool_timeout_seconds=0.1,
        ),
    )

    assert contexts == ["spawn"]
    assert reason.startswith("process_pool_timeout_fallback_to_none")
    assert shutdown_calls == [(False, True)]


def test_lid_metadata_exporter_writes_parquet_shard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_fake_lid_models(monkeypatch)
    store = LocalObjectStore(tmp_path)
    parquet_path = tmp_path / "gpu-poor/dataset/processed/example.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"body": ["saya suka chicken"], "subreddit": ["Bolehland"]}),
        parquet_path,
        row_group_size=1,
    )
    config = load_recipe_config(write_lid_config(tmp_path))
    task = build_row_group_manifest_rows(config=config, object_store=store, run_id="run-001")[0]
    records = detect_lid_metadata_for_row_group(task=task, object_store=store)
    result = LidMetadataShardResult.success_from_task(
        task=task,
        records=records,
        duration_sec=0.1,
    )
    exporter = LidMetadataParquetExporter(store)

    export_result = exporter.export_batch("batch-00001", [result.to_dict()])

    metrics_key = task.output_shard_key.removesuffix(".parquet") + ".metrics.json"
    assert export_result.output_keys == [task.output_shard_key, metrics_key]
    table = pq.read_table(tmp_path / "gpu-poor" / task.output_shard_key)
    assert table.num_rows == 1
    assert table.column("sample_uid").to_pylist() == [
        "r2://gpu-poor/dataset/processed/example.parquet#row=0"
    ]
    assert table.column("cleaned_token_count").to_pylist()[0] > 0
    metrics = store.read_json(metrics_key)
    assert metrics["status"] == "success"
    assert metrics["success_count"] == 1


def test_lid_metadata_cli_and_submission_use_lid_dependency_group(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    parquet_path = tmp_path / "gpu-poor/dataset/processed/example.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"body": ["hello"], "subreddit": ["Bolehland"]}), parquet_path)
    config_path = write_lid_config(tmp_path)
    config = load_recipe_config(config_path)
    adapter = LidMetadataSubmissionAdapter(config=config, config_path=config_path)

    prepared = adapter.prepare_new_run(FakeArtifactStore(store), dry_run=True)

    assert "lid-metadata-validate" in cli.commands
    assert "lid-metadata-run" in cli.commands
    assert "lid-metadata-resume" in cli.commands
    assert "lid-metadata-remote-job" in cli.commands
    assert "--group lid_metadata --no-dev --frozen" in prepared.bootstrap.command
    assert " --group lid_metadata " in prepared.invocation.command
    assert "remote_ocr" not in prepared.bootstrap.command
    assert "source_accounting" not in prepared.bootstrap.command


def test_lid_metadata_dstack_prepare_writes_control_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = LocalObjectStore(tmp_path)
    artifact_store = FakeArtifactStore(store)
    parquet_path = tmp_path / "gpu-poor/dataset/processed/example.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"body": ["hello"], "subreddit": ["Bolehland"]}), parquet_path)
    config_path = write_lid_config(tmp_path)
    monkeypatch.setattr(
        "training_signal_processing.main.R2ArtifactStore.from_config_file",
        lambda config: artifact_store,
    )

    result = CliRunner().invoke(
        cli,
        [
            "lid-metadata-dstack-prepare",
            "--config",
            str(config_path),
            "--set",
            "ray.concurrency=128",
            "--set",
            "ray.target_num_blocks=4096",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    prepared = payload["prepared_run"]
    dstack_env = payload["dstack_env"]
    assert prepared["discovered_items"] == 1
    assert prepared["is_resume"] is False
    assert prepared["invocation"]["env_keys"] == ["R2_BUCKET"]
    assert dstack_env["LID_METADATA_RUN_ID"] == prepared["run_id"]
    assert dstack_env["LID_METADATA_UPLOADED_ITEMS"] == "0"
    assert dstack_env["LID_METADATA_CONFIG_OBJECT_KEY"] in artifact_store.json_payloads
    assert dstack_env["LID_METADATA_INPUT_MANIFEST_KEY"] in artifact_store.jsonl_payloads
    assert artifact_store.json_payloads[dstack_env["LID_METADATA_CONFIG_OBJECT_KEY"]]["ray"][
        "concurrency"
    ] == 128
    assert artifact_store.json_payloads[dstack_env["LID_METADATA_CONFIG_OBJECT_KEY"]]["ray"][
        "target_num_blocks"
    ] == 4096


def test_lid_metadata_dstack_configs_do_not_sync_secret_files() -> None:
    config_paths = [
        Path("infra/dstack/config/lid-metadata-canary-cpu2.dstack.yml"),
        Path("infra/dstack/config/lid-metadata-cpu32.dstack.yml"),
        Path("infra/dstack/config/lid-metadata-cpu64.dstack.yml"),
    ]
    for config_path in config_paths:
        text = config_path.read_text(encoding="utf-8")
        assert "files:" not in text
        assert "infra/credentials" not in text
        assert "\nr2\n" not in text

    task_text = "\n".join(path.read_text(encoding="utf-8") for path in config_paths[1:])
    assert "registry_auth:" in task_text
    assert "VCR_USERNAME" in task_text
    assert "VCR_PASSWORD" in task_text
    assert "R2_SECRET_ACCESS_KEY" in task_text

    production_text = config_paths[1].read_text(encoding="utf-8")
    assert "nodes:" not in production_text
    assert "cpu: 32.." in production_text

    cpu64_text = config_paths[2].read_text(encoding="utf-8")
    assert "nodes:" not in cpu64_text
    assert "CPU.64V.256G" in cpu64_text
    assert "cpu: 64.." in cpu64_text
    assert "memory: 128GB.." in production_text
    assert "LID_METADATA_RUN_ID" in production_text


def test_lid_prepare_skips_completed_shard_but_not_checkpoint_only(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    parquet_path = tmp_path / "gpu-poor/dataset/processed/example.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"body": ["hello"], "subreddit": ["Bolehland"]}), parquet_path)
    config = load_recipe_config(write_lid_config(tmp_path))
    task = build_row_group_manifest_rows(config=config, object_store=store, run_id="run-001")[0]
    op = PrepareLidMetadataRowGroupOp().bind_runtime(
        SimpleNamespace(
            completed_source_keys={row_group_source_key(task)},
            allow_overwrite=False,
        )
    )

    assert op.process_row(task.to_dict()) is None

    checkpoint_key = build_checkpoint_key(task)
    store.write_json(checkpoint_key, {"rows_completed": 1})
    op = PrepareLidMetadataRowGroupOp().bind_runtime(
        SimpleNamespace(completed_source_keys=set(), allow_overwrite=False)
    )

    assert op.process_row(task.to_dict()) == task.to_dict()


def test_lid_exporter_writes_error_metadata_without_completed_shard(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    config = load_recipe_config(write_lid_config(tmp_path))
    parquet_path = tmp_path / "gpu-poor/dataset/processed/example.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"body": ["hello"], "subreddit": ["Bolehland"]}), parquet_path)
    task = build_row_group_manifest_rows(config=config, object_store=store, run_id="run-001")[0]
    result = LidMetadataShardResult.failed_from_task(
        task=task,
        error_message="boom",
        duration_sec=0.1,
    )

    export_result = LidMetadataParquetExporter(store).export_batch(
        "batch-00001",
        [result.to_dict()],
    )

    error_key = task.output_shard_key.removesuffix(".parquet") + ".error.json"
    assert export_result.output_keys == [error_key]
    assert not store.exists(task.output_shard_key)
    error_payload = store.read_json(error_key)
    assert error_payload["status"] == "failed"
    assert error_payload["metrics"]["failed_count"] == 1


def test_lid_metadata_cli_validate_loads_recipe(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["lid-metadata-validate", "--config", str(write_lid_config(tmp_path))],
    )

    assert result.exit_code == 0
    assert "Validated LID metadata recipe" in result.output
    assert "Declared sources: 1" in result.output


def install_fake_lid_models(monkeypatch: pytest.MonkeyPatch) -> None:
    from training_signal_processing.pipelines.lid_metadata import ops

    monkeypatch.setattr(ops, "_LINGUA_DETECTOR", FakeLinguaDetector())
    monkeypatch.setattr(ops, "_MALAYA_RUNTIME_QUANTIZED", True)
    monkeypatch.setattr(
        ops,
        "_MALAYA_RUNTIME",
        MalayaRuntime(
            fasttext_model=FakeFastTextModel(),
            word_model=FakeWordModel(),
            tokenizer=FakeTokenizer(),
        ),
    )


class FakeLinguaDetector:
    def detect_multiple_languages_of(self, text: str):
        del text
        return [
            SimpleNamespace(
                start_index=0,
                end_index=8,
                language=SimpleNamespace(name="MALAY"),
            ),
            SimpleNamespace(
                start_index=9,
                end_index=21,
                language=SimpleNamespace(name="ENGLISH"),
            ),
            SimpleNamespace(
                start_index=22,
                end_index=31,
                language=SimpleNamespace(name="INDONESIAN"),
            ),
        ]


class FakeFastTextModel:
    def predict_proba(self, texts: list[str]):
        del texts
        return [{"standard-malay": 0.8, "standard-indonesian": 0.2}]


class FakeWordModel:
    def predict(self, tokens: list[str]):
        return ["MS" if token in {"saya", "suka"} else "EN" for token in tokens]


class FakeTokenizer:
    def tokenize(self, text: str):
        del text
        return ["saya", "suka", "chicken"]


def write_lid_config(tmp_path: Path, text_column: str = "body") -> Path:
    config_path = tmp_path / "lid_metadata.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: lid-metadata-test",
                "  config_version: 1",
                "ssh:",
                "  host: 127.0.0.1",
                "  port: 22",
                "  user: root",
                "  identity_file: ~/.ssh/id_ed25519",
                "remote:",
                "  root_dir: /tmp/training-signal-processing",
                "  python_version: '3.12'",
                "  remote_jobs_root: /tmp/lid-metadata-jobs",
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
                "  output_prefix: dataset/processed/lid-metadata",
                "input:",
                "  source_root_key: dataset",
                "lid:",
                "  lingua_languages:",
                "    - english",
                "    - malay",
                "    - indonesian",
                "  malaya_fasttext_quantized: true",
                "mlflow:",
                "  enabled: false",
                "  tracking_uri: ''",
                "  experiment_name: lid-metadata",
                "observability:",
                "  flush_interval_sec: 5",
                "  log_per_file_events: true",
                "  heartbeat_interval_sec: 10",
                "resumability:",
                "  strategy: batch_manifest",
                "  commit_every_batches: 1",
                "  resume_mode: latest",
                "ops:",
                "  - name: prepare_lid_metadata_row_group",
                "    type: mapper",
                "  - name: detect_lid_metadata_row_group",
                "    type: mapper",
                "sources:",
                "  - name: Reddit Bolehland",
                "    format: parquet",
                "    r2_relative_glob_path: dataset/processed/example.parquet",
                f"    text_column: {text_column!r}",
                "    filters:",
                "      subreddit: Bolehland",
                "    reference_removal:",
                "      enabled: false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path
