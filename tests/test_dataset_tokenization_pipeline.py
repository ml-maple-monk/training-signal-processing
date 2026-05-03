from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner

from training_signal_processing.core.storage import ObjectStore
from training_signal_processing.core.submission import ArtifactStore
from training_signal_processing.main import cli
from training_signal_processing.pipelines.dataset_tokenization.config import load_recipe_config
from training_signal_processing.pipelines.dataset_tokenization.models import (
    DatasetTokenizationResult,
    DatasetTokenizationTask,
)
from training_signal_processing.pipelines.dataset_tokenization.ops import (
    tokenized_output_schema,
    write_dataset_tokenization_part,
)
from training_signal_processing.pipelines.dataset_tokenization.runtime import (
    resolve_dataset_tokenization_transform_resources,
)
from training_signal_processing.pipelines.dataset_tokenization.submission import (
    DatasetTokenizationSubmissionAdapter,
    build_dataset_tokenization_manifest_rows,
    collect_fineweb_availability,
    discover_dataset_tokenization_inputs,
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


class FakeTokenizer:
    def encode_batch(self, texts, add_special_tokens=False):  # type: ignore[no-untyped-def]
        assert add_special_tokens is False
        return [
            SimpleNamespace(ids=[len(word) for word in text.split()] or [0])
            for text in texts
        ]


def test_dataset_tokenization_config_loads_and_cli_is_registered() -> None:
    config = load_recipe_config(Path("config/dataset_tokenization.native_superbpe.sample.yaml"))

    assert "dataset-tokenization-remote-job" in cli.list_commands(None)
    assert "dataset-tokenization-validate" in cli.commands
    assert "dataset-tokenization-run" in cli.commands
    assert "dataset-tokenization-resume" in cli.commands
    assert config.r2.output_prefix == "dataset/processed/tokenized/native-superbpe-1m-rows-max4w"
    assert (
        config.input.final_parts_prefix
        == "dataset/processed/unified-data/final-completed-20260430T160615Z/parts/"
    )
    assert config.input.fineweb_run_root == "dataset/processed/fineweb-unified/20260501T144323Z"
    assert config.tokenizer.name == "native_superbpe_1m_rows_max4w"
    assert config.tokenizer.json_path == "tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json"


def test_dataset_tokenization_validate_command_outputs_config_summary() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "dataset-tokenization-validate",
            "--config",
            "config/dataset_tokenization.native_superbpe.sample.yaml",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Tokenizer: native_superbpe_1m_rows_max4w" in result.output
    assert "Resolved pipeline: prepare_dataset_tokenization_part" in result.output


def test_dataset_tokenization_dry_run_uses_dependency_group(
    tmp_path: Path,
) -> None:
    store = LocalObjectStore(tmp_path)
    write_minimal_dataset(store)
    config = load_recipe_config(
        Path("config/dataset_tokenization.native_superbpe.sample.yaml"),
        overrides=[
            "tokenizer.json_path=tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json",
        ],
    )

    prepared = DatasetTokenizationSubmissionAdapter(
        config=config,
        config_path=Path("config/dataset_tokenization.native_superbpe.sample.yaml"),
    ).prepare_new_run(FakeArtifactStore(store), dry_run=True)

    assert prepared.discovered_items == 2
    assert "--group dataset_tokenization" in prepared.bootstrap.command
    assert (
        "training_signal_processing.main dataset-tokenization-remote-job"
        in prepared.invocation.command
    )


def test_manifest_includes_done_fineweb_and_excludes_unresolved_parts(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    write_minimal_dataset(store)
    config = load_recipe_config(Path("config/dataset_tokenization.native_superbpe.sample.yaml"))

    source_parts, fineweb = discover_dataset_tokenization_inputs(
        config=config,
        object_store=store,
    )
    tasks = build_dataset_tokenization_manifest_rows(
        config=config,
        run_id="unit-run",
        source_parts=source_parts,
        tokenizer_object_key="out/unit-run/control/tokenizer.json",
        tokenizer_json_sha256="abc123",
    )

    assert [task.source_group for task in tasks] == ["final", "fineweb"]
    assert [task.source_part_key for task in tasks] == [
        "dataset/processed/unified-data/final-completed-20260430T160615Z/parts/part-000000.parquet",
        "dataset/processed/fineweb-unified/20260501T144323Z/parts/month=202111/part-000001.parquet",
    ]
    assert fineweb.error_only_part_keys == (
        "dataset/processed/fineweb-unified/20260501T144323Z/parts/month=202111/part-000002.parquet",
    )
    assert fineweb.orphan_part_keys == (
        "dataset/processed/fineweb-unified/20260501T144323Z/parts/month=202111/part-000003.parquet",
    )
    assert fineweb.stale_error_keys == (
        "dataset/processed/fineweb-unified/20260501T144323Z/errors/part-000001.error.json",
    )


def test_fineweb_availability_treats_done_plus_error_as_stale(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    write_minimal_dataset(store)

    availability = collect_fineweb_availability(
        object_store=store,
        run_root="dataset/processed/fineweb-unified/20260501T144323Z",
    )

    assert len(availability.done_part_keys) == 1
    assert len(availability.stale_error_keys) == 1
    assert len(availability.error_only_part_keys) == 1
    assert len(availability.orphan_part_keys) == 1


def test_write_dataset_tokenization_part_writes_compact_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = LocalObjectStore(tmp_path)
    source_key = "input/part-000000.parquet"
    write_parquet(
        store,
        source_key,
        [
            {
                "sample_uid": "sample-1",
                "sample_uid_hash": "hash-1",
                "source_name": "Unit",
                "cleaning_source": "unit",
                "source_row_group_index": 0,
                "source_row_index": 0,
                "row_index_in_row_group": 0,
                "cleaned_text": "hello world",
                "cleaned_text_sha256": "",
                "cleaning_is_dropped": False,
            },
            {
                "sample_uid": "sample-2",
                "sample_uid_hash": "hash-2",
                "source_name": "Unit",
                "cleaning_source": "unit",
                "source_row_group_index": 0,
                "source_row_index": 1,
                "row_index_in_row_group": 1,
                "cleaned_text": "",
                "cleaned_text_sha256": "",
                "cleaning_is_dropped": True,
            },
        ],
    )
    monkeypatch.setattr(
        "training_signal_processing.pipelines.dataset_tokenization.ops.load_tokenizer",
        lambda **_kwargs: FakeTokenizer(),
    )
    task = sample_task(source_part_key=source_key)

    metrics = write_dataset_tokenization_part(task=task, object_store=store)

    assert metrics["row_count"] == 2
    assert metrics["token_count"] == 3
    output = pq.read_table(tmp_path / store.bucket / task.part_key)
    assert "cleaned_text" not in output.column_names
    assert output.schema == tokenized_output_schema()
    assert output.column("sample_uid").to_pylist() == ["sample-1", "sample-2"]
    assert output.column("sample_uid_hash").to_pylist() == ["hash-1", "hash-2"]
    assert output.column("token_ids").to_pylist() == [[5, 5], [0]]
    assert output.column("token_count").to_pylist() == [2, 1]
    assert output.column("tokenizer_name").to_pylist() == ["unit-tokenizer", "unit-tokenizer"]
    assert output.column("cleaning_is_dropped").to_pylist() == [False, True]


def test_dataset_tokenization_result_serializes_for_ray_rows() -> None:
    task = sample_task(source_part_key="input/part-000000.parquet")

    result = DatasetTokenizationResult.success_from_task(
        task=task,
        metrics={"row_count": 2, "token_count": 3},
        duration_sec=1.25,
    ).to_dict()

    assert result["status"] == "success"
    assert result["task_index"] == 0
    assert result["part_key"] == "output/parts/part-000000.parquet"
    assert result["output_row_count"] == 2
    assert DatasetTokenizationResult.from_dict(result).metrics()["token_count"] == 3


def test_dataset_tokenization_transform_requests_worker_memory() -> None:
    config = load_recipe_config(Path("config/dataset_tokenization.native_superbpe.sample.yaml"))
    op = SimpleNamespace(name="write_dataset_tokenization_part")

    resources = resolve_dataset_tokenization_transform_resources(config, op, config.ray)

    assert resources.concurrency == 4
    assert resources.memory == 4 * 1024**3


def write_minimal_dataset(store: LocalObjectStore) -> None:
    final_key = (
        "dataset/processed/unified-data/final-completed-20260430T160615Z/"
        "parts/part-000000.parquet"
    )
    done_fineweb_key = (
        "dataset/processed/fineweb-unified/20260501T144323Z/"
        "parts/month=202111/part-000001.parquet"
    )
    error_fineweb_key = (
        "dataset/processed/fineweb-unified/20260501T144323Z/"
        "parts/month=202111/part-000002.parquet"
    )
    orphan_fineweb_key = (
        "dataset/processed/fineweb-unified/20260501T144323Z/"
        "parts/month=202111/part-000003.parquet"
    )
    for key in (final_key, done_fineweb_key, error_fineweb_key, orphan_fineweb_key):
        write_parquet(store, key, [{"cleaned_text": "hello", "cleaning_is_dropped": False}])
    fineweb_root = "dataset/processed/fineweb-unified/20260501T144323Z"
    store.write_json(
        f"{fineweb_root}/done/part-000001.done.json",
        {"status": "success", "part_key": done_fineweb_key},
    )
    store.write_json(
        f"{fineweb_root}/errors/part-000001.error.json",
        {"status": "failed", "part_key": done_fineweb_key},
    )
    store.write_json(
        f"{fineweb_root}/errors/part-000002.error.json",
        {"status": "failed", "part_key": error_fineweb_key},
    )


def write_parquet(store: LocalObjectStore, key: str, rows: list[dict[str, object]]) -> None:
    path = store.storage_root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def sample_task(source_part_key: str) -> DatasetTokenizationTask:
    return DatasetTokenizationTask(
        task_index=0,
        source_group="unit",
        source_part_key=source_part_key,
        part_key="output/parts/part-000000.parquet",
        metrics_key="output/metrics/part-000000.metrics.json",
        done_key="output/done/part-000000.done.json",
        error_key="output/errors/part-000000.error.json",
        text_column="cleaned_text",
        dropped_column="cleaning_is_dropped",
        tokenizer_name="unit-tokenizer",
        tokenizer_object_key="output/control/tokenizer.json",
        tokenizer_json_sha256="abc123",
        read_batch_rows=1,
        rows_per_row_group=2,
        parquet_compression="none",
        parquet_compression_level=1,
    )
