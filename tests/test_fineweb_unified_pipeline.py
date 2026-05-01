from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

from training_signal_processing.core.storage import ObjectStore
from training_signal_processing.core.submission import ArtifactStore
from training_signal_processing.main import cli
from training_signal_processing.pipelines.fineweb_unified.config import load_recipe_config
from training_signal_processing.pipelines.fineweb_unified.models import FINAL_UNIFIED_COLUMNS
from training_signal_processing.pipelines.fineweb_unified.ops import (
    build_month_byte_quotas,
    final_unified_schema,
    fineweb_row_to_unified_row,
    load_fineweb_stream,
    split_month_quota,
    write_fineweb_unified_part,
)
from training_signal_processing.pipelines.fineweb_unified.runtime import (
    resolve_fineweb_unified_transform_resources,
)
from training_signal_processing.pipelines.fineweb_unified.submission import (
    FineWebUnifiedSubmissionAdapter,
    build_fineweb_unified_manifest_rows,
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
    def __init__(self) -> None:
        self.bucket = "gpu-poor"
        self.json_rows: dict[str, list[dict[str, object]]] = {}
        self.json_objects: dict[str, dict[str, object]] = {}

    def exists(self, key: str) -> bool:
        return key in self.json_rows or key in self.json_objects

    def read_json(self, key: str) -> dict[str, object]:
        return self.json_objects[key]

    def read_jsonl(self, key: str) -> list[dict[str, object]]:
        return self.json_rows[key]

    def write_json(self, key: str, value: dict[str, object]) -> None:
        self.json_objects[key] = value

    def write_jsonl(self, key: str, rows: list[dict[str, object]]) -> None:
        self.json_rows[key] = rows

    def upload_file(self, path: Path, key: str) -> None:
        del path, key

    def build_remote_env(self) -> dict[str, str]:
        return {"R2_BUCKET": self.bucket}


def test_fineweb_unified_sample_config_loads_and_cli_is_registered() -> None:
    config = load_recipe_config(Path("config/fineweb_unified.sample.yaml"))

    assert "fineweb-unified-remote-job" in cli.commands
    assert "fineweb-unified-validate" in cli.commands
    assert "fineweb-unified-run" in cli.commands
    assert "fineweb-unified-resume" in cli.commands
    assert config.input.dataset_name == "HuggingFaceFW/fineweb"
    assert config.input.hf_token_env_var == "HF_TOKEN"
    assert config.input.enforce_month_filter is False
    assert config.export.byte_cap == 84_470_960_868
    assert config.export.tokenizer_encoding == "o200k_base"
    assert config.export.ray_memory_gib_per_worker == pytest.approx(16.0)


def test_fineweb_unified_transform_requests_worker_memory() -> None:
    config = load_recipe_config(Path("config/fineweb_unified.sample.yaml"))
    op = SimpleNamespace(name="write_fineweb_unified_part")

    resources = resolve_fineweb_unified_transform_resources(config, op, config.ray)

    assert resources.memory == 16 * 1024**3


def test_fineweb_unified_config_rejects_invalid_caps(tmp_path: Path) -> None:
    config_path = tmp_path / "fineweb_invalid.yaml"
    config_path.write_text(
        Path("config/fineweb_unified.sample.yaml")
        .read_text(encoding="utf-8")
        .replace("byte_cap: 84470960868", "byte_cap: 0"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="export.byte_cap must be positive"):
        load_recipe_config(config_path)


def test_month_byte_quotas_are_balanced_and_do_not_exceed_cap() -> None:
    quotas = build_month_byte_quotas(months=["2020-03", "2020-01", "2020-02"], byte_cap=10)

    assert sum(quotas.values()) == 10
    assert max(quotas.values()) - min(quotas.values()) <= 1
    assert quotas == {"2020-01": 4, "2020-02": 3, "2020-03": 3}
    assert split_month_quota(quota=10, part_target_bytes=4) == [4, 3, 3]


def test_fineweb_row_maps_to_final_unified_schema() -> None:
    task = sample_task(byte_quota=100)
    row = fineweb_row_to_unified_row(
        raw_row=fineweb_row(text="hello fineweb"),
        task=task,
        text="hello fineweb",
        source_row_index=7,
        row_index_in_part=2,
    )

    assert list(final_unified_schema().names) == list(FINAL_UNIFIED_COLUMNS)
    assert "text" not in FINAL_UNIFIED_COLUMNS
    assert "body" not in FINAL_UNIFIED_COLUMNS
    assert row["cleaned_text"] == "hello fineweb"
    assert row["text_column"] == "text"
    assert row["source_name"] == "FineWeb"
    assert row["cleaning_source"] == "fineweb"
    assert row["timestamp"] == "2020-01-15T00:00:00Z"
    assert row["crawl_id"] == "CC-MAIN-2020-05"
    assert row["source_shard"].startswith("s3://commoncrawl/")
    assert row["lid_cleaned_token_count"] is None


def test_write_fineweb_part_stops_before_exceeding_byte_quota(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = LocalObjectStore(tmp_path)
    task = sample_task(byte_quota=10)

    def fake_rows(_task):
        yield fineweb_row(text="12345", row_id="a")
        yield fineweb_row(text="123456", row_id="b")
        yield fineweb_row(text="tiny", row_id="c")

    monkeypatch.setattr(
        "training_signal_processing.pipelines.fineweb_unified.ops.iter_task_fineweb_rows",
        fake_rows,
    )

    metrics = write_fineweb_unified_part(task=task, object_store=store)

    assert metrics["row_count"] == 1
    assert metrics["cleaned_text_byte_count"] == 5
    output = pq.read_table(tmp_path / store.bucket / task.part_key)
    assert output.column("cleaned_text").to_pylist() == ["12345"]
    assert output.column("cleaned_o200k_tokenizer").to_pylist() == ["o200k_base"]


def test_write_fineweb_part_can_relax_month_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = LocalObjectStore(tmp_path)
    task = sample_task(byte_quota=100)

    def fake_rows(_task):
        yield fineweb_row(text="from a different month", row_id="a", date="2020-02-01T00:00:00Z")

    monkeypatch.setattr(
        "training_signal_processing.pipelines.fineweb_unified.ops.iter_task_fineweb_rows",
        fake_rows,
    )

    metrics = write_fineweb_unified_part(task=task, object_store=store)

    assert metrics["row_count"] == 1
    assert metrics["enforce_month_filter"] is False
    output = pq.read_table(tmp_path / store.bucket / task.part_key)
    assert output.column("month").to_pylist() == ["2020-01"]
    assert output.column("timestamp").to_pylist() == ["2020-02-01T00:00:00Z"]


def test_manifest_balances_discovered_months(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_recipe_config(
        Path("config/fineweb_unified.sample.yaml"),
        overrides=[
            "export.byte_cap=10",
            "export.part_target_bytes=4",
            "input.discovery_sample_rows_per_config=3",
        ],
    )

    monkeypatch.setattr(
        "training_signal_processing.pipelines.fineweb_unified.submission.discover_fineweb_months",
        lambda **_kwargs: ["2020-01", "2020-02", "2020-03"],
    )

    tasks = build_fineweb_unified_manifest_rows(config=config, run_id="out-run")

    assert sum(task.byte_quota for task in tasks) == 10
    assert {task.month for task in tasks} == {"2020-01", "2020-02", "2020-03"}
    assert all(task.byte_quota <= 4 for task in tasks)
    assert tasks[0].part_key.startswith("dataset/processed/fineweb-unified/out-run/parts/")


def test_submission_dry_run_uses_fineweb_group(monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = Path("config/fineweb_unified.sample.yaml")
    config = load_recipe_config(
        config_path,
        overrides=["export.byte_cap=2", "export.part_target_bytes=1"],
    )
    monkeypatch.setattr(
        "training_signal_processing.pipelines.fineweb_unified.submission.discover_fineweb_months",
        lambda **_kwargs: ["2020-01"],
    )

    prepared = FineWebUnifiedSubmissionAdapter(
        config=config,
        config_path=config_path,
    ).prepare_new_run(FakeArtifactStore(), dry_run=True)

    assert prepared.discovered_items == 2
    assert "--group fineweb_unified" in prepared.bootstrap.command
    assert (
        "training_signal_processing.main fineweb-unified-remote-job"
        in prepared.invocation.command
    )


def test_submission_forwards_hf_read_token_env(monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = Path("config/fineweb_unified.sample.yaml")
    config = load_recipe_config(
        config_path,
        overrides=["export.byte_cap=1", "export.part_target_bytes=1"],
    )
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setattr(
        "training_signal_processing.pipelines.fineweb_unified.submission.discover_fineweb_months",
        lambda **_kwargs: ["2020-01"],
    )

    prepared = FineWebUnifiedSubmissionAdapter(
        config=config,
        config_path=config_path,
    ).prepare_new_run(FakeArtifactStore(), dry_run=True)

    assert "HF_TOKEN" in prepared.invocation.env


def test_load_fineweb_stream_uses_hf_read_token(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    load_fineweb_stream(
        dataset_name="HuggingFaceFW/fineweb",
        config_name="default",
        split="train",
        hf_token_env_var="HF_TOKEN",
    )

    assert calls[0][0] == ("HuggingFaceFW/fineweb", "default")
    assert calls[0][1]["streaming"] is True
    assert calls[0][1]["token"] == "hf_test_token"


def sample_task(*, byte_quota: int):
    from training_signal_processing.pipelines.fineweb_unified.models import FineWebPartTask

    return FineWebPartTask(
        part_index=0,
        month="2020-01",
        month_part_index=0,
        month_part_count=1,
        byte_quota=byte_quota,
        dataset_name="HuggingFaceFW/fineweb",
        dataset_configs=("default",),
        split="train",
        source_name="FineWeb",
        cleaning_source="fineweb",
        text_column="text",
        hf_token_env_var="HF_TOKEN",
        shuffle_seed=42,
        shuffle_buffer_size=10,
        stream_shards_per_config=1,
        enforce_month_filter=False,
        part_key="dataset/processed/fineweb-unified/run/parts/part-000000.parquet",
        metrics_key="dataset/processed/fineweb-unified/run/metrics/part-000000.metrics.json",
        done_key="dataset/processed/fineweb-unified/run/done/part-000000.done.json",
        error_key="dataset/processed/fineweb-unified/run/errors/part-000000.error.json",
        rows_per_row_group=100,
        write_batch_rows=2,
        tokenizer_encoding="o200k_base",
        tokenizer_threads=1,
        parquet_compression="zstd",
        parquet_compression_level=1,
    )


def fineweb_row(
    *,
    text: str,
    row_id: str = "row-1",
    date: str = "2020-01-15T00:00:00Z",
) -> dict[str, Any]:
    return {
        "text": text,
        "id": row_id,
        "dump": "CC-MAIN-2020-05",
        "url": "https://example.com/doc",
        "date": date,
        "file_path": "s3://commoncrawl/crawl-data/CC-MAIN-2020-05/example.warc.gz",
        "language": "en",
        "language_score": 0.98,
        "token_count": 3,
    }
