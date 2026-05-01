from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

from training_signal_processing.core.storage import ObjectStore
from training_signal_processing.pipelines.tokenizer_training import runtime as tokenizer_runtime
from training_signal_processing.pipelines.tokenizer_training.config import load_recipe_config
from training_signal_processing.pipelines.tokenizer_training.models import (
    BudgetConfig,
    InputConfig,
)
from training_signal_processing.pipelines.tokenizer_training.ops import RoundRobinTextSampler
from training_signal_processing.pipelines.tokenizer_training.runtime import (
    LocalParquetObjectStore,
    execute_training,
    run_checkpointed_training,
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


class FakeTokenizer:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.vocab = {b"a": 0, b"b": 1}

    def save(self, path: str) -> None:
        Path(path).write_text(
            json.dumps({"kind": "bpeasy", "texts": self.texts}) + "\n",
            encoding="utf-8",
        )

    def export_to_huggingface_format(self, path: str) -> None:
        Path(path).write_text('{"kind": "huggingface"}\n', encoding="utf-8")


def test_tokenizer_training_sample_config_loads_final_accounting_sources() -> None:
    config = load_recipe_config(Path("config/tokenizer_training.final_merged.sample.yaml"))

    assert config.r2.bucket == "gpu-poor"
    assert (
        config.input.final_parts_prefix
        == "dataset/processed/unified-data/final-completed-20260430T160615Z/parts/"
    )
    assert config.input.text_column == "cleaned_text"
    assert config.input.source_column == "cleaning_source"
    assert config.input.dropped_column == "cleaning_is_dropped"
    assert config.input.sources == [
        "books-ocr",
        "cari",
        "hplt-indonesia",
        "hplt-malay",
        "lowyat",
        "reddit-bolehland",
        "reddit-indonesia",
        "fineweb",
    ]
    assert config.input.source_parts_prefixes == {
        "fineweb": "dataset/processed/fineweb-unified/20260501T144323Z/parts/"
    }
    assert config.training.vocab_size == 50_000
    assert config.budget.max_wall_seconds == 0
    assert config.budget.max_memory_gib == pytest.approx(0.0)
    assert config.checkpoint.enabled is True
    assert config.checkpoint.export_interval_seconds == 1200
    assert config.checkpoint.export_grace_seconds == 300
    assert config.checkpoint.keep_last == 18
    assert config.checkpoint.latest_name == "latest"


def test_tokenizer_training_config_rejects_invalid_budget(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_tokenizer.yaml"
    config_path.write_text(
        Path("config/tokenizer_training.final_merged.sample.yaml")
        .read_text(encoding="utf-8")
        .replace("max_wall_seconds: 0", "max_wall_seconds: -1"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="budget.max_wall_seconds must be zero or positive"):
        load_recipe_config(config_path)


def test_tokenizer_training_config_rejects_invalid_checkpoint_interval(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_tokenizer_checkpoint.yaml"
    config_path.write_text(
        Path("config/tokenizer_training.final_merged.sample.yaml")
        .read_text(encoding="utf-8")
        .replace("export_interval_seconds: 1200", "export_interval_seconds: 0"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint.export_interval_seconds must be positive"):
        load_recipe_config(config_path)


def test_round_robin_sampler_uses_strict_source_order(tmp_path: Path) -> None:
    store = write_final_parts(
        tmp_path,
        [
            {"cleaned_text": "a1", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "a2", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "b1", "cleaning_source": "b", "cleaning_is_dropped": False},
            {"cleaned_text": "b2", "cleaning_source": "b", "cleaning_is_dropped": False},
        ],
    )
    sampler = build_sampler(store, sources=["a", "b"])

    assert list(sampler) == ["a1", "b1", "a2", "b2"]
    assert sampler.stats.source_counts == {"a": 2, "b": 2}
    assert sampler.stats.stop_reason == "exhausted"


def test_round_robin_sampler_resumes_from_cursor_state(tmp_path: Path) -> None:
    store = write_final_parts(
        tmp_path,
        [
            {"cleaned_text": "a1", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "a2", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "b1", "cleaning_source": "b", "cleaning_is_dropped": False},
            {"cleaned_text": "b2", "cleaning_source": "b", "cleaning_is_dropped": False},
        ],
    )
    sampler = build_sampler(
        store,
        sources=["a", "b"],
        cursor_state={"source_offsets": {"a": 1, "b": 1}},
    )

    assert list(sampler) == ["a2", "b2"]
    assert sampler.cursor_state_dict()["source_offsets"] == {"a": 2, "b": 2}


def test_round_robin_sampler_skips_dropped_and_empty_rows(tmp_path: Path) -> None:
    store = write_final_parts(
        tmp_path,
        [
            {"cleaned_text": "", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "drop", "cleaning_source": "a", "cleaning_is_dropped": True},
            {"cleaned_text": "a1", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "b1", "cleaning_source": "b", "cleaning_is_dropped": False},
        ],
    )
    sampler = build_sampler(store, sources=["a", "b"])

    assert list(sampler) == ["a1", "b1"]
    assert sampler.stats.sampled_rows == 2


def test_round_robin_sampler_stops_on_row_budget(tmp_path: Path) -> None:
    store = write_final_parts(
        tmp_path,
        [
            {"cleaned_text": "a1", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "a2", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "b1", "cleaning_source": "b", "cleaning_is_dropped": False},
            {"cleaned_text": "b2", "cleaning_source": "b", "cleaning_is_dropped": False},
        ],
    )
    sampler = build_sampler(store, sources=["a", "b"], budget=BudgetConfig(max_sample_rows=3))

    assert list(sampler) == ["a1", "b1", "a2"]
    assert sampler.stats.stop_reason == "max_sample_rows"


def test_round_robin_sampler_stops_before_exceeding_byte_budget(tmp_path: Path) -> None:
    store = write_final_parts(
        tmp_path,
        [
            {"cleaned_text": "abc", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "def", "cleaning_source": "b", "cleaning_is_dropped": False},
        ],
    )
    sampler = build_sampler(store, sources=["a", "b"], budget=BudgetConfig(max_sample_bytes=4))

    assert list(sampler) == ["abc"]
    assert sampler.stats.sampled_bytes == 3
    assert sampler.stats.stop_reason == "max_sample_bytes"


def test_round_robin_sampler_stops_on_time_budget(tmp_path: Path) -> None:
    store = write_final_parts(
        tmp_path,
        [{"cleaned_text": "a1", "cleaning_source": "a", "cleaning_is_dropped": False}],
    )
    ticks = iter([0.0, 2.0, 2.0, 2.0])
    sampler = build_sampler(
        store,
        sources=["a"],
        budget=BudgetConfig(max_wall_seconds=1),
        clock=lambda: next(ticks),
    )

    assert list(sampler) == []
    assert sampler.stats.stop_reason == "max_wall_seconds"


def test_round_robin_sampler_stops_on_rss_budget(tmp_path: Path) -> None:
    store = write_final_parts(
        tmp_path,
        [{"cleaned_text": "a1", "cleaning_source": "a", "cleaning_is_dropped": False}],
    )
    sampler = build_sampler(
        store,
        sources=["a"],
        budget=BudgetConfig(max_memory_gib=0.001),
        rss_provider=lambda: 2.0,
    )

    assert list(sampler) == []
    assert sampler.stats.stop_reason == "max_memory_gib"


def test_execute_training_exports_tokenizer_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = write_final_parts(
        tmp_path,
        [
            {"cleaned_text": "a1", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "b1", "cleaning_source": "b", "cleaning_is_dropped": False},
        ],
    )
    config = load_recipe_config(
        Path("config/tokenizer_training.final_merged.sample.yaml"),
        overrides=[
            "input.final_parts_prefix=dataset/final/parts/",
            "input.sources=a,b",
            "budget.max_sample_rows=2",
            f"output.root_dir={tmp_path / 'tokenizers'}",
        ],
    )
    config.input.sources = ["a", "b"]

    def fake_trainer(iterator, **_kwargs: Any) -> FakeTokenizer:
        return FakeTokenizer(list(iterator))

    def fake_tiktoken_vocab(**kwargs: Any) -> None:
        Path(kwargs["out_path"]).write_text("YQ== 0\n", encoding="utf-8")

    monkeypatch.setattr(
        "training_signal_processing.pipelines.tokenizer_training.runtime.save_tiktoken_vocab",
        fake_tiktoken_vocab,
    )

    summary = execute_training(
        config=config,
        run_id="unit-run",
        run_dir=tmp_path / "tokenizers" / "unit-run",
        object_store=store,
        trainer=fake_trainer,
    )

    assert summary["status"] == "success"
    assert summary["sampled_rows"] == 2
    assert summary["source_counts"] == {"a": 1, "b": 1}
    assert (tmp_path / "tokenizers/unit-run/tokenizer.json").is_file()
    assert (tmp_path / "tokenizers/unit-run/tokenizer.hf.json").is_file()
    assert (tmp_path / "tokenizers/unit-run/vocab.tiktoken.txt").is_file()


def test_training_worker_can_use_local_parquet_cache_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_recipe_config(Path("config/tokenizer_training.final_merged.sample.yaml"))
    cache_root = tmp_path / "parquet-cache"
    monkeypatch.setenv("TOKENIZER_TRAINING_LOCAL_PARQUET_ROOT", str(cache_root))

    store = tokenizer_runtime.build_training_object_store(config)

    assert isinstance(store, LocalParquetObjectStore)
    assert store.bucket == "gpu-poor"


def test_checkpointed_training_exports_periodic_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = write_final_parts(
        tmp_path,
        [
            {"cleaned_text": "a1", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "a2", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "b1", "cleaning_source": "b", "cleaning_is_dropped": False},
            {"cleaned_text": "b2", "cleaning_source": "b", "cleaning_is_dropped": False},
        ],
    )
    config = checkpoint_test_config(tmp_path)
    trainer_calls: list[list[str]] = []

    def fake_trainer(iterator, **_kwargs: Any) -> FakeTokenizer:
        texts = list(iterator)
        trainer_calls.append(texts)
        return FakeTokenizer(texts)

    monkeypatch.setattr(
        "training_signal_processing.pipelines.tokenizer_training.runtime.save_tiktoken_vocab",
        fake_tiktoken_vocab,
    )

    summary = run_checkpointed_training(
        config=config,
        run_id="unit-run",
        run_dir=tmp_path / "tokenizers" / "unit-run",
        object_store=store,
        trainer=fake_trainer,
        use_process=False,
    )

    assert summary["status"] == "success"
    assert summary["completed_checkpoints"] == 2
    assert summary["sampled_rows"] == 4
    assert summary["latest_checkpoint"] == "checkpoint-000002"
    assert summary["cursor_state"]["source_offsets"] == {"a": 2, "b": 2}
    assert trainer_calls == [["a1", "b1"], ["a2", "b2"], []]
    assert (tmp_path / "tokenizers/unit-run/checkpoints/checkpoint-000001/tokenizer.json").is_file()
    latest_tokenizer = tmp_path / "tokenizers/unit-run/latest/tokenizer.json"
    assert latest_tokenizer.is_file()
    assert json.loads(latest_tokenizer.read_text(encoding="utf-8"))["texts"] == ["a2", "b2"]


def test_checkpointed_training_preserves_previous_checkpoint_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = write_final_parts(
        tmp_path,
        [
            {"cleaned_text": "a1", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "a2", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "b1", "cleaning_source": "b", "cleaning_is_dropped": False},
            {"cleaned_text": "b2", "cleaning_source": "b", "cleaning_is_dropped": False},
        ],
    )
    config = checkpoint_test_config(tmp_path)
    calls = 0

    def flaky_trainer(iterator, **_kwargs: Any) -> FakeTokenizer:
        nonlocal calls
        calls += 1
        texts = list(iterator)
        if calls == 2:
            raise RuntimeError("planned tranche failure")
        return FakeTokenizer(texts)

    monkeypatch.setattr(
        "training_signal_processing.pipelines.tokenizer_training.runtime.save_tiktoken_vocab",
        fake_tiktoken_vocab,
    )

    summary = run_checkpointed_training(
        config=config,
        run_id="unit-run",
        run_dir=tmp_path / "tokenizers" / "unit-run",
        object_store=store,
        trainer=flaky_trainer,
        use_process=False,
    )

    assert summary["status"] == "partial_success"
    assert summary["completed_checkpoints"] == 1
    assert summary["latest_checkpoint"] == "checkpoint-000001"
    assert summary["stop_reason"] == "tranche_failed"
    assert "planned tranche failure" in summary["error_message"]
    assert (tmp_path / "tokenizers/unit-run/latest/tokenizer.json").is_file()
    assert not (tmp_path / "tokenizers/unit-run/checkpoints/checkpoint-000002").exists()


def test_checkpointed_training_resumes_from_previous_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = write_final_parts(
        tmp_path,
        [
            {"cleaned_text": "a1", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "a2", "cleaning_source": "a", "cleaning_is_dropped": False},
            {"cleaned_text": "b1", "cleaning_source": "b", "cleaning_is_dropped": False},
            {"cleaned_text": "b2", "cleaning_source": "b", "cleaning_is_dropped": False},
        ],
    )
    first_config = checkpoint_test_config(tmp_path)
    trainer_calls: list[list[str]] = []

    def fake_trainer(iterator, **_kwargs: Any) -> FakeTokenizer:
        texts = list(iterator)
        trainer_calls.append(texts)
        return FakeTokenizer(texts)

    monkeypatch.setattr(
        "training_signal_processing.pipelines.tokenizer_training.runtime.save_tiktoken_vocab",
        fake_tiktoken_vocab,
    )
    first_summary = run_checkpointed_training(
        config=first_config,
        run_id="first-run",
        run_dir=tmp_path / "tokenizers" / "first-run",
        object_store=store,
        trainer=fake_trainer,
        use_process=False,
    )

    resume_config = checkpoint_test_config(tmp_path)
    resume_config.output.resume_from_dir = first_summary["run_dir"]
    resume_summary = run_checkpointed_training(
        config=resume_config,
        run_id="resume-run",
        run_dir=tmp_path / "tokenizers" / "resume-run",
        object_store=store,
        trainer=fake_trainer,
        use_process=False,
    )

    assert resume_summary["status"] == "success"
    assert resume_summary["completed_checkpoints"] == 2
    assert resume_summary["sampled_rows"] == 4
    assert resume_summary["resumed_from_run_dir"] == first_summary["run_dir"]
    assert resume_summary["cursor_state"] == first_summary["cursor_state"]
    assert trainer_calls == [["a1", "b1"], ["a2", "b2"], [], []]
    copied_checkpoint = tmp_path / "tokenizers/resume-run/checkpoints/checkpoint-000001"
    assert copied_checkpoint.is_dir()
    checkpoint_payload = json.loads(
        (copied_checkpoint / "training_summary.json").read_text(encoding="utf-8")
    )
    assert checkpoint_payload["run_dir"] == str(copied_checkpoint)


def test_checkpointed_training_does_not_hard_timeout_unbounded_global_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = checkpoint_test_config(tmp_path)
    config.budget.max_wall_seconds = 0
    observed_timeouts: list[int | None] = []

    def fake_run_training_once(**kwargs: Any) -> dict[str, Any]:
        observed_timeouts.append(kwargs["timeout_seconds"])
        staging_dir = kwargs["staging_dir"]
        paths = tokenizer_runtime.artifact_paths(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        paths["tokenizer_json"].write_text("{}\n", encoding="utf-8")
        paths["huggingface_json"].write_text("{}\n", encoding="utf-8")
        paths["tiktoken_vocab"].write_text("YQ== 0\n", encoding="utf-8")
        if len(observed_timeouts) == 1:
            return {
                "status": "success",
                "summary": {
                    "status": "success",
                    "sampled_rows": 2,
                    "sampled_bytes": 4,
                    "source_counts": {"a": 1, "b": 1},
                    "source_bytes": {"a": 2, "b": 2},
                    "stop_reason": "max_sample_rows",
                    "elapsed_seconds": 1.0,
                    "peak_rss_mib": 1.0,
                    "cursor_state": {
                        "source_offsets": {"a": 1, "b": 1},
                        "source_positions": {
                            "a": {"key_index": 0, "row_offset": 1},
                            "b": {"key_index": 0, "row_offset": 1},
                        },
                    },
                },
            }
        return {
            "status": "success",
            "summary": {
                "status": "success",
                "sampled_rows": 0,
                "sampled_bytes": 0,
                "source_counts": {"a": 0, "b": 0},
                "source_bytes": {"a": 0, "b": 0},
                "stop_reason": "exhausted",
                "elapsed_seconds": 1.0,
                "peak_rss_mib": 1.0,
                "cursor_state": {
                    "source_offsets": {"a": 1, "b": 1},
                    "source_positions": {
                        "a": {"key_index": 0, "row_offset": 1},
                        "b": {"key_index": 0, "row_offset": 1},
                    },
                },
            },
        }

    monkeypatch.setattr(tokenizer_runtime, "_run_training_once", fake_run_training_once)

    summary = run_checkpointed_training(
        config=config,
        run_id="unit-run",
        run_dir=tmp_path / "tokenizers" / "unit-run",
    )

    assert summary["status"] == "success"
    assert observed_timeouts == [None, None]


def write_final_parts(tmp_path: Path, rows: list[dict[str, object]]) -> LocalObjectStore:
    store = LocalObjectStore(tmp_path)
    part_path = tmp_path / store.bucket / "dataset/final/parts/part-000000.parquet"
    part_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), part_path)
    return store


def build_sampler(
    store: LocalObjectStore,
    *,
    sources: list[str],
    budget: BudgetConfig | None = None,
    clock=lambda: 0.0,
    rss_provider=lambda: 1.0,
    cursor_state: dict[str, Any] | None = None,
) -> RoundRobinTextSampler:
    return RoundRobinTextSampler(
        object_store=store,
        input_config=InputConfig(
            final_parts_prefix="dataset/final/parts/",
            text_column="cleaned_text",
            source_column="cleaning_source",
            dropped_column="cleaning_is_dropped",
            sources=sources,
            read_batch_rows=2,
        ),
        budget=budget or BudgetConfig(),
        clock=clock,
        rss_provider=rss_provider,
        cursor_state=cursor_state,
    )


def checkpoint_test_config(tmp_path: Path):
    config = load_recipe_config(
        Path("config/tokenizer_training.final_merged.sample.yaml"),
        overrides=[
            "input.final_parts_prefix=dataset/final/parts/",
            "input.sources=a,b",
            "budget.max_wall_seconds=5",
            "budget.max_sample_rows=2",
            "checkpoint.export_interval_seconds=1",
            "checkpoint.export_grace_seconds=1",
            f"output.root_dir={tmp_path / 'tokenizers'}",
        ],
    )
    config.input.sources = ["a", "b"]
    return config


def fake_tiktoken_vocab(**kwargs: Any) -> None:
    Path(kwargs["out_path"]).write_text("YQ== 0\n", encoding="utf-8")
