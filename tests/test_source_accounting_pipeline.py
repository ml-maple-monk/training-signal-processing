from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest
import tiktoken
from click.testing import CliRunner

from training_signal_processing.core.storage import ObjectStore
from training_signal_processing.core.submission import ArtifactStore
from training_signal_processing.main import cli
from training_signal_processing.pipelines.source_accounting.config import load_recipe_config
from training_signal_processing.pipelines.source_accounting.models import (
    SourceAccountingResult,
    SourceAccountingTask,
    extract_required_sources_from_plan,
    render_markdown_table,
)
from training_signal_processing.pipelines.source_accounting.ops import count_source
from training_signal_processing.pipelines.source_accounting.submission import (
    SourceAccountingSubmissionAdapter,
)

BOOKS_OCR_PARQUET_KEY = "dataset/processed/pdf_ocr/20260423T195035Z/markdown.parquet"


class LocalObjectStore(ObjectStore):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.bucket = str(root)

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()

    def list_keys(self, prefix: str) -> list[str]:
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file() and path.relative_to(self.root).as_posix().startswith(prefix)
        )

    def read_bytes(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def write_bytes(self, key: str, body: bytes) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)

    def upload_file(self, path: Path, key: str) -> None:
        self.write_bytes(key, path.read_bytes())

    def make_url(self, key: str) -> str:
        return f"file://{self.root / key}"

    def build_pyarrow_filesystem(self):
        return pafs.LocalFileSystem()


class FakeArtifactStore(ArtifactStore):
    bucket = "fake"

    def exists(self, key: str) -> bool:
        return False

    def read_json(self, key: str) -> dict[str, object]:
        raise NotImplementedError

    def read_jsonl(self, key: str) -> list[dict[str, object]]:
        raise NotImplementedError

    def write_json(self, key: str, value: dict[str, object]) -> None:
        return None

    def write_jsonl(self, key: str, rows: list[dict[str, object]]) -> None:
        return None

    def upload_file(self, path: Path, key: str) -> None:
        return None

    def build_remote_env(self) -> dict[str, str]:
        return {"R2_BUCKET": self.bucket}


def test_extract_required_sources_from_plan() -> None:
    assert extract_required_sources_from_plan(Path("plans/source-accounting")) == [
        "Books + OCR",
        "Lowyat",
        "Reddit Bolehland",
        "Reddit Indonesia",
        "Cari",
        "HPLT Malay",
        "HPLT Indonesia",
    ]


def test_source_accounting_config_rejects_missing_plan_mappings(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        [
            source_block("Books + OCR", BOOKS_OCR_PARQUET_KEY, "markdown_text"),
            source_block("HPLT Malay", "dataset/processed/malay/hplt/*_malay.parquet", "text"),
            source_block(
                "HPLT Indonesia",
                "dataset/processed/malay/hplt/*_indon.parquet",
                "text",
            ),
        ],
    )

    with pytest.raises(ValueError, match="Lowyat"):
        load_recipe_config(config_path)


def test_source_accounting_config_accepts_all_plan_sources(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        [
            source_block("Books + OCR", BOOKS_OCR_PARQUET_KEY, "markdown_text"),
            source_block("Lowyat", "dataset/raw/forums/lowyat/*.parquet", "text"),
            source_block("Reddit Bolehland", "dataset/raw/reddit/bolehland/*.parquet", "text"),
            source_block("Reddit Indonesia", "dataset/raw/reddit/indonesia/*.parquet", "text"),
            source_block("Cari", "dataset/raw/forums/cari/*.parquet", "text"),
            source_block("HPLT Malay", "dataset/processed/malay/hplt/*_malay.parquet", "text"),
            source_block(
                "HPLT Indonesia",
                "dataset/processed/malay/hplt/*_indon.parquet",
                "text",
            ),
        ],
    )

    config = load_recipe_config(config_path)

    assert config.tokenizer.encoding == "o200k_base"
    assert [source.name for source in config.sources] == [
        "Books + OCR",
        "Lowyat",
        "Reddit Bolehland",
        "Reddit Indonesia",
        "Cari",
        "HPLT Malay",
        "HPLT Indonesia",
    ]


def test_source_accounting_config_accepts_books_ocr_remote_parquet(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        [
            source_block("Books + OCR", BOOKS_OCR_PARQUET_KEY, "markdown_text"),
            source_block("Lowyat", "dataset/processed/malay/lowyat.parquet", "body_text"),
            source_block(
                "Reddit Bolehland",
                "dataset/processed/malay/reddit.parquet",
                "body",
                {"subreddit": "Bolehland"},
            ),
            source_block(
                "Reddit Indonesia",
                "dataset/processed/malay/reddit.parquet",
                "body",
                {"subreddit": "indonesia"},
            ),
            source_block("Cari", "dataset/processed/malay/cari.parquet", "body_text"),
            source_block("HPLT Malay", "dataset/processed/malay/hplt/*_malay.parquet", "text"),
            source_block(
                "HPLT Indonesia",
                "dataset/processed/malay/hplt/*_indon.parquet",
                "text",
            ),
        ],
    )

    config = load_recipe_config(config_path)

    books_source = config.sources[0]
    assert books_source.format == "parquet"
    assert books_source.r2_relative_glob_path == BOOKS_OCR_PARQUET_KEY
    assert books_source.text_column == "markdown_text"


def test_count_source_counts_text_objects_with_tiktoken(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    store.write_bytes("dataset/raw/forums/lowyat/a.txt", "hello dunia\n".encode("utf-8"))
    store.write_bytes("dataset/raw/forums/lowyat/b.txt", "apa khabar".encode("utf-8"))
    task = build_task(
        source="Lowyat",
        source_format="text",
        r2_relative_glob_path="dataset/raw/forums/lowyat/*.txt",
        count_concurrency=2,
    )

    result = count_source(task=task, object_store=store)
    encoding = tiktoken.get_encoding("o200k_base")

    assert result.document_count == 2
    assert result.word_count == 4
    assert result.byte_count == len("hello dunia\napa khabar".encode("utf-8"))
    assert result.token_count == (
        len(encoding.encode("hello dunia\n")) + len(encoding.encode("apa khabar"))
    )


def test_count_source_counts_markdown_objects_from_ocr_prefix(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    store.write_bytes(
        "dataset/processed/pdf_ocr/20260423T195035Z/markdown/alpha.md",
        "# Tajuk\n\nhello dunia\n".encode("utf-8"),
    )
    store.write_bytes(
        "dataset/processed/pdf_ocr/20260423T195035Z/markdown/beta.md",
        "apa khabar".encode("utf-8"),
    )
    task = build_task(
        source="Books + OCR",
        source_format="markdown",
        r2_relative_glob_path="dataset/processed/pdf_ocr/20260423T195035Z/markdown/*.md",
        count_concurrency=2,
    )

    result = count_source(task=task, object_store=store)

    assert result.document_count == 2
    assert result.word_count == 6
    assert result.metadata_columns == ["object_key"]


def test_count_source_reads_only_configured_parquet_text_column(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    parquet_path = tmp_path / BOOKS_OCR_PARQUET_KEY
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "markdown_text": ["satu dua", "tiga", None],
                "ignored": ["x", "y", "z"],
            }
        ),
        parquet_path,
    )
    task = build_task(
        source="Books + OCR",
        source_format="parquet",
        r2_relative_glob_path="dataset/processed/pdf_ocr/20260423T195035Z/*.parquet",
        text_column="markdown_text",
        count_concurrency=2,
    )

    result = count_source(task=task, object_store=store)

    assert result.document_count == 3
    assert result.word_count == 3
    assert result.byte_count == len("satu duatiga".encode("utf-8"))


def test_render_markdown_table_is_deterministic() -> None:
    rendered = render_markdown_table(
        [
            SourceAccountingResult(
                source_order=1,
                source="HPLT Malay",
                token_count=12345,
                word_count=9012,
                byte_count=42678,
                document_count=3000,
                r2_relative_glob_path="dataset/processed/malay/hplt/*_malay.parquet",
                filters={},
                metadata_columns=["id", "url", "language"],
                source_row_r2_key="out/sources/hplt-malay.json",
                table_r2_key="out/source-accounting.md",
            ),
            SourceAccountingResult(
                source_order=0,
                source="Books + OCR",
                token_count=5,
                word_count=4,
                byte_count=20,
                document_count=2,
                r2_relative_glob_path=BOOKS_OCR_PARQUET_KEY,
                filters={"metadata_source": "catalog"},
                metadata_columns=["book_id", "metadata_source", "title"],
                source_row_r2_key="out/sources/books-ocr.json",
                table_r2_key="out/source-accounting.md",
            ),
        ]
    )

    assert rendered == (
        "| source | token_count | word_count | byte_count | document_count | "
        "r2_relative_glob_path | filters | metadata_columns |\n"
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |\n"
        f"| Books + OCR | 5 | 4 | 20 | 2 | `{BOOKS_OCR_PARQUET_KEY}` | "
        "`metadata_source=catalog` | `book_id, metadata_source, title` |\n"
        "| HPLT Malay | 12,345 | 9,012 | 42,678 | 3,000 | "
        "`dataset/processed/malay/hplt/*_malay.parquet` | `` | `id, url, language` |\n"
    )


def test_count_source_filters_single_reddit_parquet_by_subreddit(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    parquet_path = tmp_path / "dataset" / "processed" / "malay" / "reddit.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "post_kind": ["submission", "comment", "comment"],
                "subreddit": ["Bolehland", "indonesia", "Bolehland"],
                "body": ["satu", "dua", "tiga empat"],
                "month": ["2026-01", "2026-01", "2026-02"],
            }
        ),
        parquet_path,
    )
    task = build_task(
        source="Reddit Bolehland",
        source_format="parquet",
        r2_relative_glob_path="dataset/processed/malay/reddit.parquet",
        text_column="body",
        filters={"subreddit": "Bolehland"},
        count_concurrency=2,
    )

    result = count_source(task=task, object_store=store)

    assert result.document_count == 2
    assert result.word_count == 3
    assert result.filters == {"subreddit": "Bolehland"}
    assert result.metadata_columns == ["post_kind", "subreddit", "month"]


def test_source_accounting_cli_validate_loads_recipe(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        [
            source_block("Books + OCR", BOOKS_OCR_PARQUET_KEY, "markdown_text"),
            source_block("Lowyat", "dataset/raw/forums/lowyat/*.parquet", "text"),
            source_block("Reddit Bolehland", "dataset/raw/reddit/bolehland/*.parquet", "text"),
            source_block("Reddit Indonesia", "dataset/raw/reddit/indonesia/*.parquet", "text"),
            source_block("Cari", "dataset/raw/forums/cari/*.parquet", "text"),
            source_block("HPLT Malay", "dataset/processed/malay/hplt/*_malay.parquet", "text"),
            source_block(
                "HPLT Indonesia",
                "dataset/processed/malay/hplt/*_indon.parquet",
                "text",
            ),
        ],
    )

    result = CliRunner().invoke(
        cli,
        ["source-accounting-validate", "--config", str(config_path)],
    )

    assert result.exit_code == 0
    assert "Validated source accounting recipe" in result.output
    assert "Token encoding: o200k_base" in result.output
    assert "Declared sources: 7" in result.output


def test_source_accounting_sample_config_uses_books_ocr_parquet() -> None:
    config = load_recipe_config(Path("config/source_accounting.sample.yaml"))

    books_source = config.sources[0]
    assert books_source.name == "Books + OCR"
    assert books_source.format == "parquet"
    assert books_source.r2_relative_glob_path == BOOKS_OCR_PARQUET_KEY
    assert books_source.text_column == "markdown_text"


def test_main_cli_registers_source_accounting_commands() -> None:
    assert "source-accounting-validate" in cli.commands
    assert "source-accounting-run" in cli.commands
    assert "source-accounting-remote-job" in cli.commands


def test_source_accounting_submission_uses_cpu_dependency_group(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        [
            source_block("Books + OCR", BOOKS_OCR_PARQUET_KEY, "markdown_text"),
            source_block("Lowyat", "dataset/processed/malay/lowyat.parquet", "body_text"),
            source_block(
                "Reddit Bolehland",
                "dataset/processed/malay/reddit.parquet",
                "body",
                {"subreddit": "Bolehland"},
            ),
            source_block(
                "Reddit Indonesia",
                "dataset/processed/malay/reddit.parquet",
                "body",
                {"subreddit": "indonesia"},
            ),
            source_block("Cari", "dataset/processed/malay/cari.parquet", "body_text"),
            source_block("HPLT Malay", "dataset/processed/malay/hplt/*_malay.parquet", "text"),
            source_block(
                "HPLT Indonesia",
                "dataset/processed/malay/hplt/*_indon.parquet",
                "text",
            ),
        ],
    )
    config = load_recipe_config(config_path)
    adapter = SourceAccountingSubmissionAdapter(config=config, config_path=config_path)

    prepared = adapter.prepare_new_run(FakeArtifactStore(), dry_run=True)

    assert "--group source_accounting --no-dev --frozen" in prepared.bootstrap.command
    assert " --group source_accounting " in prepared.invocation.command
    assert "remote_ocr" not in prepared.bootstrap.command
    assert "remote_ocr" not in prepared.invocation.command


def build_task(
    *,
    source: str,
    source_format: str,
    r2_relative_glob_path: str,
    text_column: str = "",
    filters: dict[str, str] | None = None,
    count_concurrency: int = 8,
) -> SourceAccountingTask:
    return SourceAccountingTask(
        source_order=0,
        source=source,
        format=source_format,
        r2_relative_glob_path=r2_relative_glob_path,
        text_column=text_column,
        parquet_batch_size=2,
        count_concurrency=count_concurrency,
        filters=filters or {},
        token_encoding="o200k_base",
        source_row_r2_key="output/sources/source.json",
        table_r2_key="output/source-accounting.md",
    )


def write_config(tmp_path: Path, source_blocks: list[str]) -> Path:
    config_path = tmp_path / "source_accounting.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: source-accounting-test",
                "  config_version: 1",
                "ssh:",
                "  host: 127.0.0.1",
                "  port: 22",
                "  user: root",
                "  identity_file: ~/.ssh/id_ed25519",
                "remote:",
                "  root_dir: /tmp/training-signal-processing",
                "  python_version: '3.12'",
                "  remote_jobs_root: /tmp/source-accounting-jobs",
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
                "  bucket: test-bucket",
                "  output_prefix: dataset/processed/source_accounting",
                "input:",
                "  plan_path: plans/source-accounting",
                "  source_root_key: dataset",
                "tokenizer:",
                "  encoding: o200k_base",
                "mlflow:",
                "  enabled: false",
                "  tracking_uri: ''",
                "  experiment_name: source-accounting",
                "observability:",
                "  flush_interval_sec: 5",
                "  log_per_file_events: true",
                "  heartbeat_interval_sec: 10",
                "resumability:",
                "  strategy: batch_manifest",
                "  commit_every_batches: 1",
                "  resume_mode: latest",
                "ops:",
                "  - name: prepare_source_accounting_source",
                "    type: mapper",
                "  - name: count_source_accounting_source",
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
    filters: dict[str, str] | None = None,
) -> str:
    block = (
        f"  - name: {name!r}\n"
        "    format: parquet\n"
        f"    r2_relative_glob_path: {r2_relative_glob_path!r}\n"
        f"    text_column: {text_column!r}\n"
        "    parquet_batch_size: 2"
    )
    if not filters:
        return block
    filter_lines = ["    filters:"]
    for key, value in filters.items():
        filter_lines.append(f"      {key}: {value!r}")
    return block + "\n" + "\n".join(filter_lines)


def object_source_block(
    name: str,
    source_format: str,
    r2_relative_glob_path: str,
) -> str:
    return (
        f"  - name: {name!r}\n"
        f"    format: {source_format!r}\n"
        f"    r2_relative_glob_path: {r2_relative_glob_path!r}"
    )
