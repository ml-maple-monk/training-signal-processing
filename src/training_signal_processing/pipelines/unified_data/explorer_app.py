"""Gradio explorer for the merged cleaned unified dataset on R2.

Streams random rows directly from R2 via PyArrow — no parquet shard is ever
downloaded to local disk. DuckDB is intentionally avoided here because v1.5.2
of its parquet stats reader fatally errors on these shards (negative-int
values in column statistics decoded as UBIGINT).

The shards are 1 row group of 500k rows each, so partial reads are not
possible — a fresh fetch costs ~30s. A small in-memory buffer amortizes that
cost: each shard read extracts ``BATCH_SIZE`` random rows and distributes
them into per-source FIFOs. Per-click pops are O(1). First click for a given
source filter is dominated by the shard read; subsequent clicks for the same
source are instant until the buffer drains.
"""

from __future__ import annotations

import os
import random
import threading
from collections import defaultdict
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from training_signal_processing.core.models import R2Config
from training_signal_processing.core.storage import R2ObjectStore

if TYPE_CHECKING:
    import gradio as gr


PARTS_PREFIX_DIR = (
    "dataset/processed/unified-data/final-completed-20260430T160615Z/parts/"
)

_PROJECTED_COLUMNS: tuple[str, ...] = (
    "cleaned_text",
    "source_name",
    "cleaning_source",
    "cleaned_o200k_token_count",
    "cleaning_is_dropped",
)

SOURCE_NAMES: list[str] = [
    "All",
    "Books + OCR",
    "Cari",
    "HPLT Indonesia",
    "HPLT Malay",
    "Lowyat",
    "Reddit Bolehland",
    "Reddit Indonesia",
]

EMPTY_PLACEHOLDER = "(no row matched — try again)"
DEFAULT_BATCH_SIZE = 200
DEFAULT_MAX_REFILLS = 5


def _resolve_store(config_file: str | None) -> R2ObjectStore:
    if config_file:
        return R2ObjectStore.from_config_file(R2Config(config_file=config_file, bucket=""))
    if os.environ.get("R2_ACCESS_KEY_ID"):
        return R2ObjectStore.from_environment(R2Config(config_file="", bucket=""))
    raise RuntimeError(
        "No R2 credentials available. Either pass --config-file pointing to an "
        "env-style file with keys R2_BUCKET_NAME, AWS_ACCESS_KEY_ID, "
        "AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, MLFLOW_S3_ENDPOINT_URL, "
        "or export the OS env vars R2_BUCKET, R2_ACCESS_KEY_ID, "
        "R2_SECRET_ACCESS_KEY, R2_REGION, R2_ENDPOINT_URL."
    )


class _SampleBuffer:
    """Thread-safe lazy pool of randomly-sampled rows, partitioned per source."""

    def __init__(
        self,
        store: R2ObjectStore,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._store = store
        self._fs = store.build_pyarrow_filesystem()
        self._batch_size = batch_size
        self._shards: list[str] = sorted(
            k for k in store.list_keys(PARTS_PREFIX_DIR) if k.endswith(".parquet")
        )
        if not self._shards:
            raise RuntimeError(
                f"No parquet shards found under s3://{store.bucket}/{PARTS_PREFIX_DIR}"
            )
        self._buffers: dict[str, list[dict[str, object]]] = defaultdict(list)
        self._failed_shards: set[str] = set()
        self._lock = threading.Lock()

    @property
    def shards(self) -> list[str]:
        return list(self._shards)

    def _read_random_shard(self) -> str | None:
        """Read one random unread shard, append rows into per-source buffers."""
        choices = [k for k in self._shards if k not in self._failed_shards]
        if not choices:
            return None
        key = random.choice(choices)
        try:
            pf = pq.ParquetFile(f"{self._store.bucket}/{key}", filesystem=self._fs)
            table = pf.read_row_group(0, columns=list(_PROJECTED_COLUMNS))
        except Exception:
            self._failed_shards.add(key)
            return None
        n = table.num_rows
        if n == 0:
            return key
        sample_size = min(self._batch_size, n)
        indices = random.sample(range(n), sample_size)
        sampled = table.take(pa.array(indices))
        for i in range(sampled.num_rows):
            row: dict[str, object] = {col: sampled[col][i].as_py() for col in _PROJECTED_COLUMNS}
            row["_filename"] = key
            row["_file_row_number"] = indices[i]
            self._buffers["All"].append(row)
            src = row.get("source_name")
            if isinstance(src, str) and src:
                self._buffers[src].append(row)
        for buf in self._buffers.values():
            random.shuffle(buf)
        return key

    def next(
        self,
        source_filter: str,
        max_refills: int = DEFAULT_MAX_REFILLS,
    ) -> dict[str, object] | None:
        with self._lock:
            for _ in range(max_refills + 1):
                buf = self._buffers.get(source_filter)
                if buf:
                    return buf.pop()
                if self._read_random_shard() is None:
                    return None
            return None


def _sample_one(buffer: _SampleBuffer, source_filter: str) -> dict[str, object] | None:
    raw = buffer.next(source_filter)
    if raw is None:
        return None
    return {
        "cleaned_text": raw.get("cleaned_text"),
        "source_name": raw.get("source_name"),
        "cleaning_source": raw.get("cleaning_source"),
        "tokens": raw.get("cleaned_o200k_token_count"),
        "is_dropped": raw.get("cleaning_is_dropped"),
        "removed_chars": None,
        "filename": raw.get("_filename"),
        "file_row_number": raw.get("_file_row_number"),
    }


def _format_row(row: dict[str, object] | None) -> tuple[str, str, str, str, str, str, str, str]:
    if row is None:
        return (EMPTY_PLACEHOLDER,) * 8
    parquet_basename = os.path.basename(str(row["filename"])) if row["filename"] else ""
    return (
        str(row["cleaned_text"] or ""),
        str(row["source_name"] or ""),
        str(row["cleaning_source"] or ""),
        str(row["tokens"]) if row["tokens"] is not None else "",
        str(row["is_dropped"]) if row["is_dropped"] is not None else "",
        str(row["removed_chars"]) if row["removed_chars"] is not None else "(unavailable)",
        parquet_basename,
        str(row["file_row_number"]) if row["file_row_number"] is not None else "",
    )


def build_app(buffer: _SampleBuffer) -> "gr.Blocks":
    import gradio as gr

    def on_next(source: str) -> tuple[str, str, str, str, str, str, str, str]:
        return _format_row(_sample_one(buffer, source))

    with gr.Blocks(title="Cleaned Data Explorer", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# 🧹 Cleaned Data Explorer\n"
            "Random rows streamed from R2 dataset "
            "`final-completed-20260430T160615Z` (~38.5M rows across "
            f"{len(buffer.shards)} parquet shards). "
            "First click after changing the source filter warms a cache "
            "(~30s); subsequent clicks for the same source are instant."
        )

        with gr.Row():
            source_dropdown = gr.Dropdown(
                choices=SOURCE_NAMES,
                value="All",
                label="Source filter",
                scale=3,
            )
            next_btn = gr.Button("🎲 Next random row", variant="primary", scale=1)

        cleaned_text_box = gr.Textbox(
            label="cleaned_text",
            lines=20,
            interactive=False,
            placeholder="Click 'Next random row' to draw a sample.",
        )

        with gr.Row():
            source_name_box = gr.Textbox(label="source_name", interactive=False)
            cleaning_source_box = gr.Textbox(label="cleaning_source", interactive=False)
            tokens_box = gr.Textbox(label="tokens (o200k)", interactive=False)

        with gr.Row():
            is_dropped_box = gr.Textbox(label="cleaning_is_dropped", interactive=False)
            removed_chars_box = gr.Textbox(label="removed_char_count", interactive=False)
            row_number_box = gr.Textbox(label="file_row_number", interactive=False)

        parquet_path_box = gr.Textbox(label="parquet shard", interactive=False)

        outputs = [
            cleaned_text_box,
            source_name_box,
            cleaning_source_box,
            tokens_box,
            is_dropped_box,
            removed_chars_box,
            parquet_path_box,
            row_number_box,
        ]

        next_btn.click(fn=on_next, inputs=[source_dropdown], outputs=outputs)

    return app


def launch(
    *,
    config_file: str | None,
    host: str,
    port: int,
    share: bool,
) -> None:
    store = _resolve_store(config_file)
    buffer = _SampleBuffer(store)
    app = build_app(buffer)
    app.launch(server_name=host, server_port=port, share=share, show_error=True)
