from __future__ import annotations

import os
import resource
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ...core.storage import ObjectStore
from .models import BudgetConfig, InputConfig, SamplerCursorState, SampleStats

Clock = Callable[[], float]
RssProvider = Callable[[], float]


def list_final_parquet_keys(object_store: ObjectStore, final_parts_prefix: str) -> list[str]:
    prefix = final_parts_prefix.strip().lstrip("/")
    keys = object_store.list_keys(prefix)
    return sorted(key for key in keys if key.endswith(".parquet"))


def list_source_parquet_keys(
    object_store: ObjectStore,
    input_config: InputConfig,
    source: str,
) -> list[str]:
    return list_final_parquet_keys(object_store, input_config.parts_prefix_for_source(source))


def current_peak_rss_mib() -> float:
    # Linux ru_maxrss is KiB. The project targets Linux for local processing.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


@dataclass
class SourceTextCursor:
    source: str
    object_store: ObjectStore
    input_config: InputConfig
    parquet_keys: list[str]
    position: dict[str, int]
    skip_accepted_rows: int = 0

    def __iter__(self) -> Iterator[str]:
        skipped = 0
        columns = [
            self.input_config.text_column,
            self.input_config.source_column,
            self.input_config.dropped_column,
        ]
        filesystem = self.object_store.build_pyarrow_filesystem()
        start_key_index = self.position["key_index"]
        start_row_offset = self.position["row_offset"]
        for key_index, key in enumerate(self.parquet_keys[start_key_index:], start=start_key_index):
            parquet_file = pq.ParquetFile(
                f"{self.object_store.bucket}/{key}",
                filesystem=filesystem,
            )
            missing = sorted(set(columns) - set(parquet_file.schema_arrow.names))
            if missing:
                raise ValueError(
                    f"Parquet part '{key}' is missing tokenizer training columns: "
                    + ", ".join(missing)
                )
            row_offset = 0
            for batch in parquet_file.iter_batches(
                batch_size=self.input_config.read_batch_rows,
                columns=columns,
            ):
                text_values = batch.column(self.input_config.text_column).to_pylist()
                source_values = batch.column(self.input_config.source_column).to_pylist()
                dropped_values = batch.column(self.input_config.dropped_column).to_pylist()
                for physical_row_delta, (text_value, source_value, dropped_value) in enumerate(
                    zip(
                        text_values,
                        source_values,
                        dropped_values,
                        strict=True,
                    )
                ):
                    physical_row_offset = row_offset + physical_row_delta
                    if key_index == start_key_index and physical_row_offset < start_row_offset:
                        continue
                    self.position["key_index"] = key_index
                    self.position["row_offset"] = physical_row_offset + 1
                    if bool(dropped_value):
                        continue
                    if str(source_value or "").strip().lower().replace("_", "-") != self.source:
                        continue
                    text = str(text_value or "")
                    if not text:
                        continue
                    if skipped < self.skip_accepted_rows:
                        skipped += 1
                        continue
                    yield text
                row_offset += len(text_values)
            self.position["key_index"] = key_index + 1
            self.position["row_offset"] = 0


class RoundRobinTextSampler:
    def __init__(
        self,
        *,
        object_store: ObjectStore,
        input_config: InputConfig,
        budget: BudgetConfig,
        clock: Clock = time.monotonic,
        rss_provider: RssProvider = current_peak_rss_mib,
        parquet_keys: list[str] | None = None,
        cursor_state: dict[str, Any] | SamplerCursorState | None = None,
    ) -> None:
        self.object_store = object_store
        self.input_config = input_config
        self.budget = budget
        self.clock = clock
        self.rss_provider = rss_provider
        self.parquet_keys = parquet_keys
        if isinstance(cursor_state, SamplerCursorState):
            self.cursor_state = cursor_state
            self.use_logical_offsets = False
        else:
            self.cursor_state = SamplerCursorState.from_mapping(
                cursor_state,
                sources=input_config.sources,
            )
            self.use_logical_offsets = bool(cursor_state) and "source_positions" not in cursor_state
        self.stats = SampleStats(
            stop_reason="not_started",
            source_counts={source: 0 for source in input_config.sources},
            source_bytes={source: 0 for source in input_config.sources},
        )
        self.prefix_parquet_keys: dict[str, list[str]] = {}
        self.started_at = 0.0

    def __iter__(self) -> Iterator[str]:
        self.started_at = self.clock()
        self.stats.stop_reason = "exhausted"
        progress = SamplerProgressReporter(
            enabled=os.environ.get("TOKENIZER_TRAINING_PROGRESS", "").lower()
            in {"1", "true", "yes", "on"},
            total_bytes=self.budget.max_sample_bytes or None,
            rss_limit_mib=(
                self.budget.max_memory_mib
                if self.budget.max_memory_gib
                else None
            ),
        )
        progress.start()
        active = {
            source: iter(
                SourceTextCursor(
                    source=source,
                    object_store=self.object_store,
                    input_config=self.input_config,
                    parquet_keys=self._source_parquet_keys(source),
                    position=self.cursor_state.source_positions[source],
                    skip_accepted_rows=(
                        self.cursor_state.source_offsets[source]
                        if self.use_logical_offsets
                        else 0
                    ),
                )
            )
            for source in self.input_config.sources
        }
        while active:
            for source in list(self.input_config.sources):
                cursor = active.get(source)
                if cursor is None:
                    continue
                budget_stop_reason = self._budget_stop_reason()
                if budget_stop_reason is not None:
                    self._finish(budget_stop_reason)
                    progress.close(self.stats.stop_reason)
                    return
                try:
                    text = next(cursor)
                except StopIteration:
                    active.pop(source, None)
                    continue
                text_bytes = len(text.encode("utf-8"))
                if (
                    self.budget.max_sample_bytes
                    and self.stats.sampled_bytes + text_bytes > self.budget.max_sample_bytes
                ):
                    self._finish("max_sample_bytes")
                    progress.close(self.stats.stop_reason)
                    return
                self.stats.sampled_rows += 1
                self.stats.sampled_bytes += text_bytes
                self.stats.source_counts[source] += 1
                self.stats.source_bytes[source] += text_bytes
                self.cursor_state.source_offsets[source] += 1
                self.stats.peak_rss_mib = max(self.stats.peak_rss_mib, self.rss_provider())
                progress.update(
                    text_bytes=text_bytes,
                    source=source,
                    sampled_rows=self.stats.sampled_rows,
                    sampled_bytes=self.stats.sampled_bytes,
                    peak_rss_mib=self.stats.peak_rss_mib,
                    source_counts=self.stats.source_counts,
                )
                try:
                    yield text
                except GeneratorExit:
                    progress.close("generator_closed")
                    raise
        self._finish("exhausted")
        progress.close(self.stats.stop_reason)

    def _budget_stop_reason(self) -> str | None:
        elapsed = self.clock() - self.started_at
        self.stats.elapsed_seconds = elapsed
        self.stats.peak_rss_mib = max(self.stats.peak_rss_mib, self.rss_provider())
        if self.budget.max_wall_seconds and elapsed >= self.budget.max_wall_seconds:
            return "max_wall_seconds"
        if self.budget.max_memory_gib and self.stats.peak_rss_mib >= self.budget.max_memory_mib:
            return "max_memory_gib"
        if self.budget.max_sample_rows and self.stats.sampled_rows >= self.budget.max_sample_rows:
            return "max_sample_rows"
        return None

    def _finish(self, reason: str) -> None:
        self.stats.stop_reason = reason
        self.stats.elapsed_seconds = self.clock() - self.started_at
        self.stats.peak_rss_mib = max(self.stats.peak_rss_mib, self.rss_provider())

    def cursor_state_dict(self) -> dict[str, Any]:
        return self.cursor_state.to_dict()

    def _source_parquet_keys(self, source: str) -> list[str]:
        if self.parquet_keys is not None:
            return self.parquet_keys
        prefix = self.input_config.parts_prefix_for_source(source)
        parquet_keys = self.prefix_parquet_keys.get(prefix)
        if parquet_keys is None:
            parquet_keys = list_source_parquet_keys(
                self.object_store,
                self.input_config,
                source,
            )
            self.prefix_parquet_keys[prefix] = parquet_keys
        if not parquet_keys:
            raise ValueError(f"No parquet parts found for source {source!r} under {prefix!r}.")
        return parquet_keys


class SamplerProgressReporter:
    def __init__(
        self,
        *,
        enabled: bool,
        total_bytes: int | None,
        rss_limit_mib: float | None,
    ) -> None:
        self.enabled = enabled
        self.total_bytes = total_bytes
        self.rss_limit_mib = rss_limit_mib
        self.progress: Any | None = None
        self.last_emit_at = 0.0
        self.interval_seconds = float(os.environ.get("TOKENIZER_TRAINING_PROGRESS_INTERVAL", "5"))

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            from tqdm import tqdm
        except Exception:
            self.progress = None
            self._write(
                "tokenizer sampler progress enabled but tqdm is unavailable; "
                "falling back to plain periodic logs."
            )
            return
        self.progress = tqdm(
            total=self.total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Parquet text -> BPEasy",
            dynamic_ncols=True,
            mininterval=1.0,
            file=sys.stdout,
        )
        self._write(
            "Progress tracks UTF-8 bytes yielded from R2 parquet into BPEasy. "
            "max_sample_bytes is a streamed raw-text cap, not a RAM target; RSS can stay "
            "far below raw bytes because BPEasy batches texts and deduplicates regex chunks "
            "before building pair indexes."
        )

    def update(
        self,
        *,
        text_bytes: int,
        source: str,
        sampled_rows: int,
        sampled_bytes: int,
        peak_rss_mib: float,
        source_counts: dict[str, int],
    ) -> None:
        if not self.enabled:
            return
        if self.progress is not None:
            self.progress.update(text_bytes)
            self.progress.set_postfix_str(
                self._status_text(
                    source=source,
                    sampled_rows=sampled_rows,
                    sampled_bytes=sampled_bytes,
                    peak_rss_mib=peak_rss_mib,
                    source_counts=source_counts,
                )
            )
        now = time.monotonic()
        if now - self.last_emit_at >= self.interval_seconds:
            self.last_emit_at = now
            self._write(
                "sampler "
                + self._status_text(
                    source=source,
                    sampled_rows=sampled_rows,
                    sampled_bytes=sampled_bytes,
                    peak_rss_mib=peak_rss_mib,
                    source_counts=source_counts,
                )
            )

    def close(self, reason: str) -> None:
        if not self.enabled:
            return
        if self.progress is not None:
            self.progress.close()
        self._write(
            f"Sampler stopped with {reason}. If the process keeps running after this line, "
            "BPEasy is in Rust-side pretokenized-count merge/vocab construction, where this "
            "Python sampler cannot emit more byte progress."
        )

    def _status_text(
        self,
        *,
        source: str,
        sampled_rows: int,
        sampled_bytes: int,
        peak_rss_mib: float,
        source_counts: dict[str, int],
    ) -> str:
        sampled_gib = sampled_bytes / 1024**3
        rss_gib = peak_rss_mib / 1024
        ratio_text = "rss/raw=warming-up"
        if sampled_bytes >= 64 * 1024**2:
            ratio = rss_gib / sampled_gib if sampled_gib else 0.0
            ratio_text = f"rss/raw={ratio:.2f}x"
        limit = "GiB"
        if self.rss_limit_mib:
            limit = f"/{self.rss_limit_mib / 1024:.1f}GiB"
        counts = ",".join(
            f"{source_name}:{count}"
            for source_name, count in source_counts.items()
            if count
        )
        return (
            f"bytes={self._format_bytes(sampled_bytes)} rows={sampled_rows} "
            f"rss={rss_gib:.2f}{limit} {ratio_text} "
            f"source={source} counts=[{counts}]"
        )

    @staticmethod
    def _format_bytes(value: int) -> str:
        if value >= 1024**3:
            return f"{value / 1024**3:.2f}GiB"
        if value >= 1024**2:
            return f"{value / 1024**2:.1f}MiB"
        return f"{value / 1024:.1f}KiB"

    def _write(self, message: str) -> None:
        if self.progress is not None:
            self.progress.write(message)
            return
        print(message, flush=True)


def artifact_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "tokenizer_json": run_dir / "tokenizer.json",
        "huggingface_json": run_dir / "tokenizer.hf.json",
        "tiktoken_vocab": run_dir / "vocab.tiktoken.txt",
        "training_summary": run_dir / "training_summary.json",
    }


def train_bpeasy_tokenizer(
    iterator: Iterator[str],
    *,
    vocab_size: int,
    max_token_length: int,
    regex_pattern: str,
    special_tokens: list[str],
    fill_to_nearest_multiple_of_eight: bool,
    name: str,
    batch_size: int,
) -> Any:
    from bpeasy.tokenizer import BPEasyTokenizer

    return BPEasyTokenizer.train(
        iterator,
        vocab_size=vocab_size,
        max_token_length=max_token_length,
        regex_pattern=regex_pattern,
        special_tokens=special_tokens,
        fill_to_nearest_multiple_of_eight=fill_to_nearest_multiple_of_eight,
        name=name,
        batch_size=batch_size,
    )


def save_tiktoken_vocab(
    *,
    vocab: dict[bytes, int],
    out_path: Path,
    special_tokens: list[str],
    fill_to_nearest_multiple_of_eight: bool,
) -> None:
    from bpeasy import save_vocab_to_tiktoken

    save_vocab_to_tiktoken(
        vocab,
        str(out_path),
        special_tokens=special_tokens,
        fill_to_nearest_multiple_of_eight=fill_to_nearest_multiple_of_eight,
    )
