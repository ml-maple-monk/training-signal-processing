from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from tokenizers import Tokenizer

WORD_PATTERN = re.compile(r"\S+")
MIB = 1024**2


@dataclass
class SourceMetrics:
    documents: int = 0
    utf8_bytes: int = 0
    characters: int = 0
    words_whitespace: int = 0
    encoded_tokens: int = 0
    roundtrip_ok: int = 0
    roundtrip_failures: int = 0
    first_roundtrip_failure: dict[str, Any] | None = None
    token_counts: list[int] | None = None
    token_word_ratios: list[float] | None = None
    char_token_ratios: list[float] | None = None

    def add(self, text: str, ids: list[int], decoded: str) -> None:
        byte_count = len(text.encode("utf-8"))
        char_count = len(text)
        word_count = len(WORD_PATTERN.findall(text))
        token_count = len(ids)
        self.documents += 1
        self.utf8_bytes += byte_count
        self.characters += char_count
        self.words_whitespace += word_count
        self.encoded_tokens += token_count
        if decoded == text:
            self.roundtrip_ok += 1
        else:
            self.roundtrip_failures += 1
            if self.first_roundtrip_failure is None:
                self.first_roundtrip_failure = {
                    "expected_prefix": text[:200],
                    "decoded_prefix": decoded[:200],
                    "token_ids_prefix": ids[:40],
                }
        assert self.token_counts is not None
        assert self.token_word_ratios is not None
        assert self.char_token_ratios is not None
        self.token_counts.append(token_count)
        if word_count:
            self.token_word_ratios.append(token_count / word_count)
        if token_count:
            self.char_token_ratios.append(char_count / token_count)

    def done(self, max_docs: int, max_bytes: int) -> bool:
        return self.documents >= max_docs or self.utf8_bytes >= max_bytes

    def finalize(self) -> dict[str, Any]:
        token_counts = sorted(self.token_counts or [])
        token_word_ratios = self.token_word_ratios or []
        char_token_ratios = self.char_token_ratios or []
        return {
            "documents": self.documents,
            "utf8_bytes": self.utf8_bytes,
            "characters": self.characters,
            "words_whitespace": self.words_whitespace,
            "encoded_tokens": self.encoded_tokens,
            "roundtrip_ok": self.roundtrip_ok,
            "roundtrip_failures": self.roundtrip_failures,
            "roundtrip_exact_rate": safe_div(self.roundtrip_ok, self.documents),
            "fertility_tokens_per_word": safe_div(
                self.encoded_tokens, self.words_whitespace
            ),
            "characters_per_token": safe_div(self.characters, self.encoded_tokens),
            "bytes_per_token": safe_div(self.utf8_bytes, self.encoded_tokens),
            "tokens_per_1000_chars": 1000 * safe_div(self.encoded_tokens, self.characters),
            "tokens_per_doc_mean": safe_div(self.encoded_tokens, self.documents),
            "tokens_per_doc_p50": percentile(token_counts, 50),
            "tokens_per_doc_p95": percentile(token_counts, 95),
            "doc_mean_tokens_per_word": mean(token_word_ratios),
            "doc_mean_chars_per_token": mean(char_token_ratios),
            "first_roundtrip_failure": self.first_roundtrip_failure,
        }


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(sorted_values: list[int], pct: int) -> float:
    if not sorted_values:
        return 0.0
    rank = (len(sorted_values) - 1) * (pct / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bytes_to_unicode() -> dict[int, str]:
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    char_values = byte_values[:]
    n = 0
    for byte in range(2**8):
        if byte not in byte_values:
            byte_values.append(byte)
            char_values.append(2**8 + n)
            n += 1
    return dict(zip(byte_values, [chr(value) for value in char_values], strict=True))


class ByteLevelDecoder:
    def __init__(self, tokenizer: Tokenizer) -> None:
        vocab = tokenizer.get_vocab()
        self.id_to_token = {token_id: token for token, token_id in vocab.items()}
        self.byte_by_char = {char: byte for byte, char in bytes_to_unicode().items()}

    def decode(self, ids: list[int]) -> str:
        data = bytearray()
        for token_id in ids:
            for char in self.id_to_token[token_id]:
                data.append(self.byte_by_char[char])
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="replace")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def cache_path(cache_root: Path, bucket: str, prefix: str) -> Path:
    return cache_root / bucket / prefix.strip("/")


def parquet_paths_for_source(config: dict[str, Any], cache_root: Path, source: str) -> list[Path]:
    input_config = config["input"]
    bucket = config["r2"]["bucket"]
    prefix = input_config.get("source_parts_prefixes", {}).get(
        source, input_config["final_parts_prefix"]
    )
    root = cache_path(cache_root, bucket, prefix)
    return sorted(root.rglob("*.parquet"))


def evenly_spaced(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    indexes = {
        round(index * (len(paths) - 1) / (limit - 1))
        for index in range(limit)
    }
    return [paths[index] for index in sorted(indexes)]


def active_sources(
    grouped_sources: list[str],
    metrics: dict[str, SourceMetrics],
    max_docs: int,
    max_bytes: int,
) -> set[str]:
    return {
        source
        for source in grouped_sources
        if not metrics[source].done(max_docs=max_docs, max_bytes=max_bytes)
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_yaml(args.config)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    decoder = ByteLevelDecoder(tokenizer)
    sources = list(config["input"]["sources"])
    metrics = {
        source: SourceMetrics(
            token_counts=[],
            token_word_ratios=[],
            char_token_ratios=[],
        )
        for source in sources
    }
    paths_by_source = {
        source: parquet_paths_for_source(config, args.cache_root, source)
        for source in sources
    }

    groups: dict[tuple[Path, ...], list[str]] = {}
    for source, paths in paths_by_source.items():
        if source == "fineweb":
            paths = evenly_spaced(paths, args.fineweb_files)
            paths_by_source[source] = paths
        groups.setdefault(tuple(paths), []).append(source)

    started = time.monotonic()
    text_column = config["input"]["text_column"]
    source_column = config["input"]["source_column"]
    dropped_column = config["input"]["dropped_column"]
    columns = [text_column, source_column, dropped_column]

    for paths_tuple, grouped_sources in groups.items():
        paths = list(paths_tuple)
        for path in paths:
            wanted = active_sources(
                grouped_sources, metrics, args.max_docs_per_source, args.max_bytes_per_source
            )
            if not wanted:
                break
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(
                batch_size=args.read_batch_rows,
                columns=columns,
            ):
                table = pa.Table.from_batches([batch])
                text = table[text_column]
                source_values = table[source_column]
                dropped = table[dropped_column]
                mask = pc.and_(
                    pc.is_in(source_values, value_set=pa.array(sorted(wanted))),
                    pc.and_(
                        pc.invert(pc.fill_null(dropped, False)),
                        pc.not_equal(pc.fill_null(text, ""), ""),
                    ),
                )
                filtered = table.filter(mask)
                if filtered.num_rows == 0:
                    continue
                texts = filtered[text_column].to_pylist()
                row_sources = filtered[source_column].to_pylist()
                for row_source, row_text in zip(row_sources, texts, strict=True):
                    if row_source not in wanted or row_text is None:
                        continue
                    source_metrics = metrics[row_source]
                    if source_metrics.done(
                        max_docs=args.max_docs_per_source,
                        max_bytes=args.max_bytes_per_source,
                    ):
                        wanted.discard(row_source)
                        continue
                    encoding = tokenizer.encode(row_text)
                    ids = encoding.ids
                    source_metrics.add(row_text, ids, decoder.decode(ids))
                    if source_metrics.done(
                        max_docs=args.max_docs_per_source,
                        max_bytes=args.max_bytes_per_source,
                    ):
                        wanted.discard(row_source)
                    if not wanted:
                        break
                if not wanted:
                    break

    aggregate = SourceMetrics(token_counts=[], token_word_ratios=[], char_token_ratios=[])
    for source in sources:
        source_metrics = metrics[source]
        aggregate.documents += source_metrics.documents
        aggregate.utf8_bytes += source_metrics.utf8_bytes
        aggregate.characters += source_metrics.characters
        aggregate.words_whitespace += source_metrics.words_whitespace
        aggregate.encoded_tokens += source_metrics.encoded_tokens
        aggregate.roundtrip_ok += source_metrics.roundtrip_ok
        aggregate.roundtrip_failures += source_metrics.roundtrip_failures
        assert aggregate.token_counts is not None
        assert aggregate.token_word_ratios is not None
        assert aggregate.char_token_ratios is not None
        aggregate.token_counts.extend(source_metrics.token_counts or [])
        aggregate.token_word_ratios.extend(source_metrics.token_word_ratios or [])
        aggregate.char_token_ratios.extend(source_metrics.char_token_ratios or [])
        if aggregate.first_roundtrip_failure is None:
            aggregate.first_roundtrip_failure = source_metrics.first_roundtrip_failure
    per_source = {
        source: metrics[source].finalize()
        for source in sources
    }
    result = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": args.run_id,
        "tokenizer_path": str(args.tokenizer_json),
        "training_summary_path": str(args.training_summary),
        "cache_root": str(args.cache_root),
        "evaluation_seconds": time.monotonic() - started,
        "sample_policy": {
            "max_docs_per_source": args.max_docs_per_source,
            "max_utf8_bytes_per_source": args.max_bytes_per_source,
            "read_batch_rows": args.read_batch_rows,
            "fineweb_file_sampling": (
                f"evenly spaced {len(paths_by_source.get('fineweb', []))} files "
                f"from {len(parquet_paths_for_source(config, args.cache_root, 'fineweb'))} "
                "cached FineWeb parquet files"
            ),
            "word_definition": r"Python regex \S+ over decoded Unicode text",
            "fertility_definition": "encoded SuperBPE tokens / whitespace words",
        },
        "training_summary": json.loads(args.training_summary.read_text(encoding="utf-8")),
        "aggregate_metrics": aggregate.finalize(),
        "per_source_metrics": per_source,
    }
    return result


def format_table(result: dict[str, Any]) -> str:
    rows = [
        "| Source | Docs | MiB | Words | Tokens | Fertility | "
        "Chars/token | Bytes/token | Roundtrip |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source, metrics in result["per_source_metrics"].items():
        rows.append(
            "| "
            f"`{source}` | "
            f"{metrics['documents']} | "
            f"{metrics['utf8_bytes'] / MIB:.2f} | "
            f"{metrics['words_whitespace']} | "
            f"{metrics['encoded_tokens']} | "
            f"{metrics['fertility_tokens_per_word']:.3f} | "
            f"{metrics['characters_per_token']:.3f} | "
            f"{metrics['bytes_per_token']:.3f} | "
            f"{metrics['roundtrip_ok']}/{metrics['documents']} |"
        )
    return "\n".join(rows)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    aggregate = result["aggregate_metrics"]
    training = result["training_summary"]
    superbpe = training["superbpe"]
    text = f"""# SuperBPE Fertility Evaluation

Run id: `{result["run_id"]}`

## Artifacts

- Tokenizer JSON: `{result["tokenizer_path"]}`
- Training summary: `{result["training_summary_path"]}`
- Evaluation JSON: `{path.with_suffix(".json")}`

## Training Snapshot

- Training rows: `{training["sampled_rows"]}`
- Training UTF-8 bytes: `{training["sampled_bytes"]}`
- Stop reason: `{training["stop_reason"]}`
- Vocab size: `{training["vocab_size"]}`
- Max token length: `{training["max_token_length"]}`
- Stage 2 inherited merges: `{superbpe["stage2_inherit_merge_pairs"]}`
- Stage 2 max words per token: `{superbpe["stage2_max_words_per_token"]}`
- Stage 2 ingest seconds: `{training["stage2_ingest_elapsed_seconds"]:.3f}`
- Stage 2 train seconds: `{training["stage2_train_elapsed_seconds"]:.3f}`

## Fertility Metrics

Fertility here means `encoded SuperBPE tokens / whitespace words`, where words are counted
with regex `\\S+`. Lower fertility means fewer tokenizer pieces per whitespace-delimited word.

Evaluation sample policy:

- Up to `{result["sample_policy"]["max_docs_per_source"]}` documents per source.
- Up to `{result["sample_policy"]["max_utf8_bytes_per_source"]}` UTF-8 bytes per source.
- {result["sample_policy"]["fineweb_file_sampling"]}.

Aggregate sample metrics:

- Documents: `{aggregate["documents"]}`
- UTF-8 bytes: `{aggregate["utf8_bytes"]}`
- Whitespace words: `{aggregate["words_whitespace"]}`
- Encoded tokens: `{aggregate["encoded_tokens"]}`
- Fertility tokens/word: `{aggregate["fertility_tokens_per_word"]:.3f}`
- Characters/token: `{aggregate["characters_per_token"]:.3f}`
- Bytes/token: `{aggregate["bytes_per_token"]:.3f}`
- Tokens/1000 chars: `{aggregate["tokens_per_1000_chars"]:.3f}`
- Roundtrip exact rate: `{aggregate["roundtrip_exact_rate"]:.6f}`

{format_table(result)}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/tokenizer_training.superbpe_balanced_50k.sample.yaml"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(".runtime/tokenizers/parquet-cache/20260501T211207Z-rclone-balanced-fineweb-1to1"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-docs-per-source", type=int, default=5000)
    parser.add_argument("--max-bytes-per-source", type=int, default=33_554_432)
    parser.add_argument("--read-batch-rows", type=int, default=8192)
    parser.add_argument("--fineweb-files", type=int, default=160)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(args)
    json_path = args.output_dir / "superbpe_tokenizer_evaluation.json"
    md_path = args.output_dir / "superbpe_tokenizer_evaluation.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
