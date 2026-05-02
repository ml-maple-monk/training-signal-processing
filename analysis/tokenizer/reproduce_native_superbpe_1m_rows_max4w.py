from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evaluate_superbpe_fertility import evaluate, write_markdown

RUN_ID = "native_superbpe_1m_rows_max4w"
TOKENIZER_JSON = Path("tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json")
CONFIG = Path("config/tokenizer_training.superbpe_balanced_50k.sample.yaml")
CACHE_ROOT = Path(
    ".runtime/tokenizers/parquet-cache/20260501T211207Z-rclone-balanced-fineweb-1to1"
)
OUTPUT_DIR = Path(".runtime/tokenizers/experiments/native_superbpe_1m_rows_max4w")
EXPECTED_JSON = Path(__file__).with_name("native_superbpe_1m_rows_max4w_expected.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the native_superbpe_1m_rows_max4w fertility numbers "
            "published in TOKENIZER_DOCS.md."
        )
    )
    parser.add_argument("--tokenizer-json", type=Path, default=TOKENIZER_JSON)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument("--expected-json", type=Path, default=EXPECTED_JSON)
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--max-docs-per-source", type=int, default=5000)
    parser.add_argument("--max-bytes-per-source", type=int, default=33_554_432)
    parser.add_argument("--read-batch-rows", type=int, default=8192)
    parser.add_argument("--fineweb-files", type=int, default=160)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Write metrics without checking them against the expected doc numbers.",
    )
    args = parser.parse_args()

    result = load_or_run(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "superbpe_tokenizer_evaluation.json"
    md_path = args.output_dir / "superbpe_tokenizer_evaluation.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(result, md_path)

    if not args.no_verify:
        expected = json.loads(args.expected_json.read_text(encoding="utf-8"))
        actual = comparable_metrics(result)
        assert_expected_metrics(expected=expected, actual=actual)

    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "verified": not args.no_verify,
            },
            indent=2,
        )
    )


def load_or_run(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_json is not None:
        return json.loads(args.input_json.read_text(encoding="utf-8"))
    require_path(args.tokenizer_json, "tokenizer")
    require_path(args.config, "config")
    require_path(args.cache_root, "local parquet cache root")
    if args.training_summary is not None:
        require_path(args.training_summary, "training summary")
    return evaluate(
        SimpleNamespace(
            run_id=RUN_ID,
            tokenizer_json=args.tokenizer_json,
            training_summary=args.training_summary,
            config=args.config,
            cache_root=args.cache_root,
            output_dir=args.output_dir,
            max_docs_per_source=args.max_docs_per_source,
            max_bytes_per_source=args.max_bytes_per_source,
            read_batch_rows=args.read_batch_rows,
            fineweb_files=args.fineweb_files,
        )
    )


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def comparable_metrics(result: dict[str, Any]) -> dict[str, Any]:
    aggregate = result["aggregate_metrics"]
    return {
        "aggregate": {
            "bytes_per_token": rounded(aggregate["bytes_per_token"]),
            "characters_per_token": rounded(aggregate["characters_per_token"]),
            "documents": aggregate["documents"],
            "encoded_tokens": aggregate["encoded_tokens"],
            "fertility_tokens_per_word": rounded(
                aggregate["fertility_tokens_per_word"]
            ),
            "roundtrip_failures": aggregate["roundtrip_failures"],
            "roundtrip_ok": aggregate["roundtrip_ok"],
            "tokens_per_1000_chars": rounded(aggregate["tokens_per_1000_chars"]),
            "utf8_bytes": aggregate["utf8_bytes"],
            "words_whitespace": aggregate["words_whitespace"],
        },
        "per_source": {
            source: {
                "documents": metrics["documents"],
                "encoded_tokens": metrics["encoded_tokens"],
                "fertility_tokens_per_word": rounded(
                    metrics["fertility_tokens_per_word"]
                ),
                "utf8_bytes": metrics["utf8_bytes"],
                "words_whitespace": metrics["words_whitespace"],
            }
            for source, metrics in sorted(result["per_source_metrics"].items())
        },
    }


def rounded(value: float) -> float:
    return round(value, 3)


def assert_expected_metrics(*, expected: dict[str, Any], actual: dict[str, Any]) -> None:
    if actual == expected:
        return
    expected_text = json.dumps(expected, indent=2, sort_keys=True).splitlines()
    actual_text = json.dumps(actual, indent=2, sort_keys=True).splitlines()
    diff = "\n".join(
        difflib.unified_diff(
            expected_text,
            actual_text,
            fromfile="expected",
            tofile="actual",
            lineterm="",
        )
    )
    raise SystemExit(f"Fertility metrics differ from TOKENIZER_DOCS.md:\n{diff}")


if __name__ == "__main__":
    main()
