# training-signal-processing

Minimal `uv`-managed Python workspace pinned to Python `3.12`.

## Quick Start

```bash
uv sync
uv run pytest
uv run ruff check .
```

## OCR Runtime

The OCR pipeline uses an isolated runtime instead of the main project env:

```bash
./scripts/setup_ocr_runtime.sh
./scripts/run_book_ocr.sh --help
./scripts/setup_marker_runtime.sh
./scripts/run_marker_ocr.sh --help
./scripts/run_marker_benchmark.sh --help
```

This bootstraps:

- `.venv-ocr/` for OCR-only Python packages
- `.runtime/ocr/` for a user-space `tesseract` install, including language data
- `.venv-marker/` for Marker-only Python packages

That keeps OCR dependencies and native binaries out of the default `.venv`.

## Marker Benchmarks

Use the Marker benchmark harness to compare throughput experiments without
changing the main OCR pipeline:

```bash
./scripts/run_marker_benchmark.sh build-manifest
./scripts/run_marker_benchmark.sh run --name baseline-force --ocr-mode force
./scripts/run_marker_benchmark.sh run --name auto-jobs2 --ocr-mode auto --jobs 2
./scripts/run_marker_benchmark.sh run --name full-auto-jobs2 --record-set all --ocr-mode auto --jobs 2
```

This writes manifests and experiment summaries under `.cache/marker_benchmark/`.
