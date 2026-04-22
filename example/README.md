# DuckDB + R2 examples

This folder shows how to query the uploaded parquet files in Cloudflare R2 with DuckDB.

## Why one dataset at a time

The `dataset/raw/` prefix contains multiple parquet families with different schemas and different folder layouts.
Some of them are hive-partitioned, for example:

- `local_ztd/*/selection_bucket=*/*.parquet`
- `seapile_v2/*/selection_bucket=*/*.parquet`

Others are simple parquet folders, for example:

- `books/*.parquet`
- `subtitle/*.parquet`
- `zhihu_article/*.parquet`
- `zhihu_qa/*.parquet`

Because of that, the easiest and most reliable pattern is to query one dataset family at a time.

## Quick start

From the repo root:

```bash
uv run python example/duckdb_r2_query.py --dataset local_ztd --describe
uv run python example/duckdb_r2_query.py --dataset local_ztd --count --where "selection_bucket = 'high_diff' AND qwen_token_count >= 1000"
uv run python example/duckdb_r2_query.py --dataset books --select "title, year, metadata_source" --where "year >= 2020" --limit 20
```

## Streaming-style batch processing

If you want to see rows flow through a small batch pipeline instead of only running one SQL query, use:

```bash
uv run python example/stream_r2_parquet.py \
  --dataset local_ztd \
  --text-column text \
  --where "selection_bucket = 'high_diff'" \
  --batch-size 500 \
  --top-k 5
```

This example:

- scans parquet directly from Cloudflare R2
- pushes the initial `WHERE` filter down into DuckDB
- fetches bounded batches with `fetchmany(...)`
- applies a few common ops in Python: normalize, annotate, filter, score
- keeps only a bounded Top-K heap at the sink

That last part matters for large datasets: it avoids a full global sort. If you need an exact global rank of every row, that becomes a blocking operation and usually needs a spillable external sort or a second-stage materialized ranking job.

## Add a small Qwen model in the stream

The same example can also run a small language-model annotation op after the cheap filters. This keeps the expensive model step off obviously bad rows.

Install the optional model dependencies first:

```bash
uv sync --group model
```

Then run:

```bash
uv run --group model python example/stream_r2_parquet.py \
  --dataset local_ztd \
  --text-column text \
  --where "selection_bucket = 'high_diff'" \
  --batch-size 32 \
  --top-k 5 \
  --with-qwen \
  --qwen-model Qwen/Qwen3-0.6B \
  --qwen-max-new-tokens 8
```

This Qwen op:

- builds a short prompt from each streamed text row
- generates a short label-like completion
- records per-generated-token probabilities and logprobs
- merges those fields back into each row as `qwen_completion`, `qwen_token_probs`, and `qwen_mean_token_prob`

The official Hugging Face model id for this example is `Qwen/Qwen3-0.6B`.

## Local book parquet + resumable Tesseract OCR

The same script can also read the local book-partition parquet and skip rows that
already exist in the processed markdown output tree. This keeps OCR reruns
resumable instead of reprocessing the same books every time.

Example:

```bash
./scripts/setup_ocr_runtime.sh

./scripts/run_book_ocr.sh \
  --book-parquet-glob "/home/geeyang/workspace/malay-data/final_dataset/books/part-*.parquet" \
  --processed-output-dir ".cache/book_tesseract_processed" \
  --with-tesseract \
  --batch-size 8 \
  --top-k 5 \
  --tesseract-lang eng
```

This local-book mode:

- reads the exported book parquet shards with DuckDB
- dedups duplicate source rows by a stable key before batching
- optionally anti-joins rows already present in legacy processed parquet shards
- skips rows whose processed markdown already exists in `--processed-output-dir`
- prefers rendering full source PDFs page by page for OCR, with `ocr/images/` as fallback
- writes each processed row to markdown immediately for the next run

Notes:

- `--with-tesseract` is currently only supported with `--book-parquet-glob`
- `./scripts/setup_ocr_runtime.sh` installs a user-space `tesseract` runtime under
  `.runtime/ocr` and a dedicated Python env at `.venv-ocr`, including the PDF renderer
  used for full-page OCR
- if your parquet rows store relative markdown paths, `--book-output-root` is used
  to resolve them back to the real `bookscrape/output/...` tree
- source PDFs are resolved from `--book-download-root`, which now defaults to
  `/home/geeyang/workspace/malay-data/bookscrape/data/downloads`
- `--tesseract-input-mode auto` is the default and prefers full PDF pages over sidecar images
- use `--tesseract-max-pages-per-row N` to cap very large PDFs
- use `--rewrite-processed-markdown` when you want to regenerate already-written OCR markdown

## Marker-first local OCR

For a simpler local PDF-to-Markdown path, use Marker instead of the current
`olmocr` Docker flow:

```bash
./scripts/setup_marker_runtime.sh

./scripts/run_marker_ocr.sh \
  --input "/home/geeyang/workspace/malay-data/bookscrape/data/downloads" \
  --output-dir ".cache/marker_output"
```

This Marker path:

- uses an isolated `.venv-marker` runtime
- reads the same source PDF tree under
  `/home/geeyang/workspace/malay-data/bookscrape/data/downloads`
- writes markdown output into a separate tree under `.cache/marker_output`
- forces OCR by default to avoid depending on bad embedded PDF text
- supports single-PDF, directory-batch, and smoke-test runs without any dedup layer

Smoke test:

```bash
./scripts/run_marker_smoke.sh
```

This runs a fixed 5-PDF comparison sample from `scripts/marker_smoke_inputs.txt`
with `--page-range 0-2`, writing logs to `.cache/run_logs/marker_smoke.log`.

### Throughput experiments

For repeatable throughput experiments, use the benchmark harness:

```bash
./scripts/run_marker_benchmark.sh build-manifest
./scripts/run_marker_benchmark.sh run --name baseline-force --ocr-mode force
./scripts/run_marker_benchmark.sh run --name no-force --ocr-mode skip
./scripts/run_marker_benchmark.sh run --name auto-jobs2 --ocr-mode auto --jobs 2
./scripts/run_marker_benchmark.sh run --name full-auto-jobs2 --record-set all --ocr-mode auto --jobs 2
```

The benchmark manifest samples three buckets automatically:

- long born-digital PDFs
- scanned or mixed PDFs
- short article PDFs

Each run writes per-PDF logs, markdown output, and a `summary.json` under
`.cache/marker_benchmark/runs/<experiment-name>/`.

Use `--record-set all` when you want to process every readable PDF discovered by
the manifest instead of just the 12-PDF benchmark sample.

## Predicate pushdown

DuckDB pushes filters down automatically when your `WHERE` clause uses parquet columns or hive partition columns.

Good examples:

```bash
uv run python example/duckdb_r2_query.py \
  --dataset local_ztd \
  --select "source_family, selection_bucket, qwen_token_count, token_count_diff_abs" \
  --where "selection_bucket = 'high_diff' AND qwen_token_count >= 1000" \
  --limit 10

uv run python example/duckdb_r2_query.py \
  --dataset books \
  --select "title, year, content_type" \
  --where "year >= 2020 AND content_type = 'application/pdf'" \
  --limit 10
```

In the first example, `selection_bucket` is read from hive-style partitions and can be pruned before scanning many files.
In both examples, the filtered parquet columns are eligible for pushdown automatically.

## Available datasets

- `books`
- `local_ztd`
- `seapile_v2`
- `subtitle`
- `zhihu_article`
- `zhihu_qa`
