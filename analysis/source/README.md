# Source Distribution Analysis

This directory contains a standalone publication-analysis pipeline for the final
merged cleaned dataset described by `analysis/final_merged_dataset_accounting.md`.

The pipeline reads the accounting markdown, discovers the final R2 parquet parts,
uses the existing LID metadata columns in the unified parquet, performs an exact
full-text word-frequency scan, and writes tables plus PDF/PNG figures.

## Setup

From the repository root:

```bash
.venv/bin/python -m pip install -r analysis/source/requirements.txt
```

or with uv:

```bash
uv pip install -r analysis/source/requirements.txt
```

## Run

Metadata-only validation:

```bash
.venv/bin/python analysis/source/source_distribution_analysis.py \
  --metadata-only \
  --accounting-md analysis/final_merged_dataset_accounting.md \
  --r2-config r2 \
  --output-dir analysis/source/outputs
```

Full analysis and figures:

```bash
.venv/bin/python analysis/source/source_distribution_analysis.py \
  --accounting-md analysis/final_merged_dataset_accounting.md \
  --r2-config r2 \
  --output-dir analysis/source/outputs \
  --figure-formats pdf,png
```

The full run scans all `cleaned_text` values to compute exact word counts, so it
is expected to be substantially slower than metadata-only validation.

## Self-Test

The script can generate and analyze a tiny local parquet fixture:

```bash
.venv/bin/python analysis/source/source_distribution_analysis.py \
  --self-test \
  --output-dir analysis/source/self_test_outputs
```

## Outputs

The output directory contains:

- `tables/*.csv`: source macro stats, LID distributions, top words, samples, and accounting checks.
- `figures/*.pdf` and `figures/*.png`: publication figures.
- `figure_manifest.json`: every expected figure artifact and verification status.
- `run_metadata.json`: input paths, schema probe, and run parameters.
- `report.md`: concise publication-facing summary and caveats.

The word-cloud and filtered top-word figures use `stopwords.txt`. Raw exact word
counts are still emitted before stopword filtering.
