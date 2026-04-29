# training-signal-processing

Remote OCR processing for PDF corpora, driven by YAML recipes and executed on
GPU machines through a small protected runtime core.

[Architecture](ARCHITECTURE.md) |
[Extending](EXTENDING.md) |
[Onboarding](ONBOARDING.md) |
[Sample OCR recipe](config/remote_ocr.sample.yaml)

`training-signal-processing` is built for batch OCR runs where local source
PDFs are uploaded to Cloudflare R2, a remote GPU host runs Marker through Ray
Data, and completed markdown outputs are written back to R2. The repo keeps the
operational contract intentionally narrow: runtime behavior belongs in YAML,
pipeline behavior belongs in `pipelines/ocr/`, and shared execution,
submission, storage, and observability stay in `core/`.

## Start Here

- New operator: follow [ONBOARDING.md](ONBOARDING.md) to prepare credentials,
  start infrastructure, and run OCR end to end.
- System modifier: read [ARCHITECTURE.md](ARCHITECTURE.md) before touching
  protected runtime, submission, storage, or execution code.
- Pipeline author: use [EXTENDING.md](EXTENDING.md) for the pipeline contracts,
  file layout, and verification workflow.
- Run OCR now: start with the commands in [Quick Start](#quick-start).

## What This Repo Does

- Uploads configured local PDF inputs to Cloudflare R2 before remote execution.
- Launches a detached OCR job on a reachable SSH GPU host or Runpod machine.
- Executes the OCR recipe with Ray Data and Marker as the OCR engine.
- Writes markdown outputs to R2 under the configured run output prefix.
- Uses direct MLflow logging only when `mlflow.enabled=true` and the tracking
  URI is reachable from the logging process.

## Source Accounting Snapshot

Exact counts were produced with `tiktoken` `o200k_base`; byte counts are UTF-8
text bytes, not compressed parquet object sizes.

| source | token_count | word_count | byte_count | document_count | r2_relative_glob_path | filters | metadata_columns |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Books + OCR | 253,526,396 | 135,195,337 | 1,072,190,417 | 8,914 | `dataset/processed/pdf_ocr/20260423T195035Z/markdown.parquet` | `` | `document_id, source_run_id, source_format, markdown_file_name, markdown_rel_path, markdown_sha256, markdown_char_count, markdown_byte_count` |
| Lowyat | 722,045,872 | 473,281,077 | 2,813,489,956 | 13,126,644 | `dataset/processed/malay/lowyat.parquet` | `` | `source, thread_id, thread_title, thread_url, forum, thread_total_pages, thread_status, page_number, page_offset, post_id, post_floor, author, author_id, posted_at, body_html, quoted_post_id, fetched_at, error_reason` |
| Reddit Bolehland | 7,200,698 | 5,143,027 | 29,967,055 | 215,284 | `dataset/processed/malay/reddit.parquet` | `subreddit=Bolehland` | `post_kind, post_id, submission_id, parent_id, subreddit, author, title, score, num_comments, created_utc, permalink, url, month` |
| Reddit Indonesia | 11,480,646 | 6,616,621 | 43,250,701 | 286,429 | `dataset/processed/malay/reddit.parquet` | `subreddit=indonesia` | `post_kind, post_id, submission_id, parent_id, subreddit, author, title, score, num_comments, created_utc, permalink, url, month` |
| Cari | 134,964,927 | 69,915,523 | 477,486,406 | 1,329,684 | `dataset/processed/malay/cari.parquet` | `` | `source, thread_id, thread_title, thread_url, forum, thread_total_pages, thread_status, page_number, page_offset, post_id, post_floor, author, author_id, posted_at, body_html, quoted_post_id, fetched_at, error_reason` |
| HPLT Malay | 18,639,630,667 | 10,270,837,791 | 71,548,965,682 | 17,365,290 | `dataset/processed/malay/hplt/*_malay.parquet` | `` | `id, url, timestamp, crawl_id, source_shard, language, row_language_code, row_language_prob` |
| HPLT Indonesia | 34,205,177,094 | 19,460,844,974 | 137,277,262,359 | 46,634,888 | `dataset/processed/malay/hplt/*_indon.parquet` | `` | `id, url, timestamp, crawl_id, source_shard, language, row_language_code, row_language_prob` |

## Quick Start

```bash
uv sync --group remote_ocr
uv run ruff check .
uv run lint-imports
uv run --group remote_ocr python -m training_signal_processing.main validate \
  --config config/remote_ocr.sample.yaml
uv run --group remote_ocr python -m training_signal_processing.main run \
  --config config/remote_ocr.sample.yaml --dry-run
```

The only supported Python entrypoint is:

```bash
python -m training_signal_processing.main ...
```

The package wrapper `python -m training_signal_processing ...` is not
supported.

## Runtime Boundary

Editable surfaces:

- `config/`: YAML recipes and runtime tuning values.
- `src/training_signal_processing/pipelines/ocr/`: OCR-owned schemas, ops,
  runtime adapter, submission adapter, and Marker runtime helpers.
- `src/training_signal_processing/pipelines/<family>/`: location for a new
  pipeline family if the project grows beyond OCR.

Protected infrastructure:

- `src/training_signal_processing/main.py`: CLI entrypoint.
- `src/training_signal_processing/core/`: shared executor, submission, storage,
  observability, typed models, and utilities.
- `src/training_signal_processing/ops/`: shared op base classes and registry.

The shared layers (`core/`, `ops/`) must not import from `pipelines/`. That
contract is enforced by import-linter and by
[tests/test_runtime_generic.py](tests/test_runtime_generic.py).

Run the boundary check directly:

```bash
uv run lint-imports
```

## Remote Image

The supported RTX 5090 path is the Runpod REST launcher:

```bash
infra/start_runpod_5090.sh
```

It requests one `NVIDIA GeForce RTX 5090` pod and defaults to this public CUDA
12.8 / PyTorch 2.8 image:

```text
runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
```

The repo's Linux dependency resolution pins Torch to the official PyTorch CUDA
12.8 wheel index. Do not switch the pod to a CUDA 13 image unless the host
driver is known to support it.

The launcher writes the active SSH target to `infra/current-machine` after the
pod is reachable. The OCR CLI uses that machine automatically unless the recipe
overrides `ssh.host` or `ssh.port`.

To use a different public base image for a run:

```bash
RUNPOD_IMAGE=<your-image> infra/start_runpod_5090.sh
```

### dstack 4x 4090 On-Demand Resume

Runpod also supports dstack for declarative pod orchestration. Use the
checked-in on-demand 4x RTX 4090 resume config when resuming an OCR run on
Runpod:

```bash
/home/geeyang/.dstack-cli-venv/bin/dstack apply \
  -f infra/dstack/config/ocr-resume-4090x4.dstack.yml \
  -y \
  -d
```

Operational notes:

- Use the same image as the Runpod REST launcher unless you have verified CUDA,
  Torch, Ray, and Marker together on the target host.
- `spot_policy: on-demand` is required for reserved or on-demand offers.
- `max_price: 16.0` is intentionally above the observed 4x RTX 4090 on-demand
  offer price so dstack does not filter viable Runpod offers by budget.
- dstack does not update `infra/current-machine`; either run OCR inside the
  dstack task or manually write the SSH target before launching from your
  workstation.

## Main Commands

```bash
uv run --group remote_ocr python -m training_signal_processing.main list-ops
uv run --group remote_ocr python -m training_signal_processing.main test-op --help
uv run --group remote_ocr python -m training_signal_processing.main validate \
  --config config/remote_ocr.sample.yaml
uv run --group remote_ocr python -m training_signal_processing.main run \
  --config config/remote_ocr.sample.yaml
uv run --group remote_ocr python -m training_signal_processing.main resume \
  --config config/remote_ocr.sample.yaml --run-id <run_id>
uv run --group remote_ocr python -m training_signal_processing.main ocr-remote-job --help
```

Use `--set key=value` overrides for one-off recipe changes, and prefer adding a
YAML field over hard-coding runtime behavior in logic.

## Project Shape

- `config/`: checked-in sample OCR recipes.
- `src/training_signal_processing/pipelines/ocr/`: OCR-specific config,
  schemas, ops, runtime, submission, and Marker conversion mechanics.
- `src/training_signal_processing/core/`: protected shared runtime,
  submission, storage, observability, dataclasses, and utilities.
- `src/training_signal_processing/ops/`: shared op contracts and registry.
- `tests/`: boundary, config, submission, runtime, launcher, and OCR behavior
  tests.

## Extension Rules

- Add or change OCR processing behavior in `pipelines/ocr/ops.py` and expose
  tunable behavior through the OCR recipe.
- Add a brand-new pipeline family under `pipelines/<name>/` only when the new
  workload has its own schema, exporter, resume semantics, and submission
  adapter.
- Do not edit `core/submission.py` to special-case a pipeline family.
- Do not edit the protected executor loop to add normal OCR transforms.
- Keep pipeline-owned row schemas, exporters, completion trackers, and remote
  commands inside the owning `pipelines/<name>/` package.

The executor loop and shared submission core are intentionally fixed. Extend
the workspace by editing recipes, pipeline-owned ops, or a pipeline package.

## Progress Checks

For long-running remote OCR tasks, check durable R2 outputs before assuming a
run is stuck. The detached launcher returns once the remote process starts;
completed OCR work is represented by markdown objects under the run output
prefix, with the remote log under `/root/ocr-jobs/<run_id>/job.log`.

MLflow is optional and direct-only. Use it only when `mlflow.enabled=true` and
`mlflow.tracking_uri` is reachable from the process doing the logging. The
framework does not create SSH reverse tunnels.

Always look up the experiment name from the active recipe:

- Shared OCR CLI recipes use `mlflow.experiment_name` from the YAML passed to
  `python -m training_signal_processing.main ...`.
- If MLflow is disabled or not updating, inspect R2 output objects and the
  remote job log.

Useful run examples:

```bash
uv run --group remote_ocr python -m training_signal_processing.main run \
  --config config/remote_ocr.sample.yaml

uv run --group remote_ocr python -m training_signal_processing.main resume \
  --config config/remote_ocr.sample.yaml \
  --run-id 20260423T132754Z
```

Useful MLflow verification example:

```bash
uv run --group remote_ocr python - <<'PY'
from mlflow.tracking import MlflowClient

tracking_uri = "http://127.0.0.1:5000"
experiment_name = "remote-ocr"  # replace with mlflow.experiment_name from YAML

client = MlflowClient(tracking_uri=tracking_uri)
experiment = next(
    exp for exp in client.search_experiments() if exp.name == experiment_name
)
runs = client.search_runs(
    [experiment.experiment_id],
    order_by=["attributes.start_time DESC"],
    max_results=5,
)
for run in runs:
    print("run_id:", run.info.run_id)
    print("status:", run.info.status)
    print("tag_status:", run.data.tags.get("status"))
    print("last_execution_event_code:", run.data.tags.get("last_execution_event_code"))
    print("metrics:", run.data.metrics)
    print("---")
PY
```

## Experiment Workflow

Experiments must stay config-driven.

Allowed experiment surface:

- vary YAML values in `config/`
- use `--set` overrides for one-off runs
- tune config-backed controls such as `ray.batch_size`,
  `ray.target_num_blocks`, `input.max_files`, `mlflow.experiment_name`, and
  op options such as `force_ocr`, `timeout_sec`, and
  `source_object_poll_interval_sec`

Do not change logic just to run an experiment:

- do not edit executor, submission, runtime, or op logic for a tuning run
- do not add one-off branches or temporary hard-coded behavior in tracked code
- put scratch scripts, notebooks, and one-off helpers in local-only
  `experiments/`

If an experiment depends on a hard-coded value instead of an exposed config
surface, stop the experiment and make that value configurable before running
again.

## Verification

Useful checks for this repo:

```bash
uv run ruff check .
uv run lint-imports
uv run --group remote_ocr pytest -q tests/test_runtime_generic.py
uv run --group remote_ocr pytest -q tests/test_cli_entrypoints.py
uv run --group remote_ocr python -m training_signal_processing.main validate \
  --config config/remote_ocr.sample.yaml
uv run --group remote_ocr python -m training_signal_processing.main run \
  --config config/remote_ocr.sample.yaml --dry-run
```
