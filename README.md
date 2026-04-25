# training-signal-processing

This repo contains a recipe-driven remote processing workspace with a protected
shared runtime core and separate pipeline families.

The intended user customization surface is small:
- recipes live in `config/`
- new pipeline families live under `src/training_signal_processing/pipelines/`
- pipeline-specific ops live inside that pipeline package, for example
  `src/training_signal_processing/pipelines/ocr/ops.py`

Start here:
- extension guide: [EXTENDING.md](EXTENDING.md)
- architecture guide: [ARCHITECTURE.md](ARCHITECTURE.md)
- sample recipe: [config/remote_ocr.sample.yaml](config/remote_ocr.sample.yaml)

Any new pipeline family should be added under `src/training_signal_processing/pipelines/`
without editing the protected `core/` runtime, submission, storage, or execution
modules.

## Runtime Boundary

The only root-level Python entrypoint is `src/training_signal_processing/main.py`.
Invoke shared commands with `python -m training_signal_processing.main ...`.
The package-level wrapper `python -m training_signal_processing ...` is not supported.

Editable surfaces:
- `config/`: YAML recipes and run-time tuning values
- `src/training_signal_processing/pipelines/<family>/`: pipeline-family schemas,
  ops, exporters, output completion trackers, submission adapters, and remote jobs

Protected infrastructure:
- `src/training_signal_processing/main.py`: CLI entrypoint
- `src/training_signal_processing/core/`: shared executor, submission, storage,
  observability, typed models, and utilities
- `src/training_signal_processing/ops/`: shared op base classes and registry

The `core/` package is intentionally pipeline-generic:
- it must not import `pipelines.ocr` or any other concrete pipeline package
- it owns generic contracts such as submission orchestration, executor flow,
  object storage, exporter interfaces, output completion tracking, and observability
- it should only depend on neutral runtime types like run bindings, artifact layout, and tracking context

Concrete behavior belongs in a pipeline family package:
- row schemas, dataset semantics, exporters, completion trackers, and remote jobs live in `pipelines/<name>/`
- OCR-specific behavior lives in `pipelines/ocr/`
- user-customized OCR transforms belong in `pipelines/ocr/ops.py` or in a new
  pipeline package, depending on ownership

## Boundary Enforcement

The shared layers (`core/`, `ops/`) must never import from `pipelines/`.
That invariant is enforced as a declarative architectural contract via
[import-linter](https://import-linter.readthedocs.io/), declared in `pyproject.toml` under
`[tool.importlinter]` and executed by `test_runtime_modules_do_not_import_pipeline_packages`
in [tests/test_runtime_generic.py](tests/test_runtime_generic.py). The contract detects
both direct and transitive leaks (e.g. `core -> ops -> pipelines.ocr`) and
fails with the full import chain.

Run it directly:

```bash
uv run lint-imports
```

Adding a new shared-layer directory? List it in the contract's `source_modules` so it
is covered.

## Quick Start

```bash
uv sync --group remote_ocr
uv run ruff check src/training_signal_processing
uv run --group remote_ocr python -m training_signal_processing.main validate --config config/remote_ocr.sample.yaml
uv run --group remote_ocr python -m training_signal_processing.main run --config config/remote_ocr.sample.yaml --dry-run
```

## Remote Image

The supported 5090 path today is the Runpod REST launcher in
[infra/start_runpod_5090.sh](infra/start_runpod_5090.sh). It requests one
`NVIDIA GeForce RTX 5090` pod and defaults to this public CUDA 12.8 / PyTorch
2.8 image:

```text
runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
```

The repo's Linux dependency resolution also pins Torch to the official PyTorch
CUDA 12.8 wheel index, so do not switch the pod to a CUDA 13 image unless the
host driver is known to support it.

That launcher writes the active SSH target to `infra/current-machine` after the pod is reachable.
The OCR CLI will use that machine automatically unless you explicitly override `ssh.host`
or `ssh.port` with recipe overrides.

If you need a different public base image for a run, override it at launch time:

```bash
RUNPOD_IMAGE=<your-image> infra/start_runpod_5090.sh
```

Use this launcher when you want the OCR CLI to follow the active 5090 pod
automatically through `infra/current-machine`.

### dstack 4x 4090 On-Demand Resume

Runpod also supports dstack for declarative pod orchestration. Keep dstack
scratch task files out of tracked product code unless they are intentionally
promoted.

Use the checked-in on-demand 4x RTX 4090 resume config when resuming the OCR
run on Runpod:

```bash
/home/geeyang/.dstack-cli-venv/bin/dstack apply \
  -f infra/dstack/config/ocr-resume-4090x4.dstack.yml \
  -y \
  -d
```

The relevant dstack settings are:

```yaml
type: task
name: training-signal-processing-ocr-4090x4

image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
shell: bash

backends: [runpod]
resources:
  gpu:
    name: "RTX4090"
    count: 4
  disk: 200GB..

spot_policy: on-demand
max_price: 16.0
max_duration: 24h

commands:
  - nvidia-smi
  - 'echo "To connect via SSH, use: ssh ${DSTACK_RUN_NAME}"'
  - tail -f /dev/null
```

Operational notes:
- Use the same image as the Runpod REST launcher unless you have verified CUDA,
  Torch, Ray, and Marker together on the target host.
- dstack examples use `image:` for custom containers and `resources.gpu` for GPU
  selection; Runpod's dstack guide uses GPU aliases such as `RTX4090`.
- `spot_policy: on-demand` is required for reserved/on-demand offers; do not use
  `spot` or `auto` for this resume path.
- `max_price: 16.0` is intentionally above the observed 4x RTX 4090 on-demand
  offer price so dstack does not filter viable Runpod offers by budget.
- dstack does not currently update this repo's `infra/current-machine` contract.
  If you launch OCR through dstack, either run the job inside the dstack task or
  manually write the SSH target to `infra/current-machine` before using the core
  OCR CLI from your workstation.

## Main Commands

```bash
uv run --group remote_ocr python -m training_signal_processing.main list-ops
uv run --group remote_ocr python -m training_signal_processing.main test-op --help
uv run --group remote_ocr python -m training_signal_processing.main run --help
uv run --group remote_ocr python -m training_signal_processing.main resume --help
uv run --group remote_ocr python -m training_signal_processing.main ocr-remote-job --help
uv run --group remote_ocr python -m training_signal_processing.pipelines.example_echo.cli validate --help
uv run --group remote_ocr python -m training_signal_processing.pipelines.example_echo.cli run --help
uv run --group remote_ocr python -m training_signal_processing.pipelines.example_echo.cli resume --help
```

## Project Shape

- `config/`: YAML recipes
- `src/training_signal_processing/pipelines/`: pipeline-family packages such as OCR and example_echo
- `src/training_signal_processing/core/`: protected shared runtime, submission,
  storage, observability, dataclasses, and utility helpers
- `src/training_signal_processing/ops/`: shared base classes and registry
- `tests/test_runtime_generic.py`: import-boundary and fake-pipeline verification for the generic core runtime

## Extension Rules

- Add or change OCR processing behavior in `pipelines/ocr/ops.py` plus the OCR recipe.
- Add a brand-new pipeline family in `pipelines/<name>/`.
- Do not edit `core/submission.py` to add a new dataset or pipeline family.
- Do not edit the protected executor loop to add normal OCR transforms.
- Keep pipeline-owned row schemas, exporters, output completion trackers, and
  remote commands inside `pipelines/<name>/`.

The executor loop and shared submission core are intentionally fixed. Extend the
workspace by editing recipes, pipeline-owned ops, or a pipeline family package.

## Progress Checks

For any long-running remote task, check durable R2 outputs before assuming a run
is stuck. The detached launcher returns once the process is started; completed
OCR work is represented by markdown objects under the run output prefix, with
the remote log under `/root/ocr-jobs/<run_id>/job.log`.

MLflow is optional and direct-only. Use it only when `mlflow.enabled=true` and
`mlflow.tracking_uri` is reachable from the process doing the logging. The
framework does not create SSH reverse tunnels.

Always look up the experiment name from the active recipe:
- shared OCR CLI recipes use `mlflow.experiment_name` from the YAML you passed to `python -m training_signal_processing.main ...`
- family-local CLIs such as `python -m training_signal_processing.pipelines.example_echo.cli ...` also use `mlflow.experiment_name` from their YAML

Useful launch examples:

```bash
uv run --group remote_ocr python -m training_signal_processing.main run \
  --config config/remote_ocr.sample.yaml

uv run --group remote_ocr python -m training_signal_processing.pipelines.example_echo.cli run \
  --config src/training_signal_processing/pipelines/example_echo/configs/baseline.yaml
```

To resume an existing OCR sink and skip PDFs whose markdown outputs already
exist in that run prefix, pass the run ID explicitly:

```bash
uv run --group remote_ocr python -m training_signal_processing.main resume \
  --config config/remote_ocr.sample.yaml \
  --run-id 20260423T132754Z
```

Useful MLflow verification example:

```bash
uv run --group remote_ocr python - <<'PY'
from mlflow.tracking import MlflowClient

tracking_uri = "http://127.0.0.1:5000"
experiment_name = "remote-ocr"  # replace with mlflow.experiment_name from your recipe

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

What to look for in MLflow:
- a new run under the recipe's `mlflow.experiment_name`
- `status=running` while the job is active
- `last_execution_event_code` advancing as the executor moves through phases
- metrics such as `execution_event_count` increasing over time

If MLflow is disabled or not updating, inspect R2 output objects and the remote
job log. The runtime no longer writes batch manifests, event objects,
`run_state.json`, or `run.json`.

## Experiment Workflow

Experiments in this repo must stay config-driven.

Allowed experiment surface:
- vary YAML values in `config/`
- use `--set` overrides for one-off runs
- tune existing config-backed controls such as `ray.batch_size`, `ray.target_num_blocks`, `input.max_files`, `mlflow.experiment_name`, and op options already exposed in YAML such as `force_ocr`

Do not change logic just to run an experiment:
- do not edit executor logic, submission logic, pipeline logic, or pipeline op logic for a tuning run
- do not add one-off branches or temporary hard-coded behavior in tracked product code

Put temporary experiment helpers in a local-only directory:
- use `experiments/` at the repo root for scratch scripts, notebooks, and one-off experiment helpers
- `experiments/` is intentionally untracked and should stay local
- do not promote experiment-only logic into the product path unless it is an intentional follow-up change

If an experiment depends on a hard-coded value in logic instead of an exposed config surface, stop the experiment and ask to make that value configurable before running again.

Examples of hard-coded categories worth escalating:
- fixed timeouts
- fixed ports or path assumptions
- fixed device or runtime defaults
- any experiment-critical behavior that can only be changed by editing logic

Concrete repo example:
- OCR timeout currently falls back to `OCR_CONVERSION_TIMEOUT_SEC = 1800` in
  [src/training_signal_processing/pipelines/ocr/ops.py](/home/geeyang/workspace/training-signal-processing/src/training_signal_processing/pipelines/ocr/ops.py)

## Verification

Useful checks for this repo:

```bash
uv run ruff check src tests README.md
uv run python -m compileall src/training_signal_processing
uv run --group remote_ocr pytest -q tests/test_runtime_generic.py
uv run --group remote_ocr pytest -q tests/test_cli_entrypoints.py
uv run --group remote_ocr python -m training_signal_processing.main validate --config config/remote_ocr.sample.yaml
uv run --group remote_ocr python -m training_signal_processing.main run --config config/remote_ocr.sample.yaml --dry-run
```
