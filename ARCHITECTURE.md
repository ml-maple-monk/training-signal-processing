# Architecture

`training_signal_processing` is a small framework for running **batch
GPU pipelines on remote machines**, plus concrete pipeline packages on
top of it: **OCR** (Marker on PDFs), **source accounting** (dataset
inventory/token metadata), **LID metadata** (language-ID sidecars for
processed parquet rows), and **example_echo** (a minimal reference
pipeline). The shared layer is intentionally pipeline-agnostic;
new pipelines plug in through small template bases and registered ops.
This document captures the static structure (layers, contracts, frozen
invariants) and the dynamic flow (submission → remote execution →
exporter/progress loop), then walks one OCR run end-to-end as the worked
example. Infrastructure gets a short closing section.

If you only want the OCR walkthrough, jump to [§12](#12-ocr-pipeline-walkthrough-end-to-end).
If you're about to modify a file, [§3](#3-frozen-invariants) tells you
which files are protected.

---

## Table of contents

- [Architecture](#architecture)
  - [Table of contents](#table-of-contents)
  - [1. TL;DR](#1-tldr)
  - [2. Layered architecture and the import contract](#2-layered-architecture-and-the-import-contract)
  - [3. Frozen invariants](#3-frozen-invariants)
  - [4. Static contracts: the ABCs and the concretes](#4-static-contracts-the-abcs-and-the-concretes)
    - [4.1 ABCs that pipelines implement](#41-abcs-that-pipelines-implement)
    - [4.2 Concretes pipelines compose](#42-concretes-pipelines-compose)
  - [5. Configuration plane](#5-configuration-plane)
  - [6. Object storage and artifacts](#6-object-storage-and-artifacts)
    - [6.1 Driver-side output writes](#61-driver-side-output-writes)
  - [7. Submission and transport](#7-submission-and-transport)
    - [7.1 Detached lifecycle](#71-detached-lifecycle)
    - [7.2 Observability without tunnels](#72-observability-without-tunnels)
  - [8. Remote execution flow](#8-remote-execution-flow)
  - [9. Ops: registration and resolution](#9-ops-registration-and-resolution)
  - [10. Observability](#10-observability)
  - [11. Resume semantics](#11-resume-semantics)
  - [12. OCR pipeline walkthrough end-to-end](#12-ocr-pipeline-walkthrough-end-to-end)
  - [13. Other pipelines](#13-other-pipelines)
  - [14. Infrastructure](#14-infrastructure)
  - [15. Glossary](#15-glossary)

---

## 1. TL;DR

Three layers, small pipeline packages, one runtime. The middle layer (`ops/`)
defines the `Op` ABC and registry. Concrete ops live inside the
pipeline package that owns them (`pipelines/<name>/ops.py`) and
self-register when that package is imported. The foundation layer
(`core/`) contains the executor, submission coordinator, object store,
observability, event sinks, and outputs-only completion tracking. The top layer
(`pipelines/`) wires per-pipeline `RecipeConfig`, runtime adapter, and
submission adapter. A run is two halves: a local **submission** that
syncs YAML-declared paths, uploads inputs to R2, and SSHes a `python -m
training_signal_processing.main ocr-remote-job …` invocation onto a
remote GPU box; and a **remote execution** that rebuilds the config
from R2, runs ops via Ray Data `map_batches`, and writes outputs +
in-memory progress per batch.

---

## 2. Layered architecture and the import contract

```
+--------------------------------------------------------------------+
|  TOP — pipeline-specific                                           |
|                                                                    |
|  pipelines/ocr | source_accounting | lid_metadata                   |
|    models.py   ops.py   runtime.py   submission.py                  |
+----------------------------|---------------------------------------+
                             |  pipelines compose ops + adapters
                             v  (concrete Op subclasses + Recipe)
+--------------------------------------------------------------------+
|  MIDDLE — Op contract                                              |
|                                                                    |
|  ops/base.py     Op (+ MapperOp / FilterOp / PipelineOp)           |
|  ops/builtin.py  RowWiseMapperOp, SkipExistingFilter, ...          |
|  ops/registry.py OpRegistry / RegisteredOpRegistry / Resolved...   |
|  ops/testing.py  RayOpTestHarness                                  |
+----------------------------|---------------------------------------+
                             |  registry.resolve_pipeline + Ops
                             v
+--------------------------------------------------------------------+
|  FOUNDATION — pipeline-agnostic                                    |
|                                                                    |
|  core/                                                             |
|    models.py       config_loading.py     utils.py                  |
|    storage.py      submission.py         remote.py                 |
|    execution.py    dataset.py            observability.py          |
|                                                                    |
+--------------------------------------------------------------------+
                                   ^
                                   | enforced by [tool.importlinter]:
                                   | "shared layers must not import
                                   |  from pipelines"
```

The runtime flow is intentionally small enough to draw:

```
YAML recipe
  -> core/models.py dataclasses
  -> SubmissionCoordinator.submit()  -> SSH/R2/rclone
  -> ObjectStoreRemoteJob.run()
  -> StreamingRayExecutor.run()
  -> observability helpers
  -> ops/base.py + concrete pipeline ops
  -> R2 outputs
```

The contract at `pyproject.toml [[tool.importlinter.contracts]]`
declares `core` and `ops` as `source_modules` that are forbidden from
importing `training_signal_processing.pipelines`.
Recent commits `a5c7c77` and `ec67025` retired the last leaks; tests
in `tests/test_runtime_generic.py` and `tests/test_cli_entrypoints.py`
are pipeline-agnostic by construction.

The contract is checked by `import-linter` (declared in the `dev`
dependency group). Run it with:

```
uv run lint-imports
```

---

## 3. Frozen invariants

A subset of files in this repo carry an explicit warning header:

> `# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE
> WITHOUT EXPLICIT USER APPROVAL.`

These are the contract-bearing files. **Project convention** (not
quoted from the warning, just the working rule): pure additions to
these files are usually fine — adding a new method, a new dataclass,
a new helper. Changes to existing public signatures, removals, or
behavioral changes need explicit approval because pipelines and tests
are coupled to the exact shape.

The authoritative list is whatever
`rg -l 'WARNING TO OTHER AGENTS' src/` returns. As of this writing:

| File | Why it's frozen |
|------|-----------------|
| `core/submission.py` | Submission/transport ABCs + `SubmissionCoordinator` glue |
| `core/execution.py` | `Executor`, `PipelineRuntimeAdapter`, `OutputCompletionTracker`, `Exporter`, `StreamingRayExecutor` |
| `core/dataset.py` | `DatasetBuilder` ABC + Ray adapters |
| `core/observability.py` | `ExecutionLogger` + `ProgressTracker` ABCs + MLflow concrete |
| `core/remote.py` | `RemoteJob` + `build_remote_job_cli` factory + guard CLI |
| `ops/base.py` | `Op` ABC + auto-registration mechanism |
| `ops/builtin.py` | Stage templates pipelines subclass |
| `ops/registry.py` | `OpRegistry` + import-sweep bootstrap |
| `ops/testing.py` | Op-level test harness |
| `ops/__init__.py` | Marker only |
| `core/models.py` | All shared dataclasses (`R2Config`, `RayConfig`, `RunState`, …) |
| `core/utils.py` | Cross-cutting helpers (`join_s3_key`, `utc_isoformat`, …) |
| `core/storage.py` | `ObjectStore` ABC + `R2ObjectStore` |
| `pipelines/<name>/ops.py` | Pipeline-owned concrete ops |
| `main.py` | OCR CLI entrypoint |

The import-linter contract above is the second invariant: shared
layers stay pipeline-agnostic.

---

## 4. Static contracts: the ABCs and the concretes

### 4.1 ABCs that pipelines implement

| ABC | File:line | Methods (abstract) |
|-----|-----------|--------------------|
| `SubmissionAdapter` | [core/submission.py:478](src/training_signal_processing/core/submission.py#L478) | `prepare_new_run`, `prepare_resume_run` |
| `PipelineRuntimeAdapter` | [core/execution.py](src/training_signal_processing/core/execution.py) | run bindings, execution/tracking config, op config, artifact layout, input loading, runtime context, op registry, exporter, completion tracker |
| `OutputCompletionTracker` | [core/execution.py](src/training_signal_processing/core/execution.py) | `completed_source_keys`, `source_key_for_input`, `output_key_for_input`, `output_listing_prefix` |
| `Exporter` / `RayExporter` | [core/execution.py](src/training_signal_processing/core/execution.py) | `export_batch`, optional `finalize_run` |
| `OpRegistry` | [ops/registry.py:50](src/training_signal_processing/ops/registry.py#L50) (template `resolve_pipeline`) | `resolve_pipeline(configs) -> ResolvedOpPipeline` |
| `Op` (+ `MapperOp` / `FilterOp` / `PipelineOp`) | [ops/base.py:17](src/training_signal_processing/ops/base.py#L17), [:82](src/training_signal_processing/ops/base.py#L82), [:90](src/training_signal_processing/ops/base.py#L90), [:101](src/training_signal_processing/ops/base.py#L101) | `process_batch(Batch) -> Batch` (and one of: `op_name`, `op_stage`) |

`PipelineRuntimeAdapter` has default-method extension hooks rather than making
every pipeline reimplement framework plumbing: `build_dataset_builder` defaults
to `ConfiguredRayDatasetBuilder`, `resolve_completed_source_keys` delegates to
the completion tracker, and `resolve_transform_resources` defaults to "no
override". OCR overrides the last to assign GPU resources to `marker_ocr`.

### 4.2 Concretes pipelines compose

These are provided by `core`; pipelines consume them as-is.

| Class | File:line | Role |
|-------|-----------|------|
| `R2ArtifactStore` | [core/submission.py:316](src/training_signal_processing/core/submission.py#L316) | `ArtifactStore` over R2/S3; `build_remote_env` at [:354](src/training_signal_processing/core/submission.py#L354) emits R2 + AWS-compat + MLflow S3 endpoint vars |
| `SshRemoteTransport` | [core/submission.py:386](src/training_signal_processing/core/submission.py#L386) | `RemoteTransport` over rsync + ssh; `launch_detached` spawns the remote job under `setsid` and records its pgid |
| `SubmissionCoordinator` | [core/submission.py:492](src/training_signal_processing/core/submission.py#L492) | Orchestrates `prepare → sync → bootstrap → (optional local upload wait) → launch_detached → return LaunchHandle` |
| `StreamingRayExecutor` | [core/execution.py:114](src/training_signal_processing/core/execution.py#L114) | The `Executor` for batch GPU pipelines; per-batch loop |
| `RegisteredOpRegistry` | [ops/registry.py:63](src/training_signal_processing/ops/registry.py#L63) | The concrete `OpRegistry`; triggers the import-sweep on module load |
| `RayDatasetBuilder` / `ConfiguredRayDatasetBuilder` | [core/dataset.py:57](src/training_signal_processing/core/dataset.py#L57), [:132](src/training_signal_processing/core/dataset.py#L132) | `DatasetBuilder` over Ray Data; `Configured` re-partitions to `target_num_blocks` |
| `MlflowExecutionLogger` / `MlflowProgressTracker` | [core/observability.py:32](src/training_signal_processing/core/observability.py#L32), [:364](src/training_signal_processing/core/observability.py#L364) | MLflow-backed `ExecutionLogger` and `ProgressTracker`; `StructuredExecutionLogger` and `NullProgressReporter` are the no-MLflow fallbacks |
| `RemoteJob` / `ObjectStoreRemoteJob` / `build_remote_job_cli` | [core/remote.py](src/training_signal_processing/core/remote.py) | The remote-side CLI factory; non-OCR pipelines expose it as `cli remote-job` |

The shared dataclasses live in [core/models.py](src/training_signal_processing/core/models.py):
`R2Config`, `RayConfig`, `RayTransformResources`, `RemoteRuntimeConfig`,
`SshConfig`, `MlflowConfig`, `OpConfig`, `RuntimeRunBindings`,
`RuntimeTrackingContext`, `RunArtifactLayout`, `BatchProgress`,
`ExportBatchResult`, `RunState`, `ExecutionLogEvent`,
`ExecutorRunSummary`, `OpRuntimeContext`.

---

## 5. Configuration plane

A run is described by a YAML recipe + zero or more YAML overlays + zero
or more dotted-path `--set` overrides. The pipeline-agnostic plumbing
lives in [core/config_loading.py](src/training_signal_processing/core/config_loading.py)
(`load_recipe_mapping` and `deep_merge_mapping`). Each pipeline's
`config.py` thin-wraps it through `AbstractRecipeConfigLoader`. The
base loader owns required-section checks, overlays, dotted overrides,
stale-key rejection, and op-config construction; the pipeline supplies
only typed recipe construction.

```
$ uv run python -m training_signal_processing.main run \
    --config src/training_signal_processing/pipelines/ocr/configs/baseline.yaml \
    --config src/training_signal_processing/pipelines/ocr/configs/experiment.example.yaml \
    --set ray.batch_size=8 \
    --set input.max_files=100
```

Resolution order:
1. Read base YAML (first `--config`).
2. For each subsequent `--config`, deep-merge mapping by mapping;
   scalars/lists/`ops` are replaced wholesale by the overlay.
3. Apply `--set key.path=value` overrides last.
4. **OCR only:** if `infra/current-machine` exists and
   neither `ssh.host` nor `ssh.port` was overridden, parse the file
   for an `ssh -p <port> user@host` command and inject those into
   `recipe.ssh`. (`apply_current_machine_target` at
   [core/config_loading.py:97](src/training_signal_processing/core/config_loading.py#L97).)
5. Expand `~` paths.
6. Hand off to the per-pipeline `build_recipe_config`.

**Guardrail for experiment overlays.** Files in
`pipelines/ocr/configs/` must NOT commit `ssh.host` or `ssh.port`.
The gitignored `infra/current-machine` carries each operator's box;
hardcoding an SSH target poisons the repo for collaborators. This
is documented in `pipelines/ocr/configs/baseline.yaml`.

Operational launch/upload knobs are explicit YAML too. `remote` owns
`sync_paths`, the detached-job root, and pgid wait loop settings; `input` owns OCR upload
parallelism, and the `marker_ocr` op options own OCR conversion timeout and
source-object polling interval. These values should not be reintroduced as
logic-level defaults.

---

## 6. Object storage and artifacts

`ObjectStore` ([core/storage.py:19](src/training_signal_processing/core/storage.py#L19))
is an ABC: `exists`, `list_keys`, `read_bytes`, `write_bytes`,
`upload_file`, `make_url`, `build_pyarrow_filesystem`, plus convenience
wrappers `read_json` / `write_json` / `read_jsonl` / `write_jsonl`.
The only concrete is `R2ObjectStore` ([:80](src/training_signal_processing/core/storage.py#L80)),
which uses `boto3` against a Cloudflare R2 endpoint and accepts
credentials via either an env file (`from_config_file`) or process
environment (`from_environment`).

Runtime ops do not construct stores through `OpRuntimeContext`. The context
only carries the current store slot and typed run metadata; `core.storage`
owns `resolve_runtime_object_store(...)`, which reuses an attached store or
builds `R2ObjectStore` from remote env/config when Ray has deserialized the
context with unpicklable services cleared.

`ArtifactStore` ([core/submission.py:284](src/training_signal_processing/core/submission.py#L284))
is the runtime-side façade — same contract plus `upload_file` and
`build_remote_env`. The concrete `R2ArtifactStore` wraps an
`R2ObjectStore`; `build_remote_env` ([:354](src/training_signal_processing/core/submission.py#L354))
returns the env-var bundle the remote process needs:
`R2_BUCKET`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_REGION`,
`R2_ENDPOINT_URL`, plus AWS-compatible aliases
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`)
and `MLFLOW_S3_ENDPOINT_URL` so MLflow's artifact store can write to
the same bucket.

Per-run keys live under `r2.output_prefix/<run_id>/`:

```
control/input_manifest.jsonl   # input rows for executor
control/recipe.json            # exact resolved recipe used
<pipeline outputs>             # markdown, source metadata, LID parquet shards, ...
```

`RunArtifactLayout` (`core/models.py`) carries the two roots
(`source_root_key`, `output_root_key`) the executor passes into the
runtime context.

### 6.1 Driver-side output writes

`RayExporter._put_bytes` ([core/execution.py](src/training_signal_processing/core/execution.py))
is deliberately synchronous: it calls `object_store.write_bytes(...)`
and returns only after the R2 `put_object` completes. `StreamingRayExecutor`
therefore advances in one obvious order for driver-side exporters:
`export_batch` writes outputs, `validate_export_result` checks the
reported batch shape, and only then does the executor update in-memory
batch progress.

This is the durability boundary for OCR markdown. If a markdown write
fails, the exception leaves `export_batch`, the executor marks the run
failed, and the missing output is not considered complete on restart.
Resume lists expected output objects and skips only source keys whose
outputs already exist. There is no `ray.async_upload` YAML surface for
remote output writes.

Local OCR input upload is separate: `OcrSubmissionAdapter` still builds
a local `rclone` command, and `SubmissionCoordinator` waits for it before
launching the remote job so the pod never races missing source PDFs.

---

## 7. Submission and transport

```
LOCAL                                                    REMOTE
+----------+    +------------------+    +-----------+    +-----------+
| user CLI |--> | Submission       |--> | Submission|--> | rsync -az |
| --config |    | Adapter          |    | Coord-    |    | sync_paths|
+----------+    | .prepare_new_run |    | inator    |    +-----------+
                |   discover       |    | .submit() |    | bootstrap |
                |   manifest+yaml  |    +-----+-----+    | uv sync   |
                |   PreparedRun    |          |          +-----------+
                +------------------+          |          | exec      |
                          |                   |          | uv run    |
                          v                   v          | python -m |
                +------------------+   +-----------+     | main      |
                | R2ArtifactStore  |   | rclone    |     | ocr-      |
                | write_jsonl      |   | parallel  |     | remote-job|
                | write_json       |   | upload    |     | --r-id... |
                +------------------+   | (background)    +-----+-----+
                                       +-----------+           |
                          detached ssh launch                  v
                          progress is written to R2            [Diagram 3]
                          optional MLflow uses direct URI
```

The flow on the local side is split between three interfaces:

- `SubmissionAdapter` (per-pipeline) returns a `PreparedRun` whose
  fields tell the coordinator what to do: `sync_paths` (rsync targets),
  `bootstrap` (shell to run after sync), `invocation` (the remote
  command + env), `artifacts` (input/output refs),
  and optionally `async_upload` (a local rclone command for source
  inputs, always awaited before remote execution starts).
- `RemoteTransport` (`SshRemoteTransport`) implements `sync`,
  `bootstrap`, `execute`, and `launch_detached` over SSH + rsync.
- `SubmissionCoordinator.submit` ([core/submission.py:492](src/training_signal_processing/core/submission.py#L492))
  orchestrates them: prepare → sync code → bootstrap → start local
  source upload (if any) → wait for upload → launch_detached → return
  LaunchHandle. The remote writes pipeline outputs to R2.

### 7.1 Detached lifecycle

`submit()` returns immediately once `setsid` writes `job.pgid` on the
pod — it is **fire-and-forget**. The remote OCR job runs in its own
process group under PID 1 (init), so SSH disconnects cannot kill it.
Remote stdout/stderr live at `/root/ocr-jobs/<run_id>/job.log`; the
local CLI prints only the `LaunchHandle` JSON and exits. **Exit code 0
means "launched successfully", not "run complete".** Completion must be
inferred by listing output objects in R2 or by tailing the remote log.

### 7.2 Observability without tunnels

Remote completion is durable in R2 output objects. Runtime progress is
in memory and optionally mirrored to MLflow. When `mlflow.enabled=true`, the YAML must
provide a `mlflow.tracking_uri` reachable from the process doing the
logging; the framework does not rewrite the URI or open SSH tunnels.

OCR-side helpers worth pointing at:

- `OcrSubmissionAdapter.build_async_upload_spec`
  ([pipelines/ocr/submission.py:263](src/training_signal_processing/pipelines/ocr/submission.py#L263))
  builds the local source-PDF rclone argv (lines 263-322). The R2-flavored env for
  rclone comes from `build_rclone_env` ([:330](src/training_signal_processing/pipelines/ocr/submission.py#L330)).
- `build_invocation_spec` ([pipelines/ocr/submission.py:156](src/training_signal_processing/pipelines/ocr/submission.py#L156))
  composes the remote `uv run python -m main ocr-remote-job …`
  command and passes R2 credentials as environment variables.

---

## 8. Remote execution flow

```
remote shell:
$ uv run python -m training_signal_processing.main \
      ocr-remote-job --run-id ... --config-object-key ... \
                     --input-manifest-key ... --uploaded-items N
        |
        v
build_remote_job_cli closure  (core/remote.py)
        |
        v
ObjectStoreRemoteJob.run()  (core/remote.py)
        |
        v
StreamingRayExecutor.run()  (core/execution.py:114)
        |
        +--- pipeline.get_run_bindings / get_execution_config /
        |    get_tracking_context / get_artifact_layout / get_op_configs
        |
        +--- _validate_contract
        |    raises PipelineContractError on misuse
        |
        +--- _build_execution_logger                 (observability.py)
        +--- MlflowProgressTracker                   (observability.py)
        +--- pipeline.load_input_rows()              # reads input_manifest.jsonl
        |
        +--- pipeline.build_completion_tracker()
        |     completed_source_keys(input_rows, artifact_layout, allow_overwrite)
        |
        +--- pipeline.build_runtime_context(logger=, completed_source_keys=)
        +--- pipeline.build_op_registry(runtime_context=)
        +--- registry.resolve_pipeline(op_configs)   (registry.py:50)
        |     -> ResolvedOpPipeline(prepare, [transforms], export)
        |
        +--- pipeline.build_dataset_builder()
        |     default: ConfiguredRayDatasetBuilder
        |
        +--- _apply_dataset_transform                # lazy Ray transforms
        |     dataset.map_batches(prepare_op, ...)
        |     for op in transform_ops:
        |         resources = pipeline.resolve_transform_resources(op, execution)
        |         dataset.map_batches(op, batch_size=..., concurrency=...,
        |                                  num_gpus=R.num_gpus,
        |                                  num_cpus=R.num_cpus)
        |     (LAZY — no actual execution yet)
        |
        +--- try:                                            # lifecycle wrap
        |    for batch in dataset_builder.iter_batches(dataset, batch_size):
        |       # FIRST iter materializes the lazy plan
        |       exporter.export_batch(batch_id, rows)         # writes outputs
        |       _build_batch_progress(rows)                  # in-memory counters
        |       progress_tracker.log_batch_progress(...)      # MLflow metrics
        |       _transition_run_phase(..., "first_batch_materialized")
        |
        |    mark_run_finished(run_state)
        |    exporter.finalize_run(run_state)
        |    return ExecutorRunSummary.to_dict()              # local CLI has already exited
        |                                                     # with a LaunchHandle (see §7.1)
        |
        +--- except Exception:
        |      mark_run_failed(run_state, ...)                 # in-memory status=failed
```

Three properties matter:

1. **Lazy → eager boundary.** `apply_pipeline_transforms` only chains
   `dataset.map_batches` calls; nothing actually runs until
   `iter_batches` pulls a batch. This is what gives Ray Data its
   pipelining behavior.
2. **Per-op resource override.** Default `resolve_transform_resources`
   ([core/execution.py:101](src/training_signal_processing/core/execution.py#L101))
   returns an empty `RayTransformResources`. OCR overrides at
   [pipelines/ocr/runtime.py:105](src/training_signal_processing/pipelines/ocr/runtime.py#L105)
   to give `marker_ocr` the configured `num_gpus` / `num_cpus` from
   `config.ray.marker_ocr_resources`.
3. **Phase telemetry.** `transition_run_phase`
   ([core/execution.py:313](src/training_signal_processing/core/execution.py#L313))
   is called at every checkpoint (`manifest_loaded`, `dataset_build_*`,
   `iter_batches_start`, `first_batch_materialized`, …) — it persists
   to `RunState.current_phase` and emits a structured event so an
   external watcher can tell where in the run you are.

---

## 9. Ops: registration and resolution

Ops are not registered explicitly — they self-register on import.
Each pipeline package imports its own `ops.py` from `__init__.py`, so
its concrete ops are registered before CLI validation or remote
execution resolves the recipe.

```
import time
=========

pipelines/ocr/__init__.py
   |
   v
   from . import ops
                |
                v
   importing pipelines/ocr/ops.py loads, e.g.:
       class PreparePdfDocumentOp(SourcePreparationOp):
           op_name = "prepare_pdf_document"
                |
                v
   ops/base.py:22  Op.__init_subclass__(cls)
       REGISTERED_OP_TYPES["prepare_pdf_document"] = cls

run time
========

   recipe.yaml:
     ops:
       - {name: prepare_pdf_document, type: mapper}
       - {name: skip_existing,        type: filter}
       - {name: marker_ocr,           type: mapper, force_ocr: true}
                |
                v
   OpRegistry.resolve_pipeline(configs)   (registry.py:50)
       cls = REGISTERED_OP_TYPES[name]
       op  = cls(**config.options)        # e.g. force_ocr=True
       op.bind_runtime(runtime_context)
       group by op.op_stage:
         prepare    -> exactly one
         transform  -> >= one
         export     -> zero or one
       -> ResolvedOpPipeline(prepare, [transforms...], export?)
```

**Adding a new transform op for OCR:**

1. Define the class in `pipelines/ocr/ops.py`:
   ```
   class MyTransformOp(RowWiseMapperOp):
       op_name = "my_transform"
       op_stage = "transform"
       def process_row(self, row):
           return {**row, "my_field": ...}
   ```
2. Reference it from the recipe `ops:` list in `baseline.yaml`:
   ```
   ops:
     - {name: prepare_pdf_document, type: mapper}
     - {name: skip_existing,        type: filter}
     - {name: my_transform,         type: mapper}
     - {name: marker_ocr,           type: mapper, force_ocr: true}
   ```

The class is registered when `training_signal_processing.pipelines.ocr`
is imported. Op-level testing uses
`build_default_ray_op_test_harness()` in `ops/testing.py` to exercise
a single op against a `LocalRayDatasetBuilder` without spinning up the
full executor.

OCR's concrete ops live in
[pipelines/ocr/ops.py](src/training_signal_processing/pipelines/ocr/ops.py):
`PreparePdfDocumentOp`, `SkipExistingDocumentsOp`, `MarkerOcrDocumentOp`,
and the markdown key helpers used by prepare/runtime resume logic. Marker
subprocess execution, source-object polling, temporary-file staging, and
OCR-specific event logging live in
[pipelines/ocr/marker_runtime.py](src/training_signal_processing/pipelines/ocr/marker_runtime.py)
so the op file stays focused on domain pipeline steps.

---

## 10. Observability

Two parallel hierarchies: an **event stream** (`ExecutionLogger`,
`StructuredTracer`, `StructuredMonitor`) and a **state stream**
(`ProgressTracker` + `ProgressReporter`).

- **Events.** `ExecutionLogEvent` ([core/models.py](src/training_signal_processing/core/models.py))
  is the unit. `MlflowExecutionLogger` ([observability.py:32](src/training_signal_processing/core/observability.py#L32))
  buffers events until the MLflow run id is known, then flushes via
  `attach_run_id`; per-event side effects update an
  `execution_event_count` metric and re-stamp `last_execution_event_*`
  tags ([:64](src/training_signal_processing/core/observability.py#L64)).
  This tag re-stamp is the run's **heartbeat** — an external watcher
  can look at the tag to confirm the run is alive without polling
  metrics.
- **Categories of events.** Four groups: `executor.*` (manifest load,
  resume load, phase transitions, dataset transforms, op-complete);
  `tracer.op.before` / `tracer.op.after` (per-op enter/exit);
  `monitor.run.start` / `finish` / `fail` (RunState snapshots);
  ad-hoc events from concrete ops via
  `op.log_runtime_event(level, code, message, **details)`.
- **Metrics.** `MlflowProgressTracker.log_batch_progress`
  ([:418](src/training_signal_processing/core/observability.py#L418))
  writes `success_count`, `failed_count`, `skipped_count`,
  `pending_items`, `batch_duration_sec` keyed by
  `step=run_state.last_committed_batch`. `log_run_started` logs run
  params (`pipeline_run_id`, `total_items`, `batch_size`,
  `concurrency`, plus `tracking.extra_params` from the adapter).
- **Buffered-event flush.** Events emitted before MLflow exists are
  buffered in `pending_events`; the flush happens automatically when
  `MlflowProgressTracker.__init__` calls
  `logger.attach_run_id(self.mlflow_run_id)`.

Human-facing progress goes through `TqdmProgressReporter` (stderr
with bracketed phase markers) when `total_items > 0`, otherwise
`NullProgressReporter`.

---

## 11. Resume semantics

A run can be resumed by SHA-equivalent recipe + the prior `run_id`.
The resume side of `SubmissionAdapter.prepare_resume_run`
([core/submission.py:482](src/training_signal_processing/core/submission.py#L482))
loads the prior `recipe.json` and `input_manifest.jsonl` from R2 and
reuses them. On the remote side, `StreamingRayExecutor.run` calls
`pipeline.build_completion_tracker().completed_source_keys(...)` and
threads the resulting set into `OpRuntimeContext.completed_source_keys`,
which skip ops consult to drop already-processed rows.

There is no durable checkpoint ledger. Each pipeline maps a source row
to its expected output object; if that object exists under the run output
prefix, the source key is complete. If the executor dies before an output
write succeeds, that object is absent and the source is retried on the next
run. `allow_overwrite=True` makes the completed set empty so every source is
processed again.

The LID metadata pipeline also writes progress-only row-group checkpoint
sidecars under `checkpoints/<source>/<file>.rgNNNNN.json`. Those objects report
rows/tokens completed and current tokens/sec during long row groups, but they
do not mark completion. The authoritative resume unit remains the final parquet
shard under `shards/<source>/<file>.rgNNNNN.parquet`; a checkpoint-only row
group reruns from the beginning on resume.

---

## 12. OCR pipeline walkthrough end-to-end

This is the worked example — one full OCR run from `python -m main run`
to the final markdown writes. Step numbers below match the diagram.

```
[1] user@local $ python -m training_signal_processing.main run \
                     --config src/training_signal_processing/pipelines/ocr/configs/baseline.yaml
                                                                |
                                                                v
[2] main.py:run_command  ->  load_recipe_config   (overlays + --set
                              + current-machine SSH resolution)
                                                                |
                                                                v
[3] OcrSubmissionAdapter.prepare_new_run         (submission.py:39)
        - glob PDFs under input.local_pdf_root
        - sort by PDF file size             (in build_pdf_tasks)
        - write input_manifest.jsonl + recipe.json to R2
        - build LocalAsyncUploadSpec (rclone)(submission.py:263)
                                                                |
                                                                v
[4] SubmissionCoordinator.submit                 (submission.py:492)
                ┌───────────┬──────────┬──────────┬────────────────┐
                │           │          │          │                │
                v           v          v          v                v
          rsync -az    bootstrap  rclone      launch_detached   return
          sync_paths   uv sync    parallel    (setsid sh -c …  LaunchHandle
                                  upload of   echo $$>pgid;   (local CLI
                                  PDFs        exec launch.sh   exits here;
                                  (awaited    & — survives    remote keeps
                                   before     SSH disconnect)  running under
                                   launch)                    setsid)
                                                                |
~~~~~~~~~~~~~~~~~~~~~~~~~~~ network boundary ~~~~~~~~~~~~~~~~~~~|~~~~~~~~~~
                                                                v
[5] (REMOTE) build_remote_job_cli closure        (core/remote.py)
        - bootstrap_store.read_json(config_object_key)  -> recipe dict
        - build generic ObjectStorePipelineRuntimeAdapter(config, bindings, object_store)
        - ObjectStoreRemoteJob.run()
                                                                |
                                                                v
[6] StreamingRayExecutor.run()                   (executor.py:114)
        per-batch loop iterates ops:
          prepare_pdf_document  ->  enrich row with markdown_r2_key
          skip_existing         ->  drop rows already in completed_source_keys
          marker_ocr            ->  spawned subprocess + GPU OCR
                                    (conversion mechanics in marker_runtime.py,
                                     resources from marker_ocr_resources)
                                                                |
                                                                v
[7] OcrMarkdownExporter.export_batch            (runtime.py)
        - for row in rows:
            if row.status != "success": continue
            self._put_bytes(markdown_r2_key, markdown_text.encode())
            cleanup_staged_pdf(row.staged_pdf_path)
        # _put_bytes is synchronous: the markdown object is durable
        # before in-memory progress records the row below.
                                                                |
                                                                v
[8] StreamingRayExecutor._build_batch_progress
        - count success / failed / skipped rows
        - update in-memory RunState
        - optionally log MLflow progress
                                                                |
                              ... loop until input exhausted ...
                                                                v
[9] mark_run_finished -> finalize_run
        status is one of: success | partial | failed
        (local CLI already exited with a LaunchHandle at step [4];
         operators list output objects or tail /root/ocr-jobs/<run_id>/job.log
         to see completion — see ONBOARDING §3 *Running-job runbook*)
```

**Worth knowing.** Steps 5-8 are framework-generic; any pipeline uses
the same loop with different ops and exporter logic. Steps 1-4 and 7-8
are OCR-specific (PDF discovery, file-size sort, markdown writes,
marker spawn isolation). The main OCR-only
runtime override is `resolve_transform_resources` at
[pipelines/ocr/runtime.py:105](src/training_signal_processing/pipelines/ocr/runtime.py#L105)
which gives `marker_ocr` a GPU.

OCR's pipeline-local files:

- [pipelines/ocr/submission.py](src/training_signal_processing/pipelines/ocr/submission.py)
  (359 lines): the `SubmissionAdapter`. The bulk is the rclone block
  ([:263-353](src/training_signal_processing/pipelines/ocr/submission.py#L263)).
- [pipelines/ocr/runtime.py](src/training_signal_processing/pipelines/ocr/runtime.py)
  (includes the markdown `Exporter`, `OutputCompletionTracker`, and
  generic runtime adapter factory).
- [pipelines/ocr/marker_runtime.py](src/training_signal_processing/pipelines/ocr/marker_runtime.py)
  owns Marker subprocess isolation, source-object polling, temporary PDF
  staging, timeout calculation, and OCR event emission.
- [pipelines/ocr/config.py](src/training_signal_processing/pipelines/ocr/config.py)
  (106 lines): builds `RecipeConfig` from the resolved YAML.
- [pipelines/ocr/models.py](src/training_signal_processing/pipelines/ocr/models.py)
  (124 lines): `RecipeConfig`, `PdfTask`, `DocumentResult`,
  `OcrRayConfig`, `InputConfig`, `ResumeConfig`.
- [pipelines/ocr/configs/baseline.yaml](src/training_signal_processing/pipelines/ocr/configs/baseline.yaml)
  + [experiment.example.yaml](src/training_signal_processing/pipelines/ocr/configs/experiment.example.yaml):
  the runnable baseline + an example overlay.

---

## 13. Other pipelines

**example_echo** (`pipelines/example_echo/`) is the minimal reference
pipeline. It declares source rows inline in YAML, timestamps them in Ray
Data, writes one JSON object per source, and exposes its remote job via
`python -m training_signal_processing.pipelines.example_echo.cli remote-job`.
Use it as the smallest working shape for new pipeline families; OCR is
the richer production example.

**source_accounting** (`pipelines/source_accounting/`) scans processed dataset
parquet sources from YAML, writes accounting shards, and keeps the same
batch-manifest resume shape as OCR.

**lid_metadata** (`pipelines/lid_metadata/`) expands prior processed parquet
sources at row-group granularity, optionally trims reference sections, runs
Lingua plus Malaya document/word LID, and writes traceable metadata parquet
shards. Its throughput experiment knobs live in `lid:` (`row_batch_size`,
`inner_parallelism`, checkpoint intervals, and experiment/variant names);
metrics are emitted as checkpoint JSON, structured progress logs, and
per-shard `.metrics.json` sidecars.

---

## 14. Infrastructure

**Provisioning a remote box.** Two paths, both under `infra/`:

- `start_runpod_5090.sh` boots a RunPod community RTX 5090 via the
  REST API. It reads the API key from `infra/credentials/runpod/`,
  picks the image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` by default, opens
  SSH, and writes the connection details into `infra/current-machine`
  (gitignored). Sister scripts: `probe_runpod_5090.sh`,
  `kill_runpod_5090.sh`.
- `infra/dstack/` carries a parallel `dstack` setup
  (`config/fleet.dstack.yml`, `config/pretrain.dstack.yml`,
  `server/`).

**Per-machine SSH target.** `infra/current-machine` is the sole link
between the operator's local box and "the GPU node currently
provisioned." The OCR config loader reads it (when not
overridden via `--set ssh.host=…`) so a single recipe.yaml is
portable across operators.

**R2 credentials.** Loaded either from a config file referenced by
`r2.config_file` in the recipe (parsed by `R2ObjectStore.from_config_file`)
or from process env (`R2ObjectStore.from_environment`). The remote
process gets the env-var bundle from `R2ArtifactStore.build_remote_env`.

**Python deps.** `pyproject.toml [dependency-groups]` declares pipeline-scoped
groups: `dev` (lint/test), `model` (`torch`, `transformers`), `remote_ocr`
(`marker-pdf` plus runtime deps), `source_accounting`, and `lid_metadata`
(`lingua-language-detector`, `malaya`, `fasttext`, `refextract`, `tiktoken`,
and shared R2/Ray deps). `[tool.uv.sources]` pins `torch` to PyTorch's CUDA-128
wheel index on linux.

---

## 15. Glossary

- **R2** — Cloudflare R2, S3-compatible object storage.
- **rclone** — CLI for parallel uploads to R2/S3; OCR uses it for
  bulk PDF transfer before detached remote execution starts.
- **rsync** — CLI for code/lockfile sync; `SshRemoteTransport.sync`
  invokes it under SSH.
- **Marker** — `marker-pdf`; the OCR engine OCR'd by `marker_ocr`.
- **MLflow** — Optional experiment tracker; when enabled, it must use
  a directly reachable `mlflow.tracking_uri`.
- **vLLM** — LLM serving runtime used by `Qwen3AsrVllmOp` for batched
  inference.
- **dstack** — Alternate provisioning path under `infra/dstack/`.
- **RunPod** — GPU cloud provider; default provisioning path via
  `infra/start_runpod_5090.sh`.
- **Ray Data** — `ray.data.Dataset`; the per-batch pipeline that
  `apply_pipeline_transforms` builds lazily and `iter_batches`
  materializes.
- **ABC** — Python `abc.ABC`; a base class with abstract methods that
  concrete subclasses must implement. Used pervasively in
  `core/` and `ops/`.
