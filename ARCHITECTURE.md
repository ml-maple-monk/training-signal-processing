# Architecture

`training_signal_processing` is a small framework for running **batch
GPU pipelines on remote machines**, plus three concrete pipelines built
on top of it: **OCR** (Marker on PDFs), **tokenizer** (HF on parquet
shards), and **youtube_asr** (vLLM Qwen3 ASR on YouTube media). The
shared layer is intentionally pipeline-agnostic; new pipelines plug in
by implementing four small ABCs and registering ops by class. This
document captures the static structure (layers, contracts, frozen
invariants) and the dynamic flow (submission → remote execution →
exporter/ledger loop), then walks one OCR run end-to-end as the worked
example. Sister pipelines and infrastructure get short closing
sections.

If you only want the OCR walkthrough, jump to [§12](#12-ocr-pipeline-walkthrough-end-to-end).
If you're about to modify a file, [§3](#3-frozen-invariants) tells you
which files are protected.

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [Layered architecture and the import contract](#2-layered-architecture-and-the-import-contract)
3. [Frozen invariants](#3-frozen-invariants)
4. [Static contracts: the ABCs and the concretes](#4-static-contracts-the-abcs-and-the-concretes)
5. [Configuration plane](#5-configuration-plane)
6. [Object storage and artifacts](#6-object-storage-and-artifacts)
7. [Submission and transport](#7-submission-and-transport)
8. [Remote execution flow](#8-remote-execution-flow)
9. [Ops: registration and resolution](#9-ops-registration-and-resolution)
10. [Observability](#10-observability)
11. [Resume semantics](#11-resume-semantics)
12. [OCR pipeline walkthrough end-to-end](#12-ocr-pipeline-walkthrough-end-to-end)
13. [Other pipelines](#13-other-pipelines)
14. [Infrastructure](#14-infrastructure)
15. [Glossary](#15-glossary)

---

## 1. TL;DR

Three layers, three pipelines, one runtime. The middle layer (`ops/`)
defines an `Op` ABC; concrete ops live under `custom_ops/` and
self-register at module import. The foundation layer (`runtime/` +
`core/` + `storage/`) contains the executor, submission coordinator,
artifact store, observability, and resume ledger. The top layer
(`pipelines/`) wires per-pipeline `RecipeConfig`, runtime adapter, and
submission adapter. A run is two halves: a local **submission** that
syncs code, uploads inputs to R2, and SSHes a `python -m
training_signal_processing.main ocr-remote-job …` invocation onto a
remote GPU box; and a **remote execution** that rebuilds the config
from R2, runs ops via Ray Data `map_batches`, and writes outputs +
ledger entries back to R2 per batch.

---

## 2. Layered architecture and the import contract

```
+--------------------------------------------------------------------+
|  TOP — pipeline-specific                                           |
|                                                                    |
|  pipelines/ocr        pipelines/tokenizer    pipelines/youtube_asr |
|  custom_ops/user_ops  custom_ops/tokenizer_  custom_ops/yt_asr_ops |
|                       ops                                          |
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
|  runtime/                |   core/             |   storage/        |
|    submission.py         |     models.py       |     object_store  |
|    executor.py           |     utils.py        |       .py         |
|    dataset.py            |     config_loading  |                   |
|    exporter.py           |       .py           |                   |
|    resume.py             |                     |                   |
|    observability.py      |                     |                   |
|    remote_job.py         |                     |                   |
|    async_upload_         |                     |                   |
|      coordinator.py      |                     |                   |
+--------------------------------------------------------------------+
                                   ^
                                   | enforced by [tool.importlinter]:
                                   | "shared layers must not import
                                   |  from pipelines"
```

The contract at `pyproject.toml [[tool.importlinter.contracts]]`
declares `core`, `ops`, `runtime`, `storage` as `source_modules` that
are forbidden from importing `training_signal_processing.pipelines`.
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
| `runtime/submission.py` | Submission/transport ABCs + `SubmissionCoordinator` glue |
| `runtime/executor.py` | `Executor` + `PipelineRuntimeAdapter` ABC + `StreamingRayExecutor` |
| `runtime/dataset.py` | `DatasetBuilder` ABC + Ray adapters |
| `runtime/observability.py` | `ExecutionLogger` + `ProgressTracker` ABCs + MLflow concrete |
| `runtime/exporter.py` | `Exporter` ABC |
| `runtime/resume.py` | `ResumeLedger` ABC |
| `runtime/remote_job.py` | `RemoteJob` + `build_remote_job_cli` factory + guard CLI |
| `runtime/__init__.py` | Marker only |
| `ops/base.py` | `Op` ABC + auto-registration mechanism |
| `ops/builtin.py` | Stage templates pipelines subclass |
| `ops/registry.py` | `OpRegistry` + import-sweep bootstrap |
| `ops/testing.py` | Op-level test harness |
| `ops/__init__.py` | Marker only |
| `core/models.py` | All shared dataclasses (`R2Config`, `RayConfig`, `RunState`, …) |
| `core/utils.py` | Cross-cutting helpers (`join_s3_key`, `utc_isoformat`, …) |
| `storage/object_store.py` | `ObjectStore` ABC + `R2ObjectStore` |
| `custom_ops/__init__.py` | The package-import sweep that triggers Op registration |
| `custom_ops/user_ops.py` | OCR ops (also tagged frozen) |
| `main.py` | OCR CLI entrypoint |

The import-linter contract above is the second invariant: shared
layers stay pipeline-agnostic.

---

## 4. Static contracts: the ABCs and the concretes

### 4.1 ABCs that pipelines implement

| ABC | File:line | Methods (abstract) |
|-----|-----------|--------------------|
| `SubmissionAdapter` | [runtime/submission.py:478](src/training_signal_processing/runtime/submission.py#L478) | `prepare_new_run`, `prepare_resume_run`, `parse_remote_summary` † |
| `PipelineRuntimeAdapter` | [runtime/executor.py:49](src/training_signal_processing/runtime/executor.py#L49) | `get_run_bindings`, `get_execution_config`, `get_tracking_context`, `get_op_configs`, `get_artifact_layout`, `load_input_rows`, `build_runtime_context`, `build_op_registry`, `build_exporter`, `build_resume_ledger` |
| `ResumeLedger` | [runtime/resume.py:11](src/training_signal_processing/runtime/resume.py#L11) | `find_latest_partial_run`, `load_run_state`, `load_completed_item_keys`, `initialize_run_state`, `commit_batch`, `write_run_state`, `mark_run_finished`, `mark_run_failed` |
| `Exporter` | [runtime/exporter.py:11](src/training_signal_processing/runtime/exporter.py#L11) | `export_batch`, `finalize_run` |
| `OpRegistry` | [ops/registry.py:50](src/training_signal_processing/ops/registry.py#L50) (template `resolve_pipeline`) | `resolve_pipeline(configs) -> ResolvedOpPipeline` |
| `Op` (+ `MapperOp` / `FilterOp` / `PipelineOp`) | [ops/base.py:17](src/training_signal_processing/ops/base.py#L17), [:82](src/training_signal_processing/ops/base.py#L82), [:90](src/training_signal_processing/ops/base.py#L90), [:101](src/training_signal_processing/ops/base.py#L101) | `process_batch(Batch) -> Batch` (and one of: `op_name`, `op_stage`) |

> † `parse_remote_summary` is not invoked by the detached-launch path in
> PR #1 (OCR). Blocking `execute`-style adapters (`example_echo`,
> `tokenizer`, `youtube_asr`) still call it to parse the remote JSON
> summary from stdout. See §7 *Detached lifecycle* for how OCR moved off it.

`PipelineRuntimeAdapter` lines [94-107](src/training_signal_processing/runtime/executor.py#L94-L107)
are **default-method extension hooks** rather than pure abstractions:
`build_dataset_builder` (defaults to `ConfiguredRayDatasetBuilder`),
`resolve_completed_item_keys` (defaults to passthrough),
`resolve_transform_resources` (defaults to "no override"). OCR
overrides the last to assign GPU resources to `marker_ocr` —
[pipelines/ocr/runtime.py:105](src/training_signal_processing/pipelines/ocr/runtime.py#L105).

### 4.2 Concretes pipelines compose

These are provided by the runtime; pipelines consume them as-is.

| Class | File:line | Role |
|-------|-----------|------|
| `R2ArtifactStore` | [runtime/submission.py:316](src/training_signal_processing/runtime/submission.py#L316) | `ArtifactStore` over R2/S3; `build_remote_env` at [:354](src/training_signal_processing/runtime/submission.py#L354) emits R2 + AWS-compat + MLflow S3 endpoint vars |
| `SshRemoteTransport` | [runtime/submission.py:386](src/training_signal_processing/runtime/submission.py#L386) | `RemoteTransport` over rsync + ssh; `launch_detached` spawns the remote job under `setsid` and records its pgid; `ensure_reverse_tunnels` opens a persistent `ssh -fN -o ControlMaster=yes` per declared `-R` tunnel before launch so the tunnel outlives the launcher SSH (see [runtime/reverse_tunnel.py](src/training_signal_processing/runtime/reverse_tunnel.py)) |
| `SubmissionCoordinator` | [runtime/submission.py:492](src/training_signal_processing/runtime/submission.py#L492) | Orchestrates `prepare → sync → bootstrap → (optional local upload wait) → ensure_reverse_tunnels → launch_detached → return LaunchHandle` |
| `StreamingRayExecutor` | [runtime/executor.py:114](src/training_signal_processing/runtime/executor.py#L114) | The `Executor` for batch GPU pipelines; per-batch loop |
| `RegisteredOpRegistry` | [ops/registry.py:63](src/training_signal_processing/ops/registry.py#L63) | The concrete `OpRegistry`; triggers the import-sweep on module load |
| `RayDatasetBuilder` / `ConfiguredRayDatasetBuilder` | [runtime/dataset.py:57](src/training_signal_processing/runtime/dataset.py#L57), [:132](src/training_signal_processing/runtime/dataset.py#L132) | `DatasetBuilder` over Ray Data; `Configured` re-partitions to `target_num_blocks` |
| `MlflowExecutionLogger` / `MlflowProgressTracker` | [runtime/observability.py:32](src/training_signal_processing/runtime/observability.py#L32), [:364](src/training_signal_processing/runtime/observability.py#L364) | MLflow-backed `ExecutionLogger` and `ProgressTracker`; `StructuredExecutionLogger` and `NullProgressReporter` are the no-MLflow fallbacks |
| `RemoteJob` / `ObjectStoreRemoteJob` / `build_remote_job_cli` | [runtime/remote_job.py](src/training_signal_processing/runtime/remote_job.py) | The remote-side CLI factory; pipelines wrap it in ~30 lines |

The shared dataclasses live in [core/models.py](src/training_signal_processing/core/models.py):
`R2Config`, `RayConfig`, `RayTransformResources`, `RemoteRuntimeConfig`,
`SshConfig`, `MlflowConfig`, `OpConfig`, `RuntimeRunBindings`,
`RuntimeTrackingContext`, `RunArtifactLayout`, `BatchCommit`,
`ExportBatchResult`, `RunState`, `ExecutionLogEvent`,
`ExecutorRunSummary`, `OpRuntimeContext`.

---

## 5. Configuration plane

A run is described by a YAML recipe + zero or more YAML overlays + zero
or more dotted-path `--set` overrides. The pipeline-agnostic plumbing
lives in [core/config_loading.py:17](src/training_signal_processing/core/config_loading.py#L17)
(`load_recipe_mapping`) and [:38](src/training_signal_processing/core/config_loading.py#L38)
(`deep_merge_mapping`). Each pipeline's `config.py` thin-wraps it
(e.g. [pipelines/ocr/config.py:34](src/training_signal_processing/pipelines/ocr/config.py#L34)
`load_recipe_config`) and supplies its own
`build_recipe_config(raw, path)` + `validate_recipe_constraints(raw)`.

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
4. **OCR + youtube_asr only:** if `infra/current-machine` exists and
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

---

## 6. Object storage and artifacts

`ObjectStore` ([storage/object_store.py:19](src/training_signal_processing/storage/object_store.py#L19))
is an ABC: `exists`, `list_keys`, `read_bytes`, `write_bytes`,
`upload_file`, `make_url`, `build_pyarrow_filesystem`, plus convenience
wrappers `read_json` / `write_json` / `read_jsonl` / `write_jsonl`.
The only concrete is `R2ObjectStore` ([:80](src/training_signal_processing/storage/object_store.py#L80)),
which uses `boto3` against a Cloudflare R2 endpoint and accepts
credentials via either an env file (`from_config_file`) or process
environment (`from_environment`).

`ArtifactStore` ([runtime/submission.py:284](src/training_signal_processing/runtime/submission.py#L284))
is the runtime-side façade — same contract plus `upload_file` and
`build_remote_env`. The concrete `R2ArtifactStore` wraps an
`R2ObjectStore`; `build_remote_env` ([:354](src/training_signal_processing/runtime/submission.py#L354))
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
manifests/<batch_id>.jsonl     # per-batch result rows (ledger)
events/<batch_id>.json         # per-batch event summary (ledger)
run_state.json                 # current run state (ledger)
run.json                       # final summary (exporter.finalize_run)
<pipeline outputs>             # markdown for OCR, jsonl.gz for tokenizer, ...
```

`RunArtifactLayout` (`core/models.py`) carries the two roots
(`source_root_key`, `output_root_key`) the executor passes into the
runtime context.

### 6.1 Async upload coordinator (driver-side)

`AsyncUploadCoordinator` ([runtime/async_upload_coordinator.py:28](src/training_signal_processing/runtime/async_upload_coordinator.py#L28))
is an opt-in component that lets the driver overlap per-batch R2
writes with downstream compute. Without it, `Exporter.export_batch`
loops through rows calling `object_store.write_bytes(...)`
synchronously — each `put_object` blocks the batch loop, so the
driver cannot advance to the next batch (or to
`resume_ledger.commit_batch`) until every PUT has returned.

The coordinator owns one daemon thread hosting an `asyncio` event
loop and a long-lived `aioboto3` S3 client. `submit(key, body)`
returns immediately after scheduling an upload coroutine on that
loop; bounds are enforced by an `asyncio.Semaphore(max_in_flight)`
on the loop side and a `threading.Semaphore(max_queued)` on the
submit side (backpressure so memory doesn't blow up under a fast
producer). A `_put_bytes(self, key, body)` helper on `RayExporter`
([runtime/exporter.py:21](src/training_signal_processing/runtime/exporter.py#L21))
routes to `coordinator.submit` when an upload coordinator is
attached, and falls back to `object_store.write_bytes` otherwise.

Lifecycle is scoped to one run, orchestrated by the executor:
constructed once after `build_exporter()`, drained before every
`resume_ledger.commit_batch` (so a failed upload surfaces as a run
failure rather than a partial commit), aborted in the outer
`except` handler, and closed in a `finally` block for idempotent
teardown. `R2ObjectStore` itself stays untouched — its
`write_bytes` signature is unchanged, and the coordinator holds no
state on the store, so the same `R2ObjectStore` instance stays safe
to pickle into Ray worker closures (workers continue to use the
sync path).

Opt-in via YAML — the default is absent:

```yaml
ray:
  async_upload:
    enabled: true
    max_in_flight: 8
    max_queued: 32
```

If the block is omitted (or `enabled: false`), `_build_upload_coordinator`
([runtime/executor.py](src/training_signal_processing/runtime/executor.py))
returns `None` and the export path uses sync writes — behavior
identical to the pre-coordinator runtime. Worker-side writes
(`custom_ops/youtube_asr_ops.py`, `custom_ops/tokenizer_ops.py`)
are intentionally not routed through the coordinator: workers live
in separate processes, so a driver-side asyncio loop doesn't help
them, and the complexity of per-worker loops would outweigh the
gain.

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
                          ssh -R 15000:127.0.0.1:5000          v
                          MLflow tunnel: remote port 15000     [Diagram 3]
                          forwards to local MLflow at 5000
```

The flow on the local side is split between three interfaces:

- `SubmissionAdapter` (per-pipeline) returns a `PreparedRun` whose
  fields tell the coordinator what to do: `sync_paths` (rsync targets),
  `bootstrap` (shell to run after sync), `invocation` (the remote
  command + env + reverse tunnels), `artifacts` (input/output refs),
  and optionally `async_upload` (a local rclone command run in
  parallel with the remote execution).
- `RemoteTransport` (`SshRemoteTransport`) implements `sync`,
  `bootstrap`, `execute`, `launch_detached`, and `ensure_reverse_tunnels`
  over SSH + rsync.
- `SubmissionCoordinator.submit` ([runtime/submission.py:492](src/training_signal_processing/runtime/submission.py#L492))
  orchestrates them: prepare → sync code → bootstrap → start async
  upload (if any) → wait for upload → ensure_reverse_tunnels →
  launch_detached → return LaunchHandle. The OCR path no longer parses
  remote stdout (no `adapter.parse_remote_summary` call); the remote
  writes its own `run_state.json` to R2 instead.

### 7.1 Detached lifecycle (PR #1)

`submit()` returns immediately once `setsid` writes `job.pgid` on the
pod — it is **fire-and-forget**. The remote OCR job runs in its own
process group under PID 1 (init), so SSH disconnects cannot kill it.
Remote stdout/stderr live at `/root/ocr-jobs/<run_id>/job.log`; the
local CLI prints only the `LaunchHandle` + `TunnelHandle` JSON and
exits. **Exit code 0 means "launched successfully", not "run
complete".** Completion must be inferred by reading `run_state.json`
from R2 or by tailing the remote log.

### 7.2 Reverse tunnel lifecycle (PR #2a)

`ensure_reverse_tunnels()` runs before `launch_detached` and spawns a
persistent `ssh -fN -o ControlMaster=yes` per declared `-R` tunnel
spec. The ControlMaster socket is stored at
`~/.cache/ocr-remote-launcher/tunnels/t-<hash>.sock`, hashed by host +
port + tunnel spec so the same tunnel is always deterministically
addressable. The call is idempotent: if `ssh -S <sock> -O check`
answers OK the existing tunnel is reused; otherwise a stale socket is
unlinked and a fresh `ssh -fN` is spawned. Manual teardown is
`ssh -S <sock> -O exit <ssh_target>`. See ONBOARDING §3 *Running-job
runbook* for the operator-facing checks.

OCR-side helpers worth pointing at:

- `OcrSubmissionAdapter.build_async_upload_spec`
  ([pipelines/ocr/submission.py:263](src/training_signal_processing/pipelines/ocr/submission.py#L263))
  builds the rclone argv (lines 263-322). The R2-flavored env for
  rclone comes from `build_rclone_env` ([:330](src/training_signal_processing/pipelines/ocr/submission.py#L330)).
- `build_invocation_spec` ([pipelines/ocr/submission.py:156](src/training_signal_processing/pipelines/ocr/submission.py#L156))
  composes the remote `uv run python -m main ocr-remote-job …`
  command; `build_reverse_tunnel_spec` ([:206](src/training_signal_processing/pipelines/ocr/submission.py#L206))
  produces the `-R 15000:127.0.0.1:5000` tunnel string when MLflow is
  enabled, so the remote process can post to the operator's local
  MLflow server.

---

## 8. Remote execution flow

```
remote shell:
$ uv run python -m training_signal_processing.main \
      ocr-remote-job --run-id ... --config-object-key ... \
                     --input-manifest-key ... --uploaded-items N
        |
        v
build_remote_job_cli closure  (runtime/remote_job.py)
        |
        v
ObjectStoreRemoteJob.run()  (runtime/remote_job.py)
        |
        v
StreamingRayExecutor.run()  (runtime/executor.py:114)
        |
        +--- pipeline.get_run_bindings / get_execution_config /
        |    get_tracking_context / get_artifact_layout / get_op_configs
        |
        +--- validate_contract                       (executor.py:444)
        |    raises PipelineContractError on misuse
        |
        +--- build MlflowExecutionLogger             (observability.py:32)
        +--- build MlflowProgressTracker             (observability.py:364)
        |
        +--- pipeline.build_resume_ledger()
        |     load_completed_item_keys(prior_run_id)  if resume
        |     initialize_run_state(...)
        |
        +--- pipeline.build_runtime_context(logger=, completed_item_keys=)
        +--- pipeline.build_op_registry(runtime_context=)
        +--- registry.resolve_pipeline(op_configs)   (registry.py:50)
        |     -> ResolvedOpPipeline(prepare, [transforms], export)
        |
        +--- pipeline.build_dataset_builder()
        |     default: ConfiguredRayDatasetBuilder
        +--- pipeline.load_input_rows()              # reads input_manifest.jsonl
        |
        +--- apply_pipeline_transforms               (executor.py:405)
        |     dataset.map_batches(prepare_op, ...)
        |     for op in transform_ops:
        |         resources = pipeline.resolve_transform_resources(op, execution)
        |         dataset.map_batches(op, batch_size=..., concurrency=...,
        |                                  num_gpus=R.num_gpus,
        |                                  num_cpus=R.num_cpus)
        |     (LAZY — no actual execution yet)
        |
        +--- coordinator = _build_upload_coordinator(execution, ...)
        |     # returns None unless ray.async_upload.enabled AND
        |     # object_store isinstance R2ObjectStore
        |     if coordinator: exporter.upload_coordinator = coordinator
        |
        +--- try:                                            # lifecycle wrap
        |    for batch in dataset_builder.iter_batches(dataset, batch_size):
        |       # FIRST iter materializes the lazy plan
        |       exporter.export_batch(batch_id, rows)         # queue writes
        |       if coordinator: coordinator.drain()           # await in-flight
        |       resume_ledger.commit_batch(...)               # ledger write
        |       progress_tracker.log_batch_commit(...)        # MLflow metrics
        |       transition_run_phase(..., "first_batch_materialized")
        |
        |    resume_ledger.mark_run_finished(run_state)
        |    exporter.finalize_run(run_state)
        |    if coordinator: coordinator.drain()              # final drain
        |    return ExecutorRunSummary.to_dict()              # written to run_state.json on R2;
        |                                                     # local CLI has already exited
        |                                                     # with a LaunchHandle (see §7.1)
        |
        +--- except Exception:
        |      if coordinator: coordinator.abort()            # cancel in-flight
        |      resume_ledger.mark_run_failed(run_state, ...)  # status=failed
        +--- finally:
               if coordinator: coordinator.close()            # idempotent
```

Three properties matter:

1. **Lazy → eager boundary.** `apply_pipeline_transforms` only chains
   `dataset.map_batches` calls; nothing actually runs until
   `iter_batches` pulls a batch. This is what gives Ray Data its
   pipelining behavior.
2. **Per-op resource override.** Default `resolve_transform_resources`
   ([runtime/executor.py:101](src/training_signal_processing/runtime/executor.py#L101))
   returns an empty `RayTransformResources`. OCR overrides at
   [pipelines/ocr/runtime.py:105](src/training_signal_processing/pipelines/ocr/runtime.py#L105)
   to give `marker_ocr` the configured `num_gpus` / `num_cpus` from
   `config.ray.marker_ocr_resources`.
3. **Phase telemetry.** `transition_run_phase`
   ([runtime/executor.py:313](src/training_signal_processing/runtime/executor.py#L313))
   is called at every checkpoint (`manifest_loaded`, `dataset_build_*`,
   `iter_batches_start`, `first_batch_materialized`, …) — it persists
   to `RunState.current_phase` and emits a structured event so an
   external watcher can tell where in the run you are.

---

## 9. Ops: registration and resolution

Ops are not registered explicitly — they self-register on import.
The trigger is one line at the top of `ops/registry.py`.

```
import time
=========

ops/registry.py:13
   |
   v
   importlib.import_module(
       "training_signal_processing.custom_ops")
                |
                v
   custom_ops/__init__.py:9-21
       import_custom_op_modules()
       for each .py in custom_ops/:
           import_module(f"{__name__}.{stem}")
                |
                v
   importing user_ops.py loads, e.g.:
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
       - {name: export_markdown,      type: mapper}
                |
                v
   OpRegistry.resolve_pipeline(configs)   (registry.py:50)
       cls = REGISTERED_OP_TYPES[name]
       op  = cls(**config.options)        # e.g. force_ocr=True
       op.bind_runtime(runtime_context)
       group by op.op_stage:
         prepare    -> exactly one
         transform  -> >= one
         export     -> exactly one
       -> ResolvedOpPipeline(prepare, [transforms...], export)
```

**Adding a new transform op for OCR:**

1. Define the class somewhere under `custom_ops/` (typical home:
   `custom_ops/user_ops.py`):
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
     - {name: export_markdown,      type: mapper}
   ```

The class is auto-discovered by the import-sweep; no other
registration step is needed. Op-level testing uses
`build_default_ray_op_test_harness()` in `ops/testing.py` to exercise
a single op against a `LocalRayDatasetBuilder` without spinning up the
full executor.

OCR's concrete ops live in
[custom_ops/user_ops.py](src/training_signal_processing/custom_ops/user_ops.py):
`PreparePdfDocumentOp` ([:208](src/training_signal_processing/custom_ops/user_ops.py#L208)),
`SkipExistingDocumentsOp` ([:238](src/training_signal_processing/custom_ops/user_ops.py#L238)),
`MarkerOcrDocumentOp` ([:248](src/training_signal_processing/custom_ops/user_ops.py#L248)),
`ExportMarkdownResultOp` ([:364](src/training_signal_processing/custom_ops/user_ops.py#L364)).

---

## 10. Observability

Two parallel hierarchies: an **event stream** (`ExecutionLogger`,
`StructuredTracer`, `StructuredMonitor`) and a **state stream**
(`ProgressTracker` + `ProgressReporter`).

- **Events.** `ExecutionLogEvent` ([core/models.py](src/training_signal_processing/core/models.py))
  is the unit. `MlflowExecutionLogger` ([observability.py:32](src/training_signal_processing/runtime/observability.py#L32))
  buffers events until the MLflow run id is known, then flushes via
  `attach_run_id`; per-event side effects update an
  `execution_event_count` metric and re-stamp `last_execution_event_*`
  tags ([:64](src/training_signal_processing/runtime/observability.py#L64)).
  This tag re-stamp is the run's **heartbeat** — an external watcher
  can look at the tag to confirm the run is alive without polling
  metrics.
- **Categories of events.** Four groups: `executor.*` (manifest load,
  resume load, phase transitions, dataset transforms, op-complete);
  `tracer.op.before` / `tracer.op.after` (per-op enter/exit);
  `monitor.run.start` / `finish` / `fail` (RunState snapshots);
  ad-hoc events from concrete ops via
  `op.log_runtime_event(level, code, message, **details)`.
- **Metrics.** `MlflowProgressTracker.log_batch_commit`
  ([:418](src/training_signal_processing/runtime/observability.py#L418))
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
([runtime/submission.py:482](src/training_signal_processing/runtime/submission.py#L482))
loads the prior `recipe.json` and `input_manifest.jsonl` from R2 and
reuses them. On the remote side, `StreamingRayExecutor.run` calls
`pipeline.build_resume_ledger().load_completed_item_keys(run_id)` and
threads the resulting set into `OpRuntimeContext.completed_item_keys`,
which `SkipExistingFilter` consults to drop already-processed rows.

Per-batch ledger writes are atomic: each batch produces three R2
writes (`manifests/<batch_id>.jsonl`, `events/<batch_id>.json`,
`run_state.json`). `commit_batch` ([pipelines/ocr/resume.py:86](src/training_signal_processing/pipelines/ocr/resume.py#L86))
writes them in that order; if the executor dies between batches, the
next resume starts from the latest committed batch.

`OcrResumeLedger.find_latest_partial_run`
([pipelines/ocr/resume.py:18](src/training_signal_processing/pipelines/ocr/resume.py#L18))
filters `list_keys` output to `*/run_state.json` keys, sorts
descending, and returns the first whose status is `running` or
`partial`. This avoids the per-key `set` dedup that the original
implementation did (cleanup landed in commit `0b303b0`).

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
        - count pages via pypdfium2          (submission.py:248)
        - sort by page count + size          (in build_pdf_tasks)
        - write input_manifest.jsonl + recipe.json to R2
        - build LocalAsyncUploadSpec (rclone)(submission.py:263)
                                                                |
                                                                v
[4] SubmissionCoordinator.submit                 (submission.py:492)
                ┌───────────┬──────────┬──────────┬────────────┬────────────────┐
                │           │          │          │            │                │
                v           v          v          v            v                v
          rsync -az    bootstrap  rclone      ensure_      launch_detached   return
          sync_paths   uv sync    parallel    reverse_    (setsid sh -c …  LaunchHandle
                                  upload of   tunnels      echo $$>pgid;   (local CLI
                                  PDFs        (ssh -fN -M  exec launch.sh   exits here;
                                  (awaited    -R 15000:    & — survives    remote keeps
                                   before     127.0.0.1:   SSH disconnect)  running under
                                   launch)    5000)                         setsid)
                                                                |
~~~~~~~~~~~~~~~~~~~~~~~~~~~ network boundary ~~~~~~~~~~~~~~~~~~~|~~~~~~~~~~
                                                                v
[5] (REMOTE) build_remote_job_cli closure        (runtime/remote_job.py)
        - bootstrap_store.read_json(config_object_key)  -> recipe dict
        - build OcrPipelineRuntimeAdapter(config, bindings, object_store)
        - ObjectStoreRemoteJob.run()
                                                                |
                                                                v
[6] StreamingRayExecutor.run()                   (executor.py:114)
        per-batch loop iterates ops:
          prepare_pdf_document  ->  enrich row with markdown_r2_key
          skip_existing         ->  drop rows already in completed_item_keys
          marker_ocr            ->  spawned subprocess + GPU OCR
                                    (resources from marker_ocr_resources)
          export_markdown       ->  validates row shape, no I/O
                                                                |
                                                                v
[7] OcrMarkdownExporter.export_batch            (exporter.py:16)
        - for row in rows:
            if row.status != "success": continue
            self._put_bytes(markdown_r2_key, markdown_text.encode())
            cleanup_staged_pdf(row.staged_pdf_path)
        # _put_bytes routes to the async coordinator if attached,
        # else falls back to sync object_store.write_bytes. The
        # executor drains the coordinator before commit_batch below.
                                                                |
                                                                v
[8] OcrResumeLedger.commit_batch                (resume.py:86)
        - write manifests/<batch_id>.jsonl   (per-batch result rows)
        - write events/<batch_id>.json       (counts + duration)
        - update run_state.json              (running pending_items, etc.)
                                                                |
                              ... loop until input exhausted ...
                                                                v
[9] mark_run_finished -> finalize_run -> remote writes run_state.json to R2
        status is one of: success | partial | failed
        run.json carries the final RunState
        (local CLI already exited with a LaunchHandle at step [4];
         operators inspect run_state.json or tail /root/ocr-jobs/<run_id>/job.log
         to see completion — see ONBOARDING §3 *Running-job runbook*)
```

**Worth knowing.** Steps 5-8 are framework-generic; they are the same
loop the tokenizer and youtube_asr pipelines run, only with different
ops. Steps 1-4 and 7-8 are OCR-specific (PDF discovery, page-count
sort, markdown writes, marker spawn isolation). The single OCR-only
runtime override is `resolve_transform_resources` at
[pipelines/ocr/runtime.py:105](src/training_signal_processing/pipelines/ocr/runtime.py#L105)
which gives `marker_ocr` a GPU.

OCR's pipeline-local files:

- [pipelines/ocr/submission.py](src/training_signal_processing/pipelines/ocr/submission.py)
  (359 lines): the `SubmissionAdapter`. The bulk is the rclone block
  ([:263-353](src/training_signal_processing/pipelines/ocr/submission.py#L263)).
- [pipelines/ocr/runtime.py](src/training_signal_processing/pipelines/ocr/runtime.py)
  (118 lines): the `PipelineRuntimeAdapter`.
- [pipelines/ocr/resume.py](src/training_signal_processing/pipelines/ocr/resume.py)
  (183 lines): the `ResumeLedger`.
- [pipelines/ocr/exporter.py](src/training_signal_processing/pipelines/ocr/exporter.py)
  (52 lines): the `Exporter` — writes markdown bytes per batch.
- [pipelines/ocr/config.py](src/training_signal_processing/pipelines/ocr/config.py)
  (106 lines): builds `RecipeConfig` from the resolved YAML.
- [pipelines/ocr/models.py](src/training_signal_processing/pipelines/ocr/models.py)
  (124 lines): `RecipeConfig`, `PdfTask`, `DocumentResult`,
  `OcrRayConfig`, `InputConfig`, `ResumeConfig`.
- [pipelines/ocr/remote_job.py](src/training_signal_processing/pipelines/ocr/remote_job.py)
  (34 lines): a 5-line wrapper around `build_remote_job_cli`.
- [pipelines/ocr/configs/baseline.yaml](src/training_signal_processing/pipelines/ocr/configs/baseline.yaml)
  + [experiment.example.yaml](src/training_signal_processing/pipelines/ocr/configs/experiment.example.yaml):
  the runnable baseline + an example overlay.

---

## 13. Other pipelines

**Tokenizer** (`pipelines/tokenizer/`, ops in
[custom_ops/tokenizer_ops.py](src/training_signal_processing/custom_ops/tokenizer_ops.py)).
Input rows are `ParquetShardTask` (one parquet shard per row).
`TokenizeHfTokenIdsOp` lazily loads the model via
`AutoTokenizer.from_pretrained` at
[custom_ops/tokenizer_ops.py:170](src/training_signal_processing/custom_ops/tokenizer_ops.py#L170),
reads each shard via `pq.read_table` directly off the R2 filesystem,
encodes the configured text column, and the exporter writes per-shard
gzipped JSONL of token IDs back to R2. CPU-only; no GPU resources.

**YouTube ASR** (`pipelines/youtube_asr/`, ops in
[custom_ops/youtube_asr_ops.py](src/training_signal_processing/custom_ops/youtube_asr_ops.py)).
Input rows are `YoutubeMediaTask`. The transcribe op `Qwen3AsrVllmOp`
([custom_ops/youtube_asr_ops.py:59](src/training_signal_processing/custom_ops/youtube_asr_ops.py#L59))
lazily loads `Qwen3ASRModel.LLM(...)` once per worker and runs
`model.transcribe(audio=media_paths, language=language)` per batch
([:124](src/training_signal_processing/custom_ops/youtube_asr_ops.py#L124))
— the model load is amortized across all rows in the batch. GPU
batch-shaped, vLLM-backed.

Both pipelines reuse all the same runtime machinery; only their
`PipelineRuntimeAdapter`, `SubmissionAdapter`, custom ops, and
`RecipeConfig` differ. Their `remote_job.py` is a 5-line wrapper
around `build_remote_job_cli`, identical in shape to OCR's.

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
provisioned." The OCR + youtube_asr config loaders read it (when not
overridden via `--set ssh.host=…`) so a single recipe.yaml is
portable across operators.

**R2 credentials.** Loaded either from a config file referenced by
`r2.config_file` in the recipe (parsed by `R2ObjectStore.from_config_file`)
or from process env (`R2ObjectStore.from_environment`). The remote
process gets the env-var bundle from `R2ArtifactStore.build_remote_env`.

**Python deps.** `pyproject.toml [dependency-groups]` declares only
three groups: `dev` (lint/test), `model` (`torch`, `transformers`),
`remote_ocr` (`boto3`, `click`, `marker-pdf`, `mlflow-skinny`,
`pyarrow`, `pyyaml`, `ray[data]`, `torch`, `tqdm`). Tokenizer and
youtube_asr reuse `remote_ocr`'s deps. `[tool.uv.sources]` pins
`torch` to PyTorch's CUDA-128 wheel index on linux.

---

## 15. Glossary

- **R2** — Cloudflare R2, S3-compatible object storage.
- **rclone** — CLI for parallel uploads to R2/S3; OCR uses it for
  bulk PDF transfer in parallel with remote execution.
- **rsync** — CLI for code/lockfile sync; `SshRemoteTransport.sync`
  invokes it under SSH.
- **Marker** — `marker-pdf`; the OCR engine OCR'd by `marker_ocr`.
- **MLflow** — Experiment tracker; MLflow metrics flow over a
  persistent `-R` reverse tunnel (see §7.2 *Reverse tunnel lifecycle*).
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
  `runtime/`, `ops/`, `storage/`.
