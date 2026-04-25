# Extending

You want to add a new pipeline family. This document walks you from
concept to a working `example_echo` pipeline that passes `validate`,
`--dry-run`, `ruff`, and `lint-imports`. Every code block below is an
extract from committed files under [pipelines/example_echo/](src/training_signal_processing/pipelines/example_echo/)
— the doc and the template can't drift.

**Completeness bar.** "Runnable" means the four commands below all
pass. It does **not** promise a successful remote execution —
`--dry-run` short-circuits before the SSH transport and real execution
drags in RunPod / R2 / bootstrap specifics covered elsewhere.

```
uv run ruff check src/training_signal_processing/pipelines/example_echo
uv run lint-imports
uv run --group remote_ocr python -m training_signal_processing.pipelines.example_echo.cli validate \
  --config src/training_signal_processing/pipelines/example_echo/configs/baseline.yaml
uv run --group remote_ocr python -m training_signal_processing.pipelines.example_echo.cli run \
  --config src/training_signal_processing/pipelines/example_echo/configs/baseline.yaml --dry-run
```

All CLI commands require `--group remote_ocr` — the base dependency
group has only `duckdb`; `click`, `pyyaml`, `boto3`, `ray[data]`,
`mlflow-skinny`, `pyarrow` are all gated behind that group.

**Cross-references.**
- [ARCHITECTURE.md](ARCHITECTURE.md) — layered structure, ABC
  contracts, frozen-file invariants, OCR walkthrough with diagrams.
- [ONBOARDING.md](ONBOARDING.md) — operator guide: start infra + run
  an existing pipeline.

---

## Table of contents

- [§1 The six building blocks](#1-the-six-building-blocks)
- [§2 Config — YAML schema and `RecipeConfig`](#2-config--yaml-schema-and-recipeconfig)
- [§3 Ops — the per-row chain](#3-ops--the-per-row-chain)
- [§4 Execute — `PipelineRuntimeAdapter`](#4-execute--pipelineruntimeadapter)
- [§5 Materialize — `Exporter`](#5-materialize--exporter)
- [§6 Resume — `OutputCompletionTracker`](#6-resume--outputcompletiontracker)
- [§7 Submit — `SubmissionAdapter`](#7-submit--submissionadapter)
- [§8 CLI wiring](#8-cli-wiring)
- [§9 Recipe + first dry run](#9-recipe--first-dry-run)
- [§10 Out of scope](#10-out-of-scope)

---

## §1 The six building blocks

A pipeline is six pieces of behavior plus CLI plumbing:

| What you're building | Where it lives | Contract |
|---|---|---|
| **config schema** | `pipelines/<new>/{models,config}.py` | A `RecipeConfig` dataclass that mirrors the YAML recipe |
| **ops** | `pipelines/<new>/ops.py` | Subclasses of `Op` — exactly one `prepare`, ≥1 `transform`, exactly one `export` |
| **execute** | `pipelines/<new>/runtime.py` | `PipelineRuntimeAdapter` — the seam `StreamingRayExecutor` talks to |
| **materialize** | `pipelines/<new>/runtime.py` | `Exporter` — one write per output batch |
| **resume** | `pipelines/<new>/runtime.py` | `OutputCompletionTracker` — maps existing outputs back to source keys |
| **submit** | `pipelines/<new>/submission.py` | `SubmissionAdapter` — what to sync, how to bootstrap, how to invoke remote |
| **CLI wiring** | `pipelines/<new>/cli.py` | Local commands plus a `remote-job` command built with `build_remote_job_cli` |

One canonical directory tree for the worked example:

```
src/training_signal_processing/
├── pipelines/
│   └── example_echo/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── configs/baseline.yaml
│       ├── models.py
│       ├── ops.py
│       ├── runtime.py
│       └── submission.py
```

Narrative order below is top-down conceptual, not import-dependency
order. Every section opens with a bold **forward-reference** to the
later section where a symbol is fully explained — skim §1's table
first, then read straight through.

---

## §2 Config — YAML schema and `RecipeConfig`

**What.** Every pipeline declares a `RecipeConfig` dataclass that
mirrors its YAML recipe. Eight framework sections are mandatory:
`run`, `ssh`, `remote`, `ray`, `r2`, `mlflow`, `observability`,
`resumability`. Plus your pipeline-specific sections and the `ops:`
list.

**Why.** `StreamingRayExecutor.validate_contract`
([core/execution.py:444](src/training_signal_processing/core/execution.py#L444))
enforces that bindings, execution config, tracking context, and
artifact layout all resolve — and your `RecipeConfig` is what your
adapter projects those from. Missing sections → a
`PipelineContractError` at the start of the run, before any op
executes.

**How.** Two files:
[pipelines/example_echo/models.py](src/training_signal_processing/pipelines/example_echo/models.py)
defines the dataclasses,
[pipelines/example_echo/config.py](src/training_signal_processing/pipelines/example_echo/config.py)
loads + validates.

`RecipeConfig` at
[models.py:30](src/training_signal_processing/pipelines/example_echo/models.py#L30):

```python
@dataclass
class RecipeConfig:
    run_name: str
    config_version: int
    ssh: SshConfig
    remote: RemoteRuntimeConfig
    ray: RayConfig
    r2: R2Config
    input: InputConfig
    mlflow: MlflowConfig
    observability: ObservabilityConfig
    resumability: ResumeConfig
    ops: list[OpConfig]
```

All the framework types come from
[core/models.py](src/training_signal_processing/core/models.py); only
`InputConfig` and `ResumeConfig` are pipeline-local. `example_echo`'s
`InputConfig` is a single field — a list of `{source_id, message}`
dicts:

```python
@dataclass
class InputConfig:
    items: list[dict[str, Any]] = field(default_factory=list)
```

And two row types the ops + exporter all round-trip through:
`EchoTask` (input row) at
[models.py:45](src/training_signal_processing/pipelines/example_echo/models.py#L45)
and `EchoResult` (output row) at
[models.py:61](src/training_signal_processing/pipelines/example_echo/models.py#L61).

`config.py` delegates the mechanical YAML + dotted-override plumbing
to `core.config_loading.load_recipe_mapping`
([core/config_loading.py:17](src/training_signal_processing/core/config_loading.py#L17)).
The pipeline-local bits are the required-section list, the
`build_recipe_config` constructor, and `validate_recipe_constraints`.

```python
# config.py:58
def build_recipe_config(raw: dict[str, Any], config_path: Path) -> RecipeConfig:
    config_loading.require_sections(raw, config_path, REQUIRED_SECTIONS)
    validate_recipe_constraints(raw)
    ops = [config_loading.build_op_config(item) for item in raw["ops"]]
    return RecipeConfig(
        run_name=raw["run"]["name"],
        ...
        input=InputConfig(items=list(raw["input"].get("items", []))),
        ...
    )
```

`validate_recipe_constraints` at
[config.py:77](src/training_signal_processing/pipelines/example_echo/config.py#L77)
rejects empty `input.items` — the executor treats an empty input
manifest as a `PipelineContractError`, so the YAML validator should
catch it first with a clearer message.

---

## §3 Ops — the per-row chain

**What.** Ops are the unit of per-row (or per-batch) logic. Every
pipeline has exactly three stages in order: `prepare` → `transform`
→ `export`. You can have multiple `transform` ops; you can have a
`transform` op that is a filter. You cannot have zero `prepare` or
zero `export`.

**Why.** The split isolates your pipeline's per-row logic from the
framework's dispatch, resume, and observability. Ops run inside Ray
as `dataset.map_batches` transforms — lazy, concurrent, and
batched — and the rest of the system reads their output through a
stable row contract.

**How.** `pipelines/<new>/ops.py` holds your concrete `Op`
subclasses. Registration is automatic: `Op.__init_subclass__` at
[ops/base.py:22](src/training_signal_processing/ops/base.py#L22)
adds every concrete subclass to the global `REGISTERED_OP_TYPES`.
Each pipeline package imports its own `ops.py` from `__init__.py`, so
registration happens when the pipeline CLI or remote job imports the package.

Two gotchas the doc flags:

- **`op_name` is globally unique across all imported pipeline ops.** If
  you name your op `skip_existing`, it collides with OCR and raises
  `TypeError: Duplicate registered op name: skip_existing` at import
  time.
- **The pipeline package must import its ops module.** The template does
  this in `pipelines/<new>/__init__.py` with `from . import ops as ops`.

Stage templates live in
[ops/builtin.py](src/training_signal_processing/ops/builtin.py):
`SourcePreparationOp` (op_stage="prepare"), `SkipExistingFilter`
(op_stage="transform", filter), `BatchTransformOp`
(op_stage="transform", batch-level), `RowWiseMapperOp` (abstract,
row-at-a-time), plus the historically-named `MarkerOcrMapper` and
`ExportMarkdownMapper` which are in fact general row-wise templates
despite the OCR-flavored names.

`example_echo` has three ops:

```python
# pipelines/example_echo/ops.py:29
class PrepareEchoOp(SourcePreparationOp):
    op_name = "prepare_echo"

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        runtime = self.require_runtime()
        task = EchoTask.from_dict(row)
        output_r2_key = join_s3_key(
            join_s3_key(runtime.output_root_key, "outputs"),
            f"{task.source_id}.json",
        )
        return {
            "run_id": runtime.run_id,
            "source_id": task.source_id,
            "message": task.message,
            "echoed_at": "",
            "status": "pending",
            ...
        }
```

`PrepareEchoOp` converts the raw `EchoTask` input into the canonical
row shape. Every downstream op + the exporter round-trip through
this shape, so **fields are initialized with sane defaults
up front**. `status="pending"` means the transform hasn't run yet.

```python
# pipelines/example_echo/ops.py:54
class TimestampEchoOp(BatchTransformOp):
    op_name = "timestamp_echo"

    def process_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for row in batch:
            started = utc_isoformat()
            annotated = dict(row)
            annotated["started_at"] = started
            try:
                now = utc_isoformat()
                annotated["echoed_at"] = now
                annotated["status"] = "success"
                ...
            except Exception as exc:
                annotated["status"] = "failed"
                annotated["error_message"] = str(exc)
            output.append(annotated)
        return output
```

`TimestampEchoOp` promotes status to `"success"` (or `"failed"` on
exception). Row mutations are additive — we never remove keys —
because **non-idempotent mutation breaks resume semantics**: if a
batch re-runs after a crash, keys disappearing would confuse the
exporter's `EchoResult.from_dict(row)` call.

```python
# pipelines/example_echo/ops.py:77
class ExportEchoOp(RowWiseMapperOp):
    op_name = "export_echo"
    op_stage = "export"

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        if row.get("status") != "success":
            return row
        if not str(row.get("output_r2_key", "")).strip():
            row["status"] = "failed"
            row["error_message"] = "missing output_r2_key"
        return row
```

`ExportEchoOp` is a **gate op, not the actual writer**. It validates
that successful rows have the fields the exporter needs; the actual
R2 write happens once per batch in `EchoExporter.export_batch`
(§5) — not per row. This split is the framework's convention: ops
stay pure and parallel; I/O is centralized in the exporter, which
runs on the driver once per batch after Ray materializes. Writes go
through `self._put_bytes`, which is synchronous; the executor records
batch progress only after output writes return successfully.

`RowWiseMapperOp` is re-used directly here (instead of the OCR-flavored
`ExportMarkdownMapper`) to demonstrate that `op_stage` can be set
directly on any `Op` subclass.

---

## §4 Execute — `PipelineRuntimeAdapter`

**What.** `StreamingRayExecutor.run()` calls roughly ten methods on
your adapter — nine accessors that project your `RecipeConfig` into
runtime-shaped values, plus `load_input_rows` which reads the input
manifest JSONL (**see §7 — the submission adapter is what wrote
it**).

**Why.** The adapter is a dependency-injection seam: the executor
doesn't know what your pipeline is; it just asks for artifacts.
Keeping this layer purely declarative means the frozen
`StreamingRayExecutor` works for any pipeline without changing.

**How.** One file —
[pipelines/example_echo/runtime.py](src/training_signal_processing/pipelines/example_echo/runtime.py)
— defines the batch exporter and subclasses `PipelineRuntimeAdapter`.
Key adapter methods:

```python
# runtime.py:62
def load_input_rows(self) -> list[dict[str, object]]:
    return self.object_store.read_jsonl(self.bindings.input_manifest_key)
```

The ABC at
[executor.py:71](src/training_signal_processing/core/execution.py#L71)
declares `list[dict[str, Any]]` as the return type, but all three
other pipelines override with `list[dict[str, object]]` —
`example_echo` follows that concrete convention. The
`input_manifest_key` is set on `RuntimeRunBindings` by the
submission adapter's `prepare_new_run` (§7).

```python
# runtime.py:41
def get_tracking_context(self) -> RuntimeTrackingContext:
    return RuntimeTrackingContext(
        enabled=self.config.mlflow.enabled,
        tracking_uri=self.config.mlflow.tracking_uri,
        experiment_name=self.config.mlflow.experiment_name,
        run_name=self.config.run_name,
        executor_type=self.config.ray.executor_type,
        batch_size=self.config.ray.batch_size,
        concurrency=self.config.ray.concurrency,
        target_num_blocks=self.config.ray.target_num_blocks,
    )
```

Note: `example_echo` does NOT override
`resolve_transform_resources`. The base default at
[executor.py:101](src/training_signal_processing/core/execution.py#L101)
returns an empty `RayTransformResources` (no GPU, no CPU pin).
Override only if one of your ops needs a GPU — OCR does
([pipelines/ocr/runtime.py:105](src/training_signal_processing/pipelines/ocr/runtime.py#L105))
to give `marker_ocr` the configured `marker_ocr_resources`.

Use `ObjectStorePipelineRuntimeAdapter` for normal R2-backed pipelines. It owns
the mechanical projections from `RecipeConfig` + `bindings` + `object_store`;
your runtime class usually implements only `source_root_key`, `build_exporter`,
and `build_completion_tracker`, plus `resolve_transform_resources` when an op
needs explicit Ray resources.

---

## §5 Materialize — `Exporter`

**What.** `Exporter.export_batch(batch_id, rows)` runs once per
batch after Ray materializes. It's where actual I/O happens — write
outputs for each successful row. `finalize_run(run_state)` is an optional
in-memory lifecycle hook; the base `RayExporter` does not write a final
checkpoint object.

**Why.** Separating "what goes out" from "what ops compute" lets ops
stay pure and parallel. The exporter is called from the main
executor (driver) thread, not inside a Ray worker — so it's the one
place in the pipeline that can safely do bulk I/O without shipping
credentials or clients across process boundaries.

**How.** The exporter lives beside the runtime adapter in
[pipelines/example_echo/runtime.py](src/training_signal_processing/pipelines/example_echo/runtime.py):

```python
# runtime.py
def export_batch(self, batch_id: str, rows: list[dict[str, object]]) -> ExportBatchResult:
    output_keys: list[str] = []
    for row in rows:
        result = EchoResult.from_dict(row)
        if result.status != "success":
            continue
        payload = {
            "source_id": result.source_id,
            "message": result.message,
            "echoed_at": result.echoed_at,
        }
        self._put_bytes(
            result.output_r2_key,
            json.dumps(payload, sort_keys=True).encode("utf-8"),
        )
        output_keys.append(result.output_r2_key)
    return ExportBatchResult(
        batch_id=batch_id,
        row_count=len(rows),
        output_keys=output_keys,
    )
```

Three conventions:

- **Filter by `status`.** The exporter only writes success rows to
  the output namespace; failed rows contribute to in-memory progress
  and optional observability, not durable checkpoint files.
- **`output_r2_key` was computed in `PrepareEchoOp`.** The
  exporter trusts the pre-populated key rather than rebuilding it,
  so the convention is "prepare decides where things land; exporter
  just puts them there."
- **Write via `self._put_bytes`.** The helper on
  [`RayExporter`](src/training_signal_processing/core/execution.py)
  calls synchronous `self.object_store.write_bytes`. The exporter
  returns only after output objects are durable, so failed writes
  surface as run failures before the executor records in-memory batch progress.

---

## §6 Resume — `OutputCompletionTracker`

**What.** `OutputCompletionTracker` lists materialized output objects for the
run and maps each expected output back to the source key in the input manifest.
If the expected output exists, the source key is considered complete.

**Why.** The output object is the durable fact the pipeline actually cares
about. Avoiding separate batch manifests, event objects, `run_state.json`, and
`run.json` keeps resume behavior easy to reason about: a restart skips exactly
the sources whose outputs are already present.

**How.** Put a tiny tracker class beside your exporter in
`pipelines/<new>/runtime.py`. Implement:

```python
def source_key_for_input(self, row: dict[str, object]) -> str: ...
def output_key_for_input(self, row: dict[str, object], layout: RunArtifactLayout) -> str: ...
def output_listing_prefix(self, layout: RunArtifactLayout) -> str: ...
```

The executor calls `completed_source_keys(input_rows, artifact_layout,
allow_overwrite)` at startup. If `allow_overwrite` is true, the completed set is
empty. Otherwise the set is handed to the runtime context and consulted by skip
ops such as OCR's `SkipExistingDocumentsOp`.

---

## §7 Submit — `SubmissionAdapter`

**What.** Builds a `PreparedRun` describing what to sync, how to
bootstrap, how to invoke remotely, and what artifacts were written.
The framework's `SubmissionCoordinator` drives the rest (rsync, SSH
bootstrap, local upload waits, and detached SSH launch).

**Why.** Submission orchestration is pipeline-agnostic; the adapter's
job is narrow and purely declarative. You provide the input manifest
+ recipe JSON in R2, the bootstrap shell, and the remote invocation
command. The coordinator at
[core/submission.py:492](src/training_signal_processing/core/submission.py#L492)
takes it from there. The coordinator is fire-and-forget
(`.submit()` returns a `LaunchHandle` once `setsid` writes `job.pgid`
on the pod). Your adapter's declarative output is the bootstrap,
invocation, artifacts, and any local input upload spec.

**How.** The common `SubmissionAdapter.prepare_new_run` and
`prepare_resume_run` methods own run-id creation, control keys,
manifest/config writes, and `PreparedRun` construction. Your concrete
adapter implements the pipeline hooks: `pipeline_family`,
`build_new_run_manifest`, `load_resolved_recipe_mapping`,
`build_bootstrap_spec`, and `build_invocation_spec`.

The worked example lives in
[pipelines/example_echo/submission.py](src/training_signal_processing/pipelines/example_echo/submission.py).

**`build_new_run_manifest`** materializes the inputs.
It iterates `config.input.items`, builds `EchoTask` dataclasses, and
writes rows through the core template as a JSONL control manifest at
`<output_prefix>/<run_id>/control/input_manifest.jsonl`. The core
template also writes the fully-resolved recipe JSON so the remote
process can rebuild the config from R2:

```python
def build_new_run_manifest(
    self,
    *,
    artifact_store: ArtifactStore,
    run_id: str,
    dry_run: bool,
) -> SubmissionManifest:
    del artifact_store, run_id, dry_run
    tasks = self.build_tasks()
    return SubmissionManifest(
        rows=[task.to_dict() for task in tasks],
        discovered_items=len(tasks),
    )
```

**Honesty paragraph — input discovery.** `example_echo.input.items`
is a list of dicts **hardcoded in the YAML**. That's the minimum
contract-satisfying pattern; real pipelines discover inputs from an
external system. Compare with OCR's richer pattern at
[pipelines/ocr/submission.py](src/training_signal_processing/pipelines/ocr/submission.py)
(PDF glob + file-size sort). The framework only requires that your
adapter writes `input_manifest.jsonl` before the remote job starts;
how you build that list is your pipeline's business.

**`build_bootstrap_spec` (`submission.py:116`)** returns a
`BootstrapSpec` with the shell command the remote runs once before
execution:

```python
def build_bootstrap_spec(self) -> BootstrapSpec:
    command = " && ".join([
        "command -v uv >/dev/null",
        f"uv python install {shlex.quote(self.config.remote.python_version)}",
        "uv sync --group remote_ocr --no-dev",
    ])
    return BootstrapSpec(command=command)
```

Bare minimum: install uv + the pinned Python + `uv sync` the OCR
dep group. OCR adds `--group model`
([pipelines/ocr/submission.py:141](src/training_signal_processing/pipelines/ocr/submission.py#L141))
for torch; `example_echo` doesn't need it.

**`build_invocation_spec` (`submission.py:126`)** returns the remote
command. Three things matter:
- Non-OCR pipelines invoke `python -m
  training_signal_processing.pipelines.<name>.cli remote-job`.
  OCR keeps the stable root command `python -m
  training_signal_processing.main ocr-remote-job`.
- The env includes R2 exports from `artifact_store.build_remote_env`
  when the pipeline needs remote R2 access.
- MLflow is optional and direct-only: set `mlflow.enabled=true` plus
  a `mlflow.tracking_uri` reachable from the logging process. The
  framework does not open SSH tunnels.

Submission does not parse remote stdout. The remote job writes outputs
to R2; the detached log contains any printed JSON summary.

No rclone, no local file upload, no file-size sort — `example_echo`
keeps the submission adapter minimal. For a pipeline that uploads
local binaries, crib the rclone block at
[pipelines/ocr/submission.py:263-353](src/training_signal_processing/pipelines/ocr/submission.py#L263).

---

## §8 CLI wiring

**What.** One `cli.py` file. Operators run `validate`, `run`, and
`resume` locally; the remote box runs the same module's `remote-job`
subcommand.

**Why.** `main.py` is frozen
([ARCHITECTURE.md §3](ARCHITECTURE.md)) and only carries OCR's CLI
commands. Every new pipeline needs its own standalone CLI because
you cannot edit `main.py` to add yours.

**How.**

`cli.py` registers the remote subcommand with
`build_remote_job_cli`:

```python
# pipelines/example_echo/cli.py
def build_adapter(
    config: RecipeConfig,
    bindings: RuntimeRunBindings,
    object_store: R2ObjectStore,
):
    return build_runtime_adapter(config, bindings, object_store)


cli.add_command(
    build_remote_job_cli(
        recipe_loader=build_recipe_config,
        adapter_factory=build_adapter,
    ),
    name="remote-job",
)
```

`build_remote_job_cli` at
[core/remote.py](src/training_signal_processing/core/remote.py)
handles the R2 bootstrap config fetch, `RuntimeRunBindings`
construction, and executor start. You pass the recipe loader + the
adapter factory; the runtime handles the rest.

The local commands take `--config` (multiple — for overlay files)
and `--set` (dotted-path overrides). Copy the example_echo CLI shape
and change the adapter + recipe-loader references.

```python
# cli.py:28
@cli.command("validate")
@click.option(
    "--config", "config_paths",
    required=True, multiple=True,
    type=click.Path(path_type=Path),
)
@click.option("--set", "overrides", multiple=True)
def validate_command(config_paths: tuple[Path, ...], overrides: tuple[str, ...]) -> None:
    try:
        base_path, overlay_paths = config_paths[0], config_paths[1:]
        config = load_recipe_config(base_path, list(overrides), overlay_paths=overlay_paths)
        pipeline = RegisteredOpRegistry().resolve_pipeline(config.ops)
        ...
```

The `submit_remote_pipeline` helper at
[cli.py:107](src/training_signal_processing/pipelines/example_echo/cli.py#L107)
assembles the `SubmissionCoordinator` and calls `.submit()`. This is
the same pattern OCR's `main.py` uses — only the adapter class
changes.

**Return-value contract.** `.submit()` returns a
`SubmissionResult(mode='launched', launch: LaunchHandle)`. Wire the
CLI to print `.to_safe_dict()` as before; the launch handle gives the
operator the remote log path and pgid path. Run completion is read
from R2 state, not SSH stdout.

---

## §9 Recipe + first dry run

The recipe at
[pipelines/example_echo/configs/baseline.yaml](src/training_signal_processing/pipelines/example_echo/configs/baseline.yaml)
has all eight framework sections and `example_echo`-local fields.
Relevant slice:

```yaml
ray:
  executor_type: ray
  batch_size: 2
  concurrency: 1
  target_num_blocks: 1

input:
  items:
    - source_id: alpha
      message: hello from alpha
    - source_id: beta
      message: hello from beta
    - source_id: gamma
      message: hello from gamma

ops:
  - name: prepare_echo
    type: mapper
  - name: timestamp_echo
    type: mapper
  - name: export_echo
    type: mapper
```

Validate + dry-run in order:

```
uv run ruff check src/training_signal_processing/pipelines/example_echo
uv run lint-imports
uv run --group remote_ocr python -m training_signal_processing.pipelines.example_echo.cli validate \
  --config src/training_signal_processing/pipelines/example_echo/configs/baseline.yaml
uv run --group remote_ocr python -m training_signal_processing.pipelines.example_echo.cli run \
  --config src/training_signal_processing/pipelines/example_echo/configs/baseline.yaml --dry-run
```

Expected on `validate`:

```
Validated example_echo recipe: src/.../baseline.yaml
Run name: example-echo-baseline
Executor type: ray
Declared items: 3
Declared ops: 3
Resolved pipeline: prepare_echo, timestamp_echo, export_echo
```

Expected on `run --dry-run`: a JSON `"mode": "dry_run"` object
showing the planned `remote_command`, `sync_paths`, `bootstrap_command`,
etc.

### Wire-up checklist

Before running a real (non-dry) remote run, tick all of:

- [ ] `pipelines/<new>/__init__.py` imports `ops.py` so concrete op classes register.
- [ ] Each concrete `Op` subclass declares `op_name` + `op_stage`
      (exactly one `prepare`, zero or one `export`, ≥1 `transform`).
- [ ] Every `op_name` is globally unique — no collisions with OCR or example_echo.
- [ ] Your recipe's `ops:` list references the `op_name` strings and
      sets `type:` per op.
- [ ] `uv run lint-imports` passes — shared `core/` and `ops/` code must not
      import back into `pipelines/`.
- [ ] Config loading fails clearly for missing required sections and
      obviously-bad pipeline-owned inputs.
- [ ] No frozen file was edited
      ([ARCHITECTURE.md §3](ARCHITECTURE.md)).

---

## §10 Out of scope

Each item below is deliberately excluded from this doc, with a
one-sentence rationale so you know where to look instead.

- **Writing pipeline-level tests.** Follow the pattern at
  [tests/test_ocr_upload_refactor.py](tests/test_ocr_upload_refactor.py);
  deferred to keep this doc bounded.
- **New dep groups in `pyproject.toml`.** Maintainer review; most
  new pipelines reuse `remote_ocr`.
- **Custom remote images / dstack templates.** Covered in
  [README.md](README.md) §Remote Image.
- **New `Exporter` backends beyond R2.** The `ArtifactStore` ABC
  supports it, but no non-R2 concrete exists — defer until needed.
- **Modifying frozen files.** See
  [ARCHITECTURE.md §3](ARCHITECTURE.md) for the list and the
  "additions OK, signature changes need approval" rule.
- **MLflow / observability semantics.** Covered in
  [ARCHITECTURE.md §10](ARCHITECTURE.md); your adapter just exposes
  `tracking_context` and the runtime handles the rest.
- **`allow_overwrite` + resume-replay interactions.**
  `OutputCompletionTracker.completed_source_keys` returns an empty
  set when overwrite is allowed, so skip ops reprocess every source.
- **Custom `DatasetBuilder`.** The default
  `ConfiguredRayDatasetBuilder`
  ([core/dataset.py:132](src/training_signal_processing/core/dataset.py#L132))
  works for everything except custom pyarrow ingestion.
- **Per-op resource overrides (`resolve_transform_resources`).**
  Skipped for `example_echo` because all three ops are CPU-only;
  OCR's override at
  [pipelines/ocr/runtime.py:105](src/training_signal_processing/pipelines/ocr/runtime.py#L105)
  is the canonical example.
- **YAML overlay + `--set` composition.** Covered in
  [ARCHITECTURE.md §5](ARCHITECTURE.md); `example_echo`'s CLI
  already supports both.

---

For operator-facing questions (start infra, run a pipeline,
troubleshoot a failure), see [ONBOARDING.md](ONBOARDING.md).
