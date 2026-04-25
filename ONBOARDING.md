# Onboarding

You have a clone of `training-signal-processing` and a shell. This
doc gets you from there to (1) a finished remote OCR run on a
freshly provisioned GPU box, then (2) a new pipeline of your own
extending the system.

Companion docs:

- [README.md](README.md) — command reference, experiment
  workflow guardrails, MLflow verification snippets.
- [ARCHITECTURE.md](ARCHITECTURE.md) — layers, contracts,
  frozen-file invariants, OCR walkthrough with diagrams.
- [EXTENDING.md](EXTENDING.md)
  — pre-existing op-authoring guide.

---

## Table of contents

- [§0. Pre-requisites (once per laptop)](#0-pre-requisites-once-per-laptop)
- [§1. Use case 1 — start new infra + run existing pipeline](#1-use-case-1--start-new-infra--run-existing-pipeline)
  - [§1.1 Path A — RunPod canonical bootstrap](#11-path-a--runpod-canonical-bootstrap)
  - [§1.2 Path B — bring your own SSH box](#12-path-b--bring-your-own-ssh-box)
  - [§1.3 Run the OCR pipeline](#13-run-the-ocr-pipeline)
  - [§1.4 Experiment without touching code](#14-experiment-without-touching-code)
- [§2. Use case 2 — add a new pipeline](#2-use-case-2--add-a-new-pipeline)
  - [§2.1 Architecture refresher](#21-architecture-refresher)
  - [§2.2 File-layout template](#22-file-layout-template)
  - [§2.3 Worked example — clone OCR into `example_echo`](#23-worked-example--clone-ocr-into-example_echo)
  - [§2.4 Wire-up checklist](#24-wire-up-checklist)
  - [§2.5 Explicit out-of-scope](#25-explicit-out-of-scope)
- [§3. Troubleshooting](#3-troubleshooting)
- [§4. See also](#4-see-also)

---

## §0. Pre-requisites (once per laptop)

Set these up once; you won't touch them again.

### Tooling

- **`uv`** — Astral's Python package manager. Pins Python and
  installs deps from `pyproject.toml`. `uv --version` must return a
  recent build (`>= 0.4`).
  ```
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **SSH key.** `~/.ssh/id_ed25519` (+ `.pub`) must exist — the
  RunPod bootstrap script hardcodes this path. If your key is
  passphrase-protected, run `ssh-add` in the shell you'll use for
  `start_runpod_5090.sh`; the script doesn't prompt.
  ```
  ls ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub
  ```
- **(optional) rclone** — only OCR uses it, for the local PDF upload
  to R2 before the detached remote job starts. If you skip it, OCR
  `run` without `--dry-run` will error at the rclone discovery step.
  Install from [rclone.org/install](https://rclone.org/install/).
- **(optional) MLflow server** — only when the logging process can
  reach `mlflow.tracking_uri` directly:
  ```
  uv run mlflow server --host 127.0.0.1 --port 5000
  ```
  This writes an `mlflow.db` in the CWD. A remote pod cannot reach
  your laptop's `127.0.0.1`, and the framework no longer creates SSH
  reverse tunnels. The default OCR path uses R2 output objects and
  `/root/ocr-jobs/<run_id>/job.log` for durable evidence. See §3
  *Running-job runbook* for the one-liners.

### Credentials

All credentials live under `infra/credentials/`. Create the subdirs
manually (they aren't tracked):

```
mkdir -p infra/credentials/runpod
```

- **RunPod API key** — the RunPod script `cat`s a file (not an env
  var). Paste your key at:
  ```
  infra/credentials/runpod/runpod_api_key
  ```
- **Cloudflare R2 credentials** — an env-style file the recipe
  points at via `r2.config_file`. Put it wherever you want (the
  sample recipes reference a literal path `r2`). Required keys:
  ```
  R2_BUCKET_NAME=<your-bucket>
  AWS_ACCESS_KEY_ID=<r2-access-key-id>
  AWS_SECRET_ACCESS_KEY=<r2-secret>
  AWS_DEFAULT_REGION=<your-region>
  MLFLOW_S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
  ```
  All five are required. R2's `auto` region often needs an explicit
  resolution — check the bucket detail page in the Cloudflare
  dashboard and use the concrete region.
- **`infra/credentials/verda/`** — a separate provider slot that
  isn't needed for the OCR path. Ignore unless you're wiring a
  non-RunPod backend.
- **`infra/current-machine`** — gitignored by design. It's written
  by `start_runpod_5090.sh` (or by hand, §1.2). Its absence means
  "no current pod."

### Install + sanity

```
uv sync --group remote_ocr
uv run ruff check src
uv run lint-imports          # import-linter contract; must pass
```

If `lint-imports` fails, something imported a pipeline module from
the shared layer — see [ARCHITECTURE.md §2](ARCHITECTURE.md) for the
contract.

---

## §1. Use case 1 — start new infra + run existing pipeline

Two provisioning paths. Both end at the same state: a populated
`infra/current-machine` that the OCR CLI reads automatically.

### §1.1 Path A — RunPod canonical bootstrap

Fully automated: boots a RunPod community RTX 5090, polls until SSH
is accepting connections, and writes `infra/current-machine`.

#### Step-by-step

1. Put the API key in place (§0 already covered this):
   ```
   cat infra/credentials/runpod/runpod_api_key   # sanity
   ```
2. Run the bootstrap script:
   ```
   infra/start_runpod_5090.sh
   ```
   Expected output — one line per pod:
   ```
   <pod_id>	ssh -i ~/.ssh/id_ed25519 -p <port> root@<ip>
   ```
   Plus `infra/current-machine` + `infra/.current-machine` populated
   with that same `ssh …` command.
3. **Override knobs via env vars** (all optional):
   - `RUNPOD_IMAGE` — default
     `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. Don't switch
     to a CUDA 13 image; the repo's torch pin is CUDA 12.8.
   - `RUNPOD_CONTAINER_DISK_GB` — default 80.
   - `RUNPOD_NAME` — default
     `training-signal-processing-5090-ssh`. Matters only for
     `kill_runpod_5090.sh`, which finds pods by name.
   - `RUNPOD_COUNT` — default 1. Multi-pod runs are uncommon.
4. **(optional smoke test)** The probe script runs 5 remote checks
   (nvidia-smi, CUDA driver, torch, Ray-GPU, marker-smoke):
   ```
   infra/probe_runpod_5090.sh
   ```
   ⚠️ **Probe expects `/root/training-signal-processing/.venv` to
   exist** — that's the remote venv built by the `uv sync` step of
   an OCR run's bootstrap. Run `probe_runpod_5090.sh` **after** your
   first OCR run, not before.
5. **Tear down** when done:
   ```
   infra/kill_runpod_5090.sh
   ```
   Finds pods matching `RUNPOD_NAME`, deletes them via the RunPod
   REST API, and removes `infra/current-machine`.

### §1.2 Path B — bring your own SSH box

If you have a GPU box provisioned some other way (dstack, another
cloud, a physical machine on your LAN), you can use it by writing
`infra/current-machine` by hand.

#### Step-by-step

1. Confirm SSH key-auth works without a password prompt:
   ```
   ssh -o StrictHostKeyChecking=accept-new user@host true
   ```
   That command also primes `~/.ssh/known_hosts` for the
   non-interactive runs later.
2. **Write `infra/current-machine` in the exact format the parser
   requires:**
   ```
   ssh -i /path/to/key -p 22 user@host
   ```
   Strict rules enforced at
   [core/config_loading.py:141](src/training_signal_processing/core/config_loading.py#L141):
   - Must start with literal `ssh`.
   - `-p <port>` is mandatory. **Include `-p 22` even though that's
     the SSH default** — without it, the parser raises
     `ValueError: ... must contain both ssh host and -p <port>`.
   - Must contain a `user@host` token.
   - The `-i` flag and the username part are accepted but **not
     used by the parser** — they are parsed out but don't reach the
     recipe. See caveat below.
3. **Caveat — host and port only.** The parser extracts *only*
   `host` and `port` from this file; `ssh.user` and
   `ssh.identity_file` come from the recipe YAML. So in your recipe
   (or a CLI overlay), set:
   ```yaml
   ssh:
     user: root             # or whatever your SSH user is
     identity_file: ~/.ssh/id_ed25519
   ```
   ...or override at invocation time with
   `--set ssh.user=root --set ssh.identity_file=~/.ssh/id_ed25519`.

### §1.3 Run the OCR pipeline

With `infra/current-machine` populated (either path), you can run
the OCR CLI. It auto-resolves the SSH target unless you explicitly
override `ssh.host` / `ssh.port` in the recipe.

1. **Validate first** (parses the recipe, resolves ops, reports the
   schema — no remote side effects):
   ```
   uv run --group remote_ocr python -m training_signal_processing.main validate \
     --config src/training_signal_processing/pipelines/ocr/configs/baseline.yaml
   ```
   Expected on success:
   ```
   Validated remote OCR recipe: src/.../baseline.yaml
   Run name: marker-ocr-baseline
   Executor type: ray
   Declared ops: 3
   Resolved pipeline: prepare_pdf_document, skip_existing, marker_ocr
   ```
2. **Dry run** (writes control artifacts to R2 but doesn't execute
   remotely):
   ```
   uv run --group remote_ocr python -m training_signal_processing.main run \
     --config src/training_signal_processing/pipelines/ocr/configs/baseline.yaml \
     --dry-run
   ```
   Expected output: a JSON object with `"mode": "dry_run"`. Nothing
   ran on the remote; the `invocation.command` field shows exactly
   what would have been executed.
3. **Real run** (takes minutes; materializes outputs to R2):
   ```
   uv run --group remote_ocr python -m training_signal_processing.main run \
     --config src/training_signal_processing/pipelines/ocr/configs/baseline.yaml
   ```
   What happens, in order:
   1. Local: discover PDFs, build manifest, write manifest + resolved
      recipe to R2 (`control/input_manifest.jsonl`,
      `control/recipe.json`).
   2. Local: rsync `src/`, `pyproject.toml`, `uv.lock`, `config/`
      to the remote box over SSH.
   3. Remote: bootstrap shell runs `uv sync --group remote_ocr
      --group model --frozen`.
   4. Local: rclone starts the parallel PDF upload to R2 in
      background.
   5. Local: wait for the source upload to finish so the detached
      remote never races ahead of its R2 inputs.
   6. Local: `launch_detached` SSHes to the pod, writes a
      `launch.sh` under `/root/ocr-jobs/<run_id>/`, and spawns it
      under `setsid`. The session leader records its own pgid in
      `.../job.pgid` before exec'ing the launcher. Remote stdout
      goes to `.../job.log`. The local CLI then prints a
      `LaunchHandle` JSON and exits. **Exit code 0 means
      *launched successfully*, not *run complete*** — check
      completion by listing OCR markdown outputs in R2 or tailing the
      remote log.
   7. Remote: `uv run python -m ...main ocr-remote-job …` runs
      under PID 1 (reparented to init), loops through the ops, and
      writes final markdown bytes to R2.
      Because it is detached, closing the terminal or losing the
      SSH link does not affect it.
4. **Watch progress.** The local CLI is fire-and-forget: it prints a
   `LaunchHandle` and exits after the detached process starts. Progress
   is durable in R2 outputs and verbose in the remote log. Watch the log with
   `ssh <pod> 'tail -F /root/ocr-jobs/<run_id>/job.log'` and list
   `dataset/processed/pdf_ocr/<run_id>/markdown/` to count completed
   documents.

   MLflow is optional. Enable it only when `mlflow.tracking_uri` is
   directly reachable from the process doing the logging; the framework
   does not forward local MLflow through SSH.
5. **Resume an interrupted run.** If the executor died mid-batch,
   pass the `run_id` (a UTC timestamp like `20260423T132754Z`) that
   was printed in the original run's JSON output:
   ```
   uv run --group remote_ocr python -m training_signal_processing.main resume \
     --config src/training_signal_processing/pipelines/ocr/configs/baseline.yaml \
     --run-id 20260423T132754Z
   ```
   `OcrCompletionTracker` lists existing markdown objects and feeds their
   source keys to `SkipExistingDocumentsOp`, so already-successful PDFs are
   filtered out without being reprocessed. Once output discovery finishes,
   the remote log shows Ray startup and new `[batch:finish]` lines.

   > ⚠ **Double-launch hazard.** `resume --run-id X` does **not**
   > currently detect a still-live run on the pod; it will blindly
   > re-launch and overwrite `/root/ocr-jobs/X/job.pgid`, orphaning
   > the old session's Ray cluster. Before resuming, verify the old
   > run is stopped:
   > ```
   > ssh <pod> 'f=/root/ocr-jobs/X/job.pgid; test -f $f && kill -0 -$(cat $f) 2>/dev/null && echo ALIVE || echo NOT_RUNNING'
   > ```
   > Only resume if this prints `NOT_RUNNING`. The `test -f` guard
   > matters for first-ever resumes where the pgid file does not yet
   > exist. A pgid-aware resume that reconciles stale `running`
   > states lands in a follow-up PR.

### §1.4 Experiment without touching code

Two config-only mechanisms — never edit logic for a tuning run.
Full details in [ARCHITECTURE.md §5](ARCHITECTURE.md).

1. **YAML overlay files** (landed in commit `25673fb`). Drop a new
   YAML under `pipelines/ocr/configs/` (only the keys you want to
   override), then:
   ```
   uv run --group remote_ocr python -m training_signal_processing.main run \
     --config src/training_signal_processing/pipelines/ocr/configs/baseline.yaml \
     --config src/training_signal_processing/pipelines/ocr/configs/experiment.example.yaml
   ```
   Order matters — later files win on scalar conflicts; mappings
   are deep-merged; lists (including `ops`) are replaced wholesale.
2. **`--set key.path=value` overrides** for one-off tweaks:
   ```
   uv run --group remote_ocr python -m training_signal_processing.main run \
     --config src/training_signal_processing/pipelines/ocr/configs/baseline.yaml \
     --set ray.batch_size=4 --set input.max_files=50
   ```

**Guardrail** (from [README.md §Experiment Workflow](README.md)): if
a tuning knob isn't already exposed in YAML, stop and escalate to
make it configurable before running. Do not add one-off code
branches for experiments.

---

## §2. Use case 2 — add a new pipeline

**See [EXTENDING.md](EXTENDING.md)** for the full walkthrough. It's a
top-down educational tour through the six building blocks (config,
ops, execute, materialize, resume, submit, plus CLI wiring) with
complete runnable code from the committed
[pipelines/example_echo/](src/training_signal_processing/pipelines/example_echo/)
reference template.

TL;DR:

- Create `pipelines/<new>/` with `models.py`, `config.py`, `ops.py`,
  `runtime.py`, `submission.py`, `cli.py`, and configs.
- Ops self-register when `pipelines/<new>/__init__.py` imports `ops.py`.
  `op_name` is globally unique across imported pipeline ops.
- `main.py` is frozen; each new pipeline gets its own `cli.py` with
  a `remote-job` command — mirror
  [pipelines/example_echo/cli.py](src/training_signal_processing/pipelines/example_echo/cli.py).
- Runnable means `ruff` + `lint-imports` + `cli validate` +
  `cli run --dry-run` all pass. Remote execution drags in
  RunPod / R2 / bootstrap specifics covered in §1 above.


---

## §3. Troubleshooting

Only errors the code actually raises, with the fix for each.

- **`ValueError: <path> must contain both ssh host and -p <port>: <text>`**
  Your `infra/current-machine` is missing either the port flag or
  the `user@host` token.
  *Fix:* rewrite as `ssh -i <key> -p <port> user@host` (see §1.2).

- **`ValueError: <path> is empty; expected an ssh command.`**
  Your `infra/current-machine` exists but has no content.
  *Fix:* regenerate via `infra/start_runpod_5090.sh`, or write by
  hand per §1.2.

- **`ValueError: <path> must start with an ssh command: <text>`**
  The file's first token isn't `ssh`. Often caused by copying the
  pod connection string without the `ssh` prefix.
  *Fix:* ensure the file begins with the literal word `ssh`.

- **RunPod bootstrap script prints `Timed out waiting for SSH on <pod_id>`**
  The pod was created but isn't accepting SSH within 5 minutes —
  usually a slow image pull or capacity hiccup.
  *Fix:* re-run in a minute. If it keeps happening, check RunPod's
  community capacity status and consider a different image.

- **`probe_runpod_5090.sh` fails with `ModuleNotFoundError` for
  torch/ray/marker**
  The remote `.venv` hasn't been bootstrapped yet — the probe
  expects
  `/root/training-signal-processing/.venv/bin/python` to exist.
  *Fix:* run one OCR run first (its bootstrap phase installs the
  venv), then re-run `probe_runpod_5090.sh`.

- **`botocore.exceptions.ClientError: InvalidAccessKeyId`**
  Your `r2` env file (referenced via `r2.config_file` in the
  recipe) is missing keys or has wrong values.
  *Fix:* verify all five keys are present — `R2_BUCKET_NAME`,
  `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  `AWS_DEFAULT_REGION`, `MLFLOW_S3_ENDPOINT_URL`.

- **`botocore.exceptions.ClientError: PermanentRedirect` or `BucketRegionError`**
  The `AWS_DEFAULT_REGION` in your `r2` env file doesn't match the
  actual region of your R2 bucket. Cloudflare's `auto` region
  often needs explicit resolution.
  *Fix:* look up your bucket's region in the R2 dashboard and put
  the concrete value (e.g. `wnam`, `enam`) into the env file.

- **`ValueError: Recipe missing required sections in <path>: <list>`**
  Recipe YAML is missing one or more top-level sections your
  pipeline's `REQUIRED_SECTIONS` list declares.
  *Fix:* cross-check against
  [config/remote_ocr.sample.yaml](config/remote_ocr.sample.yaml)
  or the sample for your pipeline; add the missing sections.

- **Resume errors out with "recipe not found" or a missing object
  key when passing `--run-id`**
  The `run_id` you passed doesn't match any run under
  `r2.output_prefix`. Typical causes: you changed `r2.bucket` or
  `r2.output_prefix` between the initial run and the resume; you
  swapped to a different recipe.
  *Fix:* confirm `r2.bucket` + `r2.output_prefix` match exactly.

- **Resume errors with `ray.async_upload was removed` or
  `Reverse-tunnel MLflow was removed`**
  The run was created by an older code version, so its R2
  `control/recipe.json` still contains removed config keys. Current
  code intentionally rejects those keys so stale behavior does not
  silently continue.
  *Fix:* back up the original R2 recipe object, remove
  `ray.async_upload`, remove `mlflow.local_tracking_uri` and
  `mlflow.remote_tunnel_port`, and keep `mlflow.enabled=false` unless
  you have a directly reachable `mlflow.tracking_uri`. Then rerun
  `resume --run-id <run_id>`.

- **Remote bootstrap fails during `uv sync --group remote_ocr --group model --frozen`**
  The remote image's Python doesn't match `pyproject.toml`'s pin,
  or `uv` can't build a wheel for the image's platform.
  *Fix:* use the default base image
  (`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`) or verify
  your custom image has Python available at the pinned version.
  Adjust `remote.python_version` in your recipe only as a last
  resort.

- **`ops.py` added but `validate` reports "no op registered for name X"**
  Either `pipelines/<new>/__init__.py` is not importing `ops.py`, or the
  class's `op_name` attribute doesn't match the recipe's `ops:` entry.
  *Fix:* import `ops.py` from the package `__init__.py`, or correct
  the `op_name` typo.

<!-- TRANSITIONAL: remove this subsection when ocr-remote CLI lands in PR #2b. Grep anchor: OPERATOR_RUNBOOK_MANUAL_SSH -->
### §3.X Running-job runbook (manual SSH until `ocr-remote` CLI lands)  <!-- OPERATOR_RUNBOOK_MANUAL_SSH -->

Because PR #1 made the launcher fire-and-forget, the local CLI no
longer tails the remote job. Until the `ocr-remote status / tail /
stop / wait` CLI ships (PR #2b), use these SSH one-liners against
the pod running the job. All commands assume `infra/current-machine`
is populated and `<run_id>` is the UTC timestamp id.

**Watch live:**
```
ssh <pod> 'tail -F /root/ocr-jobs/<run_id>/job.log'
```

**Status probe (distinguishes ALIVE from NOT_RUNNING and from
"never launched"):**
```
ssh <pod> 'f=/root/ocr-jobs/<run_id>/job.pgid; test -f $f && kill -0 -$(cat $f) 2>/dev/null && echo ALIVE || echo NOT_RUNNING'
```

**Stop cleanly** — SIGTERM the whole process group; SIGKILL on
timeout. Because `setsid` made the session leader's pid equal to the
pgid, the negative-target kill signals the Ray driver and workers
together:
```
ssh <pod> 'p=$(cat /root/ocr-jobs/<run_id>/job.pgid); kill -TERM -$p; sleep 10; kill -KILL -$p 2>/dev/null || true'
```

**Progress.** MLflow is no longer forwarded through SSH. Use R2
output objects plus the detached remote log for durable progress;
enable MLflow only with a directly reachable `mlflow.tracking_uri`.
<!-- /TRANSITIONAL -->

---

## §4. See also

- [README.md](README.md) — command reference and MLflow
  verification snippets.
- [ARCHITECTURE.md](ARCHITECTURE.md) — full layered architecture,
  ABC contracts, and the OCR end-to-end walkthrough with diagrams.
- [EXTENDING.md](EXTENDING.md)
  — op-authoring guide and stage-template reference.
- [pyproject.toml](pyproject.toml) — dependency groups
  (`[dependency-groups]`) and the import-linter contract
  (`[tool.importlinter]`).
