# Streaming Ray-Only Remote OCR

## What This Adds

This repo now includes a small remote OCR pipeline with:
- YAML-first config
- laptop-side Click CLI
- single-node remote `ray.data` execution
- Marker OCR
- R2 input/output
- R2-backed completion tracking through materialized markdown outputs
- optional MLflow only through a directly reachable tracking URI
- resumability by listing expected output objects in R2

## Main Entry Point

Use:

```bash
uv run --group remote_ocr python -m training_signal_processing.main validate --config config/remote_ocr.sample.yaml
uv run --group remote_ocr python -m training_signal_processing.main run --config config/remote_ocr.sample.yaml --dry-run
```

## Control Flow

1. Load and validate the YAML recipe.
2. Resolve the SSH target from CLI flags or `infra/current-machine`.
3. For a new run:
   - discover local PDFs
   - upload them to `s3://gpu-poor/dataset/raw/pdf/...`
   - write an input manifest and resolved recipe object to the run prefix
4. Sync only `pyproject.toml`, `uv.lock`, `src/`, and `config/` to the remote root.
5. Bootstrap `uv`, Python 3.12, and the `remote_ocr` dependency group remotely.
6. Wait for the local PDF upload (rclone) to finish so the remote never races its inputs.
7. Launch the remote job **detached** in its own process group via `launch_detached`:
   the pod writes its pgid to `/root/ocr-jobs/<run_id>/job.pgid` and stdout to
   `.../job.log`. The local CLI returns a `LaunchHandle` JSON and exits —
   **exit code 0 means launched successfully, not run complete**. R2 credentials
   are passed as env vars only.

## Remote Execution Flow

1. Read the resolved recipe and input manifest from R2.
2. Build a Ray Dataset from manifest rows.
3. Prepare source rows.
4. Apply:
   - `skip_existing`
   - `marker_ocr`
5. Materialize each Ray batch on the driver and write markdown outputs
   synchronously to R2.
6. Update in-memory run progress and optional MLflow metrics; do not write
   batch manifests, event objects, `run_state.json`, or `run.json`.

## Observability

- Materialized output objects in R2 are the durable completion source of truth.
  Runtime progress is in memory and optionally mirrored to MLflow.
- MLflow is optional. When enabled, `mlflow.tracking_uri` must be reachable from
  the logging process directly; the framework does not open SSH tunnels.

## Resumability

- Outputs are stored under `dataset/processed/pdf_ocr/<run_id>/`.
- The remote executor lists expected markdown output objects and skips source
  keys whose outputs already exist.
- `resume --config ... --run-id <run-id>` reruns the remote job against the same input manifest.
