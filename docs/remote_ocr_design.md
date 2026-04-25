# Streaming Ray-Only Remote OCR

## What This Adds

This repo now includes a small remote OCR pipeline with:
- YAML-first config
- laptop-side Click CLI
- single-node remote `ray.data` execution
- Marker OCR
- R2 input/output
- R2-backed progress through run state, manifests, and event objects
- optional MLflow only through a directly reachable tracking URI
- resumability through batch manifests in R2

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
4. Sync only `pyproject.toml`, `uv.lock`, and `src/` to the remote root.
5. Bootstrap `uv`, Python 3.12, and the `remote_ocr` dependency group remotely.
6. Wait for the local PDF upload (rclone) to finish so the remote never races its inputs.
7. Launch the remote job **detached** in its own process group via `launch_detached`:
   the pod writes its pgid to `/root/ocr-jobs/<run_id>/job.pgid` and stdout to
   `.../job.log`. The local CLI returns a `LaunchHandle` JSON and exits —
   **exit code 0 means launched successfully, not run complete**. R2 credentials
   are passed as env vars only.

## Remote Execution Flow

1. Read the resolved recipe and input manifest from R2.
2. Build an explicit `pyarrow.fs.S3FileSystem`.
3. Use `ray.data.read_binary_files(...)` for the selected input objects.
4. Prepare source rows.
5. Apply:
   - `skip_existing`
   - `marker_ocr`
   - `export_markdown`
6. Commit each finished microbatch:
   - manifest JSONL chunk
   - event JSONL chunk
   - updated `run_state.json`

## Observability

- `run_state.json`, batch manifests, and event objects in R2 are the durable
  progress and recovery source of truth.
- MLflow is optional. When enabled, `mlflow.tracking_uri` must be reachable from
  the logging process directly; the framework does not open SSH tunnels.

## Resumability

- Outputs are stored under `dataset/processed/pdf_ocr/<run_id>/`.
- The remote executor reads prior manifest chunks and skips already completed source keys.
- `resume --config ... --run-id <run-id>` reruns the remote job against the same input manifest.
