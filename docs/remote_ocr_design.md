# Streaming Ray-Only Remote OCR

## What This Adds

This repo now includes a small remote OCR pipeline with:
- YAML-first config
- laptop-side Click CLI
- single-node remote `ray.data` execution
- Marker OCR
- R2 input/output
- MLflow progress over a framework-managed persistent SSH reverse tunnel (ControlMaster)
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
7. Open (or reuse) a persistent `-R` reverse tunnel for each declared tunnel spec via
   `ensure_reverse_tunnels`. Each tunnel runs as an `ssh -fN -o ControlMaster=yes`
   process whose socket lives at `~/.cache/ocr-remote-launcher/tunnels/t-<hash>.sock`
   and outlives the launcher SSH.
8. Launch the remote job **detached** in its own process group via `launch_detached`:
   the pod writes its pgid to `/root/ocr-jobs/<run_id>/job.pgid` and stdout to
   `.../job.log`. The local CLI returns a `LaunchHandle` + `TunnelHandle` JSON and
   exits — **exit code 0 means launched successfully, not run complete**. R2
   credentials are passed as env vars only.

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

- The local machine already runs MLflow on `http://127.0.0.1:5000`.
- The framework ensures a persistent reverse tunnel exists before launch; its
  ControlMaster socket lives at `~/.cache/ocr-remote-launcher/tunnels/` and is
  reused across launches to the same pod. Manual teardown is
  `ssh -S <sock> -O exit <ssh_target>`.
- The remote `ProgressTrackerActor` logs run-level and batch-level metrics to MLflow.

## Resumability

- Outputs are stored under `dataset/processed/pdf_ocr/<run_id>/`.
- The remote executor reads prior manifest chunks and skips already completed source keys.
- `resume --config ... --run-id <run-id>` reruns the remote job against the same input manifest.
