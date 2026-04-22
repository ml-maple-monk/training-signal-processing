# Workspace Guidance

## Intent

This workspace uses a small YAML-first remote OCR pipeline.

## Rules

- YAML is the source of truth for runtime behavior.
- Keep one visible local entrypoint in `src/training_signal_processing/main.py`.
- Use `ray.data` only for dataset execution.
- Keep all OCR input and output on Cloudflare R2.
- Use Marker as the only OCR engine in v1.
- Do not sync `r2`, `infra/credentials`, `.omx`, `.cache`, or other secret/state files to the remote machine.
- Use the SSH reverse tunnel to forward MLflow progress back to the local MLflow server.
- Use batch-manifest resumability only.
- Do not hardcode bucket names, hosts, ports, paths, or batch sizes in logic; put them in YAML or fail clearly.
- Prefer typed dataclasses over loose dicts for config and result structures.
- Keep helper globals in `utils.py`.
- Keep functions small and obvious.

## Verification

- Run `uv run ruff check .`
- Run `uv run --group remote_ocr python -m training_signal_processing.main validate --config ...`
- Run `uv run --group remote_ocr python -m training_signal_processing.main run --config ... --dry-run`
