# Custom Ops And Recipes

This directory is the user-customized part of the remote OCR pipeline.
It is not the place to create an entirely new pipeline family.

## Where To Change Things

Change the recipe in `config/`.
- Start from `config/remote_ocr.sample.yaml`.
- Add, remove, or reorder ops under `ops:`.
- Keep exactly one `prepare` op and one `export` op.
- Put runtime settings like SSH, R2, MLflow, and batch size in YAML, not in code.

Change custom ops in `src/training_signal_processing/custom_ops/`.
- Add new concrete ops in `user_ops.py` if you want one simple extension file.
- If the file gets too large, add another `.py` module in this directory.
- You do not need to edit the registry or executor to wire that new module in.
- Every non-underscore Python module in this directory is auto-imported when the registry loads.
- If you want a helper-only backend module that should not be auto-imported, prefix it with `_`.
- Do not put user-specific op code in `runtime/` or in `ops/base.py`.
- Do not create a new pipeline family here. Put that under `src/training_signal_processing/pipelines/`.

Create a new pipeline family in `src/training_signal_processing/pipelines/<name>/`.
- Add that package when the dataset shape, remote job, or control-plane behavior is different enough that it is no longer “just another OCR op”.
- Reuse `runtime/submission.py` as the shared orchestration core instead of copying it.
- Keep `runtime/submission.py` and the protected executor loop unchanged when adding a normal new pipeline family.

## How To Add A New Op

1. Pick the correct base class.
   - `SourcePreparationOp` for the row-shaping step before the main transforms.
   - `BatchTransformOp` for normal batch-to-batch transforms.
   - `SkipExistingFilter` for filtering rows out.
   - `MarkerOcrMapper` or `ExportMarkdownMapper` when you want to stay close to the existing OCR/export shape.
2. Add a concrete class in `custom_ops/user_ops.py`, or add a new module beside it such as `custom_ops/my_backend_ops.py`.
3. Set a unique `op_name`.
4. Implement `process_row`, `process_batch`, or `keep_row`.
5. Add the new op name to a recipe in `config/`.

Example:

```python
from training_signal_processing.ops.base import Batch
from training_signal_processing.ops.builtin import BatchTransformOp


class MyPreviewOp(BatchTransformOp):
    op_name = "my_preview"

    def process_batch(self, batch: Batch) -> Batch:
        return [dict(row, preview="ok") for row in batch]
```

Then add it to a recipe:

```yaml
ops:
  - name: prepare_pdf_document
    type: mapper
  - name: my_preview
    type: mapper
  - name: marker_ocr
    type: mapper
    force_ocr: true
  - name: export_markdown
    type: mapper
```

## How To Add Backend-Specific Code Without Touching Protected Code

Keep backend-specific code in this same directory.

Recommended pattern:
- `_my_backend.py`: helper code, backend client setup, shared parsing, helper dataclasses
- `my_backend_ops.py`: concrete ops that use that backend
- `config/my_backend.yaml`: a recipe that references the new op names

Because the loader auto-imports every non-underscore module in `custom_ops/`, adding
`my_backend_ops.py` is enough to register the new ops. You do not need to edit:
- `ops/registry.py`
- `runtime/executor.py`
- `runtime/submission.py`

That keeps the protected execution path stable while letting the user extend the pipeline.

## How To Test Only One Op

Use the op-only Ray path:

```bash
uv run --group remote_ocr python -m training_signal_processing.main test-op \
  --config config/remote_ocr.sample.yaml \
  --op prepare_pdf_document \
  --rows-path /tmp/sample.jsonl \
  --batch-size 1
```

Useful commands:

```bash
uv run --group remote_ocr python -m training_signal_processing.main list-ops
uv run --group remote_ocr python -m training_signal_processing.main validate --config config/remote_ocr.sample.yaml
uv run --group remote_ocr python -m training_signal_processing.main run --config config/remote_ocr.sample.yaml --dry-run
```

## How The Pipeline Operates

1. `main.py` loads the recipe YAML and routes to `validate`, `test-op`, `run`, or `resume`.
2. `pipelines/ocr/submission.py` prepares OCR-specific artifacts and remote commands.
3. `runtime/submission.py` runs the protected shared submission flow.
   - sync project files
   - run bootstrap
   - execute the pipeline-specific remote command
4. `pipelines/ocr/remote_job.py` rebuilds the OCR recipe from R2 and constructs the executor.
5. `runtime/executor.py` runs the fixed Ray-only pipeline loop.
   - load manifest
   - build Ray dataset
   - run the prepare op
   - run transform ops
   - run the export op per committed batch
6. `runtime/exporter.py` writes final markdown outputs.
7. `runtime/resume.py` writes manifest chunks, event chunks, and `run_state.json` for resumability.
8. `runtime/observability.py` logs structured execution events and MLflow progress.

## Design Rule

The normal user workflow should be:
- edit a recipe in `config/`
- edit a custom op or backend-specific module in `custom_ops/`
- validate or test the op
- run the pipeline

The normal user should not need to edit the executor loop.
The normal user should also not need to edit `runtime/submission.py`.
