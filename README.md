# training-signal-processing

This repo contains a recipe-driven remote processing workspace with a protected
shared submission core and separate pipeline families.

The intended user customization surface is small:
- recipes live in `config/`
- custom ops live in `src/training_signal_processing/custom_ops/`
- new pipeline families live under `src/training_signal_processing/pipelines/`
- additional backend-specific op modules can also live in `src/training_signal_processing/custom_ops/`

Start here:
- pipeline guide: [src/training_signal_processing/custom_ops/README.md](src/training_signal_processing/custom_ops/README.md)
- sample recipe: [config/remote_ocr.sample.yaml](config/remote_ocr.sample.yaml)

Any non-underscore Python module added to `src/training_signal_processing/custom_ops/`
is auto-imported and can register new ops without editing the protected executor or registry code.
Any new pipeline family should be added under `src/training_signal_processing/pipelines/`
without editing `src/training_signal_processing/runtime/submission.py`.

## Quick Start

```bash
uv sync --group remote_ocr
uv run ruff check src/training_signal_processing
uv run --group remote_ocr python -m training_signal_processing.main validate --config config/remote_ocr.sample.yaml
uv run --group remote_ocr python -m training_signal_processing.main run --config config/remote_ocr.sample.yaml --dry-run
```

## Main Commands

```bash
uv run --group remote_ocr python -m training_signal_processing.main list-ops
uv run --group remote_ocr python -m training_signal_processing.main test-op --help
uv run --group remote_ocr python -m training_signal_processing.main run --help
uv run --group remote_ocr python -m training_signal_processing.main resume --help
```

## Project Shape

- `config/`: YAML recipes
- `src/training_signal_processing/custom_ops/`: user-defined ops and the customization README
- `src/training_signal_processing/pipelines/`: pipeline-family packages such as OCR and tokenizer
- `src/training_signal_processing/runtime/`: protected shared runtime infrastructure
- `src/training_signal_processing/ops/`: shared base classes and registry

## Extension Rules

- Add or change OCR processing behavior in `custom_ops/` plus the OCR recipe.
- Add a brand-new pipeline family in `pipelines/<name>/`.
- Do not edit `runtime/submission.py` to add a new dataset or pipeline family.
- Do not edit the protected executor loop to add normal OCR transforms.

The executor loop and shared submission core are intentionally fixed. Extend the
workspace by editing recipes, custom OCR ops, or a pipeline family package.
