# Final Tokenized Dataset R2 Location

The current final tokenized dataset is the completed
`native_superbpe_1m_rows_max4w` export on Cloudflare R2.

Run root:
`r2://ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z`

Equivalent `rclone` path:
`ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z`

## Prefixes

| item | R2 path |
| --- | --- |
| run root | `r2://ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z` |
| parquet parts | `r2://ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z/parts/` |
| done sentinels | `r2://ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z/done/` |
| metrics sidecars | `r2://ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z/metrics/` |
| errors | `r2://ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z/errors/` |
| input manifest | `r2://ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z/control/input_manifest.jsonl` |
| manifest summary | `r2://ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z/control/manifest_summary.json` |
| run recipe | `r2://ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z/control/recipe.json` |
| tokenizer copy | `r2://ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z/control/tokenizer.json` |

Output parquet parts are flat under `parts/`:

```text
r2://ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z/parts/part-000000.parquet
...
r2://ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z/parts/part-002728.parquet
```

## Verification Snapshot

R2 object counts for the completed run:

| object class | count |
| --- | ---: |
| parquet parts | 2,729 |
| done sentinels | 2,729 |
| metrics sidecars | 2,729 |
| error sentinels | 0 |
| control files | 4 |

`rclone size` over the `parts/` prefix reported `2,729` objects and
`53,475,794,332` bytes (`49.803 GBytes`) of parquet payload.

The manifest summary records `2,729` tokenization tasks:

| source group | input parts |
| --- | ---: |
| final cleaned dataset | 78 |
| FineWeb done-confirmed add-on parts | 2,651 |

The FineWeb source manifest excluded `3` error-only parts and `3` orphan parts
from the tokenized output.

## Dataset Shape

The tokenized parquet is compact: it keeps provenance columns, text hashes,
`text_byte_count`, `cleaning_is_dropped`, `token_ids`, `token_count`,
`tokenizer_name`, and `tokenizer_json_sha256`. It does not keep the full
`cleaned_text` column.

Tokenizer:
`native_superbpe_1m_rows_max4w`

Tokenizer SHA-256:
`55118b71012dfdc40a22a8ac4448abaf71db5e2be5381b051e0fea3864dcfbd0`

## Useful Commands

List the final tokenized parquet parts:

```bash
rclone lsf \
  ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z/parts
```

Read the manifest summary:

```bash
rclone cat \
  ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z/control/manifest_summary.json
```

Reconcile completion counts:

```bash
rclone lsf -R \
  ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/20260503T002359Z
```

Older sibling runs under
`r2://ocrresults:gpu-poor/dataset/processed/tokenized/native-superbpe-1m-rows-max4w/`
are not the final tokenized dataset unless a later doc explicitly promotes them.
