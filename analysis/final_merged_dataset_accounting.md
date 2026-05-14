# Final Merged Cleaned Dataset Accounting

Counts were produced from the final consolidated parquet dataset by scanning
`cleaned_text` and `cleaned_o200k_token_count`. Token counts use exact
`tiktoken` `o200k_base` counts. Byte counts are UTF-8 bytes of `cleaned_text`,
not compressed parquet object sizes.

Final R2 prefix:
`r2://ocrresults:gpu-poor/dataset/processed/unified-data/final-completed-20260430T160615Z`

FineWeb add-on R2 prefix:
`r2://ocrresults:gpu-poor/dataset/processed/fineweb-unified/20260501T144323Z`

Parquet parts: `78`

Cleaned text column: `cleaned_text`

Token column: `cleaned_o200k_token_count`

| source | cleaning_source | final_r2_prefix | text_column | token_count | byte_count | sample_count | dropped_sample_count | source_object_count | original_source_glob | filters |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Books + OCR | `books-ocr` | `dataset/processed/unified-data/final-completed-20260430T160615Z/parts/` | `cleaned_text` (from `markdown_text`) | 223,965,046 | 898,466,770 | 8,786 | 1 | 1 | `dataset/processed/pdf_ocr/20260423T195035Z/markdown.parquet` | `` |
| Cari | `cari` | `dataset/processed/unified-data/final-completed-20260430T160615Z/parts/` | `cleaned_text` (from `body_text`) | 116,943,092 | 431,641,324 | 1,329,684 | 8,245 | 1 | `dataset/processed/malay/cari.parquet` | `` |
| HPLT Indonesia | `hplt-indonesia` | `dataset/processed/unified-data/final-completed-20260430T160615Z/parts/` | `cleaned_text` (from `text`) | 11,452,756,231 | 45,368,173,019 | 14,724,821 | 16 | 4 | `dataset/processed/malay/hplt/*_indon.parquet` | `` |
| HPLT Malay | `hplt-malay` | `dataset/processed/unified-data/final-completed-20260430T160615Z/parts/` | `cleaned_text` (from `text`) | 9,391,236,570 | 35,175,717,072 | 9,023,619 | 54 | 4 | `dataset/processed/malay/hplt/*_malay.parquet` | `` |
| Lowyat | `lowyat` | `dataset/processed/unified-data/final-completed-20260430T160615Z/parts/` | `cleaned_text` (from `body_text`) | 598,615,295 | 2,527,778,839 | 12,960,898 | 317,753 | 1 | `dataset/processed/malay/lowyat.parquet` | `` |
| Reddit Bolehland | `reddit-bolehland` | `dataset/processed/unified-data/final-completed-20260430T160615Z/parts/` | `cleaned_text` (from `body`) | 6,494,781 | 28,437,578 | 215,284 | 16,649 | 1 | `dataset/processed/malay/reddit.parquet` | `subreddit=Bolehland` |
| Reddit Indonesia | `reddit-indonesia` | `dataset/processed/unified-data/final-completed-20260430T160615Z/parts/` | `cleaned_text` (from `body`) | 10,432,974 | 40,746,266 | 286,429 | 24,966 | 1 | `dataset/processed/malay/reddit.parquet` | `subreddit=indonesia` |
| **Total** | `` | `dataset/processed/unified-data/final-completed-20260430T160615Z/parts/` | `cleaned_text` | 21,800,443,989 | 84,470,960,868 | 38,549,521 | 367,684 |  |  |  |

## Additional Totals

| metric | value |
| --- | ---: |
| cleaned_char_count | 83,980,609,165 |
| removed_char_count | 1,605,781,875 |
| cleaned_text_byte_count | 84,470,960,868 |
| cleaned_o200k_token_count | 21,800,443,989 |
| sample_count | 38,549,521 |
| dropped_sample_count | 367,684 |

## Consolidation Inputs

| source_run_id | label | copied_parts | row_count | token_count | excluded_parts |
| --- | --- | ---: | ---: | ---: | --- |
| `20260429T214659Z` | base-completed | 61 | 30,049,521 | 14,987,206,599 | `part-000036 error` |
| `unified-data-delta-lid-20260430T0228Z` | delta-lid-completed | 17 | 8,500,000 | 6,813,237,390 | `part-000017 no-done` |

## FineWeb Add-On Run

FineWeb is being written as a separate unified-format dataset prefix, not copied
into the completed final prefix above. **In-progress, not merged into the
totals tables above.**

### Pipeline at a glance

```
   HuggingFace FineWeb  (config = default · split = train · text column = text)
                │
                │   stream_shards_per_config = 1
                │   enforce_month_filter     = false  (relaxed bucket assignment)
                ▼
   Ray executor   (concurrency = 4 · batch = 1 · target_num_blocks = 8192)
                │
                │   clean → unified schema (76 cols)
                │   compute_exact_token_counts = false  →  cleaned_o200k_token_count = NULL
                ▼
   r2://ocrresults:gpu-poor/dataset/processed/fineweb-unified/20260501T144323Z/
       ├─ parts/month={202111,202112}/part-NNNNNN.parquet
       ├─ done/part-NNNNNN.done.json
       ├─ metrics/part-NNNNNN.metrics.json
       ├─ errors/part-NNNNNN.error.json
       └─ control/{input_manifest.jsonl, recipe.json}
```

### R2 prefixes

| item | R2 path |
| --- | --- |
| run root | `r2://ocrresults:gpu-poor/dataset/processed/fineweb-unified/20260501T144323Z` |
| parquet parts prefix | `r2://ocrresults:gpu-poor/dataset/processed/fineweb-unified/20260501T144323Z/parts/` |
| done sentinels prefix | `r2://ocrresults:gpu-poor/dataset/processed/fineweb-unified/20260501T144323Z/done/` |
| metrics prefix | `r2://ocrresults:gpu-poor/dataset/processed/fineweb-unified/20260501T144323Z/metrics/` |
| error sentinels prefix | `r2://ocrresults:gpu-poor/dataset/processed/fineweb-unified/20260501T144323Z/errors/` |
| input manifest | `r2://ocrresults:gpu-poor/dataset/processed/fineweb-unified/20260501T144323Z/control/input_manifest.jsonl` |
| run recipe | `r2://ocrresults:gpu-poor/dataset/processed/fineweb-unified/20260501T144323Z/control/recipe.json` |

Output path pattern:
`r2://ocrresults:gpu-poor/dataset/processed/fineweb-unified/20260501T144323Z/parts/month=<YYYYMM>/part-<NNNNNN>.parquet`

### Run plan (from `control/recipe.json`)

| item | value |
| --- | ---: |
| planned manifest rows | 5,280 |
| manifest months | `2021-11`: 2,640 · `2021-12`: 2,640 |
| configured stream shards per config | 1 |
| `enforce_month_filter` (strict real collection-date filtering) | disabled |
| `compute_exact_token_counts` | disabled |
| byte_quota cap | 84.47 GB |

### R2 progress — verified 2026-05-01 23:55 UTC

| metric | value | status |
| --- | ---: | --- |
| written parquet parts | 2,655 | OK |
| done sentinels | 2,651 | OK |
| metrics sidecars | 2,651 | OK |
| error sentinels | 3 | WARN — see Anomalies |
| orphan parquet (no done, no error) | 3 | WARN — see Anomalies |

**Parts ledger** (must reconcile to `parts_written`):

| bucket | count |
| --- | ---: |
| `done`-sentinel only (clean success) | 2,649 |
| `done` + stale `error` (retried OK, error sentinel never deleted) | 2 |
| `error`-sentinel only (no successful retry) | 1 |
| orphan parquet (no `done`, no `error`) | 3 |
| **parts_written total** | **2,655** |

Cross-checks: `done = 2,649 + 2 = 2,651` · `error = 2 + 1 = 3` · `metrics = done = 2,651`.

### Per-month progress

```
2021-11  ▇▇▇▇▇▇▇▇▇▇░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  25.1%   1,328 parts · 6,674,302 rows · 21.19 GB
2021-12  ▇▇▇▇▇▇▇▇▇▇░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  25.1%   1,327 parts · 6,694,979 rows · 21.20 GB
         └── % of 84.47 GB byte_quota cap ─────┘  combined 50.2 %  →  run halted byte-bounded
```

The run stopped at ≈ 50.2 % of the `byte_quota` cap, so 2,625 of the 5,280
planned manifest rows were not processed. The 5,280-vs-2,655 gap is by design
(byte-bounded), not a failure.

### FineWeb aggregates (in-progress · NOT MERGED into the totals above)

Sums over all 2,651 success metric sidecars:

| field | value |
| --- | ---: |
| `row_count` (sample_count) | 13,369,281 |
| `cleaned_text_byte_count` | 42,392,275,825 |
| `cleaned_o200k_token_count` | NULL (skipped — see Anomalies) |
| `streamed_row_count` | 13,371,932 |
| `quota_stop_row_count` | 2,651 (one boundary row per part) |
| `dropped_sample_count` | 0 |
| `byte_quota_consumed` | 42,411,461,716 (≈ 50.2 % of 84.47 GB cap) |
| `source_object_count` | 1 (`hf://datasets/HuggingFaceFW/fineweb`, config `default`) |
| total parquet bytes on R2 | 50,414,062,559 (46.95 GiB) |
| native_source / language | `HuggingFaceFW/fineweb` · 100 % `en` |

Per-month aggregates:

| month | sidecars | rows | bytes |
| ---: | ---: | ---: | ---: |
| 2021-11 | 1,325 | 6,674,302 | 21,191,907,000 |
| 2021-12 | 1,326 | 6,694,979 | 21,200,368,825 |
| **Total** | **2,651** | **13,369,281** | **42,392,275,825** |

> Difference between sidecar counts (1,325 / 1,326) and parquet-part counts
> (1,328 / 1,327) is fully explained by the parts ledger above:
> `2,655 − 2,651 = 4 = 1 error-only + 3 orphans`.

### Anomalies / open items

- **3 error sentinels.**
  - `month=202111/part-000496` — snappy decode failure, **no successful retry** — this is the one true failure.
  - `month=202111/part-001304` — R2 `UploadPart` `BadDigest` (CRC64NVME mismatch), retried OK; stale error sentinel still present.
  - `month=202112/part-003935` — R2 `UploadPart` `BadDigest` (CRC64NVME mismatch), retried OK; stale error sentinel still present.
  - The runtime writes `done` on retry success but never deletes the prior
    `error` sentinel
    ([runtime.py:24-69](src/training_signal_processing/pipelines/fineweb_unified/runtime.py#L24-L69)),
    which is why two parts hold both kinds of sentinel.
- **3 orphan parquet parts** (parquet present, neither `done` nor `error` sentinel):
  - `month=202111/part-001327.parquet`
  - `month=202111/part-001328.parquet`
  - `month=202112/part-003966.parquet`
  - Either still in flight or silently failed; status to be re-listed before any consolidation step.
- **`cleaned_o200k_token_count` is NULL for every FineWeb row.** Exact
  tokenization was disabled for this fast-path run. The column is null in
  parquet, **not zero** — a SUM over the merged dataset would silently undercount
  by ≈ 13.4 M FineWeb rows of tokens. Tokens MUST be computed before merging
  FineWeb into the unified totals.

**Gating checklist before merging FineWeb into the unified totals** (all five must hold, in order):

1. The 3 orphan parts are re-listed and reclassified into `done` / `error`,
   or rewritten and given a sentinel.
2. The 1 unrecovered `error`-only part (`part-000496`) is either retried
   successfully or explicitly excluded with a documented reason.
3. The 2 stale `error` sentinels for already-`done` parts (`1304`, `3935`)
   are deleted, **or** the consolidation path is updated to ignore `error`
   when a matching `done` exists.
4. Exact `o200k_base` tokens are computed for every row and the
   `cleaned_o200k_token_count` column is populated (no NULL allowed).
5. Aggregate totals are re-verified after token computation and a fresh
   "verified" timestamp is written into this doc.

### Previous snapshot — 2026-05-01 16:50 UTC (kept for audit)

The status block below was the first snapshot taken ~6 minutes after the final
manifest replan and is preserved verbatim so the deltas to the current numbers
remain auditable. **Do not use for accounting.**

| item | value |
| --- | ---: |
| planned manifest rows / parquet parts | 5,280 |
| manifest months | `2021-11`: 2,640; `2021-12`: 2,640 |
| configured stream shards per config | 1 |
| strict real collection-date filtering | disabled |
| exact token counting | disabled |
| written parquet parts | 86 |
| done sentinels | 86 |
| metrics sidecars | 86 |
| error sentinels | 0 |

## Notes

- The final parquet is not partitioned by source; all source rows are colocated
  under the single `parts/` prefix above.
- Source identity is stored in the columns `source_name` and `cleaning_source`.
  The original configured text column is stored in `text_column`; the canonical
  final text is `cleaned_text`.
- Dropped rows remain present with empty `cleaned_text` and
  `cleaning_is_dropped=true`.
- This report covers only done-confirmed final parquet copied into the
  consolidated prefix. It excludes any upstream payload without a done sentinel
  or with an error sentinel.
- FineWeb output is tracked separately under the FineWeb add-on prefix. The run
  is byte-bounded-complete (≈ 50.2 % of `byte_quota`) and all 2,651 success
  `done/` + `metrics/` sidecars are present, **but** `cleaned_o200k_token_count`
  is NULL for every FineWeb row and 4 parts are still unresolved (1 error-only,
  3 orphan). Do not merge FineWeb into the final totals until the gating
  checklist in "Anomalies / open items" is fully satisfied.
- FineWeb month directories are relaxed assignment buckets for this speed-up
  run. The row-level `timestamp` still preserves FineWeb metadata, but output is
  no longer strictly filtered to real collection-date month partitions.
