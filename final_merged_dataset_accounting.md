# Final Merged Cleaned Dataset Accounting

Counts were produced from the final consolidated parquet dataset by scanning
`cleaned_text` and `cleaned_o200k_token_count`. Token counts use exact
`tiktoken` `o200k_base` counts. Byte counts are UTF-8 bytes of `cleaned_text`,
not compressed parquet object sizes.

Final R2 prefix:
`r2://ocrresults:gpu-poor/dataset/processed/unified-data/final-completed-20260430T160615Z`

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
