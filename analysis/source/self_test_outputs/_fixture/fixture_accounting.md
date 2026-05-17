# Fixture Accounting

Final R2 prefix:
`r2://fixture:fixture-bucket/fixture/final`

Parquet parts: `2`

Cleaned text column: `cleaned_text`

Token column: `cleaned_o200k_token_count`

| source | cleaning_source | final_r2_prefix | text_column | token_count | byte_count | sample_count | dropped_sample_count | source_object_count | original_source_glob | filters |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Alpha Source | `alpha` | `fixture/parts/` | `cleaned_text` | 9 | 47 | 2 | 1 | 1 | `fixture` | `` |
| Beta Source | `beta` | `fixture/parts/` | `cleaned_text` | 10 | 63 | 1 | 0 | 1 | `fixture` | `` |
| **Total** | `` | `fixture/parts/` | `cleaned_text` | 19 | 110 | 3 | 1 |  |  |  |

## Additional Totals

| metric | value |
| --- | ---: |
| cleaned_text_byte_count | 110 |
| cleaned_o200k_token_count | 19 |
| sample_count | 3 |
| dropped_sample_count | 1 |
