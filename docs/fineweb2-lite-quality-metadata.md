# FineWeb2-Lite Quality Metadata

`fineweb2_lite_quality_metadata` stores per-document quality metrics only. The
pipeline does not filter, delete, rewrite, deduplicate, or language-ID documents.

Rows are keyed by:

```sql
(run_id, input_source, source_doc_id)
```

For the current unified-only run, use `input_source = 'unified'`; SEA-PILE v2
documents should be read through `unified_documents` after ingestion there.

## Schema Shape

Provenance columns:

| column | meaning |
| --- | --- |
| `run_id` | pipeline run identifier |
| `input_source` | source namespace, normally `unified` |
| `source_doc_id` | source-local id; equals `doc_id` for unified rows |
| `doc_id` | FK to `unified_documents.doc_id` for unified rows |
| `sea_pile_malay_id` | raw SEA-PILE id, unused for unified-only runs |
| `cleaning_source` | source label from `unified_documents` |
| `source_is_dropped` | original `unified_documents.cleaning_is_dropped` value |
| `language_profile` | source-mapped profile, for example `ind_Latn` or `zsm_Latn` |
| `created_at` | insertion timestamp |

Metric columns:

```sql
text_char_count,
paragraph_count,
line_count,
non_empty_line_count,
word_count,
duplicate_line_count,
duplicate_line_fraction,
top_2gram_fraction,
top_3gram_fraction,
top_4gram_fraction,
duplicated_5gram_fraction,
duplicated_6gram_fraction,
duplicated_7gram_fraction,
duplicated_8gram_fraction,
duplicated_9gram_fraction,
duplicated_10gram_fraction,
line_punctuation_fraction,
duplicated_line_character_ratio,
newline_count,
newline_word_ratio,
non_symbol_word_count,
average_non_symbol_word_length,
hash_symbol_ratio,
ellipsis_ratio,
bullet_line_ratio,
ending_ellipsis_line_ratio,
alpha_word_ratio,
stopword_count,
indonesian_stopword_count,
at_least_2_profile_stopwords_present,
at_least_2_indonesian_stopwords_present
```

`metrics JSONB` mirrors the scalar metric columns plus source metadata. Prefer
the scalar columns for queries and aggregations.

## Read Metrics

Get all metric columns for one run:

```sql
SELECT
    doc_id,
    cleaning_source,
    source_is_dropped,
    language_profile,
    duplicate_line_fraction,
    top_2gram_fraction,
    top_3gram_fraction,
    top_4gram_fraction,
    duplicated_5gram_fraction,
    duplicated_6gram_fraction,
    duplicated_7gram_fraction,
    duplicated_8gram_fraction,
    duplicated_9gram_fraction,
    duplicated_10gram_fraction,
    line_punctuation_fraction,
    duplicated_line_character_ratio,
    newline_count,
    word_count,
    newline_word_ratio,
    non_symbol_word_count,
    average_non_symbol_word_length,
    hash_symbol_ratio,
    ellipsis_ratio,
    bullet_line_ratio,
    ending_ellipsis_line_ratio,
    alpha_word_ratio,
    at_least_2_indonesian_stopwords_present
FROM fineweb2_lite_quality_metadata
WHERE run_id = :run_id
  AND input_source = 'unified';
```

Get one metric independently:

```sql
SELECT doc_id, duplicate_line_fraction
FROM fineweb2_lite_quality_metadata
WHERE run_id = :run_id
  AND input_source = 'unified';
```

`newline_word_ratio` is the stored `newline_count / word_count` metric with a
zero-word guard. Use it instead of recomputing division in most queries.

## Indonesian Threshold Predicate

Rows satisfying the Indonesian FineWeb2-like thresholds:

```sql
SELECT doc_id
FROM fineweb2_lite_quality_metadata
WHERE run_id = :run_id
  AND input_source = 'unified'
  AND duplicate_line_fraction <= 0.241
  AND top_2gram_fraction <= 0.194
  AND top_3gram_fraction <= 0.166
  AND top_4gram_fraction <= 0.145
  AND duplicated_5gram_fraction <= 0.166
  AND duplicated_6gram_fraction <= 0.155
  AND duplicated_7gram_fraction <= 0.144
  AND duplicated_8gram_fraction <= 0.133
  AND duplicated_9gram_fraction <= 0.122
  AND duplicated_10gram_fraction <= 0.111
  AND line_punctuation_fraction >= 0.111
  AND duplicated_line_character_ratio <= 0.1
  AND newline_word_ratio <= 0.157
  AND non_symbol_word_count BETWEEN 50 AND 10000000
  AND average_non_symbol_word_length BETWEEN 3 AND 12
  AND hash_symbol_ratio <= 0.1
  AND ellipsis_ratio <= 0.1
  AND bullet_line_ratio <= 0.9
  AND ending_ellipsis_line_ratio <= 0.3
  AND alpha_word_ratio >= 0.805
  AND at_least_2_indonesian_stopwords_present;
```

Get each threshold as an independent boolean:

```sql
SELECT
    doc_id,
    duplicate_line_fraction <= 0.241 AS ok_duplicate_line_fraction,
    top_2gram_fraction <= 0.194 AS ok_top_2gram_fraction,
    top_3gram_fraction <= 0.166 AS ok_top_3gram_fraction,
    top_4gram_fraction <= 0.145 AS ok_top_4gram_fraction,
    duplicated_5gram_fraction <= 0.166 AS ok_duplicated_5gram_fraction,
    duplicated_6gram_fraction <= 0.155 AS ok_duplicated_6gram_fraction,
    duplicated_7gram_fraction <= 0.144 AS ok_duplicated_7gram_fraction,
    duplicated_8gram_fraction <= 0.133 AS ok_duplicated_8gram_fraction,
    duplicated_9gram_fraction <= 0.122 AS ok_duplicated_9gram_fraction,
    duplicated_10gram_fraction <= 0.111 AS ok_duplicated_10gram_fraction,
    line_punctuation_fraction >= 0.111 AS ok_line_punctuation_fraction,
    duplicated_line_character_ratio <= 0.1 AS ok_duplicated_line_character_ratio,
    newline_word_ratio <= 0.157 AS ok_newline_word_ratio,
    non_symbol_word_count BETWEEN 50 AND 100000 AS ok_non_symbol_word_count,
    average_non_symbol_word_length BETWEEN 3 AND 12 AS ok_average_non_symbol_word_length,
    hash_symbol_ratio <= 0.1 AS ok_hash_symbol_ratio,
    ellipsis_ratio <= 0.1 AS ok_ellipsis_ratio,
    bullet_line_ratio <= 0.9 AS ok_bullet_line_ratio,
    ending_ellipsis_line_ratio <= 0.3 AS ok_ending_ellipsis_line_ratio,
    alpha_word_ratio >= 0.805 AS ok_alpha_word_ratio,
    at_least_2_indonesian_stopwords_present AS ok_indonesian_stopwords
FROM fineweb2_lite_quality_metadata
WHERE run_id = :run_id
  AND input_source = 'unified';
```

Count pass/fail per independent rule:

```sql
SELECT
    count(*) AS total_rows,
    count(*) FILTER (WHERE duplicate_line_fraction <= 0.241) AS pass_duplicate_line_fraction,
    count(*) FILTER (WHERE top_2gram_fraction <= 0.194) AS pass_top_2gram_fraction,
    count(*) FILTER (WHERE top_3gram_fraction <= 0.166) AS pass_top_3gram_fraction,
    count(*) FILTER (WHERE top_4gram_fraction <= 0.145) AS pass_top_4gram_fraction,
    count(*) FILTER (WHERE duplicated_5gram_fraction <= 0.166) AS pass_duplicated_5gram_fraction,
    count(*) FILTER (WHERE duplicated_6gram_fraction <= 0.155) AS pass_duplicated_6gram_fraction,
    count(*) FILTER (WHERE duplicated_7gram_fraction <= 0.144) AS pass_duplicated_7gram_fraction,
    count(*) FILTER (WHERE duplicated_8gram_fraction <= 0.133) AS pass_duplicated_8gram_fraction,
    count(*) FILTER (WHERE duplicated_9gram_fraction <= 0.122) AS pass_duplicated_9gram_fraction,
    count(*) FILTER (WHERE duplicated_10gram_fraction <= 0.111) AS pass_duplicated_10gram_fraction,
    count(*) FILTER (WHERE line_punctuation_fraction >= 0.111) AS pass_line_punctuation_fraction,
    count(*) FILTER (WHERE duplicated_line_character_ratio <= 0.1) AS pass_duplicated_line_character_ratio,
    count(*) FILTER (WHERE newline_word_ratio <= 0.157) AS pass_newline_word_ratio,
    count(*) FILTER (WHERE non_symbol_word_count BETWEEN 50 AND 100000) AS pass_non_symbol_word_count,
    count(*) FILTER (WHERE average_non_symbol_word_length BETWEEN 3 AND 12) AS pass_average_non_symbol_word_length,
    count(*) FILTER (WHERE hash_symbol_ratio <= 0.1) AS pass_hash_symbol_ratio,
    count(*) FILTER (WHERE ellipsis_ratio <= 0.1) AS pass_ellipsis_ratio,
    count(*) FILTER (WHERE bullet_line_ratio <= 0.9) AS pass_bullet_line_ratio,
    count(*) FILTER (WHERE ending_ellipsis_line_ratio <= 0.3) AS pass_ending_ellipsis_line_ratio,
    count(*) FILTER (WHERE alpha_word_ratio >= 0.805) AS pass_alpha_word_ratio,
    count(*) FILTER (WHERE at_least_2_indonesian_stopwords_present) AS pass_indonesian_stopwords
FROM fineweb2_lite_quality_metadata
WHERE run_id = :run_id
  AND input_source = 'unified';
```

