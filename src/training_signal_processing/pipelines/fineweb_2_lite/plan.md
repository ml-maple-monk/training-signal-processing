I’m treating “Indonesia language” as **Bahasa Indonesia**, which in FineWeb2 is the config file:

```text
configs/ind_Latn.yml
```

FineWeb2 uses ISO 639-3 language code `ind` plus script `Latn`.

## Exact order for Indonesian FineWeb2 filtering

The FineWeb2 repo says the multilingual pipeline starts from the **non-English leftovers of FineWeb**, then applies: language ID/filtering, per-language deduplication, per-language filtering, and final PII/fix steps. ([GitHub][1])

For Indonesian specifically, the flow is:

```text
Common Crawl / FineWeb non-English pool
→ GlotLID language routing
→ keep only ind_Latn docs with language_score >= 0.685
→ global MinHash dedup within ind_Latn
→ Gopher repetition filter with Indonesian thresholds
→ FineWeb quality filter with Indonesian thresholds
→ Gopher quality filter with Indonesian thresholds + Indonesian stopwords
→ no C4 filters
→ FTFY + PII anonymization + trafilatura table-artifact fix
```

## 1. Language identification: keep Indonesian Latin-script docs

FineWeb2 uses **GlotLID** instead of the original FineWeb fastText language detector, because GlotLID covers more labels and also identifies script. The pipeline writes documents into folders like `${language}_${language_script}`, so Indonesian Latin-script text goes to `ind_Latn`. ([GitHub][1])

Then it applies the Indonesian language-score threshold from `ind_Latn.yml`:

```yaml
language_score: 0.685
```

So the exact rule is:

```python
keep if doc.metadata["language_score"] >= 0.685
```

The code loads `configs/{lang_script}.yml`, then keeps only docs whose `language_score` is above that language’s threshold. ([GitHub][2]) The Indonesian config sets `language_score: 0.685`. ([GitHub][3])

## 2. Deduplicate Indonesian globally with MinHash

After language-score filtering, FineWeb2 deduplicates **per language globally**, not per Common Crawl snapshot. The repo says this differs from FineWeb English, where dedup was per snapshot. ([GitHub][1])

For `ind_Latn`, the code uses the same MinHash config as other languages:

```python
MinhashConfig(
    hash_config=HashConfig(
        hash_fc="xxhash",
        precision=64,
    ),
    num_buckets=14,
    hashes_per_bucket=8,
    n_grams=5,
)
```

So near-duplicate detection is based on **5-gram MinHash**, with **14 buckets × 8 hashes per bucket**, using **64-bit xxhash**. ([GitHub][2])

The dedup stages are:

```text
1. compute MinHash signatures
2. put signatures into buckets
3. form duplicate clusters
4. remove duplicate IDs
```

The pipeline also saves `minhash_cluster_size`, so the kept document knows how many duplicates were in its cluster. ([GitHub][2])

## 3. Gopher repetition filter for Indonesian

This removes documents that are too repetitive. For Indonesian, the config is:

```yaml
dup_line_frac: 0.241

top_n_grams:
  2: 0.194
  3: 0.166
  4: 0.145

dup_n_grams:
  5: 0.166
  6: 0.155
  7: 0.144
  8: 0.133
  9: 0.122
  10: 0.111
```

These values come directly from `ind_Latn.yml`. ([GitHub][3])

The exact rules are:

```python
drop if duplicate_line_count / num_lines > 0.241

drop if top_2gram_char_fraction > 0.194
drop if top_3gram_char_fraction > 0.166
drop if top_4gram_char_fraction > 0.145

drop if duplicated_5gram_char_fraction  > 0.166
drop if duplicated_6gram_char_fraction  > 0.155
drop if duplicated_7gram_char_fraction  > 0.144
drop if duplicated_8gram_char_fraction  > 0.133
drop if duplicated_9gram_char_fraction  > 0.122
drop if duplicated_10gram_char_fraction > 0.111
```

In FineWeb2, the paragraph-related repetition filters are explicitly disabled:

```python
dup_para_frac=0
dup_line_char_frac=0
dup_para_char_frac=0
```

The pipeline comments say these are disabled because trafilatura mostly removes paragraph structure, and FineWeb2 uses a different threshold for duplicated line characters later. ([GitHub][2])

## 4. FineWeb quality filter for Indonesian

This is the FineWeb custom heuristic filter. For Indonesian, the config contributes:

```yaml
line_punct_thr: 0.111
new_line_ratio: 0.157
```

FineWeb2 also globally changes:

```python
short_line_thr = 999      # disabled
char_duplicates_ratio = 0.1
```

The repo README says FineWeb2 removed `short_line_thr`, changed `char_dup_ratio` from `0.01` to `0.1`, disabled paragraph-related repetition filters, and did not use C4 filters. ([GitHub][1])

The exact rules from `FineWebQualityFilter` are:

```python
lines = non_empty_lines(doc.text)

drop if len(lines) == 0

drop if fraction_of_lines_ending_with_terminal_punctuation < 0.111

# disabled because threshold is 999
drop if fraction_of_lines_with_length <= 30 > 999

drop if duplicated_line_character_ratio > 0.1

drop if number_of_newlines / number_of_words > 0.157
```

The filter implementation checks line punctuation, short-line ratio, duplicated line characters, and newline-per-word ratio in that order. ([GitHub][4])

## 5. Gopher quality filter for Indonesian

For Indonesian, the config gives:

```yaml
min_avg_word_length: 3
max_avg_word_length: 12
max_non_alpha_words_ratio: 0.805
stopwords:
  - dan
  - yang
  - di
  - pada
  - dari
  - ini
  - dengan
  - adalah
  - untuk
  - dalam
```

These are directly from `ind_Latn.yml`. ([GitHub][3])

The pipeline passes those into `GopherQualityFilter` and sets:

```python
min_stop_words = 2
```

while leaving several Gopher defaults unchanged. ([GitHub][2])

So the exact Indonesian Gopher-quality rules are:

```python
words = split_into_words(text, "ind_Latn")
non_symbol_words = words that contain at least one non-punctuation char

drop if len(non_symbol_words) < 50
drop if len(non_symbol_words) > 100000

drop if average_length(non_symbol_words) < 3
drop if average_length(non_symbol_words) > 12

drop if text.count("#") / len(words) > 0.1
drop if (text.count("...") + text.count("…")) / len(words) > 0.1

drop if fraction_of_lines_starting_with("•" or "-") > 0.9
drop if fraction_of_lines_ending_with("..." or "…") > 0.3

drop if fraction_of_words_containing_at_least_one_alpha_char < 0.805

drop if document contains fewer than 2 of:
    dan, yang, di, pada, dari, ini, dengan, adalah, untuk, dalam
```

One naming detail: `max_non_alpha_words_ratio` is misleading in DataTrove. The code actually checks whether the fraction of words containing at least one alphabetic character is **below** that threshold. So for Indonesian, at least **80.5%** of words must contain an alphabetic character. ([GitHub][5])

## 6. No C4 filters

FineWeb2 explicitly does **not** apply the C4 filters because they degraded performance in multilingual settings. ([GitHub][1])

In the pipeline, after Gopher repetition, FineWeb quality, and Gopher quality, the comment says:

```python
# we do not apply the C4 filters
```

then writes the filtered output. ([GitHub][2])

## 7. Final cleanup

After filtering, FineWeb2 applies:

```python
FTFYFormatter()        # fix encoding issues
PIIFormatter()         # anonymize PII
SymbolLinesFormatter(symbols_to_remove=["|"], replace_char="\n")
```

So the final Indonesian output has encoding fixes, email/IP-style PII anonymization, and a fix for trafilatura table artifacts where `|` separators are replaced by newlines. ([GitHub][2])

## Compact version

For `ind_Latn`, a document survives only if:

```python
language_score >= 0.685
and not removed_by_global_per_language_minhash
and duplicate_line_fraction <= 0.241
and top_2gram_fraction <= 0.194
and top_3gram_fraction <= 0.166
and top_4gram_fraction <= 0.145
and duplicated_5gram_fraction <= 0.166
and duplicated_6gram_fraction <= 0.155
and duplicated_7gram_fraction <= 0.144
and duplicated_8gram_fraction <= 0.133
and duplicated_9gram_fraction <= 0.122
and duplicated_10gram_fraction <= 0.111
and line_punctuation_fraction >= 0.111
and duplicated_line_character_ratio <= 0.1
and newline_count / word_count <= 0.157
and 50 <= non_symbol_word_count <= 100000
and 3 <= average_non_symbol_word_length <= 12
and hash_symbol_ratio <= 0.1
and ellipsis_ratio <= 0.1
and bullet_line_ratio <= 0.9
and ending_ellipsis_line_ratio <= 0.3
and alpha_word_ratio >= 0.805
and at_least_2_indonesian_stopwords_present
```

Then FineWeb2 applies final formatting/PII fixes.

[1]: https://github.com/huggingface/fineweb-2 "GitHub - huggingface/fineweb-2 · GitHub"
[2]: https://github.com/huggingface/fineweb-2/blob/main/fineweb-2-pipeline.py "fineweb-2/fineweb-2-pipeline.py at main · huggingface/fineweb-2 · GitHub"
[3]: https://github.com/huggingface/fineweb-2/blob/main/configs/ind_Latn.yml "fineweb-2/configs/ind_Latn.yml at main · huggingface/fineweb-2 · GitHub"
[4]: https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/fineweb_quality_filter.py "datatrove/src/datatrove/pipeline/filters/fineweb_quality_filter.py at main · huggingface/datatrove · GitHub"
[5]: https://github.com/huggingface/datatrove/blob/main/src/datatrove/pipeline/filters/gopher_quality_filter.py "datatrove/src/datatrove/pipeline/filters/gopher_quality_filter.py at main · huggingface/datatrove · GitHub"
