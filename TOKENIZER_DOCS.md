# Tokenizer Performance Notes

## Best Tokenizer To Use

Use `native_superbpe_1m_rows_max4w` as the current best tokenizer.

- Tokenizer JSON: `tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json`
- Source run id: `superbpe-interleaved-1m-oneline-besteffort-20260502T215607Z`
- Training shape: native SuperBPE, `1000000` best-effort interleaved rows, Stage 2 ingestion cap `stage2_max_words_per_token = 4`
- Vocab size: `50000`
- Max token length: `32`

Tracked comparison tokenizer artifacts:

- `local_bpeasy_balanced_1to1`: `tokenizers/local_bpeasy_balanced_1to1/tokenizer.json`
- `local_bpeasy_previous_40g_cache`: `tokenizers/local_bpeasy_previous_40g_cache/tokenizer.json`

The SuperBPE artifact is a HuggingFace `tokenizers` JSON. The two BPEEasy
comparison artifacts are BPEEasy JSON files and should be loaded with
`bpeasy.tokenizer.BPEasyTokenizer`.

Load it with HuggingFace `tokenizers` for encoding:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json"
)

ids = tokenizer.encode("Saya pergi ke pasar untuk membeli ikan dan sayur.").ids
```

For byte-exact decoding, use the byte-level decoder helper in
`analysis/tokenizer/evaluate_superbpe_fertility.py`. Plain
`Tokenizer.decode()` exposes GPT-2 byte-level space markers for this exported
tokenizer.

## Evaluation Policy

Fertility means `encoded tokens / whitespace words`, where whitespace words are
counted with regex `\S+`. Lower is better.

The evaluation sample is the same bounded domain sample used for the previous
tokenizer comparisons:

- up to `5000` documents per source
- up to `33554432` UTF-8 bytes per source
- FineWeb files sampled evenly across `2098` cached FineWeb parquet files

## Rerun The Fertility Experiment

The reproduction command re-encodes the same bounded local parquet sample with
the tracked SuperBPE tokenizer, writes fresh evaluation JSON/Markdown under
`.runtime/tokenizers/experiments/native_superbpe_1m_rows_max4w/`, and checks the
result against `analysis/tokenizer/native_superbpe_1m_rows_max4w_expected.json`.

```bash
uv run --group tokenizer_training python analysis/tokenizer/reproduce_native_superbpe_1m_rows_max4w.py
```

Prerequisite: the local parquet cache must exist at
`.runtime/tokenizers/parquet-cache/20260501T211207Z-rclone-balanced-fineweb-1to1`.

To verify the published expected metrics against an already generated evaluation
JSON without rereading the parquet cache:

```bash
uv run --group tokenizer_training python analysis/tokenizer/reproduce_native_superbpe_1m_rows_max4w.py \
  --input-json .runtime/tokenizers/experiments/native_superbpe_1m_rows_max4w/superbpe_tokenizer_evaluation.json
```

## Aggregate Performance

`native_superbpe_1m_rows_max4w` has the best measured fertility in this repo.

| Tokenizer | Vocab | Fertility | Chars/token | Bytes/token | Tokens/1000 chars | Roundtrip | Encoded tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `native_superbpe_1m_rows_max4w` | 50000 | 1.294 | 5.380 | 5.401 | 185.891 | 30649/30650 | 22603105 |
| `local_bpeasy_balanced_1to1` | 50000 | 1.406 | 4.953 | 4.973 | 201.913 | 30650/30650 | 24551254 |
| `local_bpeasy_previous_40g_cache` | 50000 | 1.369 | 5.084 | 5.105 | 196.677 | 30650/30650 | 23914634 |
| `olmo3_7b_instruct` | 100278 | 1.955 | 3.561 | 3.575 | 280.846 | 30650/30650 | 34149062 |
| `qwen3_0_6b` | 151669 | 1.982 | 3.513 | 3.527 | 284.676 | 30650/30650 | 34614789 |

Relative to `local_bpeasy_balanced_1to1`, the current SuperBPE tokenizer uses
`7.94%` fewer encoded tokens overall (`22603105` vs `24551254`). Compared with
the earlier 100k-row SuperBPE run, it uses `3.77%` fewer tokens (`22603105` vs
`23487948`).

## Per-Source Fertility

| Source | Docs | MiB | Words | `native_superbpe_1m_rows_max4w` Tokens | `native_superbpe_1m_rows_max4w` Fertility | BPEasy Fertility | Token Reduction vs BPEasy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `books-ocr` | 330 | 32.08 | 4711340 | 6796032 | 1.442 | 1.545 | 6.64% |
| `cari` | 5000 | 1.63 | 253497 | 374889 | 1.479 | 1.509 | 1.97% |
| `hplt-indonesia` | 3655 | 32.14 | 4714412 | 5592178 | 1.186 | 1.288 | 7.87% |
| `hplt-malay` | 1665 | 32.00 | 4551574 | 5952227 | 1.308 | 1.429 | 8.47% |
| `lowyat` | 5000 | 0.85 | 151248 | 193805 | 1.281 | 1.400 | 8.46% |
| `reddit-bolehland` | 5000 | 0.64 | 122018 | 140982 | 1.155 | 1.286 | 10.15% |
| `reddit-indonesia` | 5000 | 0.73 | 123673 | 177140 | 1.432 | 1.488 | 3.72% |
| `fineweb` | 5000 | 16.34 | 2836499 | 3375852 | 1.190 | 1.327 | 10.31% |

## Per-Source Comparison

| Source | `native_superbpe_1m_rows_max4w` | Current local BPEasy | Previous local BPEasy | OLMo 3 | Qwen3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `books-ocr` | 1.442 | 1.545 | 1.492 | 2.128 | 2.169 |
| `cari` | 1.479 | 1.509 | 1.444 | 2.143 | 2.171 |
| `hplt-indonesia` | 1.186 | 1.288 | 1.247 | 1.909 | 1.924 |
| `hplt-malay` | 1.308 | 1.429 | 1.357 | 2.263 | 2.288 |
| `lowyat` | 1.281 | 1.400 | 1.417 | 1.435 | 1.462 |
| `reddit-bolehland` | 1.155 | 1.286 | 1.305 | 1.308 | 1.313 |
| `reddit-indonesia` | 1.432 | 1.488 | 1.451 | 1.859 | 1.867 |
| `fineweb` | 1.190 | 1.327 | 1.378 | 1.295 | 1.321 |

## Evaluation Artifacts

- Reproduction command:
  `uv run --group tokenizer_training python analysis/tokenizer/reproduce_native_superbpe_1m_rows_max4w.py`
- Expected metrics checked by the reproduction command:
  `analysis/tokenizer/native_superbpe_1m_rows_max4w_expected.json`
- Tracked SuperBPE tokenizer: `tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json`
- Tracked BPEasy baseline tokenizer: `tokenizers/local_bpeasy_balanced_1to1/tokenizer.json`
- Tracked previous BPEasy tokenizer: `tokenizers/local_bpeasy_previous_40g_cache/tokenizer.json`
- SuperBPE evaluation JSON: `.runtime/tokenizers/experiments/superbpe-interleaved-1m-oneline-besteffort-20260502T215607Z/superbpe_tokenizer_evaluation.json`
- SuperBPE evaluation Markdown: `.runtime/tokenizers/experiments/superbpe-interleaved-1m-oneline-besteffort-20260502T215607Z/superbpe_tokenizer_evaluation.md`
- BPEasy baseline evaluation JSON: `.runtime/tokenizers/experiments/20260501T213354Z-balanced-fineweb-1to1-full-b1024-t16/tokenizer_evaluation.json`
- Full external-tokenizer comparison JSON: `.runtime/tokenizers/experiments/20260501T213354Z-balanced-fineweb-1to1-full-b1024-t16/tokenizer_fertility_comparison.json`

## Caveats

- Fertility was measured on a bounded stratified sample, not the entire corpus.
- `native_superbpe_1m_rows_max4w` round-tripped `30649/30650` sampled documents
  exactly. The single mismatch was in a Reddit Bolehland row containing Unicode
  symbol/emoji-style text; fertility counts are unaffected.
- OLMo 3 and Qwen3 emitted model-context warnings for some long OCR documents
  during comparison. That means those documents exceed model context limits, not
  that tokenizer encoding failed.
