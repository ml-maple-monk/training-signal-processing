# SuperBPE Fertility Evaluation

Run id: `native_superbpe_1m_rows_max4w`

## Artifacts

- Tokenizer JSON: `tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json`
- Training summary: `None`
- Evaluation JSON: `.runtime/tokenizers/experiments/native_superbpe_1m_rows_max4w/superbpe_tokenizer_evaluation.json`

## Training Snapshot

- Training rows: `unknown`
- Training UTF-8 bytes: `unknown`
- Stop reason: `unknown`
- Vocab size: `unknown`
- Max token length: `unknown`
- Stage 2 inherited merges: `unknown`
- Stage 2 max words per token: `unknown`
- Stage 2 ingest seconds: `unknown`
- Stage 2 train seconds: `unknown`

## Fertility Metrics

Fertility here means `encoded SuperBPE tokens / whitespace words`, where words are counted
with regex `\S+`. Lower fertility means fewer tokenizer pieces per whitespace-delimited word.

Evaluation sample policy:

- Up to `5000` documents per source.
- Up to `33554432` UTF-8 bytes per source.
- evenly spaced 160 files from 2098 cached FineWeb parquet files.

Aggregate sample metrics:

- Documents: `30650`
- UTF-8 bytes: `122083251`
- Whitespace words: `17464261`
- Encoded tokens: `22603105`
- Fertility tokens/word: `1.294`
- Characters/token: `5.380`
- Bytes/token: `5.401`
- Tokens/1000 chars: `185.891`
- Roundtrip exact rate: `0.999967`

| Source | Docs | MiB | Words | Tokens | Fertility | Chars/token | Bytes/token | Roundtrip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `books-ocr` | 330 | 32.08 | 4711340 | 6796032 | 1.442 | 4.927 | 4.950 | 330/330 |
| `cari` | 5000 | 1.63 | 253497 | 374889 | 1.479 | 4.518 | 4.555 | 5000/5000 |
| `hplt-indonesia` | 3655 | 32.14 | 4714412 | 5592178 | 1.186 | 6.008 | 6.027 | 3655/3655 |
| `hplt-malay` | 1665 | 32.00 | 4551574 | 5952227 | 1.308 | 5.624 | 5.638 | 1665/1665 |
| `lowyat` | 5000 | 0.85 | 151248 | 193805 | 1.281 | 4.591 | 4.610 | 5000/5000 |
| `reddit-bolehland` | 5000 | 0.64 | 122018 | 140982 | 1.155 | 4.746 | 4.778 | 4999/5000 |
| `reddit-indonesia` | 5000 | 0.73 | 123673 | 177140 | 1.432 | 4.311 | 4.333 | 5000/5000 |
| `fineweb` | 5000 | 16.34 | 2836499 | 3375852 | 1.190 | 5.042 | 5.076 | 5000/5000 |
