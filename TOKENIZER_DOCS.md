# BPEasy Tokenizer Documentation

Run id: `20260501T213354Z-balanced-fineweb-1to1-full-b1024-t16`

This document describes the completed tokenizer trained on the FineWeb-balanced local cache and records structural checks, exact roundtrip checks, and fertility metrics.

## Artifacts

- Tokenizer JSON: `.runtime/tokenizers/20260501T213354Z-balanced-fineweb-1to1-full-b1024-t16/tokenizer.json`
- Training summary: `.runtime/tokenizers/20260501T213354Z-balanced-fineweb-1to1-full-b1024-t16/training_summary.json`
- Evaluation JSON: `.runtime/tokenizers/experiments/20260501T213354Z-balanced-fineweb-1to1-full-b1024-t16/tokenizer_evaluation.json`
- Time log: `.runtime/tokenizers/experiments/20260501T213354Z-balanced-fineweb-1to1-full-b1024-t16/20260501T213354Z-balanced-fineweb-1to1-full-b1024-t16.time.txt`

Only `tokenizer.json` was exported for this run. HuggingFace and tiktoken exports were disabled in the training command.

## Training Data

- Local cache: `.runtime/tokenizers/parquet-cache/20260501T211207Z-rclone-balanced-fineweb-1to1`
- Cache balance by compressed parquet bytes: non-FineWeb `37.082 GiB`, FineWeb `37.091 GiB`.
- Training sampled text bytes: `45904405188`.
- Training sampled rows: `16014943`.
- Stop reason: `exhausted`.

Training source row counts:

- `books-ocr`: `1152`
- `cari`: `158751`
- `fineweb`: `10569035`
- `hplt-indonesia`: `2015431`
- `hplt-malay`: `1463783`
- `lowyat`: `1722367`
- `reddit-bolehland`: `59293`
- `reddit-indonesia`: `25131`

## Tokenizer Configuration

- Name: `final-merged-bpeasy`
- Vocab size: `50000`
- Special tokens: `[]`
- Regex pattern: `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
- Max token length: `128`
- BPEasy batch size during training: `1024`
- Rayon threads during training: `16`

The vocab keys in `tokenizer.json` are base64-encoded byte pieces. This is expected for BPEasy byte-level BPE output.

## Structural Validation

- IDs contiguous from `0` to `49999`: `True`
- Unique IDs: `50000`
- Base64 decode errors: `0`
- Complete single-byte fallback `0..255`: `True`
- Duplicate decoded byte pieces: `0`
- Empty decoded byte pieces: `0`
- UTF-8-valid pieces: `49750`
- Non-UTF-8 byte pieces: `250`
- Max decoded piece length: `64` bytes
- Pieces over max token length `128`: `0`

## Roundtrip Checks

- Fixed smoke samples exact roundtrip: `True`
- Evaluation sample exact roundtrip rate: `1.000000`
- Evaluation sample roundtrip failures: `0`

Example load and roundtrip code:

```python
from bpeasy.tokenizer import BPEasyTokenizer

tokenizer = BPEasyTokenizer.from_file(
    ".runtime/tokenizers/20260501T213354Z-balanced-fineweb-1to1-full-b1024-t16/tokenizer.json"
)
text = "Saya pergi ke pasar untuk membeli ikan dan sayur."
ids = tokenizer.encode(text)
assert tokenizer.decode(ids) == text
```

## Fertility Metrics

Fertility here means `encoded BPE tokens / whitespace words`, where words are counted with regex `\S+`. Lower fertility means fewer tokenizer pieces per whitespace-delimited word. Character and byte efficiency are reported as complementary metrics.

Evaluation sample policy:

- Up to `5000` documents per source.
- Up to `33554432` UTF-8 bytes per source.
- FineWeb files sampled evenly across `2098` cached FineWeb parquet files.

Aggregate sample metrics:

- Documents: `30650`
- UTF-8 bytes: `122083251`
- Whitespace words: `17464261`
- Encoded tokens: `24551254`
- Fertility tokens/word: `1.406`
- Characters/token: `4.953`
- Bytes/token: `4.973`
- Tokens/1000 chars: `201.913`

| Source | Docs | MiB | Words | Tokens | Fertility | Chars/token | Bytes/token | Roundtrip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `books-ocr` | 330 | 32.08 | 4711340 | 7279243 | 1.545 | 4.600 | 4.622 | 330/330 |
| `cari` | 5000 | 1.63 | 253497 | 382437 | 1.509 | 4.429 | 4.466 | 5000/5000 |
| `hplt-indonesia` | 3655 | 32.14 | 4714412 | 6070103 | 1.288 | 5.535 | 5.553 | 3655/3655 |
| `hplt-malay` | 1665 | 32.00 | 4551574 | 6502798 | 1.429 | 5.148 | 5.161 | 1665/1665 |
| `lowyat` | 5000 | 0.85 | 151248 | 211715 | 1.400 | 4.203 | 4.220 | 5000/5000 |
| `reddit-bolehland` | 5000 | 0.64 | 122018 | 156916 | 1.286 | 4.264 | 4.292 | 5000/5000 |
| `reddit-indonesia` | 5000 | 0.73 | 123673 | 183989 | 1.488 | 4.151 | 4.172 | 5000/5000 |
| `fineweb` | 5000 | 16.34 | 2836499 | 3764053 | 1.327 | 4.522 | 4.552 | 5000/5000 |

## Runtime Evidence

- User time (seconds): `28288.14`
- System time (seconds): `2508.68`
- Percent of CPU this job got: `719%`
- Maximum resident set size (kbytes): `23223440`
- Exit status: `0`

## Caveats

- Fertility was measured on a bounded stratified sample from the local cache, not the entire training corpus.
- FineWeb and non-FineWeb were balanced by compressed parquet bytes in the cache; accepted cleaned-text bytes still differ after filtering.
- The tokenizer is byte-level, so non-UTF-8 byte pieces in the vocab are expected and preserve arbitrary-byte fallback behavior.
