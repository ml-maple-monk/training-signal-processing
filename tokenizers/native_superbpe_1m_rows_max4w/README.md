# native_superbpe_1m_rows_max4w

This directory contains the current best tokenizer artifact for this repo.

- Tokenizer JSON: `tokenizer.json`
- Source run id: `superbpe-interleaved-1m-oneline-besteffort-20260502T215607Z`
- Training shape: native SuperBPE, `1000000` best-effort interleaved rows,
  Stage 2 ingestion cap `stage2_max_words_per_token = 4`
- Vocab size: `50000`
- Max token length: `32`
- Measured fertility: `1.294` tokens per whitespace word
- Token reduction vs `local_bpeasy_balanced_1to1`: `7.94%`

Usage:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "tokenizers/native_superbpe_1m_rows_max4w/tokenizer.json"
)
ids = tokenizer.encode("Saya pergi ke pasar untuk membeli ikan dan sayur.").ids
```

See `TOKENIZER_DOCS.md` for performance tables and caveats.

Reproduce the published fertility numbers with:

```bash
uv run --group tokenizer_training python analysis/tokenizer/reproduce_native_superbpe_1m_rows_max4w.py
```
