# local_bpeasy_balanced_1to1

Tracked BPEasy baseline tokenizer used in `TOKENIZER_DOCS.md`.

- Tokenizer JSON: `tokenizer.json`
- Source run id: `20260501T213354Z-balanced-fineweb-1to1-full-b1024-t16`
- Training sampled rows: `16014943`
- Training sampled bytes: `45904405188`
- Vocab size: `50000`
- Max token length: `128`
- Measured fertility: `1.406` tokens per whitespace word

Usage:

```python
from bpeasy.tokenizer import BPEasyTokenizer

tokenizer = BPEasyTokenizer.from_file(
    "tokenizers/local_bpeasy_balanced_1to1/tokenizer.json"
)
ids = tokenizer.encode("Saya pergi ke pasar untuk membeli ikan dan sayur.")
```
