# local_bpeasy_previous_40g_cache

Tracked earlier BPEasy tokenizer used in `TOKENIZER_DOCS.md`.

- Tokenizer JSON: `tokenizer.json`
- Source run id: `20260501T201106Z-rclone-40g-t16-full-b1024-t16`
- Training sampled rows: `6874279`
- Training sampled bytes: `16955218883`
- Vocab size: `50000`
- Max token length: `128`
- Measured fertility: `1.369` tokens per whitespace word

Usage:

```python
from bpeasy.tokenizer import BPEasyTokenizer

tokenizer = BPEasyTokenizer.from_file(
    "tokenizers/local_bpeasy_previous_40g_cache/tokenizer.json"
)
ids = tokenizer.encode("Saya pergi ke pasar untuk membeli ikan dan sayur.")
```
