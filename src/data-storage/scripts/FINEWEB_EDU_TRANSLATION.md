# FineWeb-Edu ingest + English→Malay translation pipeline

Self-contained `scripts/` tooling (not Ray/remote) that:

1. **Ingests** `HuggingFaceFW/fineweb-edu` and promotes it directly into the
   unified corpus layer as `cleaning_source='fineweb-edu'`.
2. **Translates** those English docs to Malay with `Qwen/Qwen3-14B-AWQ` (vLLM,
   thinking mode) and writes them back as a derived source
   `cleaning_source='fine-web-edu-malay-translate'`, keeping the Qwen3
   reasoning trace.

Everything lands in the **same** `unified_documents` / `unified_document_texts`
tables — no separate raw table for the translation (provenance is implicit in
`sample_uid`).

## Data flow

```
HuggingFaceFW/fineweb-edu (parquet, sample/100BT)
        │  ingest_fineweb_edu.py
        ▼
fineweb_edu_documents (raw)  ─┐
        ├─► unified_documents            cleaning_source='fineweb-edu'
        ├─► unified_document_texts_fineweb_edu        (English text)
        └─► document_language_detection  'standard-english' (direct, no model)
        │
        │  translate_fineweb_edu_malay.py  (local vLLM, Qwen3-14B-AWQ, thinking)
        ▼
        ├─► unified_documents            cleaning_source='fine-web-edu-malay-translate'
        ├─► unified_document_texts_fine_web_edu_malay_translate   (Malay text)
        ├─► document_language_detection  'standard-malay' (direct, no model)
        └─► document_translation_thinking  (the <think>…</think> trace)
```

`sample_uid` links a translation to its origin:
`fine-web-edu-malay-translate://<hf_id>` ↔ `fineweb-edu://<hf_id>`.
`original_text_sha256` = English-source hash; `cleaned_text_sha256` = Malay hash;
`cleaning_rules_triggered = {qwen3-14b-awq-en-ms-translate}`.

## Files

| File | Role |
|---|---|
| `ingest_fineweb_edu.py` | Download HF parquet, promote into unified layer (+direct `standard-english` LID). Resume-safe (`ON CONFLICT DO NOTHING`, JSON progress). |
| `migrations/add_fineweb_edu.sql` | Register `fineweb-edu` source, raw table, `…_fineweb_edu` partition. |
| `migrations/add_fine_web_edu_malay_translate.sql` | Register the derived translate source + `…_fine_web_edu_malay_translate` partition (no raw table). |
| `migrations/add_document_translation_thinking_table.sql` | `document_translation_thinking` (one row per translation `doc_id`, FK→`unified_documents`). |
| `serve_qwen3_translate.sh` | Brings up the local vLLM OpenAI server for Qwen3-14B-AWQ (the proven recipe). |
| `translate_fineweb_edu_malay.py` | Reads fineweb-edu unified docs, translates via the local server, writes the 4 tables. Resume-safe anti-join. |

## DB objects

- `data_sources`: `fineweb-edu` and `fine-web-edu-malay-translate` (both `web`).
- Partitions: `unified_document_texts_fineweb_edu`,
  `unified_document_texts_fine_web_edu_malay_translate`.
- `document_translation_thinking(doc_id PK/FK, thinking_text lz4,
  thinking_char_count, thinking_token_count, created_at)` — keyed by the
  **translation** doc_id; multi-chunk docs concatenate their `<think>` bodies.

See `../DATABASE.md` for full schema/ERD and example queries (incl. the
origin-link join).

## Setup — the vLLM recipe (do not improvise)

Replicates `/home/geeyang/workspace/minimind-mfu-working/evaluation`
(`run_pretrained_suite.sh`, `sea_helm_ms/README.md`). Quirks that bit us and are
now baked into `serve_qwen3_translate.sh`:

- **Isolated venv** `.venv-qwen-translate` (repo root): `uv pip install vllm
  transformers openai asyncpg ninja`. `vllm` pulls a self-consistent torch +
  flashinfer — **do not** add standalone `flash-attn`.
- **`ninja` must be on the global PATH**: flashinfer JIT-compiles its sampling
  kernel at runtime via a bare `ninja` subprocess. vLLM's `EngineCore` child
  doesn't get the venv PATH, so symlink it:
  `ln -sf <repo>/.venv-qwen-translate/bin/ninja ~/.local/bin/ninja`.
- **`CUDA_HOME` must match torch**: torch is built for cu130 but
  `/usr/local/cuda` may point at 12.8 — the script pins
  `CUDA_HOME=/usr/local/cuda-13.0`.
- `--quantization awq_marlin` (Qwen3-14B-AWQ is 4-bit AWQ; `fp4` needs a
  Blackwell GPU). `--enforce-eager` **always**. **No** `--reasoning-parser`
  (the `<think>` block must stay in `content`; the runner splits it). **No**
  `VLLM_ATTENTION_BACKEND` override (flashinfer is implicit).
- **16 GB GPU note:** Qwen3-14B-AWQ weights ≈ 9.4 GiB; on a 16 GB card use
  `GPU_UTIL≈0.85` and `MAX_MODEL_LEN=8192` (KV pool ≈ 22,992 tokens). 32k ctx
  OOMs there. First start also pays a one-time flashinfer JIT compile.

## Running

```bash
# 0. one-time: migrations (idempotent)
docker compose -f src/data-storage/docker-compose.yml exec -T postgres \
  psql -U corpus -f /scripts/migrations/add_fineweb_edu.sql
docker compose -f src/data-storage/docker-compose.yml exec -T postgres \
  psql -U corpus -f /scripts/migrations/add_fine_web_edu_malay_translate.sql
docker compose -f src/data-storage/docker-compose.yml exec -T postgres \
  psql -U corpus -f /scripts/migrations/add_document_translation_thinking_table.sql

# 1. ingest fineweb-edu (resumable)
python3 src/data-storage/scripts/ingest_fineweb_edu.py --workers 2 --batch-size 5000

# 2. terminal A: serve the model (stays foreground; Ctrl-C cleans the GPU)
CUDA_HOME=/usr/local/cuda-13.0 MAX_MODEL_LEN=8192 GPU_UTIL=0.85 \
  bash src/data-storage/scripts/serve_qwen3_translate.sh

# 3. terminal B: translate (after "vLLM ready")
.venv-qwen-translate/bin/python src/data-storage/scripts/translate_fineweb_edu_malay.py \
  --batch-size 6 --concurrency 6 --max-model-len 8192 --max-output-tokens 5120 \
  --base-url http://127.0.0.1:8000/v1 --api-key translate
```

Useful flags: `--dry-run` (roll back all writes), `--no-llm` (passthrough
English — exercises the DB path without a GPU), `--limit N`,
`--temperature/--top-p/--top-k` (default Qwen3 thinking: 0.6 / 0.95 / 20).

### Resume semantics

The translator selects fineweb-edu docs that have **no** matching
`fine-web-edu-malay-translate://<hf_id>` row (anti-join, ordered by `doc_id`).
All four inserts are `ON CONFLICT DO NOTHING` in one per-batch transaction. So a
crash/stop loses at most the in-flight batch; just re-run the same command. No
MAX-watermark (avoids the poisoning issue from commit `f2d019b`). First page is a
one-time slow PK scan past non-fineweb-edu doc_ids; later pages are fast.

## Measured throughput (RTX 4090 Laptop, AWQ 4-bit, ctx 8192, thinking mode)

- Single stream ≈ 30 tok/s; **concurrency 6 ≈ ~190–200 tok/s aggregate**, KV
  ~50–62 % (headroom).
- Concurrency **scaling tested**: at concurrency 18 vLLM only keeps ~4–6 running
  (KV pegged 78–98 %), aggregate ~100–158 tok/s — **lower** than concurrency 6.
  At ctx 8192 the KV pool (22,992 tokens) caps effective concurrency at ~6;
  **concurrency 6 is the operating point** (raising it does not scale).
- Steady-state ≈ ~0.035 docs/s ≈ ~3,000 docs/day on one GPU; highly
  length-dependent (short ~10 s/doc, long multi-chunk ~45 s/doc). Thinking mode
  ~doubles output tokens vs plain translation.
- Full-corpus scale on a single laptop GPU is impractical (~years for millions
  of docs). Levers: shard across a GPU fleet (linear), bigger GPU (A100/H100 →
  full 32k ctx, higher concurrency), or narrow scope (e.g. `edu_int_score>=4`).

## Examples

Real committed rows (Malay translation truncated to the first ~1000 chars; the
Qwen3 `<think>` trace truncated for readability — full text is in
`unified_document_texts` / `document_translation_thinking`). Query used:

```sql
SELECT u.doc_id, split_part(u.sample_uid,'://',2) AS hf_id,
       u.cleaned_char_count, dtt.thinking_token_count,
       left(t.cleaned_text,1000), left(dtt.thinking_text,1100)
FROM unified_documents u
JOIN unified_document_texts t USING(doc_id)
JOIN document_translation_thinking dtt USING(doc_id)
WHERE u.cleaning_source='fine-web-edu-malay-translate'
ORDER BY u.doc_id LIMIT 3;
```

### Example 1 — doc_id 81475641 (Jane Austen)

`hf_id=<urn:uuid:0d8a309d-25c5-405d-a08a-c11239f0d717>` ·
`cleaned_char_count=19173` · `thinking_token_count=565`

**Malay translation (first ~1000 chars):**

```
Jane yang Merdeka
Walaupun buku-buku Jane Austen penuh dengan cinta, romansa dan skandal, sebenarnya ia lebih tentang kebebasan dan kemerdekaan. Kemerdekaan berfikir dan kebebasan untuk memilih.
Pengakuan Elizabeth terhadap tawaran perkahwinan Tuan Collins menunjukkan kemerdekaan yang jarang dilihat dalam tokoh utama pada masa itu. Penolakannya terhadap Tuan Darcy, walaupun disebabkan oleh marah, menunjukkan tahap kemerdekaan yang mengejutkan beliau.
Kebebasan yang ditunjukkan Elizabeth apabila akhirnya menerima beliau secara langsung menentang Lady Catherine, dan mengetahui ayahnya tidak bersetuju, adalah sesuatu yang tidak biasa bahkan bagi Austen. Dalam buku terakhirnya, Anne Elliot dipersuasikan oleh Lady Russel untuk menolak Kapten Wentworth.
Walaupun Jane mengikuti peraturan pada masa itu, semua tulisannya penuh dengan bagaimana dia ingin kehidupan itu berjalan. Dia "menjerit" kekesalannya terhadap keterbatasan yang dihadapi wanita dalam Emma.
Apabila dihampiri oleh Mrs. Elton, J
```

**Thinking trace (truncated):**

```
Okay, the user wants me to translate this English text about Jane Austen into Malay. Let me start by reading through the original text to understand the context and main points. The text discusses Austen's themes of freedom and independence, her characters like Elizabeth Bennet and Anne Elliot, and even compares her impact to historical figures like John Adams and Thomas Jefferson.

First, I need to make sure I capture the nuances of terms like "independence of thought" and "freedom to choose" accurately in Malay. Words like "independen" and "kebebasan" are direct translations, but I should check if they're commonly used in that context. Also, names like Lady Catherine and characters like Mr. Collins need to remain as they are, but I should ensure that the titles are appropriately translated, like "Tuan Collins" for Mr. Collins.

The user mentioned preserving the structure and paragraph breaks, so I'll need to maintain the original formatting. There are several quotes from the text, so I have to be careful with punctuation and quotation marks in Malay...
```

### Example 2 — doc_id 81475642 (tornadoes / wind shear)

`hf_id=<urn:uuid:7400301c-e625-46d5-be90-1020cf8d52f8>` ·
`cleaned_char_count=2844` · `thinking_token_count=465`

**Malay translation (first ~1000 chars):**

```
Badai puting beliung merupakan badai yang paling hebat di planet ini, dan ia selalu dibincangkan tanpa sekurang-kurangnya sedikit pendedahan tentang istilah "shear angin". Namun, ramai daripada kita yang duduk di rumah mungkin tidak tahu apa itu shear angin, atau jika mereka tahu, bagaimana ia mempengaruhi pembentukan puting beliung.

Apa Itu Shear Angin
Shear angin, walaupun mungkin terdengar rumit, sebenarnya merupakan konsep yang mudah. Shear angin hanyalah perubahan dalam kekuatan dan arah angin dengan ketinggian. Saya rasa kita semua memahami bahawa angin biasanya lebih kuat di atmosfer di atas kepala kita berbanding di permukaan bumi, dan jika kita memikirkan atmosfer dalam tiga dimensi yang ia miliki, ia tidak sepatutnya mengejutkan jika angin di atas kita juga bertiup dari arah berbeza berbanding angin di permukaan bumi. Apabila berlaku demikian - iaitu apabila kekuatan dan arah angin berubah mengikut ketinggian - shear angin sedang berlaku.

Shear Angin dan Badai Petir Supercel
```

**Thinking trace (truncated):**

```
Okay, let's tackle this translation. The user wants the English text translated into natural Malay. First, I need to understand the content thoroughly. The text explains tornadoes, wind shear, and how they relate to supercell thunderstorms. It's technical but not overly complex.

I should start by translating each paragraph, maintaining the structure and meaning. Terms like "wind shear" need to be accurately translated. In Malay, "shear" might be "pemotongan" or "perubahan". But since wind shear is a specific meteorological term, I should check if there's a standard translation. Maybe "shear angin" is acceptable, but I need to ensure it's commonly understood.

Next, the paragraph about wind shear and supercell thunderstorms. The key here is to keep the scientific accuracy while making the language flow naturally in Malay. Phrases like "updraft" can be translated as "aliran udara ke atas"...
```

### Example 3 — doc_id 81475643 (Neanderthal "flute")

`hf_id=<urn:uuid:f166f15d-9976-40ed-8a49-8bed360001ff>` ·
`cleaned_char_count=10928` · `thinking_token_count=680`

**Malay translation (first ~1000 chars):**

```
Adakah tulang ini merupakan fluit Neanderthal?
Fragmen femur kitar kubur dari Slovenia, 43 ribu tahun yang lalu
KERAGUAN DI KELUARAN TENTANG TULANG NEANDERTHAL 'FLUIT'
(DAN JAWAPAN DARI MUSIKOLOGIS BOB FINK)
Science News 153 (4 April, 1998): 215.
Oleh B. Bower
Dalam kegemilangan media yang besar, satu kumpulan penyelidik pada tahun 1996 mengumumkan tulang kosong yang ditemui dalam gua Eropah Timur, berlubang pada satu sisi dengan empat lubang lengkap atau sebahagian, sebagai alat muzik tertua yang diketahui. Para saintis menyatakan bahawa fluit yang berlubang itu, ditemui dalam gua Eropah Timur, merupakan fluit yang dibuat dan dimainkan oleh Neanderthal sekurang-kurangnya 43,000 tahun yang lalu.
Kini, tiba masanya untuk berhenti memainkan muzik, kata dua arkeolog yang memeriksa fluit yang dikatakan itu pada musim lalu. Dengan pemerhatian yang lebih hampir, tulang itu kelihatan telah dilubang dan dikunyah oleh gigi haiwan - mungkin serigala - semasa mengupas daging dan sum-sum dari tula
```

**Thinking trace (truncated):**

```
Okay, I need to translate this long English text into Malay. Let me start by reading through the entire text to understand the context. It's about a Neanderthal bone possibly being a flute, with arguments from archaeologists and a musicologist.

First, I'll break down the text into paragraphs to ensure I translate each section accurately. The title is "DOUBTS AIRED OVER NEANDERTHAL BONE 'FLUTE'" which I should keep as is but maybe adjust to natural Malay. The names like Bob Fink, Science News, and others should remain in English as proper nouns.

The user mentioned preserving meaning, structure, and paragraph breaks. I need to make sure that the translation flows naturally in Malay, using appropriate terms for scientific concepts. For example, "femur fragment" becomes "fragmen femur". Terms like "carnivores" should be translated to "karnivor" or "pemangsa" depending on context...
```
