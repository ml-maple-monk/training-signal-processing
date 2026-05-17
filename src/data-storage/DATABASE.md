# Corpus Database Reference

PostgreSQL 18 corpus DB for the NLP pretraining pipeline. Stores ~38M cleaned documents from 9 heterogeneous sources with language-ID metadata, quality signals, and a training manifest view.

## Connecting

```bash
# Local (Docker)
psql "host=localhost port=5432 dbname=corpus user=corpus password=corpus_secret"

# From inside the container
docker compose -f src/data-storage/docker-compose.yml exec postgres psql -U corpus
```

---

## Architecture Overview

The schema has three layers:

```
┌─────────────────────────────────────────────────────────────────────┐
│  REGISTRY                                                           │
│  data_sources · pipeline_runs                                       │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PER-SOURCE TABLES  (raw data, one table per source)               │
│                                                                     │
│  Forums:   lowyat_threads → lowyat_posts                           │
│            cari_threads   → cari_posts                             │
│  Reddit:   reddit_bolehland_posts                                  │
│            reddit_indonesia_posts                                   │
│  Web:      hplt_malay_documents                                    │
│            hplt_indonesia_documents                                 │
│            sea_pile_malay_documents   ← raw, not yet in unified    │
│            fineweb_documents                                        │
│            fineweb_edu_documents      → promoted into unified      │
│  Derived:  fine-web-edu-malay-translate  (no raw table; Qwen3 MT)  │
│  Books:    ocr_books                                               │
└─────────────────────────────────────────────────────────────────────┘
                            │  (ingest pipeline populates both layers)
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  UNIFIED LAYER  (cleaning pipeline output)                         │
│                                                                     │
│  unified_documents          ← metadata hub (non-partitioned)       │
│  unified_document_texts     ← lz4 text, partitioned by source      │
│  lid_metadata               ← language-ID per document             │
│  quality_runs + quality_signals ← quality scoring                  │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  VIEWS                                                              │
│  training_manifest (materialized) · source_stats · quality_filtered_manifest │
└─────────────────────────────────────────────────────────────────────┘
```

**Why two layers?** PostgreSQL requires unique constraints on partitioned tables to include the partition key. Keeping `unified_documents` non-partitioned gives it a plain `BIGINT` PK that `lid_metadata` and `quality_signals` can reference with single-column FKs. The 84GB of text lives in `unified_document_texts`, partitioned by `cleaning_source`, so each source's TOAST I/O is isolated (HPLT queries don't page in forum TOAST).

---

## Full ERD

```
┌─────────────────────────┐
│      data_sources       │
├─────────────────────────┤
│ PK source_id  SMALLINT  │◄──────────────────────────────────────────┐
│    source_name TEXT      │                                           │
│    source_type TEXT      │                                           │
│    description TEXT      │                                           │
│    created_at  TSTZ      │                                           │
└─────────────────────────┘                                           │
           │                                                           │
           │ 1:N                                                       │
           ▼                                                           │
┌─────────────────────────┐                                           │
│     pipeline_runs       │                                           │
├─────────────────────────┤                                           │
│ PK run_id     TEXT       │                                           │
│    pipeline_name TEXT    │                                           │
│ FK source_id  SMALLINT  │                                           │
│    status     TEXT       │                                           │
│    row_count  BIGINT     │                                           │
│    token_count BIGINT    │                                           │
│    started_at  TSTZ      │                                           │
│    completed_at TSTZ     │                                           │
└─────────────────────────┘                                           │
                                                                       │
┌──────────────────────────────────────────────────────────────────┐  │
│                    UNIFIED LAYER                                  │  │
│                                                                   │  │
│  ┌────────────────────────────────────────────────────────────┐  │  │
│  │                   unified_documents                        │  │  │
│  ├────────────────────────────────────────────────────────────┤  │  │
│  │ PK doc_id                  BIGINT IDENTITY                 │  │  │
│  │    sample_uid              TEXT NOT NULL                   │  │  │
│  │    sample_uid_hash         TEXT NOT NULL  (SHA-256 hex)    │  │  │
│  │ FK source_id               SMALLINT ──────────────────────────┘  │
│  │    cleaning_source         TEXT  ('lowyat','hplt-malay',…) │  │
│  │    source_bucket           TEXT                            │  │
│  │    source_object_key       TEXT                            │  │
│  │    source_parquet_url      TEXT                            │  │
│  │    source_row_group_index  BIGINT                          │  │
│  │    source_row_index        BIGINT                          │  │
│  │    row_index_in_row_group  BIGINT                          │  │
│  │    original_text_sha256    TEXT                            │  │
│  │    cleaned_text_sha256     TEXT                            │  │
│  │    original_char_count     BIGINT                          │  │
│  │    cleaned_char_count      BIGINT                          │  │
│  │    removed_char_count      BIGINT                          │  │
│  │    approx_original_tokens  BIGINT                          │  │
│  │    approx_cleaned_tokens   BIGINT                          │  │
│  │    approx_removed_tokens   BIGINT                          │  │
│  │    cleaning_is_dropped     BOOLEAN DEFAULT FALSE            │  │
│  │    cleaning_rules_triggered TEXT[]                         │  │
│  │    cleaned_o200k_token_count BIGINT  (NULL if not yet tok.)│  │
│  │    cleaned_o200k_tokenizer TEXT                            │  │
│  │    sample_key              BIGINT  (for range sampling)    │  │
│  │    created_at              TSTZ                            │  │
│  └────────────────────┬───────────────────────────────────────┘  │
│                       │ 1:1                      │ 1:1            │
│                       │                          │                │
│          ┌────────────▼──────────┐   ┌───────────▼─────────────┐ │
│          │ unified_document_texts│   │      lid_metadata        │ │
│          │  (partitioned LIST)   │   ├─────────────────────────┤ │
│          ├───────────────────────┤   │ PK/FK doc_id  BIGINT    │ │
│          │ PK cleaning_source    │   │ lid_cleaned_token_count  │ │
│          │ PK doc_id    BIGINT   │   │ reference_removed BOOL   │ │
│          │    cleaned_text TEXT  │   │ reference_removal_method │ │
│          │             (lz4)     │   │ removed_reference_chars  │ │
│          │                       │   │ lingua_primary_language  │ │
│          │ Partitions:           │   │ lingua_spans     JSONB   │ │
│          │  …_books_ocr          │   │ malaya_document_label    │ │
│          │  …_lowyat             │   │ malaya_document_scores   │ │
│          │  …_cari               │   │ malaya_word_detections   │ │
│          │  …_reddit_bolehland   │   │ malaya_word_label_counts │ │
│          │  …_reddit_indonesia   │   └─────────────────────────┘ │
│          │  …_hplt_malay         │                                │
│          │  …_hplt_indonesia     │   ┌─────────────────────────┐ │
│          │  …_fineweb            │   │      quality_signals     │ │
│          │  …_fineweb_edu        │   ├─────────────────────────┤ │
│          │  …_fine_web_edu_      │   │                          │ │
│          │     malay_translate   │   │                          │ │
│          │  …_sea_pile_malay     │   │                          │ │
│          │  …_default            │   │ PK/FK doc_id  BIGINT    │◄┘
│          └───────────────────────┘   │ PK/FK run_id  INTEGER   │◄──┐
│                                      │    score      REAL       │   │
│                                      │    label      SMALLINT   │   │
│                                      │    raw_output JSONB      │   │
│                                      └─────────────────────────┘   │
│                                                                      │
│                                      ┌─────────────────────────┐   │
│                                      │      quality_runs        │   │
│                                      ├─────────────────────────┤   │
│                                      │ PK run_id   INTEGER      │───┘
│                                      │    signal_name TEXT      │
│                                      │    model_name  TEXT      │
│                                      │    model_version TEXT    │
│                                      │    prompt_version TEXT   │
│                                      │    code_version  TEXT    │
│                                      │    created_at    TSTZ    │
│                                      └─────────────────────────┘
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    PER-SOURCE TABLES                              │
│                                                                   │
│  ┌──────────────────┐    ┌──────────────────────────────────┐    │
│  │  lowyat_threads  │    │         lowyat_posts             │    │
│  ├──────────────────┤    ├──────────────────────────────────┤    │
│  │ PK thread_id TEXT│◄──FK│ PK post_id       TEXT           │    │
│  │  thread_title    │    │ FK thread_id      TEXT           │    │
│  │  thread_url      │    │    post_floor     INTEGER        │    │
│  │  forum           │    │    page_number    INTEGER        │    │
│  │  total_pages INT │    │    page_offset    INTEGER        │    │
│  │  thread_status   │    │    author         TEXT           │    │
│  │  fetched_at TSTZ │    │    author_id      TEXT           │    │
│  └──────────────────┘    │    posted_at      TSTZ           │    │
│                           │    body_text      TEXT (lz4)    │    │
│  ┌──────────────────┐    │    body_html      TEXT (lz4)    │    │
│  │   cari_threads   │    │    quoted_post_id TEXT           │    │
│  ├──────────────────┤    │    fetched_at     TSTZ           │    │
│  │ PK thread_id TEXT│◄──FK│    error_reason   TEXT          │    │
│  │  (same columns   │    └──────────────────────────────────┘    │
│  │   as lowyat)     │                                             │
│  └──────────────────┘    ┌──────────────────────────────────┐    │
│                           │          cari_posts              │    │
│                           ├──────────────────────────────────┤    │
│                           │  (same columns as lowyat_posts)  │    │
│                           │  posted_at parsed from           │    │
│                           │  '%d-%m-%Y %I:%M %p' UTC         │    │
│                           └──────────────────────────────────┘    │
│                                                                   │
│  ┌──────────────────────────────────┐                            │
│  │     reddit_bolehland_posts       │                            │
│  │     reddit_indonesia_posts       │                            │
│  ├──────────────────────────────────┤                            │
│  │ PK post_id       TEXT            │                            │
│  │    post_kind     TEXT            │  'submission' | 'comment'  │
│  │    submission_id TEXT            │  NULL for submissions      │
│  │    parent_id     TEXT            │  Reddit fullname t3_/t1_   │
│  │    subreddit     TEXT            │                            │
│  │    author        TEXT            │                            │
│  │    title         TEXT            │  NULL for comments         │
│  │    score         INTEGER         │                            │
│  │    num_comments  INTEGER         │  NULL for comments         │
│  │    created_utc   TSTZ            │                            │
│  │    permalink     TEXT            │                            │
│  │    url           TEXT            │                            │
│  │    month         TEXT            │  '2024-01' bucket          │
│  │    body          TEXT (lz4)      │                            │
│  └──────────────────────────────────┘                            │
│                                                                   │
│  ┌──────────────────────────────────┐                            │
│  │     hplt_malay_documents         │                            │
│  │     hplt_indonesia_documents     │                            │
│  ├──────────────────────────────────┤                            │
│  │ PK id             BIGINT IDENTITY │                           │
│  │    hplt_id        TEXT            │  source 'id' field        │
│  │    url            TEXT            │                           │
│  │    crawl_timestamp TSTZ           │  WARC UTC                 │
│  │    crawl_id       TEXT            │                           │
│  │    source_shard   TEXT            │                           │
│  │    language       TEXT            │                           │
│  │    row_language_code TEXT         │                           │
│  │    row_language_prob REAL         │                           │
│  │    content        TEXT (lz4)      │                           │
│  └──────────────────────────────────┘                            │
│                                                                   │
│  ┌──────────────────────────────────┐                            │
│  │       sea_pile_malay_documents   │  ← RAW, not yet in unified │
│  ├──────────────────────────────────┤                            │
│  │ PK id             BIGINT IDENTITY │                           │
│  │    warc_record_id TEXT NOT NULL   │  47-char WARC UUID, UNIQUE│
│  │    url            TEXT            │                           │
│  │    dump           TEXT            │  'CC-MAIN-2020-45' etc.   │
│  │    crawl_timestamp TSTZ           │  WARC UTC                 │
│  │    content        TEXT (lz4)      │  sole copy until cleaning │
│  └──────────────────────────────────┘                            │
│                                                                   │
│  ┌──────────────────────────────────┐                            │
│  │           ocr_books              │                            │
│  ├──────────────────────────────────┤                            │
│  │ PK document_id    TEXT           │                            │
│  │    source_run_id  TEXT           │                            │
│  │    source_format  TEXT           │  'markdown'                │
│  │    markdown_file_name TEXT       │                            │
│  │    markdown_rel_path  TEXT       │                            │
│  │    markdown_sha256    TEXT       │                            │
│  │    markdown_char_count INTEGER   │                            │
│  │    markdown_byte_count INTEGER   │                            │
│  │    markdown_text  TEXT (lz4)     │                            │
│  └──────────────────────────────────┘                            │
│                                                                   │
│  ┌──────────────────────────────────┐                            │
│  │        fineweb_documents         │  ← stub, not yet ingested  │
│  ├──────────────────────────────────┤                            │
│  │ PK id         BIGINT IDENTITY    │                            │
│  │    hf_id      TEXT               │  HuggingFace dataset ID    │
│  │    url        TEXT               │                            │
│  │    date       TSTZ               │                            │
│  │    month      TEXT               │  '2024-01' bucket          │
│  │    language   TEXT               │                            │
│  │    content    TEXT (lz4)         │                            │
│  └──────────────────────────────────┘                            │
│                                                                   │
│  ┌──────────────────────────────────┐  → promoted into unified   │
│  │      fineweb_edu_documents       │                            │
│  ├──────────────────────────────────┤                            │
│  │ PK id            BIGINT IDENTITY │                            │
│  │    hf_id         TEXT NOT NULL   │  'id', 47-char UUID, UNIQUE│
│  │    url           TEXT            │                            │
│  │    dump          TEXT            │  CC-MAIN-2024-51 etc.      │
│  │    file_path     TEXT            │  S3 source path            │
│  │    language      TEXT            │  always 'en'               │
│  │    language_score REAL           │                            │
│  │    source_token_count BIGINT     │  dataset's own token_count │
│  │    edu_score     REAL            │  'score' continuous        │
│  │    edu_int_score SMALLINT        │  'int_score' 3-5           │
│  │    content       TEXT (lz4)      │  'text'; sole copy         │
│  └──────────────────────────────────┘                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## Table Reference

### `data_sources`

Registry of all corpus sources. One row per source, seeded at DB init.

| Column | Type | Notes |
|---|---|---|
| `source_id` | SMALLINT IDENTITY | PK |
| `source_name` | TEXT UNIQUE | `'lowyat'`, `'cari'`, `'reddit-bolehland'`, `'reddit-indonesia'`, `'hplt-malay'`, `'hplt-indonesia'`, `'books-ocr'`, `'fineweb'`, `'sea-pile-malay'`, `'fineweb-edu'`, `'fine-web-edu-malay-translate'` |
| `source_type` | TEXT | `'forum'`, `'reddit'`, `'web'`, `'books'` |
| `description` | TEXT | Human-readable description |
| `created_at` | TIMESTAMPTZ | |

```sql
SELECT * FROM data_sources ORDER BY source_id;
```

---

### `pipeline_runs`

Tracks each pipeline ingest run for provenance. Insert one row at run start, update on completion.

| Column | Type | Notes |
|---|---|---|
| `run_id` | TEXT | PK, e.g. `'20260429T214659Z'` |
| `pipeline_name` | TEXT | `'unified_data'`, `'ocr'`, `'lid_metadata'`, … |
| `source_id` | SMALLINT | FK → data_sources |
| `status` | TEXT | `'running'` / `'completed'` / `'failed'` |
| `r2_output_prefix` | TEXT | R2 object key prefix for this run's output |
| `row_count` | BIGINT | Rows produced |
| `token_count` | BIGINT | Tokens produced |
| `byte_count` | BIGINT | Bytes produced |
| `started_at` | TIMESTAMPTZ | |
| `completed_at` | TIMESTAMPTZ | |

---

### `unified_documents`

**Central metadata hub.** Every document that passes through the cleaning pipeline gets one row here. All quality and LID metadata FK back to this table.

| Column | Type | Notes |
|---|---|---|
| `doc_id` | BIGINT IDENTITY | PK |
| `sample_uid` | TEXT | Global unique document identifier |
| `sample_uid_hash` | TEXT | SHA-256 hex of `sample_uid` |
| `source_id` | SMALLINT | FK → data_sources |
| `cleaning_source` | TEXT | Same as `data_sources.source_name` |
| `source_bucket` | TEXT | R2 bucket of the source parquet |
| `source_object_key` | TEXT | R2 object key of the source parquet |
| `source_parquet_url` | TEXT | Full URL to source parquet file |
| `source_row_group_index` | BIGINT | Row group within the parquet file |
| `source_row_index` | BIGINT | Row within the full file |
| `row_index_in_row_group` | BIGINT | Row within the row group |
| `original_text_sha256` | TEXT | SHA-256 of pre-cleaning text |
| `cleaned_text_sha256` | TEXT | SHA-256 of post-cleaning text |
| `original_char_count` | BIGINT | |
| `cleaned_char_count` | BIGINT | |
| `removed_char_count` | BIGINT | |
| `approximate_original_token_count` | BIGINT | Estimated via fast tokenizer |
| `approximate_cleaned_token_count` | BIGINT | |
| `approximate_removed_token_count` | BIGINT | |
| `cleaning_is_dropped` | BOOLEAN | `TRUE` = document was rejected |
| `cleaning_rules_triggered` | TEXT[] | Which cleaning rules fired |
| `cleaned_o200k_token_count` | BIGINT | Exact o200k token count; NULL for sources not yet tokenized |
| `cleaned_o200k_tokenizer` | TEXT | Tokenizer identifier |
| `sample_key` | BIGINT | Deterministic 60-bit hash for range sampling |
| `created_at` | TIMESTAMPTZ | |

**Common queries:**
```sql
-- Count active docs per source
SELECT cleaning_source, COUNT(*) FROM unified_documents
WHERE cleaning_is_dropped = FALSE GROUP BY 1 ORDER BY 1;

-- Find docs that triggered a specific cleaning rule
SELECT doc_id, sample_uid FROM unified_documents
WHERE cleaning_rules_triggered @> ARRAY['remove_boilerplate'];

-- Repeatable 1% sample (no ORDER BY random())
SELECT * FROM unified_documents
WHERE sample_key BETWEEN 0 AND 0.01 * (1::BIGINT << 62)
  AND cleaning_is_dropped = FALSE;
```

---

### `unified_document_texts`

Partitioned table holding the cleaned text for each document. Partitioned by `cleaning_source` (LIST) so TOAST segments are isolated per source.

**Partitions:** `…_books_ocr`, `…_lowyat`, `…_cari`, `…_reddit_bolehland`, `…_reddit_indonesia`, `…_hplt_malay`, `…_hplt_indonesia`, `…_fineweb`, `…_fineweb_edu`, `…_fine_web_edu_malay_translate`, `…_sea_pile_malay`, `…_default`

| Column | Type | Notes |
|---|---|---|
| `doc_id` | BIGINT | PK component; FK → unified_documents (validated post-load) |
| `cleaning_source` | TEXT | PK component; partition key |
| `cleaned_text` | TEXT COMPRESSION lz4 | Post-cleaning document text |

> **Note:** `sea_pile_malay_documents` has raw content in its own table; `unified_document_texts_sea_pile_malay` will be populated after the cleaning pipeline runs on that source.

```sql
-- Fetch text for a document
SELECT t.cleaned_text
FROM unified_documents d
JOIN unified_document_texts t USING (doc_id)
WHERE d.sample_uid = 'lowyat://thread_id/post_id';
```

---

### `lid_metadata`

Language-ID results per document. Only populated for documents where `lingua_primary_language IS NOT NULL`. One-to-one with `unified_documents`.

| Column | Type | Notes |
|---|---|---|
| `doc_id` | BIGINT | PK, FK → unified_documents |
| `lid_cleaned_token_count` | BIGINT | Token count used for LID |
| `reference_removed` | BOOLEAN | Whether reference text was stripped before LID |
| `reference_removal_method` | TEXT | |
| `removed_reference_char_count` | BIGINT | |
| `lingua_primary_language` | TEXT | Primary detected language (BCP 47) |
| `lingua_spans` | JSONB | `[{start_index, end_index, language_label}, …]` |
| `malaya_document_label` | TEXT | Malaya classifier top label |
| `malaya_document_scores` | JSONB | `[{label, score}, …]` |
| `malaya_word_detections` | JSONB | `[{word_index, start_index, end_index, token, label}, …]` |
| `malaya_word_label_counts` | JSONB | `[{label, count}, …]` |

```sql
-- Documents where Lingua detected mixed language
SELECT d.doc_id, d.cleaning_source, l.lingua_primary_language,
       jsonb_array_length(l.lingua_spans) AS lang_span_count
FROM lid_metadata l
JOIN unified_documents d USING (doc_id)
WHERE jsonb_array_length(l.lingua_spans) > 1
LIMIT 20;
```

---

### `quality_runs`

One row per quality scoring run (e.g. a single LLM judge run over the corpus).

| Column | Type | Notes |
|---|---|---|
| `run_id` | INTEGER IDENTITY | PK |
| `signal_name` | TEXT | e.g. `'edu'`, `'toxicity'`, `'fluency'` |
| `model_name` | TEXT | e.g. `'gpt-4o'`, `'claude-3-5-sonnet'` |
| `model_version` | TEXT | |
| `prompt_version` | TEXT | |
| `code_version` | TEXT | |
| `created_at` | TIMESTAMPTZ | |

---

### `quality_signals`

One row per (document, quality run) pair. PK is `(doc_id, run_id)` for fast per-document access; secondary index on `(run_id, score DESC, doc_id)` for manifest export.

| Column | Type | Notes |
|---|---|---|
| `doc_id` | BIGINT | PK, FK → unified_documents |
| `run_id` | INTEGER | PK, FK → quality_runs |
| `score` | REAL | Continuous quality score |
| `label` | SMALLINT | Discrete class label |
| `raw_output` | JSONB | Full model output for audit |

```sql
-- Top-scoring 10k docs for a quality run
SELECT doc_id, score FROM quality_signals
WHERE run_id = 1
ORDER BY score DESC
LIMIT 10000;
```

---

### Forum source tables — `lowyat_threads`, `lowyat_posts`, `cari_threads`, `cari_posts`

Two-table structure per forum: threads are parents, posts are children. A post without a corresponding thread row cannot be inserted (FK constraint).

**Thread columns** (`lowyat_threads`, `cari_threads`):

| Column | Type | Notes |
|---|---|---|
| `thread_id` | TEXT | PK |
| `thread_title` | TEXT | |
| `thread_url` | TEXT | |
| `forum` | TEXT | Sub-forum name |
| `thread_total_pages` | INTEGER | |
| `thread_status` | TEXT | e.g. `'open'`, `'closed'` |
| `fetched_at` | TIMESTAMPTZ | Crawl timestamp |

**Post columns** (`lowyat_posts`, `cari_posts`):

| Column | Type | Notes |
|---|---|---|
| `post_id` | TEXT | PK |
| `thread_id` | TEXT | FK → threads table |
| `post_floor` | INTEGER | Post number within thread |
| `page_number` | INTEGER | Forum page the post was on |
| `page_offset` | INTEGER | Position within the page |
| `author` | TEXT | |
| `author_id` | TEXT | |
| `posted_at` | TIMESTAMPTZ | Cari: parsed from `'%d-%m-%Y %I:%M %p'` UTC. Lowyat: relative strings, stored NULL. |
| `body_text` | TEXT (lz4) | Cleaned text |
| `body_html` | TEXT (lz4) | Raw HTML |
| `quoted_post_id` | TEXT | Post being replied to |
| `fetched_at` | TIMESTAMPTZ | |
| `error_reason` | TEXT | If post failed to parse |

```sql
-- Thread with all its posts in order
SELECT p.post_floor, p.author, p.posted_at, left(p.body_text, 200)
FROM lowyat_posts p
WHERE p.thread_id = 'some_thread_id'
ORDER BY p.post_floor;
```

---

### Reddit tables — `reddit_bolehland_posts`, `reddit_indonesia_posts`

Flat table. Both submissions and comments live in the same table, discriminated by `post_kind`.

| Column | Type | Notes |
|---|---|---|
| `post_id` | TEXT | PK; Reddit fullname without prefix |
| `post_kind` | TEXT | `'submission'` or `'comment'` |
| `submission_id` | TEXT | Root submission; NULL for standalone submissions |
| `parent_id` | TEXT | Reddit fullname: `t3_xxx` = submission, `t1_xxx` = comment |
| `subreddit` | TEXT | `'Bolehland'` or `'indonesia'` (defaults) |
| `author` | TEXT | |
| `title` | TEXT | NULL for comments |
| `score` | INTEGER | Upvotes minus downvotes |
| `num_comments` | INTEGER | NULL for comments |
| `created_utc` | TIMESTAMPTZ | |
| `permalink` | TEXT | |
| `url` | TEXT | Link URL for submissions |
| `month` | TEXT | `'2024-01'` crawl bucket |
| `body` | TEXT (lz4) | |

```sql
-- Thread: submission + all its top-level comments
SELECT post_kind, author, score, left(body, 100)
FROM reddit_indonesia_posts
WHERE submission_id = 'abc123' OR post_id = 'abc123'
ORDER BY post_kind DESC, score DESC;
```

---

### HPLT tables — `hplt_malay_documents`, `hplt_indonesia_documents`

Web crawl documents from the HPLT multilingual corpus. Plain INSERT (no UPSERT) — no natural unique key in source data; `id` is a synthetic IDENTITY.

| Column | Type | Notes |
|---|---|---|
| `id` | BIGINT IDENTITY | PK (synthetic) |
| `hplt_id` | TEXT | Source `id` field |
| `url` | TEXT | |
| `crawl_timestamp` | TIMESTAMPTZ | WARC UTC |
| `crawl_id` | TEXT | |
| `source_shard` | TEXT | HPLT shard file name |
| `language` | TEXT | ISO language code from HPLT |
| `row_language_code` | TEXT | Per-row LID code |
| `row_language_prob` | REAL | LID confidence |
| `content` | TEXT (lz4) | Document text |

---

### `sea_pile_malay_documents`

**Raw source table for SEA-PILE-v2 Malay.** Ingested directly from HuggingFace (`aisingapore/SEA-PILE-v2`, `ms` config). Not yet in `unified_documents` — the cleaning pipeline runs later.

| Column | Type | Notes |
|---|---|---|
| `id` | BIGINT IDENTITY | PK (synthetic) |
| `warc_record_id` | TEXT NOT NULL UNIQUE | 47-char WARC UUID e.g. `<urn:uuid:…>` |
| `url` | TEXT | Source URL |
| `dump` | TEXT | CC snapshot: `CC-MAIN-2020-45` etc. |
| `crawl_timestamp` | TIMESTAMPTZ | WARC UTC |
| `content` | TEXT (lz4) | Raw document text (sole copy until cleaning) |

```sql
-- Distribution by CC snapshot
SELECT dump, COUNT(*) FROM sea_pile_malay_documents GROUP BY 1 ORDER BY 2 DESC;
```

---

### `ocr_books`

PDF books converted to Markdown via Marker OCR. Small corpus (~8,800 docs).

| Column | Type | Notes |
|---|---|---|
| `document_id` | TEXT | PK |
| `source_run_id` | TEXT | OCR pipeline run |
| `source_format` | TEXT | `'markdown'` |
| `markdown_file_name` | TEXT | |
| `markdown_rel_path` | TEXT | Path within the OCR output directory |
| `markdown_sha256` | TEXT | |
| `markdown_char_count` | INTEGER | |
| `markdown_byte_count` | INTEGER | |
| `markdown_text` | TEXT (lz4) | Full markdown content |

---

### `fineweb_documents`

Schema placeholder for HuggingFace FineWeb. Not yet ingested.

| Column | Type | Notes |
|---|---|---|
| `id` | BIGINT IDENTITY | PK |
| `hf_id` | TEXT | HuggingFace dataset ID |
| `url` | TEXT | |
| `date` | TIMESTAMPTZ | |
| `month` | TEXT | `'2024-01'` bucket |
| `language` | TEXT | |
| `content` | TEXT (lz4) | |

---

### `fineweb_edu_documents`

**Raw provenance table for HuggingFace FineWeb-Edu** (English educational-filtered
web corpus). Ingested directly from HuggingFace (`HuggingFaceFW/fineweb-edu`),
default subset `sample/100BT`, by `scripts/ingest_fineweb_edu.py`.

Unlike `sea-pile-malay`, fineweb-edu is **promoted into the unified layer during
ingest** (no cleaning pipeline): each row also lands in `unified_documents`
(`cleaning_source='fineweb-edu'`, `sample_uid='fineweb-edu://<hf_id>'`,
cleaned text == original — no cleaning applied) and `unified_document_texts`
(`…_fineweb_edu` partition). Because it is a curated, language-filtered English
corpus, the Malaya LID model is **not** run — every doc gets a
`document_language_detection` row with `language_label='standard-english'`,
`confidence=1.0` assigned directly.

`hf_id` is the source `id` (47-char UUID), `UNIQUE` for resume-safe
`ON CONFLICT DO NOTHING` ingest. `edu_score` / `edu_int_score` are the dataset's
own intrinsic quality fields (not produced by our `quality_signals` pipeline).
The HF `date` field is empty in the parquet sample subsets, so it is not stored.

| Column | Type | Notes |
|---|---|---|
| `id` | BIGINT IDENTITY | PK (synthetic) |
| `hf_id` | TEXT NOT NULL UNIQUE | Source `id`, 47-char UUID |
| `url` | TEXT | |
| `dump` | TEXT | CC snapshot: `CC-MAIN-2024-51` etc. |
| `file_path` | TEXT | S3 source path |
| `language` | TEXT | Always `en` |
| `language_score` | REAL | LID confidence (0.65–1.0) |
| `source_token_count` | BIGINT | Dataset's own `token_count` |
| `edu_score` | REAL | Continuous edu quality (2.52–5.06) |
| `edu_int_score` | SMALLINT | Discrete edu rating (3–5) |
| `content` | TEXT (lz4) | Source `text`; copied verbatim into `unified_document_texts` |
| `created_at` | TIMESTAMPTZ | |

```sql
-- Distribution by edu rating
SELECT edu_int_score, COUNT(*) FROM fineweb_edu_documents
GROUP BY 1 ORDER BY 1;
```

---

### `fine-web-edu-malay-translate` (derived source — no raw table)

**Malay machine translation of `fineweb-edu`.** Produced by
`scripts/translate_fineweb_edu_malay.py`, which reads `cleaning_source=
'fineweb-edu'` unified docs, translates them to Malay with `Qwen/Qwen3-14B-AWQ`
served by vLLM (thinking mode; the proven recipe in
`scripts/serve_qwen3_translate.sh`), and writes straight into the unified layer.
There is **no per-source raw table** — provenance is implicit in `sample_uid`.

Conventions for the `unified_documents` rows it writes:

| Field | Value |
|---|---|
| `cleaning_source` | `fine-web-edu-malay-translate` |
| `sample_uid` | `fine-web-edu-malay-translate://<hf_id>` (origin is `fineweb-edu://<hf_id>`) |
| `original_text_sha256` | hash of the English source (origin `cleaned_text_sha256`) |
| `cleaned_text_sha256` | hash of the Malay translation (distinct → no `cleaned_hash_ux` clash) |
| `original_char_count` | origin `original_char_count` (English) |
| `cleaned_char_count` | length of the Malay text |
| `approximate_*_token_count` | origin (original) / Qwen-tokenizer count (Malay) |
| `cleaning_rules_triggered` | `{qwen3-14b-awq-en-ms-translate}` (tags the model/transform) |
| `source_object_key` / `source_parquet_url` / `source_row_index` | carried from the origin fineweb-edu row |

The translator also writes `document_language_detection` rows with
`language_label='standard-malay'`, `confidence=1.0` (direct assignment, no LID
model run — analogous to fineweb-edu's `standard-english`).

The Qwen3 thinking-mode reasoning trace that produced each translation is kept
in **`document_translation_thinking`** (one row per translation `doc_id`,
FK → `unified_documents`; `thinking_text` lz4 + `thinking_char_count` +
`thinking_token_count`; multi-chunk docs concatenate their `<think>` bodies).
Created by `scripts/migrations/add_document_translation_thinking_table.sql`.

```sql
-- A translation joined back to its English origin doc
SELECT mt.doc_id        AS ms_doc_id,
       en.doc_id         AS en_doc_id,
       mt_t.cleaned_text AS malay,
       en_t.cleaned_text AS english
FROM unified_documents mt
JOIN unified_document_texts mt_t USING (doc_id)
JOIN unified_documents en
  ON en.cleaning_source = 'fineweb-edu'
 AND en.sample_uid = 'fineweb-edu://'
     || split_part(mt.sample_uid, '://', 2)
JOIN unified_document_texts en_t ON en_t.doc_id = en.doc_id
WHERE mt.cleaning_source = 'fine-web-edu-malay-translate'
LIMIT 5;
```

---

## Views

### `training_manifest` (materialized)

Lightweight manifest for training data loading. Excludes dropped documents and documents without token counts. Refresh after each ingest batch.

```sql
-- Refresh
REFRESH MATERIALIZED VIEW CONCURRENTLY training_manifest;

-- Use: select ~1% sample for training
SELECT source_parquet_url, source_row_index, cleaned_o200k_token_count
FROM training_manifest
WHERE sample_key < (1::BIGINT << 62) / 100
ORDER BY sample_key;
```

| Column | Notes |
|---|---|
| `doc_id` | |
| `cleaning_source` | |
| `source_parquet_url` | Use to read text from source parquet |
| `source_row_group_index` | Row group within parquet |
| `source_row_index` | Row within parquet file |
| `cleaned_o200k_token_count` | For token budget calculations |
| `cleaned_char_count` | |
| `sample_key` | Use for deterministic sampling |
| `cleaning_is_dropped` | Always FALSE (filtered in view definition) |

---

### `source_stats` (view)

Quick corpus-level statistics, always current.

```sql
SELECT * FROM source_stats ORDER BY total_tokens DESC;
```

---

### `quality_filtered_manifest` (view)

Manifest joined with quality scores. Parameterise at query time with a WHERE clause.

```sql
-- Docs scoring ≥ 3.0 on the 'edu' signal
SELECT doc_id, source_parquet_url, source_row_index, score
FROM quality_filtered_manifest
WHERE signal_name = 'edu' AND score >= 3.0
ORDER BY score DESC;
```

---

## Indexes

| Index | Table | Type | Purpose |
|---|---|---|---|
| `unified_documents_source_dropped_idx` | unified_documents | B-tree (cleaning_source, cleaning_is_dropped, doc_id) | Common filter: per-source active-doc scans |
| `unified_documents_doc_id_brin` | unified_documents | BRIN (doc_id) | Cheap range scan; doc_id is append-ordered |
| `unified_documents_sample_key_idx` | unified_documents | B-tree (sample_key) | Repeatable sampling |
| `unified_documents_cleaning_rules_gin` | unified_documents | GIN (cleaning_rules_triggered) | Array containment: `@>`, `&&` |
| `lowyat_posts_thread_id_idx` | lowyat_posts | B-tree | Thread → posts lookup |
| `lowyat_posts_author_idx` | lowyat_posts | B-tree | Author search |
| `lowyat_posts_posted_at_idx` | lowyat_posts | B-tree | Time-range queries |
| *(same for cari_posts)* | | | |
| `reddit_bolehland_submission_idx` | reddit_bolehland_posts | B-tree | Comment thread reconstruction |
| `reddit_bolehland_parent_idx` | reddit_bolehland_posts | B-tree | Parent reply lookup |
| `hplt_malay_url_idx` | hplt_malay_documents | B-tree | URL lookup |
| `hplt_malay_lang_idx` | hplt_malay_documents | B-tree | Language filter |
| `sea_pile_malay_dump_idx` | sea_pile_malay_documents | B-tree | Filter by CC snapshot |
| `fineweb_edu_dump_idx` | fineweb_edu_documents | B-tree | Filter by CC snapshot |
| `fineweb_edu_int_score_idx` | fineweb_edu_documents | B-tree | Edu quality-rating filter |
| `quality_signals_run_score_idx` | quality_signals | B-tree (run_id, score DESC) | Per-run manifest export |
| `lid_metadata_lingua_lang_idx` | lid_metadata | B-tree | Language filter |
| `training_manifest_doc_id_ux` | training_manifest | UNIQUE B-tree | Enables CONCURRENTLY refresh |
| `training_manifest_sample_key_idx` | training_manifest | B-tree | Sampling |

**Post-load only** (run `scripts/post_load/04b_post_load_indexes.sql` after bulk ingest):
- `unified_documents_sample_uid_ux` — UNIQUE on sample_uid (dedup guard)
- `unified_documents_cleaned_hash_ux` — UNIQUE on cleaned_text_sha256
- `unified_document_texts_doc_id_fk` — FK validated with NOT VALID + VALIDATE

---

## Common Query Patterns

### 1. Get cleaned text for a document

```sql
SELECT d.doc_id, d.cleaning_source, t.cleaned_text
FROM unified_documents d
JOIN unified_document_texts t USING (doc_id)
WHERE d.sample_uid = 'lowyat://12345/67890';
```

### 2. Per-source token counts

```sql
SELECT * FROM source_stats ORDER BY total_tokens DESC NULLS LAST;
```

### 3. Random 0.5% repeatable sample

```sql
-- sample_key is a precomputed 60-bit hash; range query avoids ORDER BY random()
SELECT doc_id, source_parquet_url, source_row_index
FROM unified_documents
WHERE sample_key < ((1::BIGINT << 62) * 0.005)
  AND cleaning_is_dropped = FALSE;
```

### 4. Documents dropped by a specific rule

```sql
SELECT COUNT(*) FROM unified_documents
WHERE cleaning_rules_triggered @> ARRAY['filter_short_text']
  AND cleaning_is_dropped = TRUE;
```

### 5. LID: which documents are mixed-language?

```sql
SELECT d.cleaning_source, l.lingua_primary_language, COUNT(*)
FROM lid_metadata l
JOIN unified_documents d USING (doc_id)
WHERE jsonb_array_length(l.lingua_spans) > 2
GROUP BY 1, 2
ORDER BY 3 DESC;
```

### 6. Forum thread reconstruction

```sql
-- Full lowyat thread in order
SELECT p.post_floor, p.author, p.posted_at,
       left(p.body_text, 500) AS preview
FROM lowyat_posts p
WHERE p.thread_id = 'your_thread_id'
ORDER BY p.post_floor;
```

### 7. Reddit comment tree

```sql
-- All comments in a submission
SELECT post_kind, author, score, left(body, 200)
FROM reddit_indonesia_posts
WHERE submission_id = 'the_submission_id'
ORDER BY score DESC;
```

### 8. SEA-PILE snapshot distribution

```sql
SELECT dump, COUNT(*), MIN(crawl_timestamp), MAX(crawl_timestamp)
FROM sea_pile_malay_documents
GROUP BY 1
ORDER BY 1;
```

---

## Source Ingest Status

| Source | Per-source table | In unified_documents | Notes |
|---|---|---|---|
| `lowyat` | `lowyat_threads`, `lowyat_posts` | Yes | |
| `cari` | `cari_threads`, `cari_posts` | Yes | |
| `reddit-bolehland` | `reddit_bolehland_posts` | Yes | |
| `reddit-indonesia` | `reddit_indonesia_posts` | Yes | |
| `hplt-malay` | `hplt_malay_documents` | Yes | |
| `hplt-indonesia` | `hplt_indonesia_documents` | Yes | |
| `books-ocr` | `ocr_books` | Yes | |
| `sea-pile-malay` | `sea_pile_malay_documents` | **No** — raw ingest only; cleaning pipeline pending |
| `fineweb` | `fineweb_documents` | No | Schema placeholder; not yet ingested |
| `fineweb-edu` | `fineweb_edu_documents` | **Yes** — promoted during ingest (no cleaning); LID assigned directly as `standard-english` (no model run) |
| `fine-web-edu-malay-translate` | *derived (no raw table)* | **Yes** — translated from fineweb-edu via Qwen3-14B-AWQ (vLLM, thinking mode); LID assigned directly as `standard-malay` (no model run) |
