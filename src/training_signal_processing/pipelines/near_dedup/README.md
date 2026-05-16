# Near-Duplicate Detection Pipeline

Finds near-duplicate documents across all 9 sources in `unified_document_texts` using
**MinHash Locality-Sensitive Hashing (LSH)**. Near-duplicates are *marked*, never deleted.
Results land in `document_near_duplicates` as a normalized signal table.

---

## Why MinHash LSH?

Exact-duplicate detection (SHA-256) is already handled by the `cleaned_text_sha256` UNIQUE
constraint on `unified_documents`. MinHash LSH catches the cases SHA-256 misses:

- The same article scraped from two different mirrors with minor whitespace differences
- A forum post that was copied and slightly edited
- An OCR book chapter that was also republished on a web crawl domain

Comparing every pair of 7M+ documents exactly is O(n²) and infeasible. MinHash LSH reduces
this to O(n) by sketching each document as a compact signature and only examining document
pairs that share a hash bucket — without reading the full text again.

---

## Algorithm Overview

```
  unified_document_texts
  (7M+ documents, 84 GB lz4)
         │
         │  1. Read once, streaming batch by batch (10K docs/batch)
         │     Filter: cleaning_is_dropped=FALSE, len > 50 chars
         ▼
  ┌──────────────────────────────────────────────────┐
  │  For each document:                              │
  │                                                  │
  │  "the quick brown fox" → word trigrams           │
  │  {"the quick brown",                             │
  │   "quick brown fox", ...}                        │
  │                                                  │
  │         ↓  MinHash(128 permutations)             │
  │                                                  │
  │  [h₀, h₁, h₂, ..., h₁₂₇]   (uint32 × 128)      │
  │                                                  │
  │         ↓  Band decomposition (16 bands × 8 rows)│
  │                                                  │
  │  band₀ = SHA256(h₀…h₇)   → 64-bit band_hash     │
  │  band₁ = SHA256(h₈…h₁₅)  → 64-bit band_hash     │
  │  ...                                             │
  │  band₁₅= SHA256(h₁₂₀…h₁₂₇)→ 64-bit band_hash   │
  └──────────────────────────────────────────────────┘
         │
         │  COPY bulk-insert (50K rows per flush)
         ▼
  lsh_candidate_bands  ← UNLOGGED, no WAL overhead
  ┌──────────┬────────────┬───────────────┐
  │  doc_id  │ band_index │   band_hash   │
  ├──────────┼────────────┼───────────────┤
  │  100001  │     0      │  -3821749...  │
  │  100001  │     1      │   9182736...  │
  │  ...     │   ...      │   ...         │
  │  100002  │     0      │  -3821749...  │  ← same hash as doc 100001!
  │  ...     │   ...      │   ...         │
  └──────────┴────────────┴───────────────┘
  (7M docs × 16 bands = ~116M rows, ~4 GB)
         │
         │  GROUP BY (band_index, band_hash) HAVING count > 1
         │  Stream groups via server-side cursor (no client materialisation)
         ▼
  ┌──────────────────────────────────────────────────┐
  │  candidate pairs in Python dict                  │
  │                                                  │
  │  pair_bands = {                                  │
  │    (100001, 100002): 3,   ← shared 3/16 bands    │
  │    (55000,  88421): 1,    ← shared 1/16 bands    │
  │    ...                                           │
  │  }                                               │
  └──────────────────────────────────────────────────┘
         │
         │  Union-Find (path-halving, O(n·α(n)))
         │  Canonical = lowest doc_id in each cluster
         ▼
  ┌──────────────────────────────────────────────────┐
  │  Cluster resolution example:                     │
  │                                                  │
  │  Before UF:            After UF:                 │
  │  A(1)─B(2)─C(3)        A(1)─B(2)─C(3)           │
  │                         ↑         ↑              │
  │  B.canonical=1          └─────────┘              │
  │  C.canonical=2          C.canonical=1  ✓         │
  │                                                  │
  │  Every non-root doc points directly to cluster   │
  │  root. No chains. Single hop guaranteed.         │
  └──────────────────────────────────────────────────┘
         │
         │  Bulk upsert ON CONFLICT DO UPDATE
         ▼
  document_near_duplicates  ← permanent, logged
  ┌──────────┬──────────────────┬─────────────┬─────────────────────────┐
  │  doc_id  │ canonical_doc_id │ shared_bands│ detected_at             │
  ├──────────┼──────────────────┼─────────────┼─────────────────────────┤
  │  100002  │     100001       │      3      │ 2026-05-14 16:20:59+00  │
  │  100003  │     100001       │      0      │ 2026-05-14 16:20:59+00  │
  │   88421  │      55000       │      1      │ 2026-05-14 16:20:59+00  │
  │   ...    │      ...         │     ...     │ ...                     │
  └──────────┴──────────────────┴─────────────┴─────────────────────────┘
  shared_bands=0 means C reached root A only transitively (via B), not directly
```

---

## Parameters

### Quick-reference table

```
  ┌──────────────────┬─────────┬──────────────────────────────────────────────────────┐
  │ Parameter        │ Default │ What it controls                                     │
  ├──────────────────┼─────────┼──────────────────────────────────────────────────────┤
  │ num_permutations │   128   │ MinHash accuracy (σ of Jaccard estimate)             │
  │ num_bands        │    16   │ Number of LSH bands (sensitivity of detection)       │
  │ rows_per_band    │     8   │ Rows per band (strictness of band match)             │
  │ min_text_length  │    50   │ Minimum characters to process a document             │
  │ batch_size       │ 10 000  │ Documents fetched per DB round-trip per worker       │
  │ copy_buffer_rows │ 50 000  │ Band rows buffered before each COPY flush            │
  │ num_workers      │     4   │ Parallel worker processes (each uses one DB conn)    │
  └──────────────────┴─────────┴──────────────────────────────────────────────────────┘

  Constraint: num_bands × rows_per_band == num_permutations  (16 × 8 = 128 ✓)
```

---

### `num_permutations = 128` — MinHash accuracy

MinHash approximates Jaccard similarity by applying `num_permutations` independent hash
functions to a document's shingle set and keeping the minimum hash value from each.
The resulting vector of 128 values is the **MinHash signature**.

The variance of the Jaccard estimate is governed by:

```
  σ(Ĵ) ≈ 1 / √num_permutations

  128 permutations → σ ≈ 0.089
  256 permutations → σ ≈ 0.063  (2× memory, 2× compute)
   64 permutations → σ ≈ 0.125  (noisier but faster)
```

128 is the industry standard for near-dedup pipelines. The signatures are computed
in memory per document and never stored — only the 16 band hashes derived from them
are written to disk.

---

### `num_bands = 16` and `rows_per_band = 8` — The LSH threshold

The 128-value signature is split into 16 groups (bands) of 8 consecutive values each.
A **band hash** is computed for each group using SHA-256 (deterministic; not Python's
`hash()` which is randomised by `PYTHONHASHSEED`).

Two documents become **candidate pairs** iff their band hashes match in **at least one
band**. This is the core LSH trick — it converts a global comparison into 16 independent
binary tests.

The detection probability follows an S-curve:

```
  P(pair detected | Jaccard = J)  ≈  1 − (1 − Jʳ)ᵇ

  With b = 16 bands,  r = 8 rows per band:

  ┌──────────┬──────────────────────────────────────────┐
  │  Jaccard │  P(detected)   Interpretation            │
  ├──────────┼──────────────────────────────────────────┤
  │   0.50   │   ~25%         mostly noise, filtered out│
  │   0.70   │   ~72%         borderline similar         │
  │   0.78   │   ~50%         inflection / threshold ←  │
  │   0.80   │   ~94%         near-duplicate target      │
  │   0.90   │   ~99.9%       very high overlap          │
  │   1.00   │   100%         identical content          │
  └──────────┴──────────────────────────────────────────┘

  Approximate inflection threshold:  t ≈ (1/b)^(1/r) = (1/16)^(1/8) ≈ 0.78

  ↑ Documents with Jaccard ≥ 0.8 are detected with ~94% probability.
  ↓ Documents with Jaccard < 0.5 are almost never detected.
```

**How to tune:**
- Lower threshold (catch more pairs): increase `num_bands` or decrease `rows_per_band`
- Higher threshold (stricter matching): decrease `num_bands` or increase `rows_per_band`
- Always maintain `num_bands × rows_per_band == num_permutations`

---

### `min_text_length = 50` — Short document filter

A document with fewer than 50 characters might contain only 1–2 word trigrams. With
so few shingles, the Jaccard estimate is meaningless (high variance, spurious matches).
Such documents are skipped in the fetch query (`char_length(cleaned_text) > 50`).

---

### `batch_size = 10 000` — Memory vs. DB round-trips

Each worker fetches `batch_size` documents at a time using keyset pagination
(`WHERE doc_id > last_seen LIMIT batch_size`). This is a bounded-memory O(1) seek
regardless of table position — no OFFSET scan degradation.

```
  10 000 docs × ~2 KB avg cleaned_text  ≈  20 MB fetched per batch
  10 000 docs × 512 B MinHash signature ≈   5 MB computed per batch
  Total peak per worker                 ≈  25 MB (excluding Python overhead)
```

---

### `copy_buffer_rows = 50 000` — Write batching

Each document produces `num_bands = 16` rows in `lsh_candidate_bands`.
The buffer holds rows until it reaches `copy_buffer_rows`, then flushes via
PostgreSQL `COPY FROM` (bulk protocol, ~10× faster than individual INSERTs).

```
  50 000 rows ÷ 16 bands/doc  =  ~3 125 docs per COPY call
  At 10 000 docs/batch → ~3 COPY flushes per batch
```

---

### `num_workers = 4` — Parallelism vs. memory

Each worker is a separate Python process with its own DB connection. The range
`(watermark, snapshot_max]` is split into `num_workers` equal chunks; the last
chunk is open-ended so docs ingested while the pipeline runs are processed.

```
  Memory per worker ≈ 300 MB  (Python interpreter + datasketch + numpy + OS buffers)
  4 workers × 300 MB = 1.2 GB worker overhead
  + PostgreSQL page cache (can grow to several GB under load)

  On WSL2 with limited swap: keep num_workers ≤ 4.
  On a dedicated Linux server with 64 GB RAM: num_workers = 8–16 is safe.
```

---

### Shingling: word 3-grams

```
  "saya tidak faham kenapa dia buat begitu"
       ↓ split on whitespace
  words = ["saya","tidak","faham","kenapa","dia","buat","begitu"]
       ↓ sliding window of 3
  trigrams = {
    "saya tidak faham",
    "tidak faham kenapa",
    "faham kenapa dia",
    "kenapa dia buat",
    "dia buat begitu"
  }
```

Word 3-grams are robust to short insertions/deletions and work across Malay, Indonesian,
and English without a tokenizer. Jaccard over trigram sets ≈ content overlap.

Character n-grams (e.g. char 5-grams) would be more fine-grained but are ~5× more
shingles per document, increasing MinHash update cost proportionally.

---

## Pipeline Phases

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │  PHASE 1 — Band Computation  (parallel, hours)                     │
  │                                                                     │
  │  Coordinator                                                        │
  │  ┌────────────────────────────────────────────────────┐            │
  │  │ 1. Determine start_id:                             │            │
  │  │    incremental: MAX(doc_id) from lsh_candidate_bands│           │
  │  │    full:        TRUNCATE + start_id = 0            │            │
  │  │ 2. Find max eligible doc_id in unified_doc_texts   │            │
  │  │ 3. Split (start_id, max_id] into N equal chunks    │            │
  │  │ 4. Launch N ProcessPoolExecutor workers            │            │
  │  │ 5. After all workers done: rebuild band index      │            │
  │  └────────────────────────────────────────────────────┘            │
  │          │              │              │                            │
  │    Worker 0       Worker 1  ...  Worker N-1                        │
  │    doc_id          doc_id               doc_id                     │
  │    (0, 912K]    (912K, 1.8M]      (6.4M, 7.3M]                   │
  │       │                │                  │                        │
  │       └────────────────┴──────────────────┘                        │
  │                         │                                          │
  │                  lsh_candidate_bands                               │
  │                  (concurrent COPY, no conflicts)                   │
  └─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  PHASE 2 — Candidate Pair Extraction  (single process, minutes)    │
  │                                                                     │
  │  SELECT array_agg(doc_id ORDER BY doc_id)                          │
  │  FROM lsh_candidate_bands                                          │
  │  GROUP BY band_index, band_hash                                    │
  │  HAVING count(*) > 1                                               │
  │                                                                     │
  │  Streamed via server-side named cursor (itersize=5000).            │
  │  Each group generates O(k²) pairs (k = bucket size).              │
  │  Result: pair_bands dict {(a, b): shared_band_count}              │
  └─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  PHASE 3 — Union-Find Closure  (in-memory, seconds)                │
  │                                                                     │
  │  1. Collect all distinct doc_ids from pair_bands                   │
  │  2. Map to contiguous indices (sorted → canonical = index 0)       │
  │  3. For each pair (a,b): union(idx[a], idx[b])                    │
  │     _union routes by index value → lower index = root always      │
  │  4. Iterate all doc_ids; for each non-root:                        │
  │     canonical = all_ids[find(parent, idx[doc_id])]                │
  │     shared = pair_bands.get((min(doc,root), max(doc,root)), 0)    │
  └─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  PHASE 4 — Upsert Results  (single process, seconds)               │
  │                                                                     │
  │  INSERT INTO document_near_duplicates                              │
  │  ON CONFLICT (doc_id) DO UPDATE                                    │
  │  ... execute_values, page_size=10000                               │
  └─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  PHASE 5 — Cleanup                                                 │
  │                                                                     │
  │  incremental mode (--workers N, no --full):                        │
  │    lsh_candidate_bands kept  → watermark advances for next run     │
  │                                                                     │
  │  full mode (--full):                                               │
  │    lsh_candidate_bands dropped → disk reclaimed                    │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## Incremental Runs (growing table)

```
  Time ──────────────────────────────────────────────────────────────►

  Run 1 (--full):
  ┌──────────────────────────────────────────────┐
  │ Process ALL docs: doc_id in (0, 7.3M]        │
  │ lsh_candidate_bands: 116M rows, 4 GB         │
  │ Watermark after run: MAX(doc_id) = 7,282,832 │
  └──────────────────────────────────────────────┘

  ... new documents ingested, doc_ids 7.3M – 7.5M added ...

  Run 2 (--workers 8, incremental, default):
  ┌─────────────────────────────────────────────────────┐
  │ Read watermark: 7,282,832                           │
  │ Process ONLY new docs: doc_id in (7.3M, 7.5M]      │
  │ ADD 3.2M new band rows to lsh_candidate_bands       │
  │ Rebuild index on full bands table (old + new)       │
  │ Re-run Phase 2 on FULL bands table                  │
  │   → new pairs involving new docs are detected       │
  │   → old pairs re-detected, ON CONFLICT DO UPDATE    │
  └─────────────────────────────────────────────────────┘

  Crash / unclean DB shutdown:
  ┌─────────────────────────────────────────────────────┐
  │ lsh_candidate_bands is UNLOGGED → auto-truncated    │
  │ document_near_duplicates is logged → preserved      │
  │ Next run: watermark = 0, falls back to full scan    │
  │ Re-runs are safe: ON CONFLICT DO UPDATE handles it  │
  └─────────────────────────────────────────────────────┘
```

---

## Database Schema

### `document_near_duplicates` (permanent)

```sql
CREATE TABLE document_near_duplicates (
    doc_id           BIGINT   NOT NULL PRIMARY KEY,
    canonical_doc_id BIGINT   NOT NULL,
    shared_bands     SMALLINT NOT NULL,   -- 0-16; 0 = transitively resolved
    detected_at      TIMESTAMPTZ DEFAULT now(),

    CONSTRAINT dndup_doc_id_fk    FOREIGN KEY (doc_id)           REFERENCES unified_documents(doc_id),
    CONSTRAINT dndup_canonical_fk FOREIGN KEY (canonical_doc_id) REFERENCES unified_documents(doc_id),
    CONSTRAINT dndup_canonical_lower CHECK (doc_id > canonical_doc_id)
);
CREATE INDEX document_near_duplicates_canonical_idx ON document_near_duplicates (canonical_doc_id);
```

One row per **duplicate** document. Roots (canonical documents) do NOT appear as `doc_id`
in this table — they are the "winners" by virtue of having the lowest `doc_id` in their cluster.

`shared_bands` interpretation:
- `0`  — document is in the cluster only by transitivity (A≈B, B≈C → C is in A's cluster, but A and C never directly collided in a band)
- `1–16` — number of bands where this document and its canonical directly collided

### `lsh_candidate_bands` (UNLOGGED work table)

```sql
CREATE UNLOGGED TABLE lsh_candidate_bands (
    doc_id     BIGINT   NOT NULL,
    band_index SMALLINT NOT NULL,
    band_hash  BIGINT   NOT NULL
);
-- Index built post-load:
CREATE INDEX lsh_candidate_bands_band_idx ON lsh_candidate_bands (band_index, band_hash);
```

Persists between incremental runs. Truncated on DB crash (UNLOGGED). Holds the full
MinHash band sketch for every processed document — ~4 GB at 7M docs, growing linearly.

---

## Running the Pipeline

```bash
# Install dependency
uv sync --group near_dedup

# Apply migration (once)
python -c "
import psycopg2
sql = open('src/data-storage/scripts/migrations/add_near_dedup_table.sql').read()
conn = psycopg2.connect('postgresql://corpus:corpus_secret@localhost:5432/corpus')
conn.cursor().execute(sql); conn.commit()
"

# First full run (all docs, 8 parallel workers)
python -m training_signal_processing.pipelines.near_dedup.run \
    --conn "postgresql://corpus:corpus_secret@localhost:5432/corpus" \
    --full \
    --workers 8

# Every subsequent run as new docs are ingested (incremental)
python -m training_signal_processing.pipelines.near_dedup.run \
    --conn "postgresql://corpus:corpus_secret@localhost:5432/corpus" \
    --workers 8
```

---

## Querying Results

```sql
-- How many near-duplicates were found?
SELECT count(*) FROM document_near_duplicates;

-- Sanity: canonical always has lower doc_id (expect 0)
SELECT count(*) FROM document_near_duplicates WHERE canonical_doc_id > doc_id;

-- Sanity: no chains — canonical itself is never a duplicate (expect 0)
SELECT count(*)
FROM document_near_duplicates d
JOIN document_near_duplicates c ON d.canonical_doc_id = c.doc_id;

-- Near-duplicate rate by source
SELECT u.cleaning_source,
       count(*) AS near_dup_count,
       round(count(*) * 100.0 / sum(count(*)) OVER (), 2) AS pct
FROM document_near_duplicates nd
JOIN unified_documents u ON nd.doc_id = u.doc_id
GROUP BY u.cleaning_source
ORDER BY near_dup_count DESC;

-- Find all docs in the same cluster as a given doc
-- (works for both roots and duplicates)
WITH target AS (
    SELECT COALESCE(nd.canonical_doc_id, 55000) AS root
    FROM unified_documents u
    LEFT JOIN document_near_duplicates nd ON u.doc_id = nd.doc_id
    WHERE u.doc_id = 55000
)
SELECT u.doc_id, u.cleaning_source, nd.shared_bands
FROM unified_documents u
LEFT JOIN document_near_duplicates nd ON u.doc_id = nd.doc_id
WHERE u.doc_id = (SELECT root FROM target)
   OR nd.canonical_doc_id = (SELECT root FROM target)
ORDER BY u.doc_id;

-- Exclude near-duplicates from training (keep only canonical docs)
SELECT doc_id
FROM unified_documents u
WHERE cleaning_is_dropped = FALSE
  AND NOT EXISTS (
    SELECT 1 FROM document_near_duplicates nd WHERE nd.doc_id = u.doc_id
  );
```

---

## File Map

```
src/training_signal_processing/pipelines/near_dedup/
├── README.md                  ← this file
├── __init__.py
├── config.py                  ← NearDedupConfig dataclass
├── compute_bands.py           ← Phase 1: parallel MinHash band computation
├── detect_pairs.py            ← Phase 2-5: pair detection, Union-Find, upsert
└── run.py                     ← CLI entrypoint

src/data-storage/scripts/migrations/
└── add_near_dedup_table.sql   ← creates document_near_duplicates + lsh_candidate_bands
```
