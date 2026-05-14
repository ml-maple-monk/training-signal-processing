For your case, optimize Postgres as an **append-only corpus index**, not as a normal OLTP app database.

**Goal:** 200GB corpus, 8 CPU, 16GB RAM, maximum ingest + filtering speed.

My strongest recommendation:

```text
Raw text:          Parquet / JSONL.zstd shards outside Postgres
Postgres:          document IDs, hashes, source pointers, quality scores, enrichment metadata
Optional text in PG: separate document_text table only
```

Postgres can store the raw text, but if you want speed and low disk usage, keep the raw payload in compressed shards and use Postgres to generate manifests.

---

## 1. Aggressive `postgresql.conf` for 8 CPU / 16GB RAM

This is a good starting profile for a **single-node corpus DB on NVMe SSD**.

```conf
# -------------------------
# Connections
# -------------------------
max_connections = 20

# -------------------------
# Memory
# -------------------------
shared_buffers = 4GB
effective_cache_size = 12GB
work_mem = 64MB
hash_mem_multiplier = 2.0
maintenance_work_mem = 2GB
autovacuum_work_mem = 512MB
temp_buffers = 128MB
huge_pages = try

# -------------------------
# Parallelism: 8 CPU box
# -------------------------
max_worker_processes = 16
max_parallel_workers = 8
max_parallel_workers_per_gather = 4
max_parallel_maintenance_workers = 4

# -------------------------
# WAL / checkpoint
# -------------------------
wal_buffers = 64MB
wal_compression = on
synchronous_commit = off
checkpoint_timeout = 30min
checkpoint_completion_target = 0.9
max_wal_size = 32GB
min_wal_size = 4GB

# -------------------------
# Planner, SSD/NVMe bias
# -------------------------
random_page_cost = 1.1
seq_page_cost = 1.0
effective_io_concurrency = 200
maintenance_io_concurrency = 200

# -------------------------
# Autovacuum: keep enabled, but less annoying
# -------------------------
autovacuum = on
autovacuum_max_workers = 3
autovacuum_naptime = 30s

# -------------------------
# Monitoring
# -------------------------
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.track = all
track_io_timing = on
```

Why these values: `shared_buffers = 4GB` follows the usual Postgres starting point of ~25% RAM for a dedicated server; Postgres also relies heavily on the OS page cache, so going much higher than 40% is usually not better. `work_mem` is intentionally not huge because it is **per sort/hash operation**, and one query can use it multiple times. `maintenance_work_mem` is much larger because index creation and maintenance operations benefit from it and normally run in fewer sessions. ([PostgreSQL][1])

`effective_io_concurrency = 200` is aggressive for NVMe-style storage. Postgres documents it as the number of concurrent I/O operations a session may try to issue, with higher values most useful on high-IOPS devices, though unnecessarily high values can increase latency. ([PostgreSQL][1])

---

## 2. More aggressive but unsafe rebuild mode

Use this only if the DB can be fully rebuilt from external corpus files.

```conf
# Use only during rebuildable bulk ingestion
fsync = off
full_page_writes = off
synchronous_commit = off
wal_level = minimal
archive_mode = off
max_wal_senders = 0
```

This is fast, but dangerous. Postgres explicitly warns that turning `fsync` off can cause unrecoverable data corruption after power loss or OS crash; `synchronous_commit = off` is safer because it can lose recent transactions after a crash but should not corrupt the database. ([PostgreSQL][2])

My practical rule:

```text
During one-time import from reproducible shards:
  unsafe rebuild mode is acceptable.

For a long-lived metadata DB:
  keep fsync = on, but use synchronous_commit = off.
```

---

## 3. Schema optimized for speed

Use **hot metadata tables** and **cold payload tables**.

```sql
CREATE TABLE documents (
    doc_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    source_id SMALLINT NOT NULL,
    snapshot_id INTEGER NOT NULL,

    lang TEXT NOT NULL,
    n_bytes INTEGER NOT NULL,
    n_tokens INTEGER,

    content_hash BYTEA NOT NULL,
    url_hash BYTEA,

    shard_uri TEXT NOT NULL,
    row_idx BIGINT NOT NULL,

    sample_key BIGINT NOT NULL,

    created_at TIMESTAMPTZ DEFAULT now()
) WITH (fillfactor = 100);

CREATE TABLE quality_runs (
    run_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    signal_name TEXT NOT NULL,
    model_name TEXT,
    model_version TEXT,
    prompt_version TEXT,
    code_version TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE quality_signals (
    doc_id BIGINT NOT NULL REFERENCES documents(doc_id),
    run_id BIGINT NOT NULL REFERENCES quality_runs(run_id),

    score REAL,
    label SMALLINT,
    raw_output JSONB,

    PRIMARY KEY (run_id, doc_id)
) WITH (fillfactor = 100);
```

For optional raw text in Postgres:

```sql
CREATE TABLE document_text (
    doc_id BIGINT PRIMARY KEY REFERENCES documents(doc_id),
    text TEXT COMPRESSION lz4
);
```

Keep `document_text` separate. Large `TEXT` values are TOASTed out-of-line once rows exceed roughly 2KB, and Postgres notes that this keeps the main table smaller so more hot rows fit in cache. ([PostgreSQL][3])

---

## 4. Indexes: minimal, targeted, no vanity indexes

Create only the indexes that directly support your corpus operations.

```sql
-- Dedup
CREATE UNIQUE INDEX documents_content_hash_idx
ON documents (content_hash);

-- Common filtering
CREATE INDEX documents_source_lang_idx
ON documents (source_id, lang, doc_id);

-- Fast append/range scans
CREATE INDEX documents_doc_id_brin
ON documents USING brin (doc_id);

-- Quality filtering
CREATE INDEX quality_run_score_doc_idx
ON quality_signals (run_id, score DESC, doc_id);

-- Common high-quality subset
CREATE INDEX quality_high_score_idx
ON quality_signals (run_id, score DESC, doc_id)
WHERE score >= 3.0;
```

Use **BRIN** for append-correlated columns like `doc_id`, `created_at`, or `snapshot_id`. BRIN indexes are designed for very large tables where column values correlate with physical table order and are much smaller than full B-tree indexes. ([PostgreSQL][4])

Avoid:

```text
GIN index on every JSONB field
full-text index over all raw text
trigram index over raw text
many single-column indexes
foreign keys during massive ingest
```

---

## 5. Fast ingestion path

Use this pipeline:

```text
1. Preprocess outside Postgres:
   hash, language ID, token count, source ID, shard URI, row index

2. COPY into UNLOGGED staging tables

3. Insert from staging into final tables

4. Build indexes after bulk load

5. ANALYZE

6. Export train manifests from Postgres
```

Postgres explicitly recommends `COPY` over many `INSERT`s, creating indexes after bulk loading, increasing `maintenance_work_mem`, increasing `max_wal_size`, and running `ANALYZE` after large loads. ([PostgreSQL][5])

Example:

```sql
CREATE UNLOGGED TABLE staging_documents (
    source_id SMALLINT,
    snapshot_id INTEGER,
    lang TEXT,
    n_bytes INTEGER,
    n_tokens INTEGER,
    content_hash BYTEA,
    url_hash BYTEA,
    shard_uri TEXT,
    row_idx BIGINT,
    sample_key BIGINT
);

COPY staging_documents
FROM '/data/staging/documents.csv'
WITH (FORMAT csv);

INSERT INTO documents (
    source_id, snapshot_id, lang, n_bytes, n_tokens,
    content_hash, url_hash, shard_uri, row_idx, sample_key
)
SELECT
    source_id, snapshot_id, lang, n_bytes, n_tokens,
    content_hash, url_hash, shard_uri, row_idx, sample_key
FROM staging_documents
ON CONFLICT DO NOTHING;

ANALYZE documents;
```

For maximum ingest speed, run **4 parallel COPY workers**, not 8. With only 16GB RAM, 8 concurrent writers can become WAL/checkpoint/IO contention rather than faster ingestion.

---

## 6. Session settings for bulk jobs

For one big load or index build:

```sql
SET synchronous_commit = off;
SET maintenance_work_mem = '4GB';
SET work_mem = '256MB';
```

Do **not** set `work_mem = 256MB` globally. It is per operation, not per query, so one parallel query can multiply memory usage quickly. ([PostgreSQL][1])

---

## 7. Sampling optimization: avoid `ORDER BY random()`

Do not sample training data like this:

```sql
ORDER BY random()
LIMIT 100000;
```

Instead, precompute `sample_key` once per document.

```sql
SELECT d.shard_uri, d.row_idx
FROM quality_signals q
JOIN documents d ON d.doc_id = q.doc_id
WHERE q.run_id = 7
  AND q.score >= 3.0
  AND d.lang = 'en'
  AND d.sample_key BETWEEN 100000000000 AND 200000000000
LIMIT 100000;
```

This makes sampling index-friendly and repeatable.

---

## 8. Manifest-first training export

For training, do not repeatedly join giant tables at runtime. Create manifest tables.

```sql
CREATE MATERIALIZED VIEW manifest_fw_edu_score3 AS
SELECT d.doc_id, d.shard_uri, d.row_idx, q.score
FROM quality_signals q
JOIN documents d ON d.doc_id = q.doc_id
WHERE q.run_id = 7
  AND q.score >= 3.0
  AND d.lang = 'en';

CREATE INDEX manifest_fw_edu_score3_sample_idx
ON manifest_fw_edu_score3 (doc_id);
```

Then export:

```sql
COPY (
    SELECT shard_uri, row_idx, score
    FROM manifest_fw_edu_score3
    ORDER BY doc_id
) TO '/data/manifests/fw_edu_score3.csv'
WITH (FORMAT csv, HEADER true);
```

This gives you a cheap bridge:

```text
Postgres decides what data to train on.
Training loader reads raw text from compressed shards.
```

---

## 9. Extensions I would actually use

```sql
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_prewarm;
CREATE EXTENSION IF NOT EXISTS pg_buffercache;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
```

Use `pg_stat_statements` first; it tracks planning/execution statistics for SQL statements and is the fastest way to find your real bottleneck queries. ([PostgreSQL][6])

Use `pg_prewarm` only for hot manifest/index tables after restart; it can preload relation data into the OS or Postgres buffer cache. ([PostgreSQL][7])

Use `pg_buffercache` for debugging cache behavior, not as part of normal runtime.

For embeddings or semantic dedup, add `pgvector`, but I would not put a huge embedding index on the same 16GB machine unless it is a small inspection subset.

---

## 10. Final aggressive design

Use this:

```text
Storage:
  raw text in zstd Parquet/JSONL shards
  Postgres stores metadata + scores + shard pointers

Postgres memory:
  shared_buffers = 4GB
  effective_cache_size = 12GB
  work_mem = 64MB global
  maintenance_work_mem = 2GB global, 4GB per solo index build

Writes:
  COPY only
  UNLOGGED staging
  indexes after load
  synchronous_commit = off
  4 parallel loaders

Reads:
  precomputed sample_key
  materialized manifests
  BRIN for append/range columns
  B-tree only for common filters
  no full-text index on raw corpus

Durability:
  safe fast mode: fsync on, synchronous_commit off
  rebuild mode: fsync off, full_page_writes off, wal_level minimal
```

The biggest speed win is not one config parameter. It is this design rule:

> **Never make Postgres repeatedly scan 200GB of raw text. Let Postgres select document pointers; let your training pipeline read compressed shards directly.**

[1]: https://www.postgresql.org/docs/current/runtime-config-resource.html "PostgreSQL: Documentation: 18: 19.4. Resource Consumption"
[2]: https://www.postgresql.org/docs/current/runtime-config-wal.html "PostgreSQL: Documentation: 18: 19.5. Write Ahead Log"
[3]: https://www.postgresql.org/docs/current/storage-toast.html "PostgreSQL: Documentation: 18: 66.2. TOAST"
[4]: https://www.postgresql.org/docs/current/brin.html "PostgreSQL: Documentation: 18: 65.5. BRIN Indexes"
[5]: https://www.postgresql.org/docs/current/populate.html "PostgreSQL: Documentation: 18: 14.4. Populating a Database"
[6]: https://www.postgresql.org/docs/current/pgstatstatements.html?utm_source=chatgpt.com "F.32. pg_stat_statements — track statistics of SQL planning ..."
[7]: https://www.postgresql.org/docs/current/pgprewarm.html?utm_source=chatgpt.com "F.30. pg_prewarm — preload relation data into buffer caches"
