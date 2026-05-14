-- Extensions for the pretraining corpus database.
-- Runs automatically at container first start (docker-entrypoint-initdb.d).

-- Core monitoring
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS pg_buffercache;
CREATE EXTENSION IF NOT EXISTS pg_prewarm;

-- Hashing and crypto utilities
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Index type extensions
CREATE EXTENSION IF NOT EXISTS btree_gin;    -- combined B-tree/GIN for multi-column indexes
CREATE EXTENSION IF NOT EXISTS btree_gist;
CREATE EXTENSION IF NOT EXISTS bloom;        -- bloom filter indexes for hash set-membership tests
CREATE EXTENSION IF NOT EXISTS pg_trgm;      -- trigram similarity (short-text only; NOT over cleaned_text)

-- Text normalization
CREATE EXTENSION IF NOT EXISTS unaccent;

-- Array utilities
CREATE EXTENSION IF NOT EXISTS intarray;

-- Partition management (schema must exist before the extension is created)
CREATE SCHEMA IF NOT EXISTS partman;
CREATE EXTENSION IF NOT EXISTS pg_partman SCHEMA partman;

-- Embeddings
-- CAUTION: do NOT create an ivfflat or hnsw index on the full 38M-document corpus on 16GB RAM.
-- Only use vector indexes on small inspection subsets (<100K rows).
CREATE EXTENSION IF NOT EXISTS vector;
