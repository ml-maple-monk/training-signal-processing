-- Unified hub table (non-partitioned metadata) + partitioned text store + LID + quality tables.
--
-- Two-layer design rationale:
--   PostgreSQL requires unique constraints on partitioned tables to include the partition key.
--   Keeping unified_documents non-partitioned gives it a simple BIGINT PK that lid_metadata
--   and quality_signals can reference with single-column FKs.
--   The text (84GB) lives in unified_document_texts, partitioned by cleaning_source so each
--   source's TOAST segments are isolated — HPLT queries don't page in forum TOAST.

-- ── UNIFIED DOCUMENTS (metadata hub, non-partitioned) ────────────────────────

CREATE TABLE unified_documents (
    doc_id                           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    sample_uid                       TEXT NOT NULL,
    sample_uid_hash                  TEXT NOT NULL,       -- SHA-256 hex of sample_uid
    source_id                        SMALLINT NOT NULL REFERENCES data_sources(source_id),
    cleaning_source                  TEXT NOT NULL,       -- 'lowyat', 'cari', 'hplt-malay', etc.
    source_bucket                    TEXT,
    source_object_key                TEXT,
    source_parquet_url               TEXT,
    text_column                      TEXT,                -- name of raw text field in source parquet
    source_row_group_index           BIGINT,
    source_row_index                 BIGINT,
    row_index_in_row_group           BIGINT,
    original_text_sha256             TEXT,
    cleaned_text_sha256              TEXT,
    original_char_count              BIGINT,
    cleaned_char_count               BIGINT,
    removed_char_count               BIGINT,
    approximate_original_token_count BIGINT,
    approximate_cleaned_token_count  BIGINT,
    approximate_removed_token_count  BIGINT,
    cleaning_is_dropped              BOOLEAN NOT NULL DEFAULT FALSE,
    cleaning_rules_triggered         TEXT[],
    cleaned_o200k_token_count        BIGINT,              -- NULL for FineWeb (not yet tokenized)
    cleaned_o200k_tokenizer          TEXT,
    sample_key                       BIGINT,              -- precomputed for fast repeatable sampling
    created_at                       TIMESTAMPTZ DEFAULT now()
);

-- ── UNIFIED DOCUMENT TEXTS (partitioned TOAST text store) ────────────────────
-- Partitioned by cleaning_source (LIST). Each partition owns its own TOAST file.
-- FK to unified_documents is added after bulk load via 04b_post_load_indexes.sql.

CREATE TABLE unified_document_texts (
    doc_id          BIGINT NOT NULL,
    cleaning_source TEXT NOT NULL,   -- LIST partition key
    cleaned_text    TEXT COMPRESSION lz4,
    PRIMARY KEY (cleaning_source, doc_id)
) PARTITION BY LIST (cleaning_source);

CREATE TABLE unified_document_texts_books_ocr
    PARTITION OF unified_document_texts FOR VALUES IN ('books-ocr');
CREATE TABLE unified_document_texts_lowyat
    PARTITION OF unified_document_texts FOR VALUES IN ('lowyat');
CREATE TABLE unified_document_texts_cari
    PARTITION OF unified_document_texts FOR VALUES IN ('cari');
CREATE TABLE unified_document_texts_reddit_bolehland
    PARTITION OF unified_document_texts FOR VALUES IN ('reddit-bolehland');
CREATE TABLE unified_document_texts_reddit_indonesia
    PARTITION OF unified_document_texts FOR VALUES IN ('reddit-indonesia');
CREATE TABLE unified_document_texts_hplt_malay
    PARTITION OF unified_document_texts FOR VALUES IN ('hplt-malay');
CREATE TABLE unified_document_texts_hplt_indonesia
    PARTITION OF unified_document_texts FOR VALUES IN ('hplt-indonesia');
CREATE TABLE unified_document_texts_fineweb
    PARTITION OF unified_document_texts FOR VALUES IN ('fineweb');
CREATE TABLE unified_document_texts_default
    PARTITION OF unified_document_texts DEFAULT;

-- ── LID METADATA ─────────────────────────────────────────────────────────────

CREATE TABLE lid_metadata (
    doc_id                       BIGINT PRIMARY KEY REFERENCES unified_documents(doc_id),
    lid_cleaned_token_count      BIGINT,
    reference_removed            BOOLEAN,
    reference_removal_method     TEXT,
    removed_reference_char_count BIGINT,
    lingua_primary_language      TEXT,
    lingua_spans                 JSONB,  -- list<struct{start_index, end_index, language_label}>
    malaya_document_label        TEXT,
    malaya_document_scores       JSONB,  -- list<struct{label, score}>
    malaya_word_detections       JSONB,  -- list<struct{word_index, start_index, end_index, token, label}>
    malaya_word_label_counts     JSONB   -- list<struct{label, count}>
);

-- ── QUALITY SIGNALS ───────────────────────────────────────────────────────────

CREATE TABLE quality_runs (
    run_id         INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    signal_name    TEXT NOT NULL,
    model_name     TEXT NOT NULL,
    model_version  TEXT,
    prompt_version TEXT,
    code_version   TEXT,
    created_at     TIMESTAMPTZ DEFAULT now()
);

-- PK (doc_id, run_id): optimised for per-document access pattern.
-- Secondary index on (run_id, score DESC, doc_id) in 04_indexes.sql covers per-run manifest export.
CREATE TABLE quality_signals (
    doc_id     BIGINT  NOT NULL REFERENCES unified_documents(doc_id),
    run_id     INTEGER NOT NULL REFERENCES quality_runs(run_id),
    score      REAL,
    label      SMALLINT,
    raw_output JSONB,
    PRIMARY KEY (doc_id, run_id)
);
