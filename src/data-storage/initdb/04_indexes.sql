-- Non-PK indexes safe to build at container init (empty tables, no CONCURRENTLY, no FK).
-- Dedup unique indexes and FK are in scripts/post_load/04b_post_load_indexes.sql —
-- run that script MANUALLY after bulk COPY ingest.

-- ── unified_documents ─────────────────────────────────────────────────────────

-- Common filter pattern: source + drop flag + sequential doc scan
CREATE INDEX unified_documents_source_dropped_idx
    ON unified_documents (cleaning_source, cleaning_is_dropped, doc_id);

-- BRIN: cheap range scan on append-correlated doc_id (one entry per 128 pages)
CREATE INDEX unified_documents_doc_id_brin
    ON unified_documents USING BRIN (doc_id) WITH (pages_per_range = 128);

-- Repeatable sampling: range query on precomputed sample_key avoids ORDER BY random()
CREATE INDEX unified_documents_sample_key_idx
    ON unified_documents (sample_key);

-- Rule-triggered filtering: GIN for TEXT[] containment (@>, &&)
CREATE INDEX unified_documents_cleaning_rules_gin
    ON unified_documents USING GIN (cleaning_rules_triggered);

-- ── unified_document_texts ────────────────────────────────────────────────────
-- PK (cleaning_source, doc_id) covers the primary join. No additional indexes needed.

-- ── lowyat ────────────────────────────────────────────────────────────────────
CREATE INDEX lowyat_posts_thread_id_idx ON lowyat_posts (thread_id);
CREATE INDEX lowyat_posts_author_idx    ON lowyat_posts (author);
CREATE INDEX lowyat_posts_posted_at_idx ON lowyat_posts (posted_at);

-- ── cari ──────────────────────────────────────────────────────────────────────
CREATE INDEX cari_posts_thread_id_idx   ON cari_posts (thread_id);
CREATE INDEX cari_posts_author_idx      ON cari_posts (author);
CREATE INDEX cari_posts_posted_at_idx   ON cari_posts (posted_at);

-- ── reddit ────────────────────────────────────────────────────────────────────
CREATE INDEX reddit_bolehland_submission_idx ON reddit_bolehland_posts (submission_id);
CREATE INDEX reddit_bolehland_parent_idx     ON reddit_bolehland_posts (parent_id);
CREATE INDEX reddit_bolehland_author_idx     ON reddit_bolehland_posts (author);
CREATE INDEX reddit_bolehland_created_idx    ON reddit_bolehland_posts (created_utc);

CREATE INDEX reddit_indonesia_submission_idx ON reddit_indonesia_posts (submission_id);
CREATE INDEX reddit_indonesia_parent_idx     ON reddit_indonesia_posts (parent_id);
CREATE INDEX reddit_indonesia_author_idx     ON reddit_indonesia_posts (author);
CREATE INDEX reddit_indonesia_created_idx    ON reddit_indonesia_posts (created_utc);

-- ── hplt ──────────────────────────────────────────────────────────────────────
CREATE INDEX hplt_malay_url_idx      ON hplt_malay_documents (url);
CREATE INDEX hplt_malay_crawl_idx    ON hplt_malay_documents (crawl_id);
CREATE INDEX hplt_malay_lang_idx     ON hplt_malay_documents (row_language_code);

CREATE INDEX hplt_indonesia_url_idx  ON hplt_indonesia_documents (url);
CREATE INDEX hplt_indonesia_crawl_idx ON hplt_indonesia_documents (crawl_id);
CREATE INDEX hplt_indonesia_lang_idx ON hplt_indonesia_documents (row_language_code);

-- ── ocr_books ─────────────────────────────────────────────────────────────────
CREATE INDEX ocr_books_run_idx       ON ocr_books (source_run_id);

-- ── fineweb ───────────────────────────────────────────────────────────────────
CREATE INDEX fineweb_url_idx         ON fineweb_documents (url);
CREATE INDEX fineweb_month_idx       ON fineweb_documents (month);

-- ── quality_signals ───────────────────────────────────────────────────────────
-- Secondary index for per-run manifest export (inverse of the PK access pattern)
CREATE INDEX quality_signals_run_score_idx
    ON quality_signals (run_id, score DESC, doc_id)
    WHERE score IS NOT NULL;

-- ── lid_metadata ──────────────────────────────────────────────────────────────
CREATE INDEX lid_metadata_lingua_lang_idx ON lid_metadata (lingua_primary_language);
