-- Post-load indexes: run MANUALLY after bulk COPY ingest — NOT at container init.
-- This file lives in scripts/post_load/ (not initdb/) so Docker does NOT auto-execute it.
--
-- Why separate from initdb/04_indexes.sql:
--   docker-entrypoint-initdb.d runs each .sql file inside a transaction.
--   CREATE INDEX CONCURRENTLY is forbidden inside a transaction block.
--   These indexes also need data present to be meaningful (dedup, FK validation).
--
-- To run after ingest:
--   docker compose exec postgres psql -U corpus -f /scripts/post_load/04b_post_load_indexes.sql

-- ── Dedup unique indexes ───────────────────────────────────────────────────────
-- CONCURRENTLY lets the DB serve reads+writes while building the index.
-- Takes longer than non-concurrent but does not block ingest pipelines.

CREATE UNIQUE INDEX CONCURRENTLY unified_documents_sample_uid_ux
    ON unified_documents (sample_uid);

CREATE UNIQUE INDEX CONCURRENTLY unified_documents_cleaned_hash_ux
    ON unified_documents (cleaned_text_sha256);

-- ── FK: unified_document_texts → unified_documents ────────────────────────────
-- NOT VALID: adds the constraint without scanning existing rows (instant, minimal lock).
-- VALIDATE:  scans with SHARE UPDATE EXCLUSIVE lock — non-blocking for concurrent reads.
-- Run these together after a full ingest batch to enforce referential integrity.

ALTER TABLE unified_document_texts
    ADD CONSTRAINT unified_document_texts_doc_id_fk
    FOREIGN KEY (doc_id) REFERENCES unified_documents(doc_id)
    NOT VALID;

ALTER TABLE unified_document_texts
    VALIDATE CONSTRAINT unified_document_texts_doc_id_fk;
