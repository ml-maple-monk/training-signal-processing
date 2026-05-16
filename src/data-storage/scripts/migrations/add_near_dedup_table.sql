-- Near-duplicate detection tables for MinHash LSH dedup pipeline.
-- Run this migration once before executing near_dedup/run.py.
--
-- document_near_duplicates: permanent marking table (never deleted).
-- lsh_candidate_bands:      persistent work table; survives crashes to enable resume.

-- Permanent marking table: one row per near-duplicate document.
-- canonical_doc_id is always the lowest doc_id in the cluster (earliest ingest).
-- shared_bands is the direct band overlap count (0 for transitively resolved members).
CREATE TABLE IF NOT EXISTS document_near_duplicates (
    doc_id           BIGINT   NOT NULL PRIMARY KEY,
    canonical_doc_id BIGINT   NOT NULL,
    shared_bands     SMALLINT NOT NULL,
    detected_at      TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT dndup_doc_id_fk    FOREIGN KEY (doc_id)           REFERENCES unified_documents(doc_id),
    CONSTRAINT dndup_canonical_fk FOREIGN KEY (canonical_doc_id) REFERENCES unified_documents(doc_id),
    CONSTRAINT dndup_canonical_lower CHECK (doc_id > canonical_doc_id)
);

CREATE INDEX IF NOT EXISTS document_near_duplicates_canonical_idx
    ON document_near_duplicates (canonical_doc_id);

-- Persistent work table: logged so data survives a DB crash, enabling incremental resume.
-- --full mode TRUNCATEs this before starting; incremental mode preserves existing rows.
-- Index is built after bulk-load (not during COPY) for speed.
CREATE TABLE IF NOT EXISTS lsh_candidate_bands (
    doc_id     BIGINT   NOT NULL,
    band_index SMALLINT NOT NULL,
    band_hash  BIGINT   NOT NULL
);
-- Note: index lsh_candidate_bands_band_idx is created by compute_bands.py after COPY is done.

-- doc_id index backs the per-batch anti-join in incremental mode
-- (NOT EXISTS against this table). compute_bands.py also ensures this index
-- exists at runtime via CREATE INDEX IF NOT EXISTS; declared here so fresh
-- deployments have it from the start.
CREATE INDEX IF NOT EXISTS lsh_candidate_bands_doc_id_idx
    ON lsh_candidate_bands (doc_id);
