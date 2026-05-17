-- Migration: add HuggingFace FineWeb-Edu source
-- Run manually after the initial unified-data ingest:
--   docker compose -f src/data-storage/docker-compose.yml exec -T postgres \
--     psql -U corpus -f /scripts/migrations/add_fineweb_edu.sql
--
-- fineweb-edu is promoted directly into the unified layer by
-- scripts/ingest_fineweb_edu.py (it is NOT raw-only). That ingest also writes
-- document_language_detection rows with language_label='standard-english'
-- (no LID model run), so add_document_language_detection_table.sql must already
-- be applied before running the ingest. This migration does not recreate it.

-- 1. Register source
INSERT INTO data_sources (source_name, source_type, description)
VALUES ('fineweb-edu', 'web', 'HuggingFace FineWeb-Edu, English educational-filtered web corpus (CC-MAIN)')
ON CONFLICT (source_name) DO NOTHING;

-- 2. Source table (raw text; also copied verbatim into unified_document_texts)
CREATE TABLE IF NOT EXISTS fineweb_edu_documents (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    hf_id               TEXT        NOT NULL UNIQUE,   -- 'id', 47-char UUID; dedup key
    url                 TEXT,
    dump                TEXT,                           -- CC snapshot: CC-MAIN-2024-51 etc.
    file_path           TEXT,                           -- S3 source path
    language            TEXT,                           -- always 'en'
    language_score      REAL,
    source_token_count  BIGINT,                         -- 'token_count' from the dataset itself
    edu_score           REAL,                           -- 'score', continuous edu quality
    edu_int_score       SMALLINT,                       -- 'int_score', 3-5 discrete edu rating
    content             TEXT COMPRESSION lz4,           -- 'text' field; sole copy until cleaning runs
    created_at          TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS fineweb_edu_url_idx       ON fineweb_edu_documents (url);
CREATE INDEX IF NOT EXISTS fineweb_edu_dump_idx      ON fineweb_edu_documents (dump);
CREATE INDEX IF NOT EXISTS fineweb_edu_int_score_idx ON fineweb_edu_documents (edu_int_score);

-- 3. unified_document_texts partition (populated by ingest_fineweb_edu.py's unified-promotion step)
CREATE TABLE IF NOT EXISTS unified_document_texts_fineweb_edu
    PARTITION OF unified_document_texts FOR VALUES IN ('fineweb-edu');

-- 4. Guard: verify the default partition is still empty (catches partition routing bugs)
DO $$
BEGIN
    IF (SELECT COUNT(*) FROM unified_document_texts_default) > 0 THEN
        RAISE EXCEPTION 'unified_document_texts_default is not empty — partition routing broken';
    END IF;
END $$;
