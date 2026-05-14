-- Migration: add SEA-PILE-v2 Malay source
-- Run manually after the initial unified-data ingest:
--   docker compose -f src/data-storage/docker-compose.yml exec -T postgres \
--     psql -U corpus -f /scripts/migrations/add_sea_pile_malay.sql

-- 1. Register source
INSERT INTO data_sources (source_name, source_type, description)
VALUES ('sea-pile-malay', 'web', 'AIsingapore SEA-PILE v2, Malay language subset (CC-MAIN web crawls)')
ON CONFLICT (source_name) DO NOTHING;

-- 2. Source table (raw text; content is source of truth for the cleaning pipeline)
CREATE TABLE IF NOT EXISTS sea_pile_malay_documents (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    warc_record_id  TEXT        NOT NULL UNIQUE,   -- 'warc-record-id', 47-char UUID-in-brackets
    url             TEXT,
    dump            TEXT,                           -- CC snapshot: CC-MAIN-2020-45 etc.
    crawl_timestamp TIMESTAMPTZ,                    -- 'timestamp' field, WARC UTC
    content         TEXT COMPRESSION lz4,           -- 'text' field; sole copy until cleaning runs
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sea_pile_malay_url_idx  ON sea_pile_malay_documents (url);
CREATE INDEX IF NOT EXISTS sea_pile_malay_dump_idx ON sea_pile_malay_documents (dump);

-- 3. unified_document_texts partition (populated after cleaning pipeline runs, not during raw ingest)
CREATE TABLE IF NOT EXISTS unified_document_texts_sea_pile_malay
    PARTITION OF unified_document_texts FOR VALUES IN ('sea-pile-malay');

-- 4. Guard: verify the default partition is still empty (catches partition routing bugs)
DO $$
BEGIN
    IF (SELECT COUNT(*) FROM unified_document_texts_default) > 0 THEN
        RAISE EXCEPTION 'unified_document_texts_default is not empty — partition routing broken';
    END IF;
END $$;
