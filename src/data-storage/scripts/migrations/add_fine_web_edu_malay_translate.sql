-- Migration: add the FineWeb-Edu → Malay translation derived source
-- Run manually (after add_fineweb_edu.sql + the fineweb-edu ingest):
--   docker compose -f src/data-storage/docker-compose.yml exec -T postgres \
--     psql -U corpus -f /scripts/migrations/add_fine_web_edu_malay_translate.sql
--
-- This is a DERIVED source — there is no raw per-source table. Rows are produced
-- by scripts/translate_fineweb_edu_malay.py, which reads cleaning_source=
-- 'fineweb-edu' unified docs, translates them to Malay with Qwen3-14B-AWQ
-- (vLLM, thinking mode), and writes them straight into the unified layer.
-- Provenance is implicit in the sample_uid:
--   fine-web-edu-malay-translate://<hf_id>  ↔  fineweb-edu://<hf_id>
-- The translator also writes document_language_detection rows with
-- language_label='standard-malay' (no LID model run), so
-- add_document_language_detection_table.sql must already be applied. This
-- migration does not recreate it.

-- 1. Register source
INSERT INTO data_sources (source_name, source_type, description)
VALUES ('fine-web-edu-malay-translate', 'web', 'FineWeb-Edu Malay translation (Qwen3-14B-AWQ, thinking mode; machine translation of the fineweb-edu English corpus)')
ON CONFLICT (source_name) DO NOTHING;

-- 2. unified_document_texts partition (populated by translate_fineweb_edu_malay.py)
CREATE TABLE IF NOT EXISTS unified_document_texts_fine_web_edu_malay_translate
    PARTITION OF unified_document_texts FOR VALUES IN ('fine-web-edu-malay-translate');

-- 3. Guard: verify the default partition is still empty (catches partition routing bugs)
DO $$
BEGIN
    IF (SELECT COUNT(*) FROM unified_document_texts_default) > 0 THEN
        RAISE EXCEPTION 'unified_document_texts_default is not empty — partition routing broken';
    END IF;
END $$;
