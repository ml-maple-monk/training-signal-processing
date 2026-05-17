-- Qwen3 thinking-mode reasoning traces for translated documents.
-- One row per translated unified document; populated by
-- scripts/translate_fineweb_edu_malay.py. Keyed by the TRANSLATION doc_id
-- (cleaning_source='fine-web-edu-malay-translate' in unified_documents) — the
-- stripped <think>…</think> content that produced that doc's Malay text
-- (chunks concatenated with a blank line when a doc was split).
--
-- Run manually:
--   docker compose -f src/data-storage/docker-compose.yml exec -T postgres \
--     psql -U corpus -f /scripts/migrations/add_document_translation_thinking_table.sql
CREATE TABLE IF NOT EXISTS document_translation_thinking (
    doc_id               BIGINT NOT NULL PRIMARY KEY,
    thinking_text        TEXT COMPRESSION lz4,
    thinking_char_count  BIGINT,
    thinking_token_count BIGINT,
    created_at           TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT dtt_doc_id_fk FOREIGN KEY (doc_id)
        REFERENCES unified_documents(doc_id)
);

CREATE INDEX IF NOT EXISTS document_translation_thinking_token_idx
    ON document_translation_thinking (thinking_token_count);
