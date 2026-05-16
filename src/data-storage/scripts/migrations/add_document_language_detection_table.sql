-- Document-level language detection using Malaya fasttext.
-- One row per document; populated by lang_detect/run.py.
CREATE TABLE IF NOT EXISTS document_language_detection (
    doc_id         BIGINT NOT NULL PRIMARY KEY,
    language_label TEXT   NOT NULL,
    confidence     REAL   NOT NULL,
    detected_at    TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT dld_doc_id_fk FOREIGN KEY (doc_id)
        REFERENCES unified_documents(doc_id)
);

CREATE INDEX IF NOT EXISTS document_language_detection_label_idx
    ON document_language_detection (language_label);
