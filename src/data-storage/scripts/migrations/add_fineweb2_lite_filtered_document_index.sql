-- FineWeb2-Lite retained document index manifest.
--
-- Lean retained-doc table: no cleaned text. Export jobs join text later using
-- (source_domain, doc_id), which matches unified_document_texts partitioning.

CREATE TABLE IF NOT EXISTS fineweb2_lite_filtered_document_index (
    profile_name          TEXT        NOT NULL,
    run_id                TEXT        NOT NULL,
    doc_id                BIGINT      NOT NULL,
    source_doc_id         BIGINT      NOT NULL,
    source_domain         TEXT        NOT NULL,
    lid_label             TEXT        NOT NULL,
    lid_confidence        REAL        NOT NULL,
    metadata_created_at   TIMESTAMPTZ,
    lid_detected_at       TIMESTAMPTZ,
    filter_sql_sha256     TEXT        NOT NULL,
    indexed_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (profile_name, run_id, doc_id)
);

CREATE TABLE IF NOT EXISTS fineweb2_lite_filtered_document_index_refreshes (
    refresh_id        BIGSERIAL PRIMARY KEY,
    profile_name      TEXT        NOT NULL,
    run_id            TEXT        NOT NULL,
    mode              TEXT        NOT NULL CHECK (mode IN ('full', 'incremental')),
    status            TEXT        NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at      TIMESTAMPTZ,
    lower_watermark   TIMESTAMPTZ,
    upper_watermark   TIMESTAMPTZ,
    filter_sql_sha256 TEXT        NOT NULL,
    rows_upserted     BIGINT,
    rows_deleted      BIGINT,
    error_text        TEXT
);

CREATE INDEX IF NOT EXISTS fineweb2_lite_filtered_doc_idx_source_doc
    ON fineweb2_lite_filtered_document_index (profile_name, run_id, source_domain, doc_id);

CREATE INDEX IF NOT EXISTS fineweb2_lite_filtered_doc_idx_lid_source_doc
    ON fineweb2_lite_filtered_document_index (
        profile_name,
        run_id,
        lid_label,
        source_domain,
        doc_id
    );

CREATE INDEX IF NOT EXISTS fineweb2_lite_filtered_doc_idx_refresh_status
    ON fineweb2_lite_filtered_document_index_refreshes (
        profile_name,
        run_id,
        filter_sql_sha256,
        status,
        completed_at
    );
