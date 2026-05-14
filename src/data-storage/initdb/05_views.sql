-- Materialized views and regular views for training manifest generation and monitoring.

-- ── Training manifest (metadata only — no text join) ─────────────────────────
-- Empty at init time — this is expected, not an error.
-- After each ingest batch: REFRESH MATERIALIZED VIEW CONCURRENTLY training_manifest;
-- (training_manifest_doc_id_ux below enables the CONCURRENTLY form.)
CREATE MATERIALIZED VIEW training_manifest AS
SELECT
    ud.doc_id,
    ud.cleaning_source,
    ud.source_parquet_url,
    ud.source_row_group_index,
    ud.source_row_index,
    ud.cleaned_o200k_token_count,
    ud.cleaned_char_count,
    ud.sample_key,
    ud.cleaning_is_dropped
FROM unified_documents ud
WHERE ud.cleaning_is_dropped = FALSE
  AND ud.cleaned_o200k_token_count IS NOT NULL;

-- Required for REFRESH MATERIALIZED VIEW CONCURRENTLY
CREATE UNIQUE INDEX training_manifest_doc_id_ux    ON training_manifest (doc_id);
CREATE INDEX training_manifest_source_idx          ON training_manifest (cleaning_source, doc_id);
CREATE INDEX training_manifest_sample_key_idx      ON training_manifest (sample_key);

-- ── Per-source statistics (regular view — always current) ─────────────────────
CREATE VIEW source_stats AS
SELECT
    ds.source_name,
    ds.source_type,
    COUNT(*)                                          AS total_docs,
    COUNT(*) FILTER (WHERE NOT ud.cleaning_is_dropped) AS active_docs,
    SUM(ud.cleaned_o200k_token_count)                 AS total_tokens,
    SUM(ud.cleaned_char_count)                        AS total_chars
FROM unified_documents ud
JOIN data_sources ds USING (source_id)
GROUP BY ds.source_name, ds.source_type;

-- ── Quality-filtered manifest (regular view — parameterise at query time) ─────
-- Example: SELECT * FROM quality_filtered_manifest WHERE signal_name='edu' AND score >= 3.0;
CREATE VIEW quality_filtered_manifest AS
SELECT
    ud.doc_id,
    ud.cleaning_source,
    ud.source_parquet_url,
    ud.source_row_index,
    ud.cleaned_o200k_token_count,
    ud.sample_key,
    qs.score,
    qs.label,
    qr.signal_name
FROM unified_documents ud
JOIN quality_signals qs USING (doc_id)
JOIN quality_runs    qr USING (run_id)
WHERE ud.cleaning_is_dropped = FALSE;
