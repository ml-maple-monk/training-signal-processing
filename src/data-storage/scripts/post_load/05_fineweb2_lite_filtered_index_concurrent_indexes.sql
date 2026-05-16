-- Live indexes for FineWeb2-Lite retained-index refresh/export.
--
-- Run with psql, not inside an explicit transaction:
--   docker exec corpus-postgres psql -U corpus -d corpus \
--     -f /path/to/05_fineweb2_lite_filtered_index_concurrent_indexes.sql

CREATE INDEX CONCURRENTLY IF NOT EXISTS fineweb2_lite_quality_metadata_retained_unified_idx
    ON fineweb2_lite_quality_metadata (run_id, doc_id, source_doc_id, cleaning_source)
    INCLUDE (created_at)
    WHERE input_source = 'unified'
        AND duplicate_line_fraction <= 0.241
        AND top_2gram_fraction <= 0.194
        AND top_3gram_fraction <= 0.2
        AND top_4gram_fraction <= 0.145
        AND duplicated_5gram_fraction <= 0.166
        AND duplicated_6gram_fraction <= 0.155
        AND duplicated_7gram_fraction <= 0.144
        AND duplicated_8gram_fraction <= 0.133
        AND duplicated_9gram_fraction <= 0.122
        AND duplicated_10gram_fraction <= 0.111
        AND line_punctuation_fraction >= 0.08
        AND duplicated_line_character_ratio <= 0.1
        AND newline_word_ratio <= 0.157
        AND non_symbol_word_count BETWEEN 5 AND 10000000
        AND average_non_symbol_word_length BETWEEN 3 AND 12
        AND hash_symbol_ratio <= 0.1
        AND ellipsis_ratio <= 0.1
        AND bullet_line_ratio <= 0.9
        AND ending_ellipsis_line_ratio <= 0.4
        AND alpha_word_ratio >= 0.5;

CREATE INDEX CONCURRENTLY IF NOT EXISTS document_language_detection_label_doc_idx
    ON document_language_detection (language_label, doc_id)
    INCLUDE (confidence, detected_at);
