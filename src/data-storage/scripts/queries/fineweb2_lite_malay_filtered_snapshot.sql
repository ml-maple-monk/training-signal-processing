-- FineWeb2-Lite Malay filtered current snapshot export.
--
-- This is intentionally read-only: it selects from the metadata, unified text,
-- LID, and near-dedup tables without writing to any database table.
--
-- Snapshot refreshed for the active metadata run on 2026-05-15:
--   run_id: fineweb2-lite-metadata-full-20260515T183115Z
--   snapshot_captured_at: 2026-05-15 21:40:19.631041+00:00
--   snapshot_max_doc_id: 3388000
--
-- Filtering choices:
--   * unified documents only
--   * LID labels local-malay and standard-malay only
--   * source_domain is the source label from cleaning_source
--   * near-duplicate doc_id rows are excluded via document_near_duplicates
--   * alpha_word_ratio threshold is 0.805
--   * non_symbol_word_count is constrained to 5..10000000
--   * Indonesian stopword filter is not applied
with params as (
    select
        'fineweb2-lite-metadata-full-20260515T183115Z'::text as run_id,
        '2026-05-15 21:40:19.631041+00:00'::timestamptz as snapshot_captured_at,
        3388000::bigint as snapshot_max_doc_id
)
select
    m.run_id,
    m.doc_id,
    coalesce(ud.cleaning_source, m.cleaning_source, 'unknown') as source_domain,
    dld.language_label as lid_label,
    dld.confidence as lid_confidence,
    t.cleaned_text as completed_text
from fineweb2_lite_quality_metadata m
join params p
    on p.run_id = m.run_id
join unified_documents ud
    on ud.doc_id = m.doc_id
join unified_document_texts t
    on t.doc_id = m.doc_id
join document_language_detection dld
    on dld.doc_id = m.doc_id
where m.input_source = 'unified'
    and m.doc_id <= p.snapshot_max_doc_id
    and m.created_at <= p.snapshot_captured_at
    and ud.cleaning_is_dropped = false
    and t.cleaned_text is not null
    and dld.language_label in ('local-malay', 'standard-malay')
    and dld.detected_at <= p.snapshot_captured_at
    and m.duplicate_line_fraction <= 0.241
    and m.top_2gram_fraction <= 0.194
    and m.top_3gram_fraction <= 0.166
    and m.top_4gram_fraction <= 0.145
    and m.duplicated_5gram_fraction <= 0.166
    and m.duplicated_6gram_fraction <= 0.155
    and m.duplicated_7gram_fraction <= 0.144
    and m.duplicated_8gram_fraction <= 0.133
    and m.duplicated_9gram_fraction <= 0.122
    and m.duplicated_10gram_fraction <= 0.111
    and m.line_punctuation_fraction >= 0.111
    and m.duplicated_line_character_ratio <= 0.1
    and m.newline_word_ratio <= 0.157
    and m.non_symbol_word_count between 5 and 10000000
    and m.average_non_symbol_word_length between 3 and 12
    and m.hash_symbol_ratio <= 0.1
    and m.ellipsis_ratio <= 0.1
    and m.bullet_line_ratio <= 0.9
    and m.ending_ellipsis_line_ratio <= 0.3
    and m.alpha_word_ratio >= 0.5
order by m.doc_id;
