-- FineWeb2-Lite retained document index.
--
-- This is the editable source of truth for retained-document filtering.
-- It intentionally does not select cleaned text; parquet export joins text later.
--
-- Parameters:
--   $1::text = profile_name
--   $2::text = run_id

with params as (
    select
        $1::text as profile_name,
        $2::text as run_id
)
select
    p.profile_name,
    m.run_id,
    m.doc_id,
    m.source_doc_id,
    coalesce(ud.cleaning_source, m.cleaning_source, 'unknown') as source_domain,
    dld.language_label as lid_label,
    dld.confidence as lid_confidence,
    m.created_at as metadata_created_at,
    dld.detected_at as lid_detected_at
from fineweb2_lite_quality_metadata m
join params p
    on p.run_id = m.run_id
join unified_documents ud
    on ud.doc_id = m.doc_id
join document_language_detection dld
    on dld.doc_id = m.doc_id
where m.input_source = 'unified'
    and ud.cleaning_is_dropped = false
    and not exists (
        select 1
        from document_near_duplicates nd
        where nd.doc_id = m.doc_id
    )
    and dld.language_label in (
        'local-malay',
        'standard-malay',
        'socialmedia-indonesian',
        'standard-english',
        'standard-indonesian',
        'manglish',
        'local-english',
        'other',
        'local-mandarin',
        'standard-mandarin'
    )
    and m.duplicate_line_fraction <= 0.241
    and m.top_2gram_fraction <= 0.194
    and m.top_3gram_fraction <= 0.2
    and m.top_4gram_fraction <= 0.145
    and m.duplicated_5gram_fraction <= 0.166
    and m.duplicated_6gram_fraction <= 0.155
    and m.duplicated_7gram_fraction <= 0.144
    and m.duplicated_8gram_fraction <= 0.133
    and m.duplicated_9gram_fraction <= 0.122
    and m.duplicated_10gram_fraction <= 0.111
    and m.line_punctuation_fraction >= 0.08
    and m.duplicated_line_character_ratio <= 0.1
    and m.newline_word_ratio <= 0.157
    and m.non_symbol_word_count between 5 and 10000000
    and m.average_non_symbol_word_length between 3 and 12
    and m.hash_symbol_ratio <= 0.1
    and m.ellipsis_ratio <= 0.1
    and m.bullet_line_ratio <= 0.9
    and m.ending_ellipsis_line_ratio <= 0.4
    and m.alpha_word_ratio >= 0.5;
