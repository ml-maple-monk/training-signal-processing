-- FineWeb2-lite text-quality metadata output.
-- Run once before executing fineweb2-lite-run without --dry-run.
-- Requires add_sea_pile_malay.sql if SEA-PILE raw rows should be processed.

CREATE TABLE IF NOT EXISTS fineweb2_lite_quality_metadata (
    run_id                                   TEXT   NOT NULL,
    input_source                             TEXT   NOT NULL,
    source_doc_id                            BIGINT NOT NULL,
    doc_id                                   BIGINT REFERENCES unified_documents(doc_id),
    sea_pile_malay_id                        BIGINT REFERENCES sea_pile_malay_documents(id),
    cleaning_source                          TEXT   NOT NULL,
    source_is_dropped                        BOOLEAN,
    language_profile                         TEXT   NOT NULL,
    text_char_count                          BIGINT NOT NULL,
    paragraph_count                          BIGINT NOT NULL,
    line_count                               BIGINT NOT NULL,
    non_empty_line_count                     BIGINT NOT NULL,
    word_count                               BIGINT NOT NULL,
    duplicate_line_count                     BIGINT NOT NULL,
    duplicate_line_fraction                  DOUBLE PRECISION NOT NULL,
    top_2gram_fraction                       DOUBLE PRECISION NOT NULL,
    top_3gram_fraction                       DOUBLE PRECISION NOT NULL,
    top_4gram_fraction                       DOUBLE PRECISION NOT NULL,
    duplicated_5gram_fraction                DOUBLE PRECISION NOT NULL,
    duplicated_6gram_fraction                DOUBLE PRECISION NOT NULL,
    duplicated_7gram_fraction                DOUBLE PRECISION NOT NULL,
    duplicated_8gram_fraction                DOUBLE PRECISION NOT NULL,
    duplicated_9gram_fraction                DOUBLE PRECISION NOT NULL,
    duplicated_10gram_fraction               DOUBLE PRECISION NOT NULL,
    line_punctuation_fraction                DOUBLE PRECISION NOT NULL,
    duplicated_line_character_ratio          DOUBLE PRECISION NOT NULL,
    newline_count                            BIGINT NOT NULL,
    newline_word_ratio                       DOUBLE PRECISION NOT NULL,
    non_symbol_word_count                    BIGINT NOT NULL,
    average_non_symbol_word_length           DOUBLE PRECISION NOT NULL,
    hash_symbol_ratio                        DOUBLE PRECISION NOT NULL,
    ellipsis_ratio                           DOUBLE PRECISION NOT NULL,
    bullet_line_ratio                        DOUBLE PRECISION NOT NULL,
    ending_ellipsis_line_ratio               DOUBLE PRECISION NOT NULL,
    alpha_word_ratio                         DOUBLE PRECISION NOT NULL,
    stopword_count                           BIGINT NOT NULL,
    indonesian_stopword_count                BIGINT NOT NULL,
    at_least_2_profile_stopwords_present     BOOLEAN NOT NULL,
    at_least_2_indonesian_stopwords_present  BOOLEAN NOT NULL,
    metrics                                  JSONB NOT NULL,
    created_at                               TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (run_id, input_source, source_doc_id),
    CONSTRAINT fineweb2_lite_quality_metadata_source_ref_check CHECK (
        (input_source = 'unified'
         AND doc_id = source_doc_id
         AND sea_pile_malay_id IS NULL)
        OR
        (input_source = 'sea-pile-malay'
         AND sea_pile_malay_id = source_doc_id
         AND doc_id IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS fineweb2_lite_quality_metadata_profile_idx
    ON fineweb2_lite_quality_metadata (run_id, language_profile, input_source, source_doc_id);

CREATE INDEX IF NOT EXISTS fineweb2_lite_quality_metadata_source_idx
    ON fineweb2_lite_quality_metadata (input_source, source_doc_id);

ALTER TABLE fineweb2_lite_quality_metadata
    DROP CONSTRAINT IF EXISTS fineweb2_lite_quality_metadata_run_id_fkey;
