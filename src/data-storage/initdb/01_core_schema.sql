-- Core registry tables: data sources and pipeline run tracking.

CREATE TABLE data_sources (
    source_id   SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_name TEXT NOT NULL UNIQUE,   -- 'lowyat', 'cari', 'reddit-bolehland', etc.
    source_type TEXT NOT NULL,          -- 'forum', 'reddit', 'web', 'books'
    description TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

INSERT INTO data_sources (source_name, source_type, description) VALUES
    ('lowyat',           'forum',  'Lowyat Malay forum threads and posts'),
    ('cari',             'forum',  'Cari Malay forum threads and posts'),
    ('reddit-bolehland', 'reddit', 'Reddit r/Bolehland submissions and comments'),
    ('reddit-indonesia', 'reddit', 'Reddit r/indonesia submissions and comments'),
    ('hplt-malay',       'web',    'HPLT multilingual web corpus, Malay language'),
    ('hplt-indonesia',   'web',    'HPLT multilingual web corpus, Indonesian language'),
    ('books-ocr',        'books',  'PDF books converted to markdown via Marker OCR'),
    ('fineweb',          'web',    'HuggingFace FineWeb multilingual web corpus');

-- Tracks each pipeline ingest run for provenance.
CREATE TABLE pipeline_runs (
    run_id           TEXT PRIMARY KEY,                -- e.g. '20260429T214659Z'
    pipeline_name    TEXT NOT NULL,                   -- 'unified_data', 'ocr', 'lid_metadata', etc.
    source_id        SMALLINT REFERENCES data_sources(source_id),
    status           TEXT NOT NULL DEFAULT 'running', -- 'running', 'completed', 'failed'
    r2_output_prefix TEXT,
    row_count        BIGINT,
    token_count      BIGINT,
    byte_count       BIGINT,
    started_at       TIMESTAMPTZ,
    completed_at     TIMESTAMPTZ,
    created_at       TIMESTAMPTZ DEFAULT now()
);
