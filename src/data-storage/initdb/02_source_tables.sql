-- Per-source permanent tables. Each maps to one cleaning_source value.
-- Text columns use COMPRESSION lz4 (stored in TOAST above ~2KB threshold).

-- ── LOWYAT ───────────────────────────────────────────────────────────────────

CREATE TABLE lowyat_threads (
    thread_id          TEXT PRIMARY KEY,
    thread_title       TEXT,
    thread_url         TEXT,
    forum              TEXT,
    thread_total_pages INTEGER,
    thread_status      TEXT,
    fetched_at         TIMESTAMPTZ,
    created_at         TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE lowyat_posts (
    post_id        TEXT PRIMARY KEY,
    thread_id      TEXT NOT NULL REFERENCES lowyat_threads(thread_id),
    post_floor     INTEGER,
    page_number    INTEGER,
    page_offset    INTEGER,
    author         TEXT,
    author_id      TEXT,
    posted_at      TIMESTAMPTZ,
    body_text      TEXT COMPRESSION lz4,
    body_html      TEXT COMPRESSION lz4,
    quoted_post_id TEXT,
    fetched_at     TIMESTAMPTZ,
    error_reason   TEXT,
    created_at     TIMESTAMPTZ DEFAULT now()
);

-- ── CARI ─────────────────────────────────────────────────────────────────────

CREATE TABLE cari_threads (
    thread_id          TEXT PRIMARY KEY,
    thread_title       TEXT,
    thread_url         TEXT,
    forum              TEXT,
    thread_total_pages INTEGER,
    thread_status      TEXT,
    fetched_at         TIMESTAMPTZ,
    created_at         TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE cari_posts (
    post_id        TEXT PRIMARY KEY,
    thread_id      TEXT NOT NULL REFERENCES cari_threads(thread_id),
    post_floor     INTEGER,
    page_number    INTEGER,
    page_offset    INTEGER,
    author         TEXT,
    author_id      TEXT,
    posted_at      TIMESTAMPTZ,
    body_text      TEXT COMPRESSION lz4,
    body_html      TEXT COMPRESSION lz4,
    quoted_post_id TEXT,
    fetched_at     TIMESTAMPTZ,
    error_reason   TEXT,
    created_at     TIMESTAMPTZ DEFAULT now()
);

-- ── REDDIT ───────────────────────────────────────────────────────────────────
-- Flat table with post_kind discriminator ('submission' | 'comment').
-- parent_id uses Reddit fullname format: t3_xxx = submission, t1_xxx = comment.
-- submission_id links comments back to their root submission.

CREATE TABLE reddit_bolehland_posts (
    post_id       TEXT PRIMARY KEY,
    post_kind     TEXT NOT NULL CHECK (post_kind IN ('submission', 'comment')),
    submission_id TEXT,        -- NULL for top-level submissions
    parent_id     TEXT,        -- Reddit fullname; NULL for top-level submissions
    subreddit     TEXT NOT NULL DEFAULT 'Bolehland',
    author        TEXT,
    title         TEXT,        -- NULL for comments
    score         INTEGER,
    num_comments  INTEGER,     -- NULL for comments
    created_utc   TIMESTAMPTZ,
    permalink     TEXT,
    url           TEXT,
    month         TEXT,        -- partition bucket: e.g. '2024-01'
    body          TEXT COMPRESSION lz4,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE reddit_indonesia_posts (
    post_id       TEXT PRIMARY KEY,
    post_kind     TEXT NOT NULL CHECK (post_kind IN ('submission', 'comment')),
    submission_id TEXT,
    parent_id     TEXT,
    subreddit     TEXT NOT NULL DEFAULT 'indonesia',
    author        TEXT,
    title         TEXT,
    score         INTEGER,
    num_comments  INTEGER,
    created_utc   TIMESTAMPTZ,
    permalink     TEXT,
    url           TEXT,
    month         TEXT,
    body          TEXT COMPRESSION lz4,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- ── HPLT ─────────────────────────────────────────────────────────────────────

CREATE TABLE hplt_malay_documents (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    hplt_id           TEXT,        -- 'id' field in source parquet
    url               TEXT,
    crawl_timestamp   TIMESTAMPTZ, -- 'timestamp' field in source parquet
    crawl_id          TEXT,
    source_shard      TEXT,
    language          TEXT,
    row_language_code TEXT,
    row_language_prob REAL,
    content           TEXT COMPRESSION lz4,  -- 'text' field in source parquet
    created_at        TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE hplt_indonesia_documents (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    hplt_id           TEXT,
    url               TEXT,
    crawl_timestamp   TIMESTAMPTZ,
    crawl_id          TEXT,
    source_shard      TEXT,
    language          TEXT,
    row_language_code TEXT,
    row_language_prob REAL,
    content           TEXT COMPRESSION lz4,
    created_at        TIMESTAMPTZ DEFAULT now()
);

-- ── OCR BOOKS ────────────────────────────────────────────────────────────────
-- Small corpus (~8,786 documents, ~900MB text) — single table, text stored inline.

CREATE TABLE ocr_books (
    document_id         TEXT PRIMARY KEY,  -- from source parquet
    source_run_id       TEXT,
    source_format       TEXT DEFAULT 'markdown',
    markdown_file_name  TEXT,
    markdown_rel_path   TEXT,
    markdown_sha256     TEXT,
    markdown_char_count INTEGER,
    markdown_byte_count INTEGER,
    markdown_text       TEXT COMPRESSION lz4,
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ── FINEWEB ──────────────────────────────────────────────────────────────────
-- In-progress source; schema covers known fields from HuggingFace dataset.

CREATE TABLE fineweb_documents (
    id         BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    hf_id      TEXT,         -- 'id' from HuggingFace dataset
    url        TEXT,
    date       TIMESTAMPTZ,
    month      TEXT,         -- partition bucket: e.g. '2024-01'
    language   TEXT,
    content    TEXT COMPRESSION lz4,
    created_at TIMESTAMPTZ DEFAULT now()
);
