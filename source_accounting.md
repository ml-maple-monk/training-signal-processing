# Source Accounting

Counts were produced with the `tiktoken` tokenizer using the `o200k_base`
encoding. Byte counts are UTF-8 text bytes, not compressed parquet object
sizes.

| source | token_count | word_count | byte_count | document_count | r2_relative_glob_path | filters | metadata_columns |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Books + OCR | 253,526,396 | 135,195,337 | 1,072,190,417 | 8,914 | `dataset/processed/pdf_ocr/20260423T195035Z/markdown.parquet` | `` | `document_id, source_run_id, source_format, markdown_file_name, markdown_rel_path, markdown_sha256, markdown_char_count, markdown_byte_count` |
| Lowyat | 722,045,872 | 473,281,077 | 2,813,489,956 | 13,126,644 | `dataset/processed/malay/lowyat.parquet` | `` | `source, thread_id, thread_title, thread_url, forum, thread_total_pages, thread_status, page_number, page_offset, post_id, post_floor, author, author_id, posted_at, body_html, quoted_post_id, fetched_at, error_reason` |
| Reddit Bolehland | 7,200,698 | 5,143,027 | 29,967,055 | 215,284 | `dataset/processed/malay/reddit.parquet` | `subreddit=Bolehland` | `post_kind, post_id, submission_id, parent_id, subreddit, author, title, score, num_comments, created_utc, permalink, url, month` |
| Reddit Indonesia | 11,480,646 | 6,616,621 | 43,250,701 | 286,429 | `dataset/processed/malay/reddit.parquet` | `subreddit=indonesia` | `post_kind, post_id, submission_id, parent_id, subreddit, author, title, score, num_comments, created_utc, permalink, url, month` |
| Cari | 134,964,927 | 69,915,523 | 477,486,406 | 1,329,684 | `dataset/processed/malay/cari.parquet` | `` | `source, thread_id, thread_title, thread_url, forum, thread_total_pages, thread_status, page_number, page_offset, post_id, post_floor, author, author_id, posted_at, body_html, quoted_post_id, fetched_at, error_reason` |
| HPLT Malay | 18,639,630,667 | 10,270,837,791 | 71,548,965,682 | 17,365,290 | `dataset/processed/malay/hplt/*_malay.parquet` | `` | `id, url, timestamp, crawl_id, source_shard, language, row_language_code, row_language_prob` |
| HPLT Indonesia | 34,205,177,094 | 19,460,844,974 | 137,277,262,359 | 46,634,888 | `dataset/processed/malay/hplt/*_indon.parquet` | `` | `id, url, timestamp, crawl_id, source_shard, language, row_language_code, row_language_prob` |
