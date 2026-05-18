"""
Translate fineweb-edu (English) → Malay with Qwen3-14B-AWQ and store the result
in the unified layer as cleaning_source='fine-web-edu-malay-translate'.

Self-contained local runner (NOT Ray/remote). It reads cleaning_source=
'fineweb-edu' docs from the corpus DB, calls a LOCAL vLLM OpenAI-compatible
server (start it with serve_qwen3_translate.sh — the proven Qwen3+vLLM+
flashinfer recipe), and writes Malay docs straight into unified_documents /
unified_document_texts / document_language_detection.

Always thinking mode. Long docs are split on paragraph boundaries to fit the
context window, each chunk translated, then concatenated. Resume-safe: it
anti-joins against already-translated docs (no MAX-watermark poisoning), and
every write is ON CONFLICT DO NOTHING.

Usage:
  # terminal A: bash src/data-storage/scripts/serve_qwen3_translate.sh
  # terminal B:
  python3 translate_fineweb_edu_malay.py --limit 5
  python3 translate_fineweb_edu_malay.py --no-llm --dry-run --limit 3   # DB path only
"""

import argparse
import asyncio
import hashlib
import logging
import sys

import asyncpg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s INFO %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

SOURCE = "fine-web-edu-malay-translate"  # == data_sources.source_name; partition key
ORIGIN_SOURCE = "fineweb-edu"
TEXT_COLUMN = "cleaned_text"
TRANSLATE_RULE = "qwen3-14b-awq-en-ms-translate"
LID_LABEL = "standard-malay"  # Malaya fasttext label 5 (translated MS corpus)
LID_CONFIDENCE = 1.0  # direct assignment, no model run

DEFAULT_DSN = "postgresql://corpus:corpus_secret@localhost:5432/corpus"
DEFAULT_MODEL = "Qwen/Qwen3-14B-AWQ"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

SYSTEM_PROMPT = (
    "You are a professional English-to-Malay translator. Translate the user's "
    "English text into natural, fluent Malay (Bahasa Melayu). Preserve meaning, "
    "structure, and paragraph breaks. Output only the Malay translation — no "
    "notes, no English, no commentary."
)


# ── helpers (─sha256 / _sample_key verbatim from ingest_fineweb_edu.py:74-81) ──


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sample_key(h: str) -> int:
    return int(h[:15], 16) % (2**62)


_TOKENIZER = None
_TOKENIZER_TRIED = False


def _get_tokenizer(model: str):
    """Qwen tokenizer for accurate chunk sizing; None → char/4 fallback."""
    global _TOKENIZER, _TOKENIZER_TRIED
    if _TOKENIZER_TRIED:
        return _TOKENIZER
    _TOKENIZER_TRIED = True
    try:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(model)
    except Exception as exc:  # offline / transformers absent → heuristic
        log.warning("tokenizer unavailable (%s); using char/4 estimate", exc)
        _TOKENIZER = None
    return _TOKENIZER


def _count_tokens(text: str, model: str) -> int:
    tok = _get_tokenizer(model)
    if tok is None:
        return max(1, len(text) // 4)
    return len(tok.encode(text))


def _split_to_budget(text: str, budget: int, model: str) -> list[str]:
    """Paragraph-first split so each chunk is <= budget tokens."""
    if _count_tokens(text, model) <= budget:
        return [text]
    units = text.split("\n\n")
    chunks: list[str] = []
    cur: list[str] = []
    for unit in units:
        candidate = "\n\n".join([*cur, unit]) if cur else unit
        if cur and _count_tokens(candidate, model) > budget:
            chunks.append("\n\n".join(cur))
            cur = [unit]
        else:
            cur = [*cur, unit]
        # a single oversize paragraph: hard-split by sentence then words
        while _count_tokens("\n\n".join(cur), model) > budget and len(cur) == 1:
            big = cur[0]
            parts = _hard_split(big, budget, model)
            chunks.extend(parts[:-1])
            cur = [parts[-1]]
            break
    if cur:
        chunks.append("\n\n".join(cur))
    return [c for c in chunks if c.strip()]


def _hard_split(text: str, budget: int, model: str) -> list[str]:
    import re

    pieces = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    cur = ""
    for p in pieces:
        cand = (cur + " " + p).strip() if cur else p
        if cur and _count_tokens(cand, model) > budget:
            out.append(cur)
            cur = p
        else:
            cur = cand
    if cur:
        out.append(cur)
    # still-too-big single sentence → split by words
    final: list[str] = []
    for seg in out:
        if _count_tokens(seg, model) <= budget:
            final.append(seg)
            continue
        words = seg.split(" ")
        buf: list[str] = []
        for w in words:
            buf.append(w)
            if _count_tokens(" ".join(buf), model) > budget:
                buf.pop()
                final.append(" ".join(buf))
                buf = [w]
        if buf:
            final.append(" ".join(buf))
    return final or [text]


def _split_think(content: str) -> tuple[str, str]:
    """Return (translation, thinking). thinking = the <think>…</think> body;
    translation = everything after the last </think> (Qwen3 thinking mode)."""
    close = "</think>"
    idx = content.rfind(close)
    if idx == -1:
        return content.strip(), ""
    translation = content[idx + len(close) :].strip()
    head = content[:idx]
    open_ = head.rfind("<think>")
    thinking = head[open_ + len("<think>") :] if open_ != -1 else head
    return translation, thinking.strip()


async def _translate_doc(client, args, english: str) -> tuple[str, str]:
    """Return (malay_text, thinking_text); chunks concatenated with a blank line.

    One in-flight doc == one worker slot; multi-chunk docs translate their chunks
    sequentially within that worker (concurrency is bounded by the worker pool).
    """
    if args.no_llm:
        return english, ""  # exercises read+chunk+write without a GPU
    budget = args.max_model_len - args.max_output_tokens - args.prompt_overhead
    chunks = _split_to_budget(english, budget, args.model)
    out: list[str] = []
    thinks: list[str] = []
    for chunk in chunks:
        resp = await client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": chunk},
            ],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_output_tokens,
            stop=[],  # thinking mode emits \n\n constantly
            extra_body={
                "top_k": args.top_k,
                "chat_template_kwargs": {"enable_thinking": True},
            },
        )
        translation, thinking = _split_think(resp.choices[0].message.content or "")
        if translation:
            out.append(translation)
        if thinking:
            thinks.append(thinking)
    return "\n\n".join(out), "\n\n".join(thinks)


# ── DB ────────────────────────────────────────────────────────────────────────

READ_SQL = """
SELECT d.doc_id, d.sample_uid, d.source_object_key, d.source_parquet_url,
       d.source_row_index, d.original_char_count,
       d.approximate_cleaned_token_count, d.cleaned_text_sha256, t.cleaned_text
FROM unified_documents d
JOIN unified_document_texts t USING (doc_id)
WHERE d.cleaning_source = $1
  AND d.cleaning_is_dropped = FALSE
  AND t.cleaned_text IS NOT NULL
  AND d.doc_id > $2
  AND NOT EXISTS (
    SELECT 1 FROM unified_documents x
    WHERE x.cleaning_source = $3
      AND x.sample_uid = $4 || split_part(d.sample_uid, '://', 2))
ORDER BY d.doc_id
LIMIT $5
"""

INSERT_DOCS_SQL = """
INSERT INTO unified_documents
    (sample_uid, sample_uid_hash, source_id, cleaning_source,
     source_bucket, source_object_key, source_parquet_url, text_column,
     source_row_group_index, source_row_index, row_index_in_row_group,
     original_text_sha256, cleaned_text_sha256,
     original_char_count, cleaned_char_count, removed_char_count,
     approximate_original_token_count, approximate_cleaned_token_count,
     approximate_removed_token_count, cleaning_is_dropped,
     cleaning_rules_triggered,
     cleaned_o200k_token_count, cleaned_o200k_tokenizer, sample_key)
SELECT
    t.sample_uid, t.sample_uid_hash, $1::SMALLINT, $2::TEXT,
    NULL, t.source_object_key, t.source_parquet_url, $3::TEXT,
    NULL, t.source_row_index, NULL,
    t.orig_sha, t.clean_sha,
    t.orig_cc, t.clean_cc, 0,
    t.approx_orig, t.approx_clean,
    0, FALSE, ARRAY[$4::TEXT]::TEXT[],
    NULL, NULL, t.sample_key
FROM unnest(
    $5::TEXT[], $6::TEXT[], $7::TEXT[], $8::TEXT[], $9::BIGINT[],
    $10::TEXT[], $11::TEXT[], $12::BIGINT[], $13::BIGINT[],
    $14::BIGINT[], $15::BIGINT[], $16::BIGINT[]
) AS t(sample_uid, sample_uid_hash, source_object_key, source_parquet_url,
       source_row_index, orig_sha, clean_sha, orig_cc, clean_cc,
       approx_orig, approx_clean, sample_key)
ON CONFLICT (sample_uid) DO NOTHING
RETURNING doc_id, sample_uid
"""


async def _write_batch(conn, source_id, rows, dry_run):
    """rows: list of dicts with origin fields + translated `malay`."""
    sample_uids, suid_hashes = [], []
    obj_keys, pq_urls, row_idxs = [], [], []
    orig_sha, clean_sha = [], []
    orig_cc, clean_cc = [], []
    approx_orig, approx_clean, skeys = [], [], []
    malay_by_uid: dict[str, str] = {}
    think_by_uid: dict[str, str] = {}

    for r in rows:
        suffix = r["sample_uid"].split("://", 1)[1]
        suid = f"{SOURCE}://{suffix}"
        h = _sha256(suid)
        malay = r["malay"]
        think_by_uid[suid] = r.get("thinking", "")
        sample_uids.append(suid)
        suid_hashes.append(h)
        obj_keys.append(r["source_object_key"])
        pq_urls.append(r["source_parquet_url"])
        row_idxs.append(r["source_row_index"])
        orig_sha.append(r["cleaned_text_sha256"])  # English source hash
        clean_sha.append(_sha256(malay))
        orig_cc.append(r["original_char_count"])
        clean_cc.append(len(malay))
        approx_orig.append(r["approximate_cleaned_token_count"])
        approx_clean.append(_count_tokens(malay, _MODEL_FOR_COUNT))
        skeys.append(_sample_key(h))
        malay_by_uid[suid] = malay

    tr = conn.transaction()
    await tr.start()
    try:
        result = await conn.fetch(
            INSERT_DOCS_SQL,
            source_id, SOURCE, TEXT_COLUMN, TRANSLATE_RULE,
            sample_uids, suid_hashes, obj_keys, pq_urls, row_idxs,
            orig_sha, clean_sha, orig_cc, clean_cc,
            approx_orig, approx_clean, skeys,
        )
        doc_id_map = {r["sample_uid"]: r["doc_id"] for r in result}
        missing = [u for u in sample_uids if u not in doc_id_map]
        if missing:
            existing = await conn.fetch(
                "SELECT doc_id, sample_uid FROM unified_documents "
                "WHERE sample_uid = ANY($1::TEXT[])",
                missing,
            )
            doc_id_map.update({r["sample_uid"]: r["doc_id"] for r in existing})

        doc_ids = [doc_id_map[u] for u in sample_uids]
        texts = [malay_by_uid[u] for u in sample_uids]
        thinks = [think_by_uid[u] for u in sample_uids]
        think_cc = [len(t) for t in thinks]
        think_tc = [_count_tokens(t, _MODEL_FOR_COUNT) if t else 0 for t in thinks]

        await conn.execute(
            """
            INSERT INTO unified_document_texts (doc_id, cleaning_source, cleaned_text)
            SELECT t.doc_id, $1::TEXT, t.cleaned_text
            FROM unnest($2::BIGINT[], $3::TEXT[]) AS t(doc_id, cleaned_text)
            ON CONFLICT (cleaning_source, doc_id) DO NOTHING
            """,
            SOURCE, doc_ids, texts,
        )
        await conn.execute(
            """
            INSERT INTO document_translation_thinking
                (doc_id, thinking_text, thinking_char_count, thinking_token_count)
            SELECT t.doc_id, t.thinking_text, t.cc, t.tc
            FROM unnest($1::BIGINT[], $2::TEXT[], $3::BIGINT[], $4::BIGINT[])
                AS t(doc_id, thinking_text, cc, tc)
            ON CONFLICT (doc_id) DO NOTHING
            """,
            doc_ids, thinks, think_cc, think_tc,
        )
        await conn.execute(
            """
            INSERT INTO document_language_detection
                (doc_id, language_label, confidence)
            SELECT doc_id, $1::TEXT, $2::REAL
            FROM unnest($3::BIGINT[]) AS t(doc_id)
            ON CONFLICT (doc_id) DO NOTHING
            """,
            LID_LABEL, LID_CONFIDENCE, doc_ids,
        )
        if dry_run:
            await tr.rollback()
        else:
            await tr.commit()
    except Exception:
        await tr.rollback()
        raise


_MODEL_FOR_COUNT = DEFAULT_MODEL  # set in main() so _write_batch can size MS tokens


async def run(args) -> None:
    global _MODEL_FOR_COUNT
    _MODEL_FOR_COUNT = args.model

    client = None
    if not args.no_llm:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    # Separate connections: producer reads, committer writes (an asyncpg
    # connection is not safe for concurrent use by multiple coroutines).
    read_conn = await asyncpg.connect(args.dsn)
    write_conn = await asyncpg.connect(args.dsn)
    try:
        srow = await read_conn.fetchrow(
            "SELECT source_id FROM data_sources WHERE source_name = $1", SOURCE
        )
        if srow is None:
            raise RuntimeError(
                f"data_sources has no '{SOURCE}' row — apply "
                "scripts/migrations/add_fine_web_edu_malay_translate.sql first"
            )
        source_id = srow["source_id"]
        prefix = f"{SOURCE}://"

        # Backpressure: producer blocks once the queue holds ~2x concurrency,
        # so we never read thousands of docs ahead of the workers.
        work_q: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 2)
        done_q: asyncio.Queue = asyncio.Queue()
        worker_fin = object()  # a worker has drained and exited
        stats = {"produced": 0, "translated": 0, "skipped": 0, "committed": 0}

        async def producer() -> None:
            last_doc_id = 0
            try:
                while True:
                    page_n = args.batch_size
                    if args.limit:
                        page_n = min(page_n, args.limit - stats["produced"])
                    if page_n <= 0:
                        break
                    page = await read_conn.fetch(
                        READ_SQL, ORIGIN_SOURCE, last_doc_id, SOURCE, prefix, page_n
                    )
                    if not page:
                        break
                    last_doc_id = page[-1]["doc_id"]
                    for r in page:
                        await work_q.put(dict(r))  # blocks when full → backpressure
                        stats["produced"] += 1
                    if args.limit and stats["produced"] >= args.limit:
                        break
            finally:
                for _ in range(args.concurrency):
                    await work_q.put(None)  # stop sentinel per worker

        async def worker() -> None:
            while True:
                item = await work_q.get()
                if item is None:
                    await done_q.put(worker_fin)
                    return
                malay, thinking = await _translate_doc(
                    client, args, item["cleaned_text"]
                )
                if malay.strip():
                    await done_q.put({**item, "malay": malay, "thinking": thinking})
                    stats["translated"] += 1
                else:
                    stats["skipped"] += 1
                    log.warning(
                        "empty translation for doc_id=%s — skipped", item["doc_id"]
                    )

        async def committer() -> None:
            buf: list[dict] = []

            async def flush() -> None:
                if not buf:
                    return
                n = len(buf)
                await _write_batch(write_conn, source_id, buf, args.dry_run)
                stats["committed"] += n
                log.info(
                    "committed %d (committed=%d translated=%d produced=%d "
                    "queued=%d)%s",
                    n, stats["committed"], stats["translated"],
                    stats["produced"], work_q.qsize(),
                    " [DRY-RUN rollback]" if args.dry_run else "",
                )
                buf.clear()

            workers_left = args.concurrency
            try:
                while workers_left > 0 or not done_q.empty():
                    try:
                        item = await asyncio.wait_for(
                            done_q.get(), timeout=args.commit_interval
                        )
                    except asyncio.TimeoutError:
                        await flush()  # bound latency: don't sit on finished docs
                        continue
                    if item is worker_fin:
                        workers_left -= 1
                        continue
                    buf.append(item)
                    if len(buf) >= args.commit_size:
                        await flush()
                await flush()
            finally:
                if buf and not args.dry_run:
                    try:
                        await flush()
                    except Exception:
                        log.exception("final flush failed; %d docs uncommitted "
                                       "(resume-safe — rerun)", len(buf))

        await asyncio.gather(producer(), committer(),
                             *[worker() for _ in range(args.concurrency)])
        log.info(
            "done — produced=%d translated=%d skipped=%d committed=%d%s",
            stats["produced"], stats["translated"], stats["skipped"],
            stats["committed"],
            " (dry-run, nothing committed)" if args.dry_run else "",
        )
    finally:
        await read_conn.close()
        await write_conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Translate fineweb-edu → Malay (Qwen3-14B-AWQ)")
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default="translate")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch-size", type=int, default=64,
                   help="DB read page size (docs fetched per anti-join query)")
    p.add_argument("--concurrency", type=int, default=8,
                   help="in-flight docs (worker-pool size); backpressure caps "
                        "the read queue at 2x this")
    p.add_argument("--commit-size", type=int, default=6,
                   help="flush a commit once this many docs have finished "
                        "(finished docs do NOT wait for slow ones)")
    p.add_argument("--commit-interval", type=float, default=30.0,
                   help="also flush any finished docs at least this often (s)")
    p.add_argument("--max-output-tokens", type=int, default=8192)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--prompt-overhead", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.6)  # Qwen3 thinking
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="max docs this run")
    p.add_argument("--dry-run", action="store_true", help="roll back all writes")
    p.add_argument("--no-llm", action="store_true",
                   help="passthrough English (exercise DB path without a GPU)")
    args = p.parse_args()

    if args.no_llm:
        log.info("NO-LLM passthrough mode (no translation performed)")
    if args.dry_run:
        log.info("DRY RUN — all writes rolled back")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
