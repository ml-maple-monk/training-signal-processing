from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import psycopg2
import psycopg2.extras

from .config import RecipeConfig, load_recipe_config
from .datatrove_lite import compute_quality_metrics

PIPELINE_NAME = "fineweb_2_lite_quality_metadata"
OUTPUT_TABLE = "fineweb2_lite_quality_metadata"
PROGRESS_LOG_EVERY_ROWS = 25_000


@dataclass(frozen=True)
class SourceDocument:
    input_source: str
    source_doc_id: int
    doc_id: int | None
    sea_pile_malay_id: int | None
    cleaning_source: str
    text: str
    source_is_dropped: bool | None = None


@dataclass(frozen=True)
class QualityMetadataRow:
    input_source: str
    source_doc_id: int
    doc_id: int | None
    sea_pile_malay_id: int | None
    cleaning_source: str
    language_profile: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class SourceDocRange:
    input_source: str
    start_after: int
    stop_at: int
    worker_index: int


@click.command("fineweb2-lite-validate")
@click.option("--config", "config_path", required=True, type=click.Path(path_type=Path))
def fineweb2_lite_validate_cli(config_path: Path) -> None:
    try:
        config = load_recipe_config(config_path)
        click.echo(f"Validated FineWeb2-lite metadata recipe: {config_path}")
        click.echo(f"Run name: {config.run_name}")
        click.echo(f"DB env var: {config.database.dsn_env_var}")
        click.echo(f"Inputs: {enabled_input_label(config)}")
        click.echo(f"Language profiles: {', '.join(sorted(config.language_profiles))}")
        click.echo(f"Workers: {config.processing.worker_count}")
        click.echo("Mode: metadata-only, no LID filter, no dedup, no data removal")
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@click.command("fineweb2-lite-run")
@click.option("--config", "config_path", required=True, type=click.Path(path_type=Path))
@click.option("--run-id", default="", help="Pipeline run id. Defaults to a UTC timestamp.")
@click.option(
    "--full",
    is_flag=True,
    default=False,
    help="Delete existing metadata for this run id.",
)
@click.option("--dry-run", is_flag=True, default=False, help="Read and compute only; do not write.")
@click.option("--limit", type=int, default=None, help="Maximum source documents to scan.")
def fineweb2_lite_run_cli(
    config_path: Path,
    run_id: str,
    full: bool,
    dry_run: bool,
    limit: int | None,
) -> None:
    try:
        config = load_recipe_config(config_path)
        result = run_fineweb2_lite(
            config=config,
            run_id=run_id.strip() or build_run_id(),
            full=full,
            dry_run=dry_run,
            limit=limit,
        )
        click.echo(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


def build_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def run_fineweb2_lite(
    *,
    config: RecipeConfig,
    run_id: str,
    full: bool,
    dry_run: bool,
    limit: int | None,
) -> dict[str, Any]:
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive when provided.")

    dsn = config.database.resolve_dsn()
    with psycopg2.connect(dsn) as conn:
        if dry_run:
            effective_limit = limit or 100
            rows = list(
                compute_metadata_rows(
                    iter_source_documents(conn, config=config, limit=effective_limit),
                    config=config,
                )
            )
            summary = summarize_metadata_rows(rows)
            summary.update(
                {
                    "run_id": run_id,
                    "dry_run": True,
                    "limit": effective_limit,
                    "would_write_table": OUTPUT_TABLE,
                }
            )
            return summary

        ensure_output_schema(conn)
        if full:
            clear_run_outputs(conn, run_id=run_id)

        worker_count = config.processing.worker_count
        if limit is None and worker_count > 1:
            return run_fineweb2_lite_parallel(
                dsn=dsn,
                config=config,
                run_id=run_id,
                full=full,
                worker_count=worker_count,
            )

        summary = compute_and_write_metadata(
            conn,
            config=config,
            run_id=run_id,
            limit=limit,
            source_range=None,
        )
        return {
            "run_id": run_id,
            "pipeline_name": PIPELINE_NAME,
            "dry_run": False,
            "full": full,
            "limit": limit,
            "worker_count": 1,
            "metadata_summary": summary,
        }


def ensure_output_schema(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.fineweb2_lite_quality_metadata')")
        output_table = cur.fetchone()[0]
    if output_table is None:
        raise ValueError(
            "FineWeb2-lite metadata table is missing. Apply "
            "src/data-storage/scripts/migrations/add_fineweb2_lite_documents.sql first."
        )


def clear_run_outputs(conn: psycopg2.extensions.connection, *, run_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute(f"DELETE FROM {OUTPUT_TABLE} WHERE run_id = %s", (run_id,))
    conn.commit()


def run_fineweb2_lite_parallel(
    *,
    dsn: str,
    config: RecipeConfig,
    run_id: str,
    full: bool,
    worker_count: int,
) -> dict[str, Any]:
    with psycopg2.connect(dsn) as conn:
        partitions = build_source_doc_ranges(
            conn,
            config=config,
            run_id=run_id,
            worker_count=worker_count,
        )

    print(
        json.dumps(
            {
                "event": "fineweb2_lite_parallel_start",
                "run_id": run_id,
                "worker_count": worker_count,
                "partition_count": len(partitions),
                "partitions": [partition_summary(partition) for partition in partitions],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    counter: Counter[str] = Counter()
    if partitions:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    compute_and_write_metadata_partition,
                    dsn,
                    config,
                    run_id,
                    partition,
                )
                for partition in partitions
            ]
            for future in as_completed(futures):
                merge_metadata_summary(counter, future.result())

    return {
        "run_id": run_id,
        "pipeline_name": PIPELINE_NAME,
        "dry_run": False,
        "full": full,
        "limit": None,
        "worker_count": worker_count,
        "partition_count": len(partitions),
        "metadata_summary": summarize_counter(counter),
    }


def compute_and_write_metadata_partition(
    dsn: str,
    config: RecipeConfig,
    run_id: str,
    source_range: SourceDocRange,
) -> dict[str, Any]:
    with psycopg2.connect(dsn) as conn:
        return compute_and_write_metadata(
            conn,
            config=config,
            run_id=run_id,
            limit=None,
            source_range=source_range,
        )


def build_source_doc_ranges(
    conn: psycopg2.extensions.connection,
    *,
    config: RecipeConfig,
    run_id: str,
    worker_count: int,
) -> list[SourceDocRange]:
    ranges: list[SourceDocRange] = []
    if config.input.include_unified:
        ranges.extend(
            split_source_doc_range(
                input_source="unified",
                start_after=get_resume_after(
                    conn,
                    run_id=run_id,
                    input_source="unified",
                ),
                stop_at=get_source_max_doc_id(conn, input_source="unified"),
                worker_count=worker_count,
            )
        )
    if config.input.include_sea_pile_malay and table_exists(conn, "sea_pile_malay_documents"):
        ranges.extend(
            split_source_doc_range(
                input_source="sea-pile-malay",
                start_after=get_resume_after(
                    conn,
                    run_id=run_id,
                    input_source="sea-pile-malay",
                ),
                stop_at=get_source_max_doc_id(
                    conn,
                    input_source="sea-pile-malay",
                ),
                worker_count=worker_count,
            )
        )
    return ranges


def split_source_doc_range(
    *,
    input_source: str,
    start_after: int,
    stop_at: int,
    worker_count: int,
) -> list[SourceDocRange]:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")
    if stop_at <= start_after:
        return []

    total_ids = stop_at - start_after
    base_width, remainder = divmod(total_ids, worker_count)
    ranges: list[SourceDocRange] = []
    lower = start_after
    for worker_index in range(worker_count):
        width = base_width + (1 if worker_index < remainder else 0)
        if width <= 0:
            continue
        upper = lower + width
        ranges.append(
            SourceDocRange(
                input_source=input_source,
                start_after=lower,
                stop_at=upper,
                worker_index=worker_index,
            )
        )
        lower = upper
    return ranges


def get_source_max_doc_id(
    conn: psycopg2.extensions.connection,
    *,
    input_source: str,
) -> int:
    with conn.cursor() as cur:
        if input_source == "unified":
            cur.execute(
                """
                SELECT COALESCE(max(d.doc_id), 0)
                FROM unified_documents d
                """
            )
        elif input_source == "sea-pile-malay":
            cur.execute(
                """
                SELECT COALESCE(max(s.id), 0)
                FROM sea_pile_malay_documents s
                """
            )
        else:
            raise ValueError(f"Unknown input source: {input_source}")
        return int(cur.fetchone()[0])


def partition_summary(source_range: SourceDocRange) -> dict[str, Any]:
    return {
        "input_source": source_range.input_source,
        "start_after": source_range.start_after,
        "stop_at": source_range.stop_at,
        "worker_index": source_range.worker_index,
    }


def merge_metadata_summary(counter: Counter[str], summary: dict[str, Any]) -> None:
    counter["processed_row_count"] += int(summary["processed_row_count"])
    for source, count in summary["source_counts"].items():
        counter[f"source_{source}"] += int(count)
    for profile, count in summary["profile_counts"].items():
        counter[f"profile_{profile}"] += int(count)


def iter_source_documents(
    conn: psycopg2.extensions.connection,
    *,
    config: RecipeConfig,
    limit: int | None,
    run_id: str | None = None,
    source_range: SourceDocRange | None = None,
) -> Iterator[SourceDocument]:
    yielded = 0
    if config.input.include_unified and source_matches(source_range, "unified"):
        for doc in iter_unified_documents(
            conn,
            config=config,
            limit=remaining_limit(limit, yielded),
            start_after=get_resume_after(
                conn,
                run_id=run_id,
                input_source="unified",
                source_range=source_range,
            ),
            stop_at=source_range.stop_at if source_range is not None else None,
        ):
            yielded += 1
            yield doc
            if limit is not None and yielded >= limit:
                return

    if (
        config.input.include_sea_pile_malay
        and source_matches(source_range, "sea-pile-malay")
        and table_exists(conn, "sea_pile_malay_documents")
    ):
        for doc in iter_sea_pile_malay_documents(
            conn,
            config=config,
            limit=remaining_limit(limit, yielded),
            start_after=get_resume_after(
                conn,
                run_id=run_id,
                input_source="sea-pile-malay",
                source_range=source_range,
            ),
            stop_at=source_range.stop_at if source_range is not None else None,
        ):
            yielded += 1
            yield doc
            if limit is not None and yielded >= limit:
                return


def remaining_limit(limit: int | None, yielded: int) -> int | None:
    if limit is None:
        return None
    return max(limit - yielded, 0)


def source_matches(source_range: SourceDocRange | None, input_source: str) -> bool:
    return source_range is None or source_range.input_source == input_source


def get_resume_after(
    conn: psycopg2.extensions.connection,
    *,
    run_id: str | None,
    input_source: str,
    source_range: SourceDocRange | None = None,
) -> int:
    if run_id is None:
        return source_range.start_after if source_range is not None else 0
    with conn.cursor() as cur:
        range_clause = ""
        params: list[Any] = [run_id, input_source]
        if source_range is not None:
            range_clause = "AND source_doc_id > %s AND source_doc_id <= %s"
            params.extend([source_range.start_after, source_range.stop_at])
        cur.execute(
            f"""
            SELECT COALESCE(max(source_doc_id), 0)
            FROM {OUTPUT_TABLE}
            WHERE run_id = %s
              AND input_source = %s
              {range_clause}
            """,
            params,
        )
        resume_after = int(cur.fetchone()[0])
    if source_range is not None:
        return max(source_range.start_after, resume_after)
    return resume_after


def iter_unified_documents(
    conn: psycopg2.extensions.connection,
    *,
    config: RecipeConfig,
    limit: int | None,
    start_after: int = 0,
    stop_at: int | None = None,
) -> Iterator[SourceDocument]:
    yielded = 0
    last_doc_id = start_after
    while True:
        if limit is not None and yielded >= limit:
            return
        batch_limit = config.input.batch_size
        if limit is not None:
            batch_limit = min(batch_limit, limit - yielded)

        with conn.cursor() as cur:
            stop_clause = ""
            params: list[Any] = [last_doc_id, config.input.min_text_length]
            if stop_at is not None:
                stop_clause = "AND d.doc_id <= %s"
                params.append(stop_at)
            params.append(batch_limit)
            cur.execute(
                f"""
                SELECT d.doc_id,
                       d.cleaning_source,
                       COALESCE(t.cleaned_text, ''),
                       d.cleaning_is_dropped
                FROM unified_documents d
                LEFT JOIN unified_document_texts t USING (doc_id)
                WHERE d.doc_id > %s
                  AND char_length(COALESCE(t.cleaned_text, '')) >= %s
                  {stop_clause}
                ORDER BY d.doc_id
                LIMIT %s
                """,
                params,
            )
            rows = cur.fetchall()

        if not rows:
            return
        for row in rows:
            last_doc_id = int(row[0])
            yielded += 1
            yield SourceDocument(
                input_source="unified",
                source_doc_id=last_doc_id,
                doc_id=last_doc_id,
                sea_pile_malay_id=None,
                cleaning_source=str(row[1] or "unknown"),
                text=str(row[2]),
                source_is_dropped=bool(row[3]),
            )


def iter_sea_pile_malay_documents(
    conn: psycopg2.extensions.connection,
    *,
    config: RecipeConfig,
    limit: int | None,
    start_after: int = 0,
    stop_at: int | None = None,
) -> Iterator[SourceDocument]:
    yielded = 0
    last_id = start_after
    while True:
        if limit is not None and yielded >= limit:
            return
        batch_limit = config.input.batch_size
        if limit is not None:
            batch_limit = min(batch_limit, limit - yielded)

        with conn.cursor() as cur:
            stop_clause = ""
            params: list[Any] = [last_id, config.input.min_text_length]
            if stop_at is not None:
                stop_clause = "AND s.id <= %s"
                params.append(stop_at)
            params.append(batch_limit)
            cur.execute(
                f"""
                SELECT s.id, COALESCE(s.content, '')
                FROM sea_pile_malay_documents s
                WHERE s.id > %s
                  AND char_length(COALESCE(s.content, '')) >= %s
                  {stop_clause}
                ORDER BY s.id
                LIMIT %s
                """,
                params,
            )
            rows = cur.fetchall()

        if not rows:
            return
        for row in rows:
            last_id = int(row[0])
            yielded += 1
            yield SourceDocument(
                input_source="sea-pile-malay",
                source_doc_id=last_id,
                doc_id=None,
                sea_pile_malay_id=last_id,
                cleaning_source="sea-pile-malay",
                text=str(row[1]),
            )


def compute_metadata_rows(
    docs: Iterator[SourceDocument],
    *,
    config: RecipeConfig,
) -> Iterator[QualityMetadataRow]:
    indonesian_profile = config.language_profiles.get("ind_Latn")
    indonesian_stopwords = indonesian_profile.stopwords if indonesian_profile else None
    for doc in docs:
        profile_name = resolve_language_profile(doc, config)
        profile = config.language_profiles[profile_name]
        metrics = compute_quality_metrics(
            doc.text,
            profile,
            indonesian_stopwords=indonesian_stopwords,
        )
        metrics.update(
            {
                "input_source": doc.input_source,
                "source_doc_id": doc.source_doc_id,
                "cleaning_source": doc.cleaning_source,
                "source_is_dropped": doc.source_is_dropped,
            }
        )
        yield QualityMetadataRow(
            input_source=doc.input_source,
            source_doc_id=doc.source_doc_id,
            doc_id=doc.doc_id,
            sea_pile_malay_id=doc.sea_pile_malay_id,
            cleaning_source=doc.cleaning_source,
            language_profile=profile_name,
            metrics=metrics,
        )


def compute_and_write_metadata(
    conn: psycopg2.extensions.connection,
    *,
    config: RecipeConfig,
    run_id: str,
    limit: int | None,
    source_range: SourceDocRange | None,
) -> dict[str, Any]:
    pending: list[QualityMetadataRow] = []
    counter: Counter[str] = Counter()
    started_at = time.monotonic()
    last_progress_row_count = 0

    print(
        json.dumps(
            {
                "event": "fineweb2_lite_progress_start",
                "run_id": run_id,
                "output_table": OUTPUT_TABLE,
                "source_range": (
                    partition_summary(source_range) if source_range is not None else None
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for row in compute_metadata_rows(
        iter_source_documents(
            conn,
            config=config,
            limit=limit,
            run_id=run_id,
            source_range=source_range,
        ),
        config=config,
    ):
        counter["processed_row_count"] += 1
        counter[f"source_{row.input_source}"] += 1
        counter[f"profile_{row.language_profile}"] += 1
        pending.append(row)
        if len(pending) >= config.processing.write_batch_size:
            flush_metadata_rows(conn, run_id=run_id, rows=pending)
            pending.clear()
            processed_row_count = int(counter["processed_row_count"])
            if processed_row_count - last_progress_row_count >= PROGRESS_LOG_EVERY_ROWS:
                emit_progress(run_id, row, processed_row_count, started_at, source_range)
                last_progress_row_count = processed_row_count
    if pending:
        flush_metadata_rows(conn, run_id=run_id, rows=pending)
        emit_progress(
            run_id,
            pending[-1],
            int(counter["processed_row_count"]),
            started_at,
            source_range,
        )

    return summarize_counter(counter)


def emit_progress(
    run_id: str,
    row: QualityMetadataRow,
    processed_row_count: int,
    started_at: float,
    source_range: SourceDocRange | None,
) -> None:
    elapsed_seconds = max(time.monotonic() - started_at, 0.001)
    print(
        json.dumps(
            {
                "event": "fineweb2_lite_progress",
                "run_id": run_id,
                "processed_rows_this_process": processed_row_count,
                "last_input_source": row.input_source,
                "last_source_doc_id": row.source_doc_id,
                "rows_per_second_this_process": round(
                    processed_row_count / elapsed_seconds,
                    2,
                ),
                "source_range": (
                    partition_summary(source_range) if source_range is not None else None
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def flush_metadata_rows(
    conn: psycopg2.extensions.connection,
    *,
    run_id: str,
    rows: list[QualityMetadataRow],
) -> None:
    values = [metadata_insert_values(run_id, row) for row in rows]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            f"""
            INSERT INTO {OUTPUT_TABLE}
                (run_id, input_source, source_doc_id, doc_id, sea_pile_malay_id,
                 cleaning_source, source_is_dropped, language_profile, text_char_count,
                 paragraph_count, line_count, non_empty_line_count, word_count,
                 duplicate_line_count, duplicate_line_fraction, top_2gram_fraction,
                 top_3gram_fraction, top_4gram_fraction, duplicated_5gram_fraction,
                 duplicated_6gram_fraction, duplicated_7gram_fraction,
                 duplicated_8gram_fraction, duplicated_9gram_fraction,
                 duplicated_10gram_fraction, line_punctuation_fraction,
                 duplicated_line_character_ratio, newline_count, newline_word_ratio,
                 non_symbol_word_count, average_non_symbol_word_length,
                 hash_symbol_ratio, ellipsis_ratio, bullet_line_ratio,
                 ending_ellipsis_line_ratio, alpha_word_ratio, stopword_count,
                 indonesian_stopword_count, at_least_2_profile_stopwords_present,
                 at_least_2_indonesian_stopwords_present, metrics)
            VALUES %s
            ON CONFLICT (run_id, input_source, source_doc_id) DO UPDATE
                SET cleaning_source = EXCLUDED.cleaning_source,
                    source_is_dropped = EXCLUDED.source_is_dropped,
                    language_profile = EXCLUDED.language_profile,
                    text_char_count = EXCLUDED.text_char_count,
                    paragraph_count = EXCLUDED.paragraph_count,
                    line_count = EXCLUDED.line_count,
                    non_empty_line_count = EXCLUDED.non_empty_line_count,
                    word_count = EXCLUDED.word_count,
                    duplicate_line_count = EXCLUDED.duplicate_line_count,
                    duplicate_line_fraction = EXCLUDED.duplicate_line_fraction,
                    top_2gram_fraction = EXCLUDED.top_2gram_fraction,
                    top_3gram_fraction = EXCLUDED.top_3gram_fraction,
                    top_4gram_fraction = EXCLUDED.top_4gram_fraction,
                    duplicated_5gram_fraction = EXCLUDED.duplicated_5gram_fraction,
                    duplicated_6gram_fraction = EXCLUDED.duplicated_6gram_fraction,
                    duplicated_7gram_fraction = EXCLUDED.duplicated_7gram_fraction,
                    duplicated_8gram_fraction = EXCLUDED.duplicated_8gram_fraction,
                    duplicated_9gram_fraction = EXCLUDED.duplicated_9gram_fraction,
                    duplicated_10gram_fraction = EXCLUDED.duplicated_10gram_fraction,
                    line_punctuation_fraction = EXCLUDED.line_punctuation_fraction,
                    duplicated_line_character_ratio =
                        EXCLUDED.duplicated_line_character_ratio,
                    newline_count = EXCLUDED.newline_count,
                    newline_word_ratio = EXCLUDED.newline_word_ratio,
                    non_symbol_word_count = EXCLUDED.non_symbol_word_count,
                    average_non_symbol_word_length =
                        EXCLUDED.average_non_symbol_word_length,
                    hash_symbol_ratio = EXCLUDED.hash_symbol_ratio,
                    ellipsis_ratio = EXCLUDED.ellipsis_ratio,
                    bullet_line_ratio = EXCLUDED.bullet_line_ratio,
                    ending_ellipsis_line_ratio = EXCLUDED.ending_ellipsis_line_ratio,
                    alpha_word_ratio = EXCLUDED.alpha_word_ratio,
                    stopword_count = EXCLUDED.stopword_count,
                    indonesian_stopword_count = EXCLUDED.indonesian_stopword_count,
                    at_least_2_profile_stopwords_present =
                        EXCLUDED.at_least_2_profile_stopwords_present,
                    at_least_2_indonesian_stopwords_present =
                        EXCLUDED.at_least_2_indonesian_stopwords_present,
                    metrics = EXCLUDED.metrics,
                    created_at = now()
            """,
            values,
            page_size=1000,
        )
    conn.commit()


def metadata_insert_values(run_id: str, row: QualityMetadataRow) -> tuple[Any, ...]:
    metrics = row.metrics
    return (
        run_id,
        row.input_source,
        row.source_doc_id,
        row.doc_id,
        row.sea_pile_malay_id,
        row.cleaning_source,
        row.metrics["source_is_dropped"],
        row.language_profile,
        int(metrics["text_char_count"]),
        int(metrics["paragraph_count"]),
        int(metrics["line_count"]),
        int(metrics["non_empty_line_count"]),
        int(metrics["word_count"]),
        int(metrics["duplicate_line_count"]),
        float(metrics["duplicate_line_fraction"]),
        float(metrics["top_2gram_fraction"]),
        float(metrics["top_3gram_fraction"]),
        float(metrics["top_4gram_fraction"]),
        float(metrics["duplicated_5gram_fraction"]),
        float(metrics["duplicated_6gram_fraction"]),
        float(metrics["duplicated_7gram_fraction"]),
        float(metrics["duplicated_8gram_fraction"]),
        float(metrics["duplicated_9gram_fraction"]),
        float(metrics["duplicated_10gram_fraction"]),
        float(metrics["line_punctuation_fraction"]),
        float(metrics["duplicated_line_character_ratio"]),
        int(metrics["newline_count"]),
        float(metrics["newline_word_ratio"]),
        int(metrics["non_symbol_word_count"]),
        float(metrics["average_non_symbol_word_length"]),
        float(metrics["hash_symbol_ratio"]),
        float(metrics["ellipsis_ratio"]),
        float(metrics["bullet_line_ratio"]),
        float(metrics["ending_ellipsis_line_ratio"]),
        float(metrics["alpha_word_ratio"]),
        int(metrics["stopword_count"]),
        int(metrics["indonesian_stopword_count"]),
        bool(metrics["at_least_2_profile_stopwords_present"]),
        bool(metrics["at_least_2_indonesian_stopwords_present"]),
        psycopg2.extras.Json(metrics),
    )


def summarize_metadata_rows(rows: list[QualityMetadataRow]) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    previews: list[dict[str, Any]] = []
    for row in rows:
        counter["processed_row_count"] += 1
        counter[f"source_{row.input_source}"] += 1
        counter[f"profile_{row.language_profile}"] += 1
        if len(previews) < 5:
            previews.append(
                {
                    "input_source": row.input_source,
                    "source_doc_id": row.source_doc_id,
                    "doc_id": row.doc_id,
                    "sea_pile_malay_id": row.sea_pile_malay_id,
                    "source_is_dropped": row.metrics["source_is_dropped"],
                    "language_profile": row.language_profile,
                    "duplicate_line_fraction": row.metrics["duplicate_line_fraction"],
                    "line_punctuation_fraction": row.metrics["line_punctuation_fraction"],
                    "non_symbol_word_count": row.metrics["non_symbol_word_count"],
                    "alpha_word_ratio": row.metrics["alpha_word_ratio"],
                }
            )
    summary = summarize_counter(counter)
    summary["preview"] = previews
    return summary


def summarize_counter(counter: Counter[str]) -> dict[str, Any]:
    return {
        "processed_row_count": int(counter["processed_row_count"]),
        "source_counts": {
            key.removeprefix("source_"): int(value)
            for key, value in sorted(counter.items())
            if key.startswith("source_")
        },
        "profile_counts": {
            key.removeprefix("profile_"): int(value)
            for key, value in sorted(counter.items())
            if key.startswith("profile_")
        },
    }


def resolve_language_profile(doc: SourceDocument, config: RecipeConfig) -> str:
    profile_name = config.source_language_defaults.get(doc.cleaning_source.strip().lower())
    if profile_name is not None:
        return profile_name
    if "zsm_Latn" in config.language_profiles:
        return "zsm_Latn"
    return sorted(config.language_profiles)[0]


def table_exists(conn: psycopg2.extensions.connection, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
        return cur.fetchone()[0] is not None


def enabled_input_label(config: RecipeConfig) -> str:
    inputs = []
    if config.input.include_unified:
        inputs.append("unified")
    if config.input.include_sea_pile_malay:
        inputs.append("sea-pile-malay")
    return ", ".join(inputs)
