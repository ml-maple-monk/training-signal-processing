from __future__ import annotations

import json
import multiprocessing
import re
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, TimeoutError, as_completed
from time import perf_counter
from typing import Any

import pyarrow.parquet as pq

from ...core.storage import ObjectStore, resolve_runtime_object_store
from ...ops.books_ocr_cleanup import clean_books_ocr_markdown
from ...ops.builtin import RowWiseMapperOp, SourcePreparationOp
from .models import (
    DEFAULT_PASS_THROUGH_COLUMNS,
    LidConfig,
    LidMetadataShardResult,
    ParquetRowGroupTask,
    ReferenceRemovalConfig,
    sample_uid,
    sample_uid_hash,
    text_sha256,
)

_LINGUA_DETECTOR: Any | None = None
_MALAYA_RUNTIME: "MalayaRuntime | None" = None
_MALAYA_RUNTIME_QUANTIZED: bool | None = None
_TIKTOKEN_ENCODINGS: dict[str, Any] = {}


class PrepareLidMetadataRowGroupOp(SourcePreparationOp):
    op_name = "prepare_lid_metadata_row_group"

    def process_row(self, row: dict[str, object]) -> dict[str, object] | None:
        runtime = self.require_runtime()
        task = ParquetRowGroupTask.from_dict(row)
        source_key = row_group_source_key(task)
        if source_key in runtime.completed_source_keys and not runtime.allow_overwrite:
            return None
        return task.to_dict()


class DetectLidMetadataRowGroupOp(RowWiseMapperOp):
    op_name = "detect_lid_metadata_row_group"
    op_stage = "transform"

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        task = ParquetRowGroupTask.from_dict(row)
        try:
            runtime = self.require_runtime()
            object_store = resolve_runtime_object_store(runtime)
            records, metrics = detect_lid_metadata_shard(
                task=task,
                object_store=object_store,
                lid_config=runtime.config.lid,
            )
            return LidMetadataShardResult.success_from_task(
                task=task,
                records=records,
                duration_sec=perf_counter() - started,
                metrics=metrics,
            ).to_dict()
        except Exception as exc:
            runtime = self.runtime
            lid_config = getattr(getattr(runtime, "config", None), "lid", LidConfig())
            return LidMetadataShardResult.failed_from_task(
                task=task,
                error_message=str(exc),
                duration_sec=perf_counter() - started,
                metrics=build_failure_metrics(
                    task=task,
                    lid_config=lid_config,
                    duration_sec=perf_counter() - started,
                ),
            ).to_dict()


class MalayaRuntime:
    def __init__(self, *, fasttext_model: Any, word_model: Any, tokenizer: Any) -> None:
        self.fasttext_model = fasttext_model
        self.word_model = word_model
        self.tokenizer = tokenizer


def row_group_source_key(task: ParquetRowGroupTask) -> str:
    return f"{task.source_object_key}#row_group={task.source_row_group_index}"


def detect_lid_metadata_for_row_group(
    *,
    task: ParquetRowGroupTask,
    object_store: ObjectStore,
    malaya_fasttext_quantized: bool = True,
) -> list[dict[str, Any]]:
    lid_config = LidConfig(malaya_fasttext_quantized=malaya_fasttext_quantized)
    records, _metrics = detect_lid_metadata_shard(
        task=task,
        object_store=object_store,
        lid_config=lid_config,
    )
    return records


def detect_lid_metadata_shard(
    *,
    task: ParquetRowGroupTask,
    object_store: ObjectStore,
    lid_config: LidConfig,
    clock: Callable[[], float] = perf_counter,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table = read_task_row_group(task=task, object_store=object_store)
    rows = table.to_pylist()
    records: list[dict[str, Any]] = []
    checkpoint_key = build_checkpoint_key(task)
    progress = {
        "row_count": 0,
        "success_count": 0,
        "failed_count": 0,
        "cleaned_token_count": 0,
        "last_source_row_index": task.source_row_group_start_index - 1,
    }
    started = clock()
    last_checkpoint_at = started
    last_checkpoint_rows = 0
    last_log_at = started
    parallelism_fallback_reasons: list[str] = []

    def maybe_emit_progress(*, force: bool = False) -> None:
        nonlocal last_checkpoint_at, last_checkpoint_rows, last_log_at
        now = clock()
        completed_rows = int(progress["success_count"]) + int(progress["failed_count"])
        elapsed = max(now - started, 0.0)
        checkpoint_due = (
            completed_rows - last_checkpoint_rows >= lid_config.checkpoint_every_rows
            or now - last_checkpoint_at >= lid_config.checkpoint_every_seconds
        )
        log_due = now - last_log_at >= lid_config.progress_log_every_seconds
        if force or checkpoint_due:
            object_store.write_json(
                checkpoint_key,
                build_checkpoint_payload(
                    task=task,
                    lid_config=lid_config,
                    progress=progress,
                    elapsed_sec=elapsed,
                    checkpoint_key=checkpoint_key,
                    status="row_group_processing" if not force else "row_group_processed",
                    fallback_reason="; ".join(dedupe(parallelism_fallback_reasons)),
                ),
            )
            last_checkpoint_at = now
            last_checkpoint_rows = completed_rows
        if force or log_due:
            log_lid_progress(
                task=task,
                lid_config=lid_config,
                progress=progress,
                elapsed_sec=elapsed,
                checkpoint_key=checkpoint_key,
                fallback_reason="; ".join(dedupe(parallelism_fallback_reasons)),
            )
            last_log_at = now

    row_batches = build_row_batches(rows=rows, row_batch_size=lid_config.row_batch_size)
    for row_batch in row_batches:
        prepared_samples: list[dict[str, Any]] = []
        for row_index_in_row_group, row in row_batch:
            if not row_matches_filters(row=row, filters=task.filters):
                continue
            progress["row_count"] = int(progress["row_count"]) + 1
            source_row_index = task.source_row_group_start_index + row_index_in_row_group
            progress["last_source_row_index"] = source_row_index
            try:
                prepared = prepare_lid_sample(
                    task=task,
                    row=row,
                    row_index_in_row_group=row_index_in_row_group,
                    source_row_index=source_row_index,
                    tokenizer_encoding=lid_config.tokenizer_encoding,
                )
                progress["cleaned_token_count"] = int(progress["cleaned_token_count"]) + int(
                    prepared["cleaned_token_count"]
                )
                prepared_samples.append(prepared)
            except Exception:
                progress["failed_count"] = int(progress["failed_count"]) + 1
                maybe_emit_progress()

        batch_fallback = maybe_run_process_pool_safety_check(
            samples=prepared_samples,
            lid_config=lid_config,
        )
        if batch_fallback:
            parallelism_fallback_reasons.append(batch_fallback)
        lingua_by_source_row = detect_lingua_for_samples(
            samples=prepared_samples,
            lid_config=lid_config,
            fallback_reasons=parallelism_fallback_reasons,
        )
        for prepared in prepared_samples:
            try:
                lingua = lingua_by_source_row[int(prepared["source_row_index"])]
                malaya_document = detect_malaya_document(
                    str(prepared["cleaned_text"]),
                    quantized=lid_config.malaya_fasttext_quantized,
                )
                malaya_words = detect_malaya_words(
                    str(prepared["cleaned_text"]),
                    quantized=lid_config.malaya_fasttext_quantized,
                )
                records.append(
                    build_lid_metadata_record(
                        task=task,
                        prepared=prepared,
                        lingua=lingua,
                        malaya_document=malaya_document,
                        malaya_words=malaya_words,
                    )
                )
                progress["success_count"] = int(progress["success_count"]) + 1
            except Exception:
                progress["failed_count"] = int(progress["failed_count"]) + 1
            maybe_emit_progress()
    maybe_emit_progress(force=True)
    duration_sec = max(clock() - started, 0.0)
    sorted_records = sorted(
        records,
        key=lambda item: (item["source_object_key"], item["source_row_index"]),
    )
    return sorted_records, build_shard_metrics(
        task=task,
        lid_config=lid_config,
        progress=progress,
        duration_sec=duration_sec,
        checkpoint_key=checkpoint_key,
        fallback_reason="; ".join(dedupe(parallelism_fallback_reasons)),
    )


def read_task_row_group(*, task: ParquetRowGroupTask, object_store: ObjectStore):
    filesystem = object_store.build_pyarrow_filesystem()
    parquet_file = pq.ParquetFile(
        f"{object_store.bucket}/{task.source_object_key}",
        filesystem=filesystem,
    )
    schema_names = set(parquet_file.schema_arrow.names)
    requested_columns = [
        task.text_column,
        *task.filters,
        *task.pass_through_columns,
        task.reference_removal.url_column,
    ]
    columns = [column for column in requested_columns if column and column in schema_names]
    if task.text_column not in schema_names:
        raise ValueError(
            f"Parquet source '{task.source_object_key}' is missing text column "
            f"'{task.text_column}'."
        )
    missing_filter_columns = sorted(set(task.filters) - schema_names)
    if missing_filter_columns:
        raise ValueError(
            f"Parquet source '{task.source_object_key}' is missing filter columns: "
            + ", ".join(missing_filter_columns)
        )
    return parquet_file.read_row_group(task.source_row_group_index, columns=dedupe(columns))


def build_row_batches(
    *,
    rows: list[dict[str, Any]],
    row_batch_size: int,
) -> Iterable[list[tuple[int, dict[str, Any]]]]:
    indexed = list(enumerate(rows))
    if row_batch_size <= 0:
        yield indexed
        return
    for start in range(0, len(indexed), row_batch_size):
        yield indexed[start : start + row_batch_size]


def prepare_lid_sample(
    *,
    task: ParquetRowGroupTask,
    row: dict[str, Any],
    row_index_in_row_group: int,
    source_row_index: int,
    tokenizer_encoding: str,
) -> dict[str, Any]:
    text = "" if row.get(task.text_column) is None else str(row.get(task.text_column))
    cleaned_text, reference_metadata = remove_references(
        text=text,
        row=row,
        config=task.reference_removal,
    )
    return {
        "row": row,
        "row_index_in_row_group": row_index_in_row_group,
        "source_row_index": source_row_index,
        "text": text,
        "cleaned_text": cleaned_text,
        "cleaned_token_count": count_text_tokens(
            cleaned_text,
            encoding_name=tokenizer_encoding,
        ),
        "reference_metadata": reference_metadata,
    }


def build_lid_metadata_record(
    *,
    task: ParquetRowGroupTask,
    prepared: dict[str, Any],
    lingua: dict[str, Any],
    malaya_document: dict[str, Any],
    malaya_words: dict[str, Any],
) -> dict[str, Any]:
    source_row_index = int(prepared["source_row_index"])
    row_index_in_row_group = int(prepared["row_index_in_row_group"])
    text = str(prepared["text"])
    cleaned_text = str(prepared["cleaned_text"])
    reference_metadata = dict(prepared["reference_metadata"])
    row = dict(prepared["row"])
    uid = sample_uid(
        bucket=task.source_bucket,
        source_object_key=task.source_object_key,
        source_row_index=source_row_index,
    )
    return {
        "sample_uid": uid,
        "sample_uid_hash": sample_uid_hash(uid),
        "source_name": task.source_name,
        "source_bucket": task.source_bucket,
        "source_object_key": task.source_object_key,
        "source_parquet_url": task.source_parquet_url,
        "source_row_group_index": task.source_row_group_index,
        "source_row_index": source_row_index,
        "row_index_in_row_group": row_index_in_row_group,
        "text_column": task.text_column,
        "original_text_sha256": text_sha256(text),
        "original_char_count": len(text),
        "cleaned_char_count": len(cleaned_text),
        "cleaned_token_count": int(prepared["cleaned_token_count"]),
        "reference_removed": reference_metadata["reference_removed"],
        "reference_removal_method": reference_metadata["reference_removal_method"],
        "removed_reference_char_count": reference_metadata["removed_reference_char_count"],
        "lingua_primary_language": lingua["primary_language"],
        "lingua_spans": lingua["spans"],
        "malaya_document_label": malaya_document["label"],
        "malaya_document_scores": malaya_document["scores"],
        "malaya_word_detections": malaya_words["detections"],
        "malaya_word_label_counts": malaya_words["label_counts"],
        **extract_pass_through_values(row, task.pass_through_columns),
    }


def count_text_tokens(text: str, *, encoding_name: str) -> int:
    if not text:
        return 0
    try:
        encoding = _TIKTOKEN_ENCODINGS.get(encoding_name)
        if encoding is None:
            import tiktoken

            encoding = tiktoken.get_encoding(encoding_name)
            _TIKTOKEN_ENCODINGS[encoding_name] = encoding
        return len(encoding.encode(text))
    except ImportError:
        return len(text.split())


def detect_lingua_for_samples(
    *,
    samples: list[dict[str, Any]],
    lid_config: LidConfig,
    fallback_reasons: list[str],
) -> dict[int, dict[str, Any]]:
    if not samples:
        return {}
    if lid_config.inner_parallelism != "thread_pool":
        return {
            int(sample["source_row_index"]): detect_lingua(str(sample["cleaned_text"]))
            for sample in samples
        }
    try:
        with ThreadPoolExecutor(max_workers=lid_config.inner_workers) as executor:
            futures = {
                executor.submit(detect_lingua, str(sample["cleaned_text"])): int(
                    sample["source_row_index"]
                )
                for sample in samples
            }
            return {
                source_row_index: future.result()
                for future, source_row_index in futures.items()
            }
    except Exception as exc:
        fallback_reasons.append(f"thread_pool_failed_fallback_to_none:{exc}")
        return {
            int(sample["source_row_index"]): detect_lingua(str(sample["cleaned_text"]))
            for sample in samples
        }


def maybe_run_process_pool_safety_check(
    *,
    samples: list[dict[str, Any]],
    lid_config: LidConfig,
) -> str:
    if lid_config.inner_parallelism != "process_pool" or not samples:
        return ""
    payloads = [
        (str(sample["cleaned_text"]), lid_config.tokenizer_encoding)
        for sample in samples[: lid_config.inner_workers]
    ]
    executor: ProcessPoolExecutor | None = None
    try:
        ctx = multiprocessing.get_context(lid_config.multiprocessing_context)
        executor = ProcessPoolExecutor(
            max_workers=lid_config.inner_workers,
            mp_context=ctx,
        )
        futures = [
            executor.submit(process_pool_token_count_probe, text, encoding_name)
            for text, encoding_name in payloads
        ]
        for future in as_completed(futures, timeout=lid_config.process_pool_timeout_seconds):
            future.result(timeout=lid_config.process_pool_timeout_seconds)
        executor.shutdown(wait=True, cancel_futures=True)
        executor = None
        return "process_pool_probe_succeeded_fallback_to_none"
    except TimeoutError as exc:
        return f"process_pool_timeout_fallback_to_none:{exc}"
    except Exception as exc:
        return f"process_pool_failed_fallback_to_none:{exc}"
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)


def process_pool_token_count_probe(text: str, encoding_name: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding(encoding_name).encode(text))
    except ImportError:
        return len(text.split())


def build_checkpoint_key(task: ParquetRowGroupTask) -> str:
    if "/shards/" in task.output_shard_key:
        return (
            task.output_shard_key.replace("/shards/", "/checkpoints/", 1)
            .removesuffix(".parquet")
            + ".json"
        )
    return task.output_shard_key.removesuffix(".parquet") + ".checkpoint.json"


def build_checkpoint_payload(
    *,
    task: ParquetRowGroupTask,
    lid_config: LidConfig,
    progress: dict[str, int],
    elapsed_sec: float,
    checkpoint_key: str,
    status: str,
    fallback_reason: str,
) -> dict[str, object]:
    metrics = build_shard_metrics(
        task=task,
        lid_config=lid_config,
        progress=progress,
        duration_sec=elapsed_sec,
        checkpoint_key=checkpoint_key,
        fallback_reason=fallback_reason,
    )
    return {
        "status": status,
        "source_name": task.source_name,
        "source_object_key": task.source_object_key,
        "source_row_group_index": task.source_row_group_index,
        "rows_completed": int(progress["success_count"]) + int(progress["failed_count"]),
        "last_source_row_index": int(progress["last_source_row_index"]),
        "elapsed_seconds": elapsed_sec,
        "current_tokens_per_sec": metrics["tokens_per_sec"],
        "metrics": metrics,
    }


def build_shard_metrics(
    *,
    task: ParquetRowGroupTask,
    lid_config: LidConfig,
    progress: dict[str, int],
    duration_sec: float,
    checkpoint_key: str,
    fallback_reason: str,
) -> dict[str, Any]:
    row_count = int(progress["row_count"])
    token_count = int(progress["cleaned_token_count"])
    elapsed = max(duration_sec, 1e-9)
    return {
        "source_name": task.source_name,
        "source_object_key": task.source_object_key,
        "source_row_group_index": task.source_row_group_index,
        "row_count": row_count,
        "success_count": int(progress["success_count"]),
        "failed_count": int(progress["failed_count"]),
        "cleaned_token_count": token_count,
        "duration_sec": duration_sec,
        "tokens_per_sec": token_count / elapsed,
        "rows_per_sec": row_count / elapsed,
        "experiment_name": lid_config.experiment_name,
        "variant_name": lid_config.variant_name,
        "inner_parallelism": lid_config.inner_parallelism,
        "inner_workers": lid_config.inner_workers,
        "row_batch_size": lid_config.row_batch_size,
        "checkpoint_key": checkpoint_key,
        "parallelism_fallback_reason": fallback_reason,
    }


def build_failure_metrics(
    *,
    task: ParquetRowGroupTask,
    lid_config: LidConfig,
    duration_sec: float,
) -> dict[str, Any]:
    return build_shard_metrics(
        task=task,
        lid_config=lid_config,
        progress={
            "row_count": 0,
            "success_count": 0,
            "failed_count": 1,
            "cleaned_token_count": 0,
            "last_source_row_index": task.source_row_group_start_index - 1,
        },
        duration_sec=duration_sec,
        checkpoint_key=build_checkpoint_key(task),
        fallback_reason="",
    )


def log_lid_progress(
    *,
    task: ParquetRowGroupTask,
    lid_config: LidConfig,
    progress: dict[str, int],
    elapsed_sec: float,
    checkpoint_key: str,
    fallback_reason: str,
) -> None:
    payload = {
        "event": "lid_metadata.progress",
        "source_name": task.source_name,
        "source_object_key": task.source_object_key,
        "source_row_group_index": task.source_row_group_index,
        "rows_completed": int(progress["success_count"]) + int(progress["failed_count"]),
        "row_group_num_rows": task.source_row_group_num_rows,
        "checkpoint_key": checkpoint_key,
        **build_shard_metrics(
            task=task,
            lid_config=lid_config,
            progress=progress,
            duration_sec=elapsed_sec,
            checkpoint_key=checkpoint_key,
            fallback_reason=fallback_reason,
        ),
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def row_matches_filters(*, row: dict[str, Any], filters: dict[str, str]) -> bool:
    for column_name, expected_value in filters.items():
        if str(row.get(column_name)) != expected_value:
            return False
    return True


def remove_references(
    *,
    text: str,
    row: dict[str, Any],
    config: ReferenceRemovalConfig,
) -> tuple[str, dict[str, Any]]:
    if not config.enabled:
        return text, reference_metadata(False, "disabled", 0)
    cleaned_text = text
    methods: list[str] = []
    if config.url_column and row.get(config.url_column):
        if refextract_url_has_references(str(row[config.url_column])):
            methods.append("refextract_url")
    if config.books_ocr_cleanup_enabled:
        cleanup_result = clean_books_ocr_markdown(cleaned_text)
        if cleanup_result.cleaned_text != cleaned_text:
            cleaned_text = cleanup_result.cleaned_text
            methods.append("books_ocr_cleanup")
    trimmed = trim_reference_section(text=cleaned_text, heading_names=config.heading_names)
    if trimmed != cleaned_text:
        cleaned_text = trimmed
        methods.append("heading_trim")
    if cleaned_text == text:
        method = "+".join(methods + ["none"]) if methods else "none"
        return text, reference_metadata(False, method, 0)
    return cleaned_text, reference_metadata(
        True,
        "+".join(methods) if methods else "unknown",
        len(text) - len(cleaned_text),
    )


def refextract_url_has_references(url: str) -> bool:
    try:
        from refextract import extract_references_from_url
    except Exception:
        return False
    try:
        return bool(extract_references_from_url(url))
    except Exception:
        return False


def trim_reference_section(*, text: str, heading_names: tuple[str, ...]) -> str:
    if not text.strip():
        return text
    heading_pattern = "|".join(re.escape(heading) for heading in heading_names)
    pattern = re.compile(
        rf"(?im)^\s*(?:#+\s*)?(?:\*\*)?\s*(?:{heading_pattern})\s*(?:\*\*)?\s*:?\s*$"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return text
    match = matches[-1]
    before = text[: match.start()].rstrip()
    after = text[match.end() :]
    if not looks_like_reference_tail(after):
        return text
    return before


def looks_like_reference_tail(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    sample = lines[:20]
    reference_like = 0
    for line in sample:
        if re.match(r"^[-*]?\s*(?:\[\d+\]|\d+\.|\w[\w.-]+,\s+[A-Z])", line):
            reference_like += 1
        elif "doi.org/" in line.lower() or "http://" in line.lower() or "https://" in line.lower():
            reference_like += 1
        elif re.search(r"\b(?:19|20)\d{2}\b", line):
            reference_like += 1
    return reference_like >= max(1, min(3, len(sample)))


def reference_metadata(
    reference_removed: bool,
    method: str,
    removed_char_count: int,
) -> dict[str, Any]:
    return {
        "reference_removed": reference_removed,
        "reference_removal_method": method,
        "removed_reference_char_count": removed_char_count,
    }


def detect_lingua(text: str) -> dict[str, Any]:
    if not text.strip():
        return {"primary_language": "", "spans": []}
    detector = get_lingua_detector()
    spans = [
        {
            "start_index": int(result.start_index),
            "end_index": int(result.end_index),
            "language_label": result.language.name,
        }
        for result in detector.detect_multiple_languages_of(text)
    ]
    coverage: Counter[str] = Counter()
    for span in spans:
        coverage[str(span["language_label"])] += int(span["end_index"]) - int(
            span["start_index"]
        )
    primary = coverage.most_common(1)[0][0] if coverage else ""
    return {"primary_language": primary, "spans": spans}


def get_lingua_detector():
    global _LINGUA_DETECTOR
    if _LINGUA_DETECTOR is None:
        _LINGUA_DETECTOR = build_lingua_detector()
    return _LINGUA_DETECTOR


def build_lingua_detector():
    from lingua import Language, LanguageDetectorBuilder

    return LanguageDetectorBuilder.from_languages(
        Language.ENGLISH,
        Language.INDONESIAN,
        Language.MALAY,
    ).build()


def detect_malaya_document(text: str, *, quantized: bool = True) -> dict[str, Any]:
    if not text.strip():
        return {"label": "", "scores": []}
    runtime = get_malaya_runtime(quantized=quantized)
    scores_by_label = runtime.fasttext_model.predict_proba([text])[0]
    scores = [
        {"label": str(label), "score": float(score)}
        for label, score in sorted(scores_by_label.items())
    ]
    label = max(scores, key=lambda item: item["score"])["label"] if scores else ""
    return {"label": label, "scores": scores}


def detect_malaya_words(text: str, *, quantized: bool = True) -> dict[str, Any]:
    if not text.strip():
        return {"detections": [], "label_counts": []}
    runtime = get_malaya_runtime(quantized=quantized)
    tokens = runtime.tokenizer.tokenize(text)
    labels = runtime.word_model.predict(tokens)
    detections = build_word_detections(text=text, tokens=tokens, labels=labels)
    counts = Counter(detection["label"] for detection in detections)
    return {
        "detections": detections,
        "label_counts": [
            {"label": label, "count": count} for label, count in sorted(counts.items())
        ],
    }


def get_malaya_runtime(*, quantized: bool = True) -> MalayaRuntime:
    global _MALAYA_RUNTIME, _MALAYA_RUNTIME_QUANTIZED
    if _MALAYA_RUNTIME is None or _MALAYA_RUNTIME_QUANTIZED != quantized:
        _MALAYA_RUNTIME = build_malaya_runtime(quantized=quantized)
        _MALAYA_RUNTIME_QUANTIZED = quantized
    return _MALAYA_RUNTIME


def build_malaya_runtime(*, quantized: bool = True) -> MalayaRuntime:
    import malaya

    fasttext_model = malaya.language_detection.fasttext(quantized=quantized)
    word_model = malaya.language_detection.substring_rules(model=fasttext_model)
    tokenizer = malaya.preprocessing.Tokenizer()
    return MalayaRuntime(
        fasttext_model=fasttext_model,
        word_model=word_model,
        tokenizer=tokenizer,
    )


def build_word_detections(
    *,
    text: str,
    tokens: list[str],
    labels: list[str],
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    cursor = 0
    for word_index, token in enumerate(tokens):
        start_index = text.find(token, cursor) if token else cursor
        if start_index < 0:
            start_index = cursor
        end_index = start_index + len(token)
        cursor = end_index
        detections.append(
            {
                "word_index": word_index,
                "start_index": start_index,
                "end_index": end_index,
                "token": token,
                "label": str(labels[word_index]) if word_index < len(labels) else "",
            }
        )
    return detections


def extract_pass_through_values(
    row: dict[str, Any],
    pass_through_columns: tuple[str, ...],
) -> dict[str, str]:
    values: dict[str, str] = {}
    for column in DEFAULT_PASS_THROUGH_COLUMNS:
        value = row.get(column) if column in pass_through_columns else None
        values[column] = "" if value is None else str(value)
    return values


def dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped
