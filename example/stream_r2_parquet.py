#!/usr/bin/env python3
"""Stream parquet rows through a small batch pipeline.

This example now supports two source modes:

- Cloudflare R2 parquet via DuckDB
- Local book-partition parquet with resumable Tesseract OCR on real sidecar images

Both modes stay streaming-friendly:

- push down simple SQL filters and dedup in DuckDB
- fetch rows in bounded batches with ``fetchmany``
- apply common batch ops in Python
- maintain a bounded Top-K heap instead of globally sorting all rows

Run from the repo root, for example:

    uv run python example/stream_r2_parquet.py \
      --dataset local_ztd \
      --text-column text \
      --where "selection_bucket = 'high_diff'" \
      --batch-size 500 \
      --top-k 5

    uv run python example/stream_r2_parquet.py \
      --book-parquet-glob "/home/geeyang/workspace/malay-data/final_dataset/books/part-*.parquet" \
      --with-tesseract \
      --batch-size 8 \
      --top-k 5
"""

from __future__ import annotations

import argparse
import heapq
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Callable

import duckdb
from duckdb_r2_query import (
    DATASETS,
    DEFAULT_CONFIG,
    build_dataset_scan,
    connect_to_r2,
    sql_literal,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOOK_PARQUET_GLOB = "/home/geeyang/workspace/malay-data/final_dataset/books/part-*.parquet"
DEFAULT_BOOK_OUTPUT_ROOT = Path(
    "/home/geeyang/workspace/malay-data/bookscrape/output/mineru-pipeline-latin-full"
)
DEFAULT_BOOK_DOWNLOAD_ROOT = Path(
    "/home/geeyang/workspace/malay-data/bookscrape/data/downloads"
)
DEFAULT_BOOK_PROCESSED_OUTPUT_DIR = ROOT / ".cache" / "book_tesseract_processed"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
WHITESPACE_RE = re.compile(r"\s+")

Batch = list[dict[str, Any]]
BatchOp = Callable[[Batch], Batch]


@dataclass
class PipelineStats:
    source_rows: int = 0
    normalized_rows: int = 0
    filtered_rows: int = 0
    emitted_rows: int = 0
    written_rows: int = 0
    batches: int = 0


class RollingTopK:
    """Maintain a bounded Top-K heap while streaming batches."""

    def __init__(self, k: int) -> None:
        self.k = k
        self._heap: list[tuple[float, int, dict[str, Any]]] = []
        self._counter = 0

    def add_batch(self, batch: Batch) -> None:
        for row in batch:
            score = float(row["quality_score"])
            item = (score, self._counter, row)
            self._counter += 1
            if len(self._heap) < self.k:
                heapq.heappush(self._heap, item)
                continue
            if score > self._heap[0][0]:
                heapq.heapreplace(self._heap, item)

    def rows(self) -> list[dict[str, Any]]:
        return [item[2] for item in sorted(self._heap, reverse=True)]


class QwenTokenProbOp:
    """Annotate rows with a short Qwen completion and per-token probabilities."""

    def __init__(
        self,
        *,
        model_name: str,
        max_input_tokens: int,
        max_new_tokens: int,
        device: str | None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional deps
            raise SystemExit(
                "Qwen support requires optional model dependencies. "
                "Install them with: uv sync --group model"
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.device = self._resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, requested: str | None) -> str:
        if requested:
            return requested
        if self.torch.cuda.is_available():
            return "cuda"
        if hasattr(self.torch.backends, "mps") and self.torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _build_prompt(self, text: str) -> str:
        return (
            "Read the text snippet and answer with a short topical label.\n\n"
            f"Snippet:\n{text}\n\n"
            "Label:"
        )

    def __call__(self, batch: Batch) -> Batch:
        if not batch:
            return batch

        prompts = [self._build_prompt(str(row["text"])) for row in batch]
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}

        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_lengths = encoded["attention_mask"].sum(dim=1)
        output: Batch = []

        for row_index, row in enumerate(batch):
            generated_ids = generated.sequences[row_index, prompt_lengths[row_index] :]
            token_probs = []
            completion_ids = []

            for step_index, token_id in enumerate(generated_ids.tolist()):
                if step_index >= len(generated.scores):
                    break
                if token_id == self.tokenizer.pad_token_id:
                    continue

                distribution = self.torch.softmax(generated.scores[step_index][row_index], dim=-1)
                probability = float(distribution[token_id].detach().cpu())
                logprob = float(self.torch.log(distribution[token_id]).detach().cpu())
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=False)

                completion_ids.append(token_id)
                token_probs.append(
                    {
                        "token": token_text,
                        "prob": round(probability, 6),
                        "logprob": round(logprob, 6),
                    }
                )

            completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            mean_prob = (
                round(sum(item["prob"] for item in token_probs) / len(token_probs), 6)
                if token_probs
                else None
            )
            output.append(
                {
                    **row,
                    "qwen_completion": completion,
                    "qwen_token_probs": token_probs,
                    "qwen_mean_token_prob": mean_prob,
                }
            )

        return output


class TesseractImageOcrOp:
    """Run Tesseract on real sidecar images referenced by book markdown rows."""

    def __init__(
        self,
        *,
        book_output_root: Path | None,
        book_download_root: Path | None,
        tesseract_binary: str,
        lang: str,
        input_mode: str,
        max_pages_per_row: int | None,
        max_images_per_row: int,
        render_dpi: int,
        psm: int | None,
        oem: int | None,
        timeout_seconds: int,
        extra_args: str,
    ) -> None:
        self.book_output_root = book_output_root
        self.book_download_root = book_download_root
        self.tesseract_binary = tesseract_binary
        self.lang = lang
        self.input_mode = input_mode
        self.max_pages_per_row = max_pages_per_row
        self.max_images_per_row = max_images_per_row
        self.render_dpi = render_dpi
        self.psm = psm
        self.oem = oem
        self.timeout_seconds = timeout_seconds
        self.extra_args = shlex.split(extra_args)

        binary_path = (
            tesseract_binary
            if Path(tesseract_binary).is_file()
            else shutil.which(tesseract_binary)
        )
        if binary_path is None:
            raise SystemExit(
                "Tesseract OCR requested but the configured binary is not available: "
                f"{tesseract_binary}"
            )
        self.tesseract_binary = binary_path

    def _command(self, image_path: Path) -> list[str]:
        command = [self.tesseract_binary, str(image_path), "stdout", "-l", self.lang]
        if self.psm is not None:
            command.extend(["--psm", str(self.psm)])
        if self.oem is not None:
            command.extend(["--oem", str(self.oem)])
        command.extend(self.extra_args)
        command.append("quiet")
        return command

    def _render_pdf_pages(
        self,
        pdf_path: Path,
    ) -> tuple[list[Path], list[str], Path | None]:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:  # pragma: no cover - depends on OCR runtime env
            raise RuntimeError(
                "Full-PDF OCR requires pypdfium2 in the OCR runtime. "
                "Re-run scripts/setup_ocr_runtime.sh."
            ) from exc

        temp_dir = Path(tempfile.mkdtemp(prefix="tesseract-pdf-pages-"))
        rendered_paths: list[Path] = []
        page_refs: list[str] = []
        try:
            pdf = pdfium.PdfDocument(str(pdf_path))
            page_total = len(pdf)
            page_limit = page_total
            if self.max_pages_per_row is not None:
                page_limit = min(page_total, self.max_pages_per_row)

            scale = max(self.render_dpi, 72) / 72
            for page_index in range(page_limit):
                page = pdf[page_index]
                bitmap = page.render(scale=scale)
                image = bitmap.to_pil()
                output_path = temp_dir / f"page-{page_index + 1:04d}.png"
                image.save(output_path, format="PNG")
                image.close()
                if hasattr(bitmap, "close"):
                    bitmap.close()
                if hasattr(page, "close"):
                    page.close()
                rendered_paths.append(output_path)
                page_refs.append(f"{pdf_path}#page={page_index + 1}")
            if hasattr(pdf, "close"):
                pdf.close()
            return rendered_paths, page_refs, temp_dir
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def __call__(self, batch: Batch) -> Batch:
        output: Batch = []
        for row in batch:
            source_pdf_path = resolve_source_pdf_path(
                row,
                book_download_root=self.book_download_root,
            )
            image_paths = resolve_tesseract_image_paths(
                row,
                book_output_root=self.book_output_root,
                max_images=self.max_images_per_row,
            )
            rendered_temp_dir: Path | None = None
            page_refs: list[str] = [str(path) for path in image_paths]
            input_kind = "images"

            if (
                self.input_mode in {"auto", "pdf"}
                and source_pdf_path is not None
                and source_pdf_path.suffix.lower() == ".pdf"
            ):
                try:
                    (
                        image_paths,
                        page_refs,
                        rendered_temp_dir,
                    ) = self._render_pdf_pages(source_pdf_path)
                    input_kind = "pdf"
                except Exception as exc:
                    if self.input_mode == "pdf":
                        raise
                    if not image_paths:
                        raise RuntimeError(
                            f"Failed to render PDF for OCR fallback: {source_pdf_path}"
                        ) from exc

            extracted_text: list[str] = []
            successful_images = 0
            last_error: str | None = None

            try:
                for image_path in image_paths:
                    try:
                        completed = subprocess.run(
                            self._command(image_path),
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=self.timeout_seconds,
                        )
                    except subprocess.TimeoutExpired:
                        last_error = f"timeout:{image_path.name}"
                        continue

                    if completed.returncode != 0:
                        stderr = completed.stderr.strip()
                        last_error = stderr or f"exit:{completed.returncode}"
                        continue

                    text = WHITESPACE_RE.sub(" ", completed.stdout.strip())
                    if not text:
                        continue
                    extracted_text.append(text)
                    successful_images += 1
            finally:
                if rendered_temp_dir is not None:
                    shutil.rmtree(rendered_temp_dir, ignore_errors=True)

            combined_text = "\n\n".join(extracted_text)
            output.append(
                {
                    **row,
                    "resolved_source_pdf_path": (
                        str(source_pdf_path) if source_pdf_path is not None else None
                    ),
                    "tesseract_text": combined_text,
                    "tesseract_char_count": len(combined_text),
                    "tesseract_input_kind": input_kind,
                    "tesseract_image_count": len(image_paths),
                    "tesseract_successful_images": successful_images,
                    "tesseract_image_paths": page_refs,
                    "tesseract_last_error": last_error,
                }
            )

        return output


class ProcessedMarkdownWriter:
    """Persist processed rows to markdown files as each batch completes."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows_written = 0
        self.files_written = 0
        self.output_paths: list[Path] = []

    def add_batch(self, batch: Batch) -> list[Path]:
        if not batch:
            return []

        written_paths: list[Path] = []
        for row in batch:
            output_path = processed_markdown_path(row, output_dir=self.output_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(render_processed_markdown(row), encoding="utf-8")
            written_paths.append(output_path)

        self.rows_written += len(written_paths)
        self.files_written += len(written_paths)
        self.output_paths.extend(written_paths)
        return written_paths

    def close(self) -> list[Path]:
        return list(self.output_paths)


def coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def sql_string_list(values: list[str]) -> str:
    return "[" + ", ".join(sql_literal(value) for value in values) + "]"


def existing_parquet_files(spec: str) -> list[str]:
    if not spec:
        return []

    candidate = Path(spec)
    if candidate.is_dir():
        return sorted(str(path) for path in candidate.glob("*.parquet") if path.is_file())

    return sorted(path for path in glob(spec) if Path(path).is_file() and path.endswith(".parquet"))


def build_stream_query(
    *,
    dataset: str,
    bucket: str,
    prefix: str,
    text_column: str,
    id_column: str,
    extra_columns: list[str],
    where: str,
) -> str:
    scan_sql = build_dataset_scan(dataset, bucket, prefix)
    selected = [
        f"CAST({id_column} AS VARCHAR) AS record_id",
        f"CAST({text_column} AS VARCHAR) AS raw_text",
    ]
    selected.extend(extra_columns)

    filters = []
    if where:
        filters.append(f"({where})")
    filters.extend(
        [
            f"{text_column} IS NOT NULL",
            f"length(trim(CAST({text_column} AS VARCHAR))) > 0",
        ]
    )

    return f"""
        SELECT
            {", ".join(selected)}
        FROM ({scan_sql}) AS dataset_scan
        WHERE {" AND ".join(filters)}
    """


def build_book_partition_query(
    *,
    source_files: list[str],
    processed_files: list[str],
    extra_columns: list[str],
    where: str,
) -> str:
    if not source_files:
        raise SystemExit("No source parquet files were found for --book-parquet-glob.")

    selected = [
        "record_id",
        "raw_text",
        "dedup_key",
        "markdown_sha256",
        "markdown_path",
        "markdown_rel_path",
        "source_pdf_path",
        "source_pdf_rel_path",
    ]
    selected.extend(extra_columns)

    processed_cte = ""
    processed_filter = ""
    if processed_files:
        processed_cte = f"""
        ,
        processed_rows AS (
            SELECT DISTINCT
                COALESCE(
                    NULLIF(CAST(dedup_key AS VARCHAR), ''),
                    NULLIF(CAST(markdown_sha256 AS VARCHAR), ''),
                    NULLIF(CAST(record_id AS VARCHAR), ''),
                    NULLIF(CAST(source_pdf_rel_path AS VARCHAR), ''),
                    NULLIF(CAST(markdown_rel_path AS VARCHAR), '')
                ) AS dedup_key
            FROM read_parquet(
                {sql_string_list(processed_files)},
                union_by_name = true
            )
        )
        """
        processed_filter = """
            AND NOT EXISTS (
                SELECT 1
                FROM processed_rows
                WHERE processed_rows.dedup_key = deduped_rows.dedup_key
            )
        """

    source_filters = [
        "markdown_text IS NOT NULL",
        "length(trim(CAST(markdown_text AS VARCHAR))) > 0",
    ]
    if where:
        source_filters.insert(0, f"({where})")

    return f"""
        WITH source_rows AS (
            SELECT
                COALESCE(
                    NULLIF(CAST(book_id AS VARCHAR), ''),
                    NULLIF(CAST(markdown_sha256 AS VARCHAR), ''),
                    NULLIF(CAST(source_pdf_rel_path AS VARCHAR), ''),
                    NULLIF(CAST(markdown_rel_path AS VARCHAR), '')
                ) AS record_id,
                CAST(markdown_text AS VARCHAR) AS raw_text,
                COALESCE(
                    NULLIF(CAST(markdown_sha256 AS VARCHAR), ''),
                    NULLIF(CAST(book_id AS VARCHAR), ''),
                    NULLIF(CAST(source_pdf_rel_path AS VARCHAR), ''),
                    NULLIF(CAST(markdown_rel_path AS VARCHAR), '')
                ) AS dedup_key,
                CAST(markdown_sha256 AS VARCHAR) AS markdown_sha256,
                CAST(markdown_path AS VARCHAR) AS markdown_path,
                CAST(markdown_rel_path AS VARCHAR) AS markdown_rel_path,
                CAST(source_pdf_path AS VARCHAR) AS source_pdf_path,
                CAST(source_pdf_rel_path AS VARCHAR) AS source_pdf_rel_path
                {", " if extra_columns else ""}{", ".join(extra_columns)}
            FROM read_parquet(
                {sql_string_list(source_files)},
                union_by_name = true
            )
            WHERE {" AND ".join(source_filters)}
        ),
        deduped_rows AS (
            SELECT * EXCLUDE (_row_number)
            FROM (
                SELECT
                    *,
                    row_number() OVER (
                        PARTITION BY dedup_key
                        ORDER BY markdown_path, markdown_rel_path
                    ) AS _row_number
                FROM source_rows
                WHERE dedup_key IS NOT NULL
            )
            WHERE _row_number = 1
        )
        {processed_cte}
        SELECT
            {", ".join(selected)}
        FROM deduped_rows
        WHERE 1 = 1
            {processed_filter}
    """


def stream_batches(
    con: duckdb.DuckDBPyConnection,
    query: str,
    *,
    batch_size: int,
    max_batches: int | None,
    stats: PipelineStats,
) -> Iterator[Batch]:
    cursor = con.execute(query)
    columns = [description[0] for description in cursor.description]

    batch_index = 0
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break

        batch_index += 1
        stats.batches += 1
        stats.source_rows += len(rows)
        batch = [dict(zip(columns, row, strict=True)) for row in rows]

        print(
            f"[source] batch={batch_index} rows={len(batch)} "
            f"total_source_rows={stats.source_rows}"
        )
        yield batch

        if max_batches is not None and batch_index >= max_batches:
            print(f"[source] stopped early after {max_batches} batches")
            break


def normalize_text(batch: Batch) -> Batch:
    output: Batch = []
    for row in batch:
        text = WHITESPACE_RE.sub(" ", str(row["raw_text"]).strip())
        if not text:
            continue
        output.append({**row, "text": text})
    return output


def add_metrics(batch: Batch) -> Batch:
    output: Batch = []
    for row in batch:
        text = str(row["text"])
        words = [word for word in text.split(" ") if word]
        unique_ratio = len(set(words)) / max(len(words), 1)
        output.append(
            {
                **row,
                "char_count": len(text),
                "word_count": len(words),
                "unique_ratio": round(unique_ratio, 4),
            }
        )
    return output


def keep_reasonable_lengths(min_words: int, max_words: int) -> BatchOp:
    def op(batch: Batch) -> Batch:
        return [
            row
            for row in batch
            if min_words <= int(row["word_count"]) <= max_words
        ]

    return op


def score_rows(batch: Batch) -> Batch:
    output: Batch = []
    for row in batch:
        word_count = int(row["word_count"])
        unique_ratio = float(row["unique_ratio"])
        quality_score = round(word_count * unique_ratio, 2)
        output.append({**row, "quality_score": quality_score})
    return output


def run_op(name: str, op: BatchOp, batch: Batch) -> Batch:
    started_at = time.perf_counter()
    result = op(batch)
    elapsed = time.perf_counter() - started_at
    print(
        f"[op:{name}] in={len(batch)} out={len(result)} "
        f"elapsed={elapsed:.4f}s"
    )
    return result


def parse_extra_columns(raw: str) -> list[str]:
    return [column.strip() for column in raw.split(",") if column.strip()]


def resolve_markdown_path(row: dict[str, Any], *, book_output_root: Path | None) -> Path | None:
    candidates: list[Path] = []

    markdown_path = coerce_text(row.get("markdown_path"))
    if markdown_path:
        candidates.append(Path(markdown_path))

    markdown_rel_path = coerce_text(row.get("markdown_rel_path"))
    if markdown_rel_path and book_output_root is not None:
        candidates.append(book_output_root / markdown_rel_path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def resolve_source_pdf_path(
    row: dict[str, Any],
    *,
    book_download_root: Path | None,
) -> Path | None:
    candidates: list[Path] = []

    source_pdf_path = coerce_text(row.get("source_pdf_path"))
    if source_pdf_path:
        candidates.append(Path(source_pdf_path))

    source_pdf_rel_path = coerce_text(row.get("source_pdf_rel_path"))
    if source_pdf_rel_path and book_download_root is not None:
        candidates.append(book_download_root / source_pdf_rel_path)
        candidates.append(book_download_root / Path(source_pdf_rel_path).name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return None


def resolve_tesseract_image_paths(
    row: dict[str, Any],
    *,
    book_output_root: Path | None,
    max_images: int,
) -> list[Path]:
    markdown_path = resolve_markdown_path(row, book_output_root=book_output_root)
    if markdown_path is None:
        return []

    discovered: list[Path] = []
    seen: set[Path] = set()

    raw_text = coerce_text(row.get("raw_text")) or ""
    for image_ref in MARKDOWN_IMAGE_RE.findall(raw_text):
        if image_ref.startswith(("http://", "https://", "data:")):
            continue
        candidate = (markdown_path.parent / image_ref).resolve()
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if candidate in seen:
            continue
        discovered.append(candidate)
        seen.add(candidate)
        if len(discovered) >= max_images:
            return discovered

    images_dir = markdown_path.parent / "images"
    if images_dir.is_dir():
        for image_path in sorted(images_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            resolved = image_path.resolve()
            if resolved in seen:
                continue
            discovered.append(resolved)
            seen.add(resolved)
            if len(discovered) >= max_images:
                break

    return discovered


def keep_rows_with_tesseract_inputs(
    *,
    book_output_root: Path | None,
    book_download_root: Path | None,
    max_images: int,
) -> BatchOp:
    def op(batch: Batch) -> Batch:
        output: Batch = []
        for row in batch:
            source_pdf_path = resolve_source_pdf_path(
                row,
                book_download_root=book_download_root,
            )
            image_paths = resolve_tesseract_image_paths(
                row,
                book_output_root=book_output_root,
                max_images=max_images,
            )
            if source_pdf_path is None and not image_paths:
                continue
            output.append(
                {
                    **row,
                    "candidate_source_pdf_path": (
                        str(source_pdf_path) if source_pdf_path is not None else None
                    ),
                    "candidate_tesseract_image_count": len(image_paths),
                    "candidate_tesseract_image_paths": [str(path) for path in image_paths],
                }
            )
        return output

    return op


def processed_markdown_path(row: dict[str, Any], *, output_dir: Path) -> Path:
    cached_path = coerce_text(row.get("processed_markdown_path"))
    if cached_path:
        return Path(cached_path)

    markdown_rel_path = coerce_text(row.get("markdown_rel_path"))
    if markdown_rel_path:
        relative_path = Path(markdown_rel_path)
        if not relative_path.is_absolute() and ".." not in relative_path.parts:
            return output_dir / relative_path

    dedup_key = coerce_text(row.get("dedup_key")) or coerce_text(row.get("record_id")) or "row"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", dedup_key).strip(".-") or "row"
    return output_dir / f"{safe_name}.md"


def keep_rows_without_processed_markdown(
    *,
    output_dir: Path,
    overwrite_existing: bool = False,
) -> BatchOp:
    def op(batch: Batch) -> Batch:
        output: Batch = []
        for row in batch:
            output_path = processed_markdown_path(row, output_dir=output_dir)
            if output_path.is_file() and not overwrite_existing:
                continue
            output.append({**row, "processed_markdown_path": str(output_path)})
        return output

    return op


def render_processed_markdown(row: dict[str, Any]) -> str:
    metadata = [
        ("record_id", coerce_text(row.get("record_id"))),
        ("dedup_key", coerce_text(row.get("dedup_key"))),
        ("markdown_sha256", coerce_text(row.get("markdown_sha256"))),
        ("markdown_rel_path", coerce_text(row.get("markdown_rel_path"))),
        ("source_pdf_rel_path", coerce_text(row.get("source_pdf_rel_path"))),
        ("tesseract_input_kind", coerce_text(row.get("tesseract_input_kind"))),
        ("quality_score", row.get("quality_score")),
        ("word_count", row.get("word_count")),
        ("char_count", row.get("char_count")),
        ("tesseract_image_count", row.get("tesseract_image_count")),
        ("tesseract_successful_images", row.get("tesseract_successful_images")),
        ("tesseract_last_error", coerce_text(row.get("tesseract_last_error"))),
    ]
    lines = ["# OCR Result", ""]
    for key, value in metadata:
        if value is None:
            continue
        lines.append(f"- {key}: {value}")

    tesseract_text = coerce_text(row.get("tesseract_text")) or ""
    text_body = tesseract_text.strip() or "_No OCR text extracted._"
    lines.extend(["", "## OCR Text", "", text_body, ""])
    return "\n".join(lines)


def print_top_k(rows: list[dict[str, Any]], *, preview_chars: int) -> None:
    if not rows:
        print("\n[top-k] no rows survived the pipeline")
        return

    print("\n[top-k] highest scoring streamed rows")
    for index, row in enumerate(rows, start=1):
        preview = str(row["text"])[:preview_chars].replace("\n", " ")
        qwen_suffix = ""
        if row.get("qwen_mean_token_prob") is not None:
            qwen_suffix = (
                f" qwen_mean_prob={row['qwen_mean_token_prob']:.4f}"
                f" completion={row.get('qwen_completion', '')!r}"
            )

        tesseract_suffix = ""
        if row.get("tesseract_char_count") is not None:
            tesseract_suffix = (
                f" tesseract_chars={row.get('tesseract_char_count', 0)}"
                f" tesseract_images={row.get('tesseract_successful_images', 0)}"
                f"/{row.get('tesseract_image_count', 0)}"
            )

        print(
            f"  {index}. score={row['quality_score']:.2f} "
            f"words={row['word_count']} id={row['record_id']}"
            f"{qwen_suffix}{tesseract_suffix}"
        )
        print(f"     preview={preview}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream-process parquet from R2 or local book partitions."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    parser.add_argument("--book-parquet-glob", default="")
    parser.add_argument("--book-output-root", type=Path, default=DEFAULT_BOOK_OUTPUT_ROOT)
    parser.add_argument("--book-download-root", type=Path, default=DEFAULT_BOOK_DOWNLOAD_ROOT)
    parser.add_argument("--processed-parquet-glob", default="")
    parser.add_argument(
        "--processed-output-dir",
        type=Path,
        default=DEFAULT_BOOK_PROCESSED_OUTPUT_DIR,
    )
    parser.add_argument("--rewrite-processed-markdown", action="store_true")
    parser.add_argument("--text-column", default="")
    parser.add_argument("--id-column", default="filename")
    parser.add_argument("--extra-columns", default="")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bucket", default="gpu-poor")
    parser.add_argument("--prefix", default="dataset/raw")
    parser.add_argument("--where", default="")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--min-words", type=int, default=50)
    parser.add_argument("--max-words", type=int, default=4000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--preview-chars", type=int, default=120)
    parser.add_argument("--show-sql", action="store_true")
    parser.add_argument("--with-qwen", action="store_true")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--qwen-device")
    parser.add_argument("--qwen-max-input-tokens", type=int, default=256)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=8)
    parser.add_argument("--with-tesseract", action="store_true")
    parser.add_argument(
        "--tesseract-binary",
        default=os.environ.get("TESSERACT_BINARY", "tesseract"),
    )
    parser.add_argument("--tesseract-lang", default="eng")
    parser.add_argument(
        "--tesseract-input-mode",
        choices=("auto", "pdf", "images"),
        default="auto",
    )
    parser.add_argument("--tesseract-max-pages-per-row", type=int)
    parser.add_argument("--tesseract-max-images-per-row", type=int, default=4)
    parser.add_argument("--tesseract-render-dpi", type=int, default=200)
    parser.add_argument("--tesseract-psm", type=int)
    parser.add_argument("--tesseract-oem", type=int)
    parser.add_argument("--tesseract-timeout-seconds", type=int, default=120)
    parser.add_argument("--tesseract-extra-args", default="")
    args = parser.parse_args()

    if bool(args.dataset) == bool(args.book_parquet_glob):
        raise SystemExit("Choose exactly one source mode: --dataset or --book-parquet-glob.")

    if args.dataset and not args.text_column:
        raise SystemExit("--text-column is required when using --dataset.")

    if args.dataset and args.with_tesseract:
        raise SystemExit("--with-tesseract currently requires --book-parquet-glob source mode.")

    extra_columns = parse_extra_columns(args.extra_columns)

    if args.book_parquet_glob:
        source_files = existing_parquet_files(args.book_parquet_glob)
        if not source_files:
            raise SystemExit(
                "No source book parquet files matched "
                f"{args.book_parquet_glob!r}. "
                f"Default expected location is {DEFAULT_BOOK_PARQUET_GLOB!r}."
            )

        processed_glob = args.processed_parquet_glob or str(args.processed_output_dir / "*.parquet")
        processed_files = existing_parquet_files(processed_glob)
        query = build_book_partition_query(
            source_files=source_files,
            processed_files=processed_files,
            extra_columns=extra_columns,
            where=args.where,
        )
        con = duckdb.connect()
        source_mode = "book_parquet"
        print(
            f"[source] mode={source_mode} source_files={len(source_files)} "
            f"processed_files={len(processed_files)}"
        )
    else:
        query = build_stream_query(
            dataset=args.dataset,
            bucket=args.bucket,
            prefix=args.prefix,
            text_column=args.text_column,
            id_column=args.id_column,
            extra_columns=extra_columns,
            where=args.where,
        )
        con = connect_to_r2(args.config)
        source_mode = "r2"
        print(f"[source] mode={source_mode} dataset={args.dataset}")

    if args.show_sql:
        print(query.strip())
        print()

    stats = PipelineStats()
    sink = RollingTopK(args.top_k)
    writer = ProcessedMarkdownWriter(args.processed_output_dir) if args.book_parquet_glob else None

    ops: list[tuple[str, BatchOp]] = [
        ("normalize_text", normalize_text),
        ("add_metrics", add_metrics),
        ("keep_reasonable_lengths", keep_reasonable_lengths(args.min_words, args.max_words)),
    ]
    if args.book_parquet_glob:
        ops.append(
            (
                "skip_existing_markdown",
                keep_rows_without_processed_markdown(
                    output_dir=args.processed_output_dir,
                    overwrite_existing=args.rewrite_processed_markdown,
                ),
            )
        )
    if args.with_tesseract:
        ops.append(
            (
                "keep_rows_with_tesseract_inputs",
                keep_rows_with_tesseract_inputs(
                    book_output_root=args.book_output_root,
                    book_download_root=args.book_download_root,
                    max_images=args.tesseract_max_images_per_row,
                ),
            )
        )
        ops.append(
            (
                "tesseract_ocr",
                TesseractImageOcrOp(
                    book_output_root=args.book_output_root,
                    book_download_root=args.book_download_root,
                    tesseract_binary=args.tesseract_binary,
                    lang=args.tesseract_lang,
                    input_mode=args.tesseract_input_mode,
                    max_pages_per_row=args.tesseract_max_pages_per_row,
                    max_images_per_row=args.tesseract_max_images_per_row,
                    render_dpi=args.tesseract_render_dpi,
                    psm=args.tesseract_psm,
                    oem=args.tesseract_oem,
                    timeout_seconds=args.tesseract_timeout_seconds,
                    extra_args=args.tesseract_extra_args,
                ),
            )
        )
    if args.with_qwen:
        ops.append(
            (
                "qwen_token_probs",
                QwenTokenProbOp(
                    model_name=args.qwen_model,
                    max_input_tokens=args.qwen_max_input_tokens,
                    max_new_tokens=args.qwen_max_new_tokens,
                    device=args.qwen_device,
                ),
            )
        )
    ops.append(("score_rows", score_rows))

    written_paths: list[Path] = []
    try:
        for batch_index, batch in enumerate(
            stream_batches(
                con,
                query,
                batch_size=args.batch_size,
                max_batches=args.max_batches,
                stats=stats,
            ),
            start=1,
        ):
            current = batch
            print(f"[pipeline] start batch={batch_index}")
            for name, op in ops:
                current = run_op(name, op, current)
                if name == "normalize_text":
                    stats.normalized_rows += len(current)
                elif name in {"keep_reasonable_lengths", "skip_existing_markdown"}:
                    stats.filtered_rows += len(current)

            sink.add_batch(current)
            if writer is not None:
                written_paths_for_batch = writer.add_batch(current)
                stats.written_rows += len(current)
                written_paths.extend(written_paths_for_batch)
                if written_paths_for_batch:
                    print(
                        f"[sink:processed_markdown] batch={batch_index} "
                        f"written={len(written_paths_for_batch)} "
                        f"output_dir={args.processed_output_dir}"
                    )

            stats.emitted_rows += len(current)
            print(
                f"[sink:top_k] batch={batch_index} kept={len(current)} "
                f"running_top_k={min(args.top_k, stats.emitted_rows)}"
            )
    finally:
        if writer is not None:
            written_paths = writer.close()
        con.close()

    print(
        "\n[summary] "
        f"batches={stats.batches} "
        f"source_rows={stats.source_rows} "
        f"normalized_rows={stats.normalized_rows} "
        f"filtered_rows={stats.filtered_rows} "
        f"emitted_rows={stats.emitted_rows}"
    )
    if stats.written_rows:
        print(
            f"[summary] written_rows={stats.written_rows} "
            f"processed_markdown_files={len(written_paths)}"
        )
    if written_paths:
        print(f"[summary] processed_markdown_dir={args.processed_output_dir}")
    if args.with_qwen:
        print(
            f"[summary] qwen_model={args.qwen_model} "
            f"qwen_device={args.qwen_device or 'auto'} "
            f"qwen_max_new_tokens={args.qwen_max_new_tokens}"
        )
    if args.with_tesseract:
        print(
            f"[summary] tesseract_lang={args.tesseract_lang} "
            f"tesseract_input_mode={args.tesseract_input_mode} "
            f"tesseract_max_pages_per_row={args.tesseract_max_pages_per_row or 'all'} "
            f"tesseract_max_images_per_row={args.tesseract_max_images_per_row}"
        )
    print(
        "[summary] note: this stays streaming-friendly because the sink only "
        "keeps a bounded Top-K heap. An exact global rank would be a blocking barrier."
    )

    print_top_k(sink.rows(), preview_chars=args.preview_chars)


if __name__ == "__main__":
    main()
