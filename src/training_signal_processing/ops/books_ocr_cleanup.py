from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .builtin import RowWiseMapperOp

IMAGE_MARKDOWN_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
PAGE_ASSET_PATTERN = re.compile(r"_page_\d+_[A-Za-z]+_\d+\.(?:jpeg|jpg|png)", re.IGNORECASE)
URL_DOI_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+|\bdoi\s*[:.]?\s*10\.\S+|\b10\.\d{4,9}/\S+",
    re.IGNORECASE,
)
REFERENCE_HEADING_PATTERN = re.compile(
    r"(?im)^\s{0,4}(?:#{1,6}\s*)?(?:\*\*)?\s*"
    r"(?:references|bibliography|works cited|rujukan|daftar pustaka)"
    r"\s*(?:\*\*)?\s*:?\s*(?:$|[-–—].*)"
)
INLINE_REFERENCE_HEADING_PATTERN = re.compile(
    r"(?im)^\s{0,4}(?:#{1,6}\s*)?(?:\*\*)?\s*"
    r"(?:references|bibliography|works cited|rujukan|daftar pustaka)"
    r"\b"
)
MIDLINE_MARKDOWN_REFERENCE_HEADING_PATTERN = re.compile(
    r"(?im)(?:^|\s)(?:#{1,6}\s+)(?:\*\*)?"
    r"(?:references|bibliography|works cited|rujukan|daftar pustaka)"
    r"\b"
)
CAPTION_PATTERN = re.compile(
    r"(?i)^\s*(?:\*\*)?\s*(?:figure|fig\.?|table|gambar|rajah|jadual)\s*\d+[\.:]?\b"
)
PUBLICATION_METADATA_PATTERN = re.compile(
    r"(?i)(?:"
    r"\b(?:issn|e-issn|isbn|corresponding author|to cite this article|copyright|"
    r"creative commons|all rights reserved)\b|"
    r"\b(?:published|received|accepted|tel\.?|faks|fax|e-?mail)\s*:|"
    r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b"
    r")"
)
STRONG_METADATA_PATTERN = re.compile(
    r"(?i)(?:to cite this article|all rights reserved|creative commons|"
    r"hak cipta|diperbolehkan.*memperbanyak|memperbanyak.*isi buku|izin penerbit)"
)
REFERENCE_LINE_PATTERN = re.compile(
    r"(?i)(?:\b(?:19|20)\d{2}\b.*(?:doi|https?://|www\.)|"
    r"(?:doi|https?://|www\.).*\b(?:19|20)\d{2}\b|"
    r"^\s*[-*]?\s*(?:\[\d+\]|\d+\.|\w[\w.-]+,\s+[A-Z]).{20,})"
)


@dataclass
class BooksOcrCleanupResult:
    cleaned_text: str
    removed_counts: Counter[str] = field(default_factory=Counter)
    removed_examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def removed_char_count(self) -> int:
        return sum(self.removed_counts.values())


class CleanBooksOcrMarkdownOp(RowWiseMapperOp):
    op_name = "clean_books_ocr_markdown"
    op_stage = "transform"

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        text_column = str(self.options.get("text_column", "markdown_text"))
        output_text_column = str(self.options.get("output_text_column", "cleaned_text"))
        if text_column not in row:
            raise ValueError(f"clean_books_ocr_markdown row missing text column '{text_column}'")
        result = clean_books_ocr_markdown(str(row.get(text_column) or ""))
        output = dict(row)
        output[output_text_column] = result.cleaned_text
        output["books_ocr_cleanup_removed_char_count"] = result.removed_char_count
        output["books_ocr_cleanup_removed_counts"] = dict(result.removed_counts)
        output["books_ocr_cleanup_removed_examples"] = {
            key: list(values) for key, values in result.removed_examples.items()
        }
        return output


def clean_books_ocr_markdown(text: str) -> BooksOcrCleanupResult:
    """Remove high-recall, low-information Books + OCR markdown artifacts.

    The cleaner is intentionally line-oriented after stripping inline image and URL
    fragments. That gives high recall for conversion residue while preserving the
    surrounding prose when a noisy token appears inside an otherwise useful line.
    """

    examples: dict[str, list[str]] = {}
    counts: Counter[str] = Counter()

    working = _remove_inline_pattern(
        text,
        pattern=IMAGE_MARKDOWN_PATTERN,
        label="markdown_image_link",
        counts=counts,
        examples=examples,
    )
    working = _remove_inline_pattern(
        working,
        pattern=PAGE_ASSET_PATTERN,
        label="page_asset_reference",
        counts=counts,
        examples=examples,
    )
    working = _trim_reference_tail(working, counts=counts, examples=examples)
    working = _clean_lines(working, counts=counts, examples=examples)
    working = re.sub(r"\n{3,}", "\n\n", working).strip()
    return BooksOcrCleanupResult(
        cleaned_text=working,
        removed_counts=counts,
        removed_examples=examples,
    )


def _remove_inline_pattern(
    text: str,
    *,
    pattern: re.Pattern[str],
    label: str,
    counts: Counter[str],
    examples: dict[str, list[str]],
) -> str:
    def replace(match: re.Match[str]) -> str:
        removed = match.group(0)
        counts[label] += len(removed)
        _add_example(examples, label, removed)
        return ""

    return pattern.sub(replace, text)


def _trim_reference_tail(
    text: str,
    *,
    counts: Counter[str],
    examples: dict[str, list[str]],
) -> str:
    matches = list(REFERENCE_HEADING_PATTERN.finditer(text))
    matches.extend(INLINE_REFERENCE_HEADING_PATTERN.finditer(text))
    matches.extend(MIDLINE_MARKDOWN_REFERENCE_HEADING_PATTERN.finditer(text))
    for match in reversed(matches):
        before = text[: match.start()].rstrip()
        after = text[match.start() :]
        if not _looks_like_reference_tail(after):
            continue
        removed = text[match.start() :]
        counts["reference_tail"] += len(removed)
        _add_example(examples, "reference_tail", removed)
        return before
    return text


def _looks_like_reference_tail(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    sample = lines[:30]
    reference_like = sum(1 for line in sample if REFERENCE_LINE_PATTERN.search(line))
    return reference_like >= max(2, min(5, len(sample) // 3))


def _clean_lines(
    text: str,
    *,
    counts: Counter[str],
    examples: dict[str, list[str]],
) -> str:
    output: list[str] = []
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            output.append(line)
            continue
        previous_line = output[-1].strip() if output else ""
        if _is_reference_heading_line(stripped):
            _record_removed_line(stripped, "reference_heading_line", counts, examples)
            continue
        if _is_publication_metadata_line(stripped):
            _record_removed_line(stripped, "publication_contact_metadata", counts, examples)
            continue
        if _is_detached_caption_line(stripped, previous_line):
            _record_removed_line(stripped, "figure_table_caption", counts, examples)
            continue
        cleaned_line = _remove_url_doi_fragments(stripped, counts=counts, examples=examples)
        if cleaned_line:
            output.append(cleaned_line)
    return "\n".join(output)


def _is_reference_heading_line(line: str) -> bool:
    if not (
        REFERENCE_HEADING_PATTERN.search(line)
        or MIDLINE_MARKDOWN_REFERENCE_HEADING_PATTERN.search(line)
    ):
        return False
    # Avoid deleting table-of-contents fragments such as "Daftar Pustaka Glosarium".
    if "glosarium" in line.lower() and not re.search(r"#{1,6}", line):
        return False
    return True


def _is_publication_metadata_line(line: str) -> bool:
    if STRONG_METADATA_PATTERN.search(line):
        return True
    if not PUBLICATION_METADATA_PATTERN.search(line):
        return False
    word_count = len(re.findall(r"\w+", line))
    return word_count <= 35 or _metadata_density(line) >= 0.16


def _metadata_density(line: str) -> float:
    metadata_hits = len(PUBLICATION_METADATA_PATTERN.findall(line))
    return metadata_hits / max(1, len(re.findall(r"\w+", line)))


def _is_detached_caption_line(line: str, previous_line: str) -> bool:
    if not CAPTION_PATTERN.search(line):
        return False
    word_count = len(re.findall(r"\w+", line))
    previous_has_asset = PAGE_ASSET_PATTERN.search(previous_line) is not None
    return word_count <= 30 or not previous_line or previous_has_asset


def _remove_url_doi_fragments(
    line: str,
    *,
    counts: Counter[str],
    examples: dict[str, list[str]],
) -> str:
    matches = list(URL_DOI_PATTERN.finditer(line))
    if not matches:
        return line
    without_links = URL_DOI_PATTERN.sub("", line)
    without_links = re.sub(r"\s+", " ", without_links).strip()
    removed_chars = sum(len(match.group(0)) for match in matches)
    label = "url_or_doi"
    counts[label] += removed_chars
    for match in matches[:2]:
        _add_example(examples, label, match.group(0))
    if _is_link_dominated_line(line, without_links):
        counts["url_or_doi_line"] += len(without_links)
        _add_example(examples, "url_or_doi_line", line)
        return ""
    return without_links


def _is_link_dominated_line(original: str, without_links: str) -> bool:
    remaining_words = len(re.findall(r"\w+", without_links))
    link_chars = sum(len(match.group(0)) for match in URL_DOI_PATTERN.finditer(original))
    return remaining_words <= 4 or link_chars / max(1, len(original)) >= 0.45


def _record_removed_line(
    line: str,
    label: str,
    counts: Counter[str],
    examples: dict[str, list[str]],
) -> None:
    counts[label] += len(line)
    _add_example(examples, label, line)


def _add_example(examples: dict[str, list[str]], label: str, text: str) -> None:
    bucket = examples.setdefault(label, [])
    if len(bucket) >= 5:
        return
    compact = re.sub(r"\s+", " ", text).strip()
    bucket.append(compact[:220])
