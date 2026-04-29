from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ahocorasick
import polars as pl

BOOKS_OCR_TEXT_FIELD = "markdown_text"
LOWYAT_TEXT_FIELD = "body_text"
REDDIT_TEXT_FIELD = "body"
CARI_TEXT_FIELD = "body_text"
HPLT_TEXT_FIELD = "text"

SOURCE_CHOICES = (
    "books-ocr",
    "lowyat",
    "reddit-bolehland",
    "reddit-indonesia",
    "cari",
    "hplt-malay",
    "hplt-indonesia",
)

IMAGE_MARKDOWN_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
PAGE_ASSET_PATTERN = re.compile(r"_page_\d+_[A-Za-z]+_\d+\.(?:jpeg|jpg|png)", re.IGNORECASE)
URL_DOI_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+|\bdoi\s*[:.]?\s*10\.\S+|\b10\.\d{4,9}/\S+",
    re.IGNORECASE,
)
TRUNCATED_URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S*\.{2,}\S*", re.IGNORECASE)
REFERENCE_HEADING_PATTERN = re.compile(
    r"(?im)^\s{0,4}(?:#{1,6}\s*)?(?:\*\*)?\s*"
    r"(?:references|bibliography|works cited|rujukan|daftar pustaka)"
    r"\s*(?:\*\*)?\s*:?\s*(?:$|[-:\u2013\u2014].*)"
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
REFERENCE_LINE_PATTERN = re.compile(
    r"(?i)(?:\b(?:19|20)\d{2}\b.*(?:doi|https?://|www\.)|"
    r"(?:doi|https?://|www\.).*\b(?:19|20)\d{2}\b|"
    r"^\s*[-*]?\s*(?:\[\d+\]|\d+\.|\w[\w.-]+,\s+[A-Z]).{20,})"
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

LOWYAT_QUOTE_PATTERN = re.compile(r"(?is)\bQUOTE\s*\([^)]{0,160}\)")
FORUM_REPLIED_PREFIX_PATTERN = re.compile(
    r"(?is)^\s*[\w.~@+-]{2,40}\s+replied\s+at\s+"
    r"\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{2}\s+[AP]M\s+"
)
FORUM_EDIT_LINE_PATTERN = re.compile(
    r"(?i)^\s*(?:this post has been edited by|edited by)\b.{0,160}$"
)
FORUM_EDIT_INLINE_PATTERN = re.compile(
    r"(?i)\b(?:this post has been edited by|edited by)\b.{0,160}$"
)
SPOILER_LINE_PATTERN = re.compile(r"(?i)^\s*(?:show|hide)?\s*spoiler\s*:?\s*$")
ATTACHMENT_PATTERN = re.compile(r"(?is)<!--\s*IBF\.ATTACHMENT_[^>]*-->")
HTML_MEDIA_PATTERN = re.compile(
    r"(?is)<(?:iframe|script|video|source|embed|object|img)\b[^>]*(?:>.*?</(?:iframe|script|video|object)>|/?>)"
)
HTML_TAG_PATTERN = re.compile(r"(?s)<[^>]+>")
SMILEY_FILENAME_PATTERN = re.compile(
    r"(?i)\b(?:static/image/smiley/)?[\w-]+\.(?:gif|jpe?g|png|webp)\b"
)
BARE_MEDIA_TOKEN_PATTERN = re.compile(r"(?i)^\s*[\w.-]+\.(?:gif|jpe?g|png|webp|mp4)\s*$")

REDDIT_MEDIA_MARKDOWN_PATTERN = re.compile(r"(?is)!\[(?:gif|img|image|video)?\]\([^)]*\)")
REDDIT_DELETED_PATTERN = re.compile(r"(?i)^\s*\[(?:deleted|removed)\]\s*$")
REDDIT_MEDIA_URL_PATTERN = re.compile(
    r"(?i)^\s*(?:https?://)?(?:preview\.redd\.it|i\.redd\.it|v\.redd\.it|giphy\.com|"
    r"media\.giphy\.com|imgur\.com|i\.imgur\.com)/\S+\s*$"
)

BOILERPLATE_LINE_PATTERN = re.compile(
    r"(?i)^\s*(?:"
    r"home|menu|search|login|log in|sign in|sign up|register|subscribe|"
    r"privacy policy|terms(?: of service)?|cookie policy|accept cookies|"
    r"all rights reserved|copyright|advertisement|share this|read more|"
    r"previous|next|back to top|skip to content"
    r")\s*$"
)
COOKIE_LINE_PATTERN = re.compile(
    r"(?i)\b(?:cookies?|privacy policy|terms of service|accept all|manage consent)\b"
)
FIXED_BOILERPLATE_PHRASES = (
    "Sent from my iPhone",
    "Login to view this content",
    "Log in to view this content",
    "Register to view this content",
    "[embedded media]",
    "embedded media",
    "attachment unavailable",
)

FAST_IMAGE_MARKDOWN_PATTERN = r"!\[[^\]]*\]\([^)]*\)"
FAST_PAGE_ASSET_PATTERN = r"(?i)_page_\d+_[A-Za-z]+_\d+\.(?:jpeg|jpg|png)"
FAST_URL_DOI_PATTERN = (
    r"(?i)(?:https?://|www\.)\S+|\bdoi\s*[:.]?\s*10\.\S+|\b10\.\d{4,9}/\S+"
)
FAST_TRUNCATED_URL_PATTERN = r"(?i)\b(?:https?://|www\.)\S*\.{2,}\S*"
FAST_BOOKS_METADATA_LINE_PATTERN = (
    r"(?im)^[^\n]{0,160}(?:\b(?:issn|e-issn|isbn|corresponding author|to cite this article|"
    r"copyright|creative commons|all rights reserved)\b|"
    r"\b(?:published|received|accepted|tel\.?|faks|fax|e-?mail)\s*:|"
    r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b)[^\n]{0,160}$\n?"
)
FAST_BOOKS_STRONG_METADATA_LINE_PATTERN = (
    r"(?im)^.*(?:to cite this article|all rights reserved|creative commons|hak cipta|"
    r"diperbolehkan.*memperbanyak|memperbanyak.*isi buku|izin penerbit).*$\n?"
)
FAST_BOOKS_CAPTION_LINE_PATTERN = (
    r"(?im)^\s*(?:\*\*)?\s*(?:figure|fig\.?|table|gambar|rajah|jadual)"
    r"\s*\d+[\.:]?[^\n]{0,220}$\n?"
)
FAST_BOOKS_REFERENCE_HEADING_LINE_PATTERN = (
    r"(?im)^\s{0,4}(?:#{1,6}\s*)?(?:\*\*)?\s*"
    r"(?:references|bibliography|works cited|rujukan|daftar pustaka)"
    r"[^\n]*\n?"
)
FAST_FORUM_EDIT_LINE_PATTERN = (
    r"(?im)^\s*(?:this post has been edited by|edited by)\b[^\n]*\n?"
)
FAST_SPOILER_LINE_PATTERN = r"(?im)^\s*(?:show|hide)?\s*spoiler\s*:?\s*$\n?"
FAST_REDDIT_DELETED_LINE_PATTERN = r"(?im)^\s*\[(?:deleted|removed)\]\s*$\n?"
FAST_BARE_MEDIA_LINE_PATTERN = r"(?im)^\s*[\w.-]+\.(?:gif|jpe?g|png|webp|mp4)\s*$\n?"
FAST_REDDIT_MEDIA_URL_LINE_PATTERN = (
    r"(?im)^\s*(?:https?://)?(?:preview\.redd\.it|i\.redd\.it|v\.redd\.it|giphy\.com|"
    r"media\.giphy\.com|imgur\.com|i\.imgur\.com)/\S+\s*$\n?"
)
FAST_BOILERPLATE_LINE_PATTERN = (
    r"(?im)^\s*(?:home|menu|search|login|log in|sign in|sign up|register|subscribe|"
    r"privacy policy|terms(?: of service)?|cookie policy|accept cookies|"
    r"all rights reserved|copyright|advertisement|share this|read more|"
    r"previous|next|back to top|skip to content)\s*$\n?"
)
FAST_COOKIE_LINE_PATTERN = (
    r"(?im)^.{0,200}\b(?:cookies?|privacy policy|terms of service|accept all|"
    r"manage consent)\b.{0,200}$\n?"
)
FAST_HTML_MEDIA_PATTERN = (
    r"(?is)<(?:iframe|script|video|source|embed|object|img)\b[^>]*"
    r"(?:>.*?</(?:iframe|script|video|object)>|/?>)"
)
FAST_HTML_TAG_PATTERN = r"(?s)<[^>]+>"
FAST_SMILEY_FILENAME_PATTERN = r"(?i)\b(?:static/image/smiley/)?[\w-]+\.(?:gif|jpe?g|png|webp)\b"
FAST_WHITESPACE_TOKEN_PATTERN = r"\S+"


@dataclass
class CleanTextResult:
    cleaned_text: str
    dropped: bool
    rules_triggered: tuple[str, ...]
    original_char_count: int
    cleaned_char_count: int
    removed_char_count: int
    approximate_original_token_count: int
    approximate_cleaned_token_count: int
    approximate_removed_token_count: int
    rule_counts: dict[str, int] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        return {
            "dropped": self.dropped,
            "rules_triggered": list(self.rules_triggered),
            "original_char_count": self.original_char_count,
            "cleaned_char_count": self.cleaned_char_count,
            "removed_char_count": self.removed_char_count,
            "approximate_original_token_count": self.approximate_original_token_count,
            "approximate_cleaned_token_count": self.approximate_cleaned_token_count,
            "approximate_removed_token_count": self.approximate_removed_token_count,
            "rule_counts": dict(self.rule_counts),
        }


@dataclass
class CleaningState:
    text: str
    rule_counts: Counter[str] = field(default_factory=Counter)

    def add(self, rule: str, count: int = 1) -> None:
        self.rule_counts[rule] += count


@dataclass(frozen=True)
class SourceSpec:
    text_field: str
    cleaner: Callable[[str], CleanTextResult]


def _build_fixed_phrase_automaton() -> Any:
    automaton = ahocorasick.Automaton()
    for phrase in FIXED_BOILERPLATE_PHRASES:
        automaton.add_word(phrase.lower(), phrase)
    automaton.make_automaton()
    return automaton


FIXED_PHRASE_AUTOMATON = _build_fixed_phrase_automaton()


def clean_books_ocr_text(text: str) -> CleanTextResult:
    state = CleaningState(_as_text(text))
    state.text = _remove_inline_pattern(state, IMAGE_MARKDOWN_PATTERN, "books_ocr.markdown_image")
    state.text = _remove_inline_pattern(state, PAGE_ASSET_PATTERN, "books_ocr.page_asset")
    state.text = _trim_reference_tail(state, "books_ocr.reference_tail")
    state.text = _clean_lines(
        state,
        drop_line=lambda line, previous: _books_ocr_drop_line(state, line, previous),
        clean_line=lambda line: _remove_url_or_doi_fragments(state, line, "books_ocr.url_or_doi"),
    )
    return _finish_result(text, state, drop_reason=_drop_reason_for_noise_only(state.text))


def clean_lowyat_text(text: str) -> CleanTextResult:
    state = CleaningState(_decode_forum_text(text))
    state.text = _remove_inline_pattern(state, ATTACHMENT_PATTERN, "lowyat.attachment_placeholder")
    state.text = _remove_inline_pattern(state, HTML_MEDIA_PATTERN, "lowyat.media_html")
    state.text = _remove_inline_pattern(state, LOWYAT_QUOTE_PATTERN, "lowyat.quote_header")
    state.text = _remove_prefix_pattern(state, FORUM_REPLIED_PREFIX_PATTERN, "lowyat.quote_leadin")
    state.text = _clean_lines(
        state,
        drop_line=lambda line, previous: _forum_drop_line(state, line, previous, "lowyat"),
        clean_line=lambda line: _clean_forum_line(state, line, "lowyat"),
    )
    return _finish_result(text, state, drop_reason=_drop_reason_for_noise_only(state.text))


def clean_reddit_bolehland_text(text: str) -> CleanTextResult:
    return _clean_reddit_text(text, source_prefix="reddit_bolehland")


def clean_reddit_indonesia_text(text: str) -> CleanTextResult:
    return _clean_reddit_text(text, source_prefix="reddit_indonesia")


def clean_cari_text(text: str) -> CleanTextResult:
    state = CleaningState(_decode_forum_text(text))
    state.text = _remove_inline_pattern(state, HTML_MEDIA_PATTERN, "cari.media_html")
    state.text = _remove_prefix_pattern(state, FORUM_REPLIED_PREFIX_PATTERN, "cari.quote_leadin")
    state.text = _clean_lines(
        state,
        drop_line=lambda line, previous: _forum_drop_line(state, line, previous, "cari"),
        clean_line=lambda line: _clean_forum_line(state, line, "cari"),
    )
    return _finish_result(text, state, drop_reason=_drop_reason_for_noise_only(state.text))


def clean_hplt_malay_text(text: str) -> CleanTextResult:
    return _clean_hplt_text(text, source_prefix="hplt_malay")


def clean_hplt_indonesia_text(text: str) -> CleanTextResult:
    return _clean_hplt_text(text, source_prefix="hplt_indonesia")


def clean_books_ocr_record(
    record: Mapping[str, Any],
    *,
    keep_metadata: bool = True,
) -> dict[str, Any] | None:
    return _clean_record(record, SOURCE_SPECS["books-ocr"], "books-ocr", keep_metadata)


def clean_lowyat_record(
    record: Mapping[str, Any],
    *,
    keep_metadata: bool = True,
) -> dict[str, Any] | None:
    return _clean_record(record, SOURCE_SPECS["lowyat"], "lowyat", keep_metadata)


def clean_reddit_bolehland_record(
    record: Mapping[str, Any],
    *,
    keep_metadata: bool = True,
) -> dict[str, Any] | None:
    return _clean_record(
        record,
        SOURCE_SPECS["reddit-bolehland"],
        "reddit-bolehland",
        keep_metadata,
    )


def clean_reddit_indonesia_record(
    record: Mapping[str, Any],
    *,
    keep_metadata: bool = True,
) -> dict[str, Any] | None:
    return _clean_record(
        record,
        SOURCE_SPECS["reddit-indonesia"],
        "reddit-indonesia",
        keep_metadata,
    )


def clean_cari_record(
    record: Mapping[str, Any],
    *,
    keep_metadata: bool = True,
) -> dict[str, Any] | None:
    return _clean_record(record, SOURCE_SPECS["cari"], "cari", keep_metadata)


def clean_hplt_malay_record(
    record: Mapping[str, Any],
    *,
    keep_metadata: bool = True,
) -> dict[str, Any] | None:
    return _clean_record(record, SOURCE_SPECS["hplt-malay"], "hplt-malay", keep_metadata)


def clean_hplt_indonesia_record(
    record: Mapping[str, Any],
    *,
    keep_metadata: bool = True,
) -> dict[str, Any] | None:
    return _clean_record(record, SOURCE_SPECS["hplt-indonesia"], "hplt-indonesia", keep_metadata)


def clean_record_for_source(
    record: Mapping[str, Any],
    *,
    source: str,
    keep_metadata: bool = True,
) -> dict[str, Any] | None:
    normalized = _normalize_source(source)
    return _clean_record(record, SOURCE_SPECS[normalized], normalized, keep_metadata)


def _clean_reddit_text(text: str, *, source_prefix: str) -> CleanTextResult:
    state = CleaningState(_as_text(text))
    state.text = _remove_inline_pattern(
        state,
        REDDIT_MEDIA_MARKDOWN_PATTERN,
        f"{source_prefix}.media_markdown",
    )
    state.text = _clean_lines(
        state,
        drop_line=lambda line, previous: _reddit_drop_line(state, line, previous, source_prefix),
        clean_line=lambda line: _remove_url_or_doi_fragments(state, line, f"{source_prefix}.url"),
    )
    return _finish_result(text, state, drop_reason=_drop_reason_for_noise_only(state.text))


def _clean_hplt_text(text: str, *, source_prefix: str) -> CleanTextResult:
    state = CleaningState(_decode_forum_text(text))
    state.text = _remove_inline_pattern(state, HTML_MEDIA_PATTERN, f"{source_prefix}.media_html")
    state.text = _remove_inline_pattern(state, HTML_TAG_PATTERN, f"{source_prefix}.html_tag")
    state.text = _clean_lines(
        state,
        drop_line=lambda line, previous: _hplt_drop_line(state, line, previous, source_prefix),
        clean_line=lambda line: _remove_url_or_doi_fragments(state, line, f"{source_prefix}.url"),
    )
    return _finish_result(text, state, drop_reason=_drop_reason_for_noise_only(state.text))


def _books_ocr_drop_line(state: CleaningState, line: str, previous: str) -> str | None:
    if _is_reference_heading_line(line):
        return "books_ocr.reference_heading"
    if _is_publication_metadata_line(line):
        return "books_ocr.publication_metadata"
    if _is_detached_caption_line(line, previous):
        return "books_ocr.figure_table_caption"
    if BARE_MEDIA_TOKEN_PATTERN.fullmatch(line):
        return "books_ocr.media_token_line"
    return None


def _forum_drop_line(
    state: CleaningState,
    line: str,
    previous: str,
    source_prefix: str,
) -> str | None:
    del state, previous
    if _has_fixed_boilerplate_phrase(line):
        return f"{source_prefix}.fixed_boilerplate"
    if FORUM_EDIT_LINE_PATTERN.match(line):
        return f"{source_prefix}.edit_boilerplate"
    if SPOILER_LINE_PATTERN.match(line):
        return f"{source_prefix}.spoiler_label"
    if BARE_MEDIA_TOKEN_PATTERN.fullmatch(line) or SMILEY_FILENAME_PATTERN.fullmatch(line):
        return f"{source_prefix}.media_token_line"
    if REDDIT_MEDIA_URL_PATTERN.match(line):
        return f"{source_prefix}.media_url_line"
    if _is_link_only_or_link_dominated(line):
        return f"{source_prefix}.url_line"
    return None


def _reddit_drop_line(
    state: CleaningState,
    line: str,
    previous: str,
    source_prefix: str,
) -> str | None:
    del state, previous
    if _has_fixed_boilerplate_phrase(line):
        return f"{source_prefix}.fixed_boilerplate"
    if REDDIT_DELETED_PATTERN.match(line):
        return f"{source_prefix}.deleted_placeholder"
    if REDDIT_MEDIA_URL_PATTERN.match(line):
        return f"{source_prefix}.media_url_line"
    if BARE_MEDIA_TOKEN_PATTERN.fullmatch(line):
        return f"{source_prefix}.media_token_line"
    if _is_link_only_or_link_dominated(line):
        return f"{source_prefix}.url_line"
    return None


def _hplt_drop_line(
    state: CleaningState,
    line: str,
    previous: str,
    source_prefix: str,
) -> str | None:
    del state, previous
    if _has_fixed_boilerplate_phrase(line):
        return f"{source_prefix}.fixed_boilerplate"
    if BOILERPLATE_LINE_PATTERN.match(line):
        return f"{source_prefix}.boilerplate_line"
    if _is_cookie_boilerplate(line):
        return f"{source_prefix}.cookie_boilerplate"
    if BARE_MEDIA_TOKEN_PATTERN.fullmatch(line):
        return f"{source_prefix}.media_token_line"
    if _is_link_only_or_link_dominated(line):
        return f"{source_prefix}.url_line"
    return None


def _has_fixed_boilerplate_phrase(line: str) -> bool:
    if len(line.split()) > 20:
        return False
    lowered = line.lower()
    return any(True for _ in FIXED_PHRASE_AUTOMATON.iter(lowered))


def _clean_forum_line(state: CleaningState, line: str, source_prefix: str) -> str:
    cleaned = _remove_inline_pattern_from_line(
        state,
        line,
        FORUM_EDIT_INLINE_PATTERN,
        f"{source_prefix}.edit_boilerplate",
    )
    cleaned = _remove_inline_pattern_from_line(
        state,
        cleaned,
        SMILEY_FILENAME_PATTERN,
        f"{source_prefix}.smiley_or_media_filename",
    )
    cleaned = _remove_url_or_doi_fragments(state, cleaned, f"{source_prefix}.url")
    return cleaned


def _remove_inline_pattern(state: CleaningState, pattern: re.Pattern[str], rule: str) -> str:
    matches = list(pattern.finditer(state.text))
    if not matches:
        return state.text
    state.add(rule, len(matches))
    return pattern.sub("", state.text)


def _remove_prefix_pattern(state: CleaningState, pattern: re.Pattern[str], rule: str) -> str:
    cleaned, count = pattern.subn("", state.text, count=1)
    if count:
        state.add(rule, count)
    return cleaned


def _remove_inline_pattern_from_line(
    state: CleaningState,
    line: str,
    pattern: re.Pattern[str],
    rule: str,
) -> str:
    cleaned, count = pattern.subn("", line)
    if count:
        state.add(rule, count)
    return re.sub(r"\s+", " ", cleaned).strip()


def _trim_reference_tail(state: CleaningState, rule: str) -> str:
    matches = list(REFERENCE_HEADING_PATTERN.finditer(state.text))
    matches.extend(INLINE_REFERENCE_HEADING_PATTERN.finditer(state.text))
    matches.extend(MIDLINE_MARKDOWN_REFERENCE_HEADING_PATTERN.finditer(state.text))
    for match in reversed(matches):
        tail = state.text[match.start() :]
        if _looks_like_reference_tail(tail):
            state.add(rule)
            return state.text[: match.start()].rstrip()
    return state.text


def _looks_like_reference_tail(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    sample = lines[:30]
    reference_like = sum(1 for line in sample if REFERENCE_LINE_PATTERN.search(line))
    return reference_like >= max(2, min(5, len(sample) // 3))


def _clean_lines(
    state: CleaningState,
    *,
    drop_line: Callable[[str, str], str | None],
    clean_line: Callable[[str], str],
) -> str:
    output: list[str] = []
    for raw_line in state.text.splitlines():
        line = raw_line.strip()
        if not line:
            output.append("")
            continue
        previous = output[-1].strip() if output else ""
        drop_rule = drop_line(line, previous)
        if drop_rule is not None:
            state.add(drop_rule)
            continue
        cleaned = clean_line(line)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            output.append(cleaned)
    return _normalize_blank_lines("\n".join(output))


def _remove_url_or_doi_fragments(state: CleaningState, line: str, rule: str) -> str:
    patterns = (URL_DOI_PATTERN, TRUNCATED_URL_PATTERN)
    cleaned = line
    total_matches = 0
    for pattern in patterns:
        cleaned, count = pattern.subn("", cleaned)
        total_matches += count
    if total_matches:
        state.add(rule, total_matches)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if total_matches and _is_link_dominated_after_removal(line, cleaned):
        state.add(f"{rule}_line")
        return ""
    return cleaned


def _is_reference_heading_line(line: str) -> bool:
    if not (
        REFERENCE_HEADING_PATTERN.search(line)
        or MIDLINE_MARKDOWN_REFERENCE_HEADING_PATTERN.search(line)
    ):
        return False
    return "glosarium" not in line.lower() or bool(re.search(r"#{1,6}", line))


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


def _is_link_only_or_link_dominated(line: str) -> bool:
    without_links = URL_DOI_PATTERN.sub("", TRUNCATED_URL_PATTERN.sub("", line)).strip()
    return _is_link_dominated_after_removal(line, without_links)


def _is_link_dominated_after_removal(original: str, without_links: str) -> bool:
    remaining_words = len(re.findall(r"\w+", without_links))
    link_chars = sum(len(match.group(0)) for match in URL_DOI_PATTERN.finditer(original))
    link_chars += sum(len(match.group(0)) for match in TRUNCATED_URL_PATTERN.finditer(original))
    if link_chars == 0:
        return False
    if remaining_words == 0:
        return True
    link_ratio = link_chars / max(1, len(original))
    if remaining_words <= 2 and link_ratio >= 0.45:
        return True
    return link_ratio >= 0.75


def _is_cookie_boilerplate(line: str) -> bool:
    if not COOKIE_LINE_PATTERN.search(line):
        return False
    word_count = len(re.findall(r"\w+", line))
    return word_count <= 40


def _drop_reason_for_noise_only(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return "drop.empty_after_cleaning"
    if _looks_like_markup_only(stripped):
        return "drop.markup_only"
    if _looks_like_punctuation_only(stripped):
        return "drop.punctuation_or_symbol_only"
    if _is_link_only_or_link_dominated(stripped):
        return "drop.url_only"
    if not re.search(r"\w", stripped):
        return "drop.no_alphanumeric_content"
    return None


def _looks_like_markup_only(text: str) -> bool:
    if not re.search(r"[<>/&;]", text):
        return False
    without_tags = HTML_TAG_PATTERN.sub("", text)
    without_entities = re.sub(r"&[A-Za-z0-9#]+;", "", without_tags)
    return not re.search(r"\w{2,}", without_entities)


def _looks_like_punctuation_only(text: str) -> bool:
    alnum_count = len(re.findall(r"\w", text))
    return alnum_count == 0 or alnum_count / max(1, len(text)) < 0.12


def _finish_result(
    original_text: Any,
    state: CleaningState,
    *,
    drop_reason: str | None,
) -> CleanTextResult:
    original = _as_text(original_text)
    cleaned = _normalize_blank_lines(state.text)
    if drop_reason:
        state.add(drop_reason)
    final_text = "" if drop_reason else cleaned
    original_tokens = _approximate_token_count(original)
    cleaned_tokens = _approximate_token_count(final_text)
    removed_chars = max(0, len(original) - len(final_text))
    removed_tokens = max(0, original_tokens - cleaned_tokens)
    rule_counts = dict(sorted(state.rule_counts.items()))
    return CleanTextResult(
        cleaned_text=final_text,
        dropped=drop_reason is not None,
        rules_triggered=tuple(rule_counts),
        original_char_count=len(original),
        cleaned_char_count=len(final_text),
        removed_char_count=removed_chars,
        approximate_original_token_count=original_tokens,
        approximate_cleaned_token_count=cleaned_tokens,
        approximate_removed_token_count=removed_tokens,
        rule_counts=rule_counts,
    )


def _clean_record(
    record: Mapping[str, Any],
    spec: SourceSpec,
    source: str,
    keep_metadata: bool,
) -> dict[str, Any] | None:
    if spec.text_field not in record:
        raise ValueError(f"source {source!r} record is missing text field {spec.text_field!r}")
    result = spec.cleaner(_as_text(record.get(spec.text_field)))
    if result.dropped:
        return None
    output = dict(record)
    output[spec.text_field] = result.cleaned_text
    if keep_metadata:
        output["_cleaning"] = {"source": source, **result.metadata()}
    return output


def _as_text(value: Any) -> str:
    return "" if value is None else str(value)


def _decode_forum_text(value: Any) -> str:
    text = html.unescape(_as_text(value))
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = HTML_TAG_PATTERN.sub(" ", text)
    return re.sub(r"[ \t]+", " ", text)


def _normalize_blank_lines(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _approximate_token_count(text: str) -> int:
    return len(text.split())


def _normalize_source(source: str) -> str:
    normalized = source.strip().lower().replace("_", "-")
    aliases = {
        "books": "books-ocr",
        "books+ocr": "books-ocr",
        "books + ocr": "books-ocr",
        "reddit-bolehland": "reddit-bolehland",
        "reddit-indonesia": "reddit-indonesia",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SOURCE_SPECS:
        choices = ", ".join(SOURCE_CHOICES)
        raise ValueError(f"unknown source {source!r}; expected one of: {choices}")
    return normalized


SOURCE_SPECS: dict[str, SourceSpec] = {
    "books-ocr": SourceSpec(BOOKS_OCR_TEXT_FIELD, clean_books_ocr_text),
    "lowyat": SourceSpec(LOWYAT_TEXT_FIELD, clean_lowyat_text),
    "reddit-bolehland": SourceSpec(REDDIT_TEXT_FIELD, clean_reddit_bolehland_text),
    "reddit-indonesia": SourceSpec(REDDIT_TEXT_FIELD, clean_reddit_indonesia_text),
    "cari": SourceSpec(CARI_TEXT_FIELD, clean_cari_text),
    "hplt-malay": SourceSpec(HPLT_TEXT_FIELD, clean_hplt_malay_text),
    "hplt-indonesia": SourceSpec(HPLT_TEXT_FIELD, clean_hplt_indonesia_text),
}


def clean_polars_frame_for_source(
    frame: Any,
    *,
    source: str,
    keep_metadata: bool = False,
    keep_dropped: bool = False,
) -> tuple[Any, dict[str, Any]]:
    normalized_source = _normalize_source(source)
    spec = SOURCE_SPECS[normalized_source]
    if isinstance(frame, pl.LazyFrame):
        frame_columns = frame.collect_schema().names()
    else:
        frame_columns = frame.columns
    if spec.text_field not in frame_columns:
        raise ValueError(
            f"source {normalized_source!r} frame is missing text field {spec.text_field!r}"
        )

    original_col = "__cleaning_original_text"
    pre_tail_col = "__cleaning_pre_tail_text"
    tail_trimmed_col = "__cleaning_tail_trimmed_text"
    cleaned_col = "__cleaning_cleaned_text"
    final_col = "__cleaning_final_text"
    dropped_col = "__cleaning_dropped"

    frame = frame.with_columns(
        pl.col(spec.text_field).cast(pl.String).fill_null("").alias(original_col)
    )
    cleaned_expr = _polars_clean_expr(pl.col(original_col), normalized_source)
    frame = frame.with_columns(cleaned_expr.alias(pre_tail_col))
    if normalized_source == "books-ocr":
        frame = frame.with_columns(
            pl.col(pre_tail_col)
            .map_elements(_trim_reference_tail_text, return_dtype=pl.String)
            .alias(tail_trimmed_col)
        )
        books_expr = pl.col(tail_trimmed_col)
        books_expr = books_expr.str.replace_all(FAST_BOOKS_STRONG_METADATA_LINE_PATTERN, "\n")
        books_expr = books_expr.str.replace_all(FAST_BOOKS_METADATA_LINE_PATTERN, "\n")
        books_expr = books_expr.str.replace_all(FAST_BOOKS_CAPTION_LINE_PATTERN, "\n")
        books_expr = _polars_remove_urls(books_expr)
        frame = frame.with_columns(books_expr.alias(cleaned_col))
    else:
        frame = frame.with_columns(
            pl.col(pre_tail_col).alias(tail_trimmed_col),
            pl.col(pre_tail_col).alias(cleaned_col),
        )
    frame = frame.with_columns(
        _polars_normalize_expr(pl.col(cleaned_col)).alias(cleaned_col)
    ).with_columns(_polars_drop_expr(pl.col(cleaned_col)).alias(dropped_col))
    frame = frame.with_columns(
        pl.when(pl.col(dropped_col))
        .then(pl.lit(""))
        .otherwise(pl.col(cleaned_col))
        .alias(final_col)
    )
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()

    metrics = _polars_metrics(frame, normalized_source, spec.text_field)
    output = frame if keep_dropped else frame.filter(~pl.col(dropped_col))
    output = output.with_columns(
        pl.col(final_col).alias(spec.text_field)
    )
    if keep_metadata:
        rules_triggered = _polars_rules_triggered_expr(
            normalized_source,
            pl.col(original_col),
            pl.col(pre_tail_col),
            pl.col(tail_trimmed_col),
        )
        output = output.with_columns(
            pl.struct(
                pl.lit(normalized_source).alias("source"),
                pl.col(dropped_col).alias("dropped"),
                rules_triggered.alias("rules_triggered"),
                pl.col(original_col).str.len_chars().alias("original_char_count"),
                pl.col(final_col).str.len_chars().alias("cleaned_char_count"),
                (
                    pl.col(original_col).str.len_chars()
                    - pl.col(final_col).str.len_chars()
                ).alias("removed_char_count"),
                pl.col(original_col)
                .str.count_matches(FAST_WHITESPACE_TOKEN_PATTERN)
                .alias("approximate_original_token_count"),
                pl.col(final_col)
                .str.count_matches(FAST_WHITESPACE_TOKEN_PATTERN)
                .alias("approximate_cleaned_token_count"),
                (
                    pl.col(original_col).str.count_matches(FAST_WHITESPACE_TOKEN_PATTERN)
                    - pl.col(final_col).str.count_matches(FAST_WHITESPACE_TOKEN_PATTERN)
                ).alias("approximate_removed_token_count"),
            ).alias("_cleaning")
        )
    output = output.drop(
        [original_col, pre_tail_col, tail_trimmed_col, cleaned_col, final_col, dropped_col]
    )
    return output, metrics


def clean_file_with_polars(
    *,
    source: str,
    input_path: Path,
    output_path: Path,
    metrics_path: Path,
    keep_metadata: bool = False,
) -> dict[str, Any]:
    if input_path.suffix == ".parquet":
        frame = pl.scan_parquet(input_path)
    elif input_path.suffix in {".jsonl", ".ndjson"}:
        frame = pl.scan_ndjson(input_path)
    else:
        raise ValueError(
            f"unsupported input format {input_path.suffix!r}; expected .jsonl, .ndjson, or .parquet"
        )
    output, metrics = clean_polars_frame_for_source(
        frame,
        source=source,
        keep_metadata=keep_metadata,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        output.write_parquet(output_path)
    else:
        output.write_ndjson(output_path)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def _polars_clean_expr(text_expr: Any, source: str) -> Any:
    expr = text_expr
    if source == "books-ocr":
        expr = expr.str.replace_all(FAST_IMAGE_MARKDOWN_PATTERN, "")
        expr = expr.str.replace_all(FAST_PAGE_ASSET_PATTERN, "")
    elif source == "lowyat":
        expr = _polars_decode_forum_expr(expr)
        expr = expr.str.replace_all(r"(?is)<!--\s*IBF\.ATTACHMENT_[^>]*-->", "")
        expr = expr.str.replace_all(FAST_HTML_MEDIA_PATTERN, "")
        expr = expr.str.replace_all(r"(?is)\bQUOTE\s*\([^)]{0,160}\)", "")
        expr = expr.str.replace_all(
            r"(?is)^\s*[\w.~@+-]{2,40}\s+replied\s+at\s+"
            r"\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{2}\s+[AP]M\s+",
            "",
        )
        expr = expr.str.replace_all(FAST_FORUM_EDIT_LINE_PATTERN, "\n")
        expr = expr.str.replace_all(FAST_SPOILER_LINE_PATTERN, "\n")
        expr = expr.str.replace_all(_fixed_phrase_line_pattern(), "\n")
        expr = expr.str.replace_all(FAST_SMILEY_FILENAME_PATTERN, "")
        expr = _polars_remove_media_and_urls(expr)
    elif source in {"reddit-bolehland", "reddit-indonesia"}:
        expr = expr.str.replace_all(r"(?is)!\[(?:gif|img|image|video)?\]\([^)]*\)", "")
        expr = expr.str.replace_all(FAST_REDDIT_DELETED_LINE_PATTERN, "\n")
        expr = expr.str.replace_all(_fixed_phrase_line_pattern(), "\n")
        expr = _polars_remove_media_and_urls(expr)
    elif source == "cari":
        expr = _polars_decode_forum_expr(expr)
        expr = expr.str.replace_all(FAST_HTML_MEDIA_PATTERN, "")
        expr = expr.str.replace_all(
            r"(?is)^\s*[\w.~@+-]{2,40}\s+replied\s+at\s+"
            r"\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{2}\s+[AP]M\s+",
            "",
        )
        expr = expr.str.replace_all(FAST_FORUM_EDIT_LINE_PATTERN, "\n")
        expr = expr.str.replace_all(_fixed_phrase_line_pattern(), "\n")
        expr = expr.str.replace_all(FAST_SMILEY_FILENAME_PATTERN, "")
        expr = _polars_remove_media_and_urls(expr)
    elif source in {"hplt-malay", "hplt-indonesia"}:
        expr = _polars_decode_forum_expr(expr)
        expr = expr.str.replace_all(FAST_HTML_MEDIA_PATTERN, "")
        expr = expr.str.replace_all(FAST_HTML_TAG_PATTERN, " ")
        expr = expr.str.replace_all(FAST_BOILERPLATE_LINE_PATTERN, "\n")
        expr = expr.str.replace_all(FAST_COOKIE_LINE_PATTERN, "\n")
        expr = expr.str.replace_all(_fixed_phrase_line_pattern(), "\n")
        expr = _polars_remove_media_and_urls(expr)
    else:
        raise ValueError(f"unsupported source {source!r}")
    return _polars_normalize_expr(expr.fill_null(pl.lit("")))


def _polars_decode_forum_expr(expr: Any) -> Any:
    return (
        expr.str.replace_all("&amp;", "&")
        .str.replace_all("&lt;", "<")
        .str.replace_all("&gt;", ">")
        .str.replace_all("&quot;", '"')
        .str.replace_all("&#39;", "'")
        .str.replace_all("(?i)<br\\s*/?>", "\n")
        .str.replace_all(FAST_HTML_TAG_PATTERN, " ")
    )


def _polars_remove_media_and_urls(expr: Any) -> Any:
    expr = expr.str.replace_all(FAST_REDDIT_MEDIA_URL_LINE_PATTERN, "\n")
    expr = expr.str.replace_all(FAST_BARE_MEDIA_LINE_PATTERN, "\n")
    return _polars_remove_urls(expr)


def _polars_remove_urls(expr: Any) -> Any:
    expr = expr.str.replace_all(FAST_URL_DOI_PATTERN, "")
    return expr.str.replace_all(FAST_TRUNCATED_URL_PATTERN, "")


def _polars_normalize_expr(expr: Any) -> Any:
    return (
        expr.str.replace_all(r"[ \t]+\n", "\n")
        .str.replace_all(r"[ \t]{2,}", " ")
        .str.replace_all(r"\n{3,}", "\n\n")
        .str.strip_chars()
    )


def _polars_drop_expr(text_expr: Any) -> Any:
    stripped = text_expr.str.strip_chars()
    word_count = stripped.str.count_matches(r"\w")
    url_chars_removed = (
        stripped.str.len_chars()
        - stripped.str.replace_all(FAST_URL_DOI_PATTERN, "")
        .str.replace_all(FAST_TRUNCATED_URL_PATTERN, "")
        .str.len_chars()
    )
    without_urls = stripped.str.replace_all(FAST_URL_DOI_PATTERN, "").str.replace_all(
        FAST_TRUNCATED_URL_PATTERN,
        "",
    )
    remaining_url_words = without_urls.str.count_matches(r"\w+")
    url_only = (url_chars_removed > 0) & (
        (remaining_url_words == 0)
        | ((remaining_url_words <= 2) & (url_chars_removed / stripped.str.len_chars() >= 0.45))
        | (url_chars_removed / stripped.str.len_chars() >= 0.75)
    )
    punctuation_only = (word_count == 0) | (word_count / stripped.str.len_chars() < 0.12)
    return (stripped == "") | punctuation_only | url_only


def _polars_metrics(frame: Any, source: str, text_field: str) -> dict[str, Any]:
    original_col = "__cleaning_original_text"
    pre_tail_col = "__cleaning_pre_tail_text"
    tail_trimmed_col = "__cleaning_tail_trimmed_text"
    final_col = "__cleaning_final_text"
    dropped_col = "__cleaning_dropped"
    metric_exprs = [
        pl.len().alias("total_records"),
        (~pl.col(dropped_col)).sum().alias("records_written"),
        pl.col(dropped_col).sum().alias("records_dropped"),
        (pl.col(original_col) != pl.col(final_col)).sum().alias("records_modified"),
        pl.col(original_col).str.len_chars().sum().alias("original_total_characters"),
        pl.col(final_col).str.len_chars().sum().alias("cleaned_total_characters"),
        pl.col(original_col).str.count_matches(FAST_WHITESPACE_TOKEN_PATTERN)
        .sum()
        .alias("approximate_original_tokens"),
        pl.col(final_col).str.count_matches(FAST_WHITESPACE_TOKEN_PATTERN)
        .sum()
        .alias("approximate_cleaned_tokens"),
    ]
    for rule, expr in _polars_rule_count_exprs(
        source,
        pl.col(original_col),
        pl.col(pre_tail_col),
        pl.col(tail_trimmed_col),
    ):
        metric_exprs.append(expr.sum().alias(f"rule::{rule}"))
    summary = frame.select(metric_exprs).to_dicts()[0]
    original_chars = int(summary["original_total_characters"] or 0)
    cleaned_chars = int(summary["cleaned_total_characters"] or 0)
    original_tokens = int(summary["approximate_original_tokens"] or 0)
    cleaned_tokens = int(summary["approximate_cleaned_tokens"] or 0)
    rule_hit_counts = {
        key.removeprefix("rule::"): int(value or 0)
        for key, value in summary.items()
        if key.startswith("rule::") and value
    }
    removed_chars = max(0, original_chars - cleaned_chars)
    removed_tokens = max(0, original_tokens - cleaned_tokens)
    return {
        "engine": "polars",
        "source": source,
        "text_field": text_field,
        "total_records": int(summary["total_records"]),
        "records_written": int(summary["records_written"]),
        "records_dropped": int(summary["records_dropped"]),
        "records_modified": int(summary["records_modified"]),
        "original_total_characters": original_chars,
        "cleaned_total_characters": cleaned_chars,
        "removed_total_characters": removed_chars,
        "percent_characters_removed": _safe_percent(removed_chars, original_chars),
        "approximate_original_tokens": original_tokens,
        "approximate_cleaned_tokens": cleaned_tokens,
        "approximate_tokens_removed": removed_tokens,
        "percent_tokens_removed": _safe_percent(removed_tokens, original_tokens),
        "rule_hit_counts": dict(sorted(rule_hit_counts.items())),
    }


def _polars_rule_count_exprs(
    source: str,
    original: Any,
    pre_tail: Any,
    tail_trimmed: Any,
) -> list[tuple[str, Any]]:
    if source == "books-ocr":
        return [
            ("books_ocr.markdown_image", original.str.count_matches(FAST_IMAGE_MARKDOWN_PATTERN)),
            ("books_ocr.page_asset", original.str.count_matches(FAST_PAGE_ASSET_PATTERN)),
            ("books_ocr.url_or_doi", original.str.count_matches(FAST_URL_DOI_PATTERN)),
            (
                "books_ocr.publication_metadata",
                original.str.count_matches(FAST_BOOKS_METADATA_LINE_PATTERN),
            ),
            (
                "books_ocr.figure_table_caption",
                original.str.count_matches(FAST_BOOKS_CAPTION_LINE_PATTERN),
            ),
            (
                "books_ocr.reference_heading",
                original.str.count_matches(FAST_BOOKS_REFERENCE_HEADING_LINE_PATTERN),
            ),
            (
                "books_ocr.reference_tail",
                (pre_tail != tail_trimmed).cast(pl.UInt32),
            ),
        ]
    fixed_phrase_count = original.str.to_lowercase().str.count_matches(_fixed_phrase_pattern())
    if source in {"lowyat", "cari"}:
        prefix = source
        return [
            (f"{prefix}.url", original.str.count_matches(FAST_URL_DOI_PATTERN)),
            (
                f"{prefix}.edit_boilerplate",
                original.str.count_matches(FAST_FORUM_EDIT_LINE_PATTERN),
            ),
            (f"{prefix}.fixed_boilerplate", fixed_phrase_count),
            (
                f"{prefix}.smiley_or_media_filename",
                original.str.count_matches(FAST_SMILEY_FILENAME_PATTERN),
            ),
        ]
    if source in {"reddit-bolehland", "reddit-indonesia"}:
        prefix = source.replace("-", "_")
        return [
            (
                f"{prefix}.media_markdown",
                original.str.count_matches(r"(?is)!\[(?:gif|img|image|video)?\]\([^)]*\)"),
            ),
            (
                f"{prefix}.deleted_placeholder",
                original.str.count_matches(FAST_REDDIT_DELETED_LINE_PATTERN),
            ),
            (f"{prefix}.fixed_boilerplate", fixed_phrase_count),
            (
                f"{prefix}.media_url_line",
                original.str.count_matches(FAST_REDDIT_MEDIA_URL_LINE_PATTERN),
            ),
        ]
    prefix = source.replace("-", "_")
    return [
        (f"{prefix}.boilerplate_line", original.str.count_matches(FAST_BOILERPLATE_LINE_PATTERN)),
        (f"{prefix}.cookie_boilerplate", original.str.count_matches(FAST_COOKIE_LINE_PATTERN)),
        (f"{prefix}.fixed_boilerplate", fixed_phrase_count),
        (f"{prefix}.url", original.str.count_matches(FAST_URL_DOI_PATTERN)),
    ]


def _polars_rules_triggered_expr(
    source: str,
    original: Any,
    pre_tail: Any,
    tail_trimmed: Any,
) -> Any:
    triggered = [
        pl.when(expr > 0).then(pl.lit(rule)).otherwise(None)
        for rule, expr in _polars_rule_count_exprs(source, original, pre_tail, tail_trimmed)
    ]
    if not triggered:
        return pl.lit([], dtype=pl.List(pl.String))
    return pl.concat_list(triggered).list.drop_nulls()


def _trim_reference_tail_text(text: str) -> str:
    state = CleaningState(_as_text(text))
    return _trim_reference_tail(state, "books_ocr.reference_tail")


def _fixed_phrase_pattern() -> str:
    phrases = [re.escape(phrase.lower()) for phrase in FIXED_BOILERPLATE_PHRASES]
    return "|".join(phrases)


def _fixed_phrase_line_pattern() -> str:
    return rf"(?im)^.*(?:{_fixed_phrase_pattern()}).*$\n?"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean source-specific uninformative text.")
    parser.add_argument("--source", required=True, choices=SOURCE_CHOICES)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-path", required=True, type=Path)
    parser.add_argument("--keep-metadata", dest="keep_metadata", action="store_true", default=True)
    parser.add_argument("--no-keep-metadata", dest="keep_metadata", action="store_false")
    args = parser.parse_args(argv)

    metrics = clean_file_with_polars(
        source=args.source,
        input_path=args.input,
        output_path=args.output,
        metrics_path=args.metrics_path,
        keep_metadata=args.keep_metadata,
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


def _safe_percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 6)


if __name__ == "__main__":
    raise SystemExit(main())
