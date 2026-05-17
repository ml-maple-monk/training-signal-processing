import re
from collections import Counter

import regex

from .config import LanguageProfile

PUNCTUATION = (
    "!/—”:％１〈&(、━\\〖#%「」，〗；+^]~“《„';’{|∶´[=-`*．"
    "（–？！：$～«〉,><》)?）。…@_.\"}►»"
    + "".join(
        chr(x)
        for start, end in ((0, 9), (11, 13), (13, 32), (127, 160))
        for x in range(start, end)
    )
)
TERMINAL_PUNCTUATION = {
    ".",
    "!",
    "?",
    "。",
    "؟",
    "！",
    "？",
    "؛",
    "…",
    "।",
    "။",
    "።",
    "៖",
    "៕",
}
PUNCTUATION_SET = set(PUNCTUATION).union(TERMINAL_PUNCTUATION)
WORD_PATTERN = regex.compile(
    r"\p{L}[\p{L}\p{M}\p{Nd}_'-]*|\p{Nd}+|[^\s]",
    regex.VERSION1,
)


def split_into_words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text)


def get_n_grams(words: list[str], n: int) -> list[str]:
    return [" ".join(words[index : index + n]) for index in range(len(words) - n + 1)]


def find_duplicates(values: list[str]) -> tuple[int, int]:
    unique_values = set()
    duplicate_chars = 0
    duplicate_elements = 0
    for value in values:
        if value in unique_values:
            duplicate_chars += len(value)
            duplicate_elements += 1
        else:
            unique_values.add(value)
    return duplicate_elements, duplicate_chars


def find_top_duplicate(values: list[str]) -> int:
    top_n_gram = Counter(values).most_common(1)[0]
    return len(top_n_gram[0]) * top_n_gram[1]


def find_all_duplicate(words: list[str], n: int) -> int:
    n_words = len(words)
    unique = set()
    repeated_chars = 0
    index = 0
    while index < n_words - n + 1:
        n_gram = "".join(words[index : index + n])
        if n_gram in unique:
            repeated_chars += len(n_gram)
            index += n
        else:
            unique.add(n_gram)
            index += 1
    return repeated_chars


def compute_quality_metrics(
    text: str,
    profile: LanguageProfile,
    *,
    indonesian_stopwords: tuple[str, ...] | None = None,
) -> dict[str, float | int | bool | str]:
    paragraphs = re.split(r"\n{2,}", text.strip()) or [text]
    repeated_lines = re.split(r"\n+", text) or [text]
    non_empty_lines = [line for line in text.split("\n") if line.strip()]
    words = split_into_words(text)
    non_symbol_words = [
        word for word in words if any(char not in PUNCTUATION_SET for char in word)
    ]
    line_duplicate_count, _line_duplicate_chars = find_duplicates(repeated_lines)
    text_without_newlines = text.replace("\n", "")
    duplicated_line_character_ratio = find_duplicates(non_empty_lines)[1] / max(
        len(text_without_newlines),
        1,
    )
    metrics: dict[str, float | int | bool | str] = {
        "paragraph_count": len(paragraphs),
        "line_count": len(repeated_lines),
        "non_empty_line_count": len(non_empty_lines),
        "word_count": len(words),
        "text_char_count": len(text),
        "duplicate_line_count": line_duplicate_count,
        "duplicate_line_fraction": line_duplicate_count / max(len(repeated_lines), 1),
        "duplicated_line_character_ratio": duplicated_line_character_ratio,
        "newline_count": text.count("\n"),
        "newline_word_ratio": text.count("\n") / max(len(words), 1),
        "non_symbol_word_count": len(non_symbol_words),
        "average_non_symbol_word_length": (
            sum(len(word) for word in non_symbol_words) / len(non_symbol_words)
            if non_symbol_words
            else 0.0
        ),
        "hash_symbol_ratio": text.count("#") / max(len(words), 1),
        "ellipsis_ratio": (text.count("...") + text.count("…")) / max(len(words), 1),
        "alpha_word_ratio": (
            sum(any(char.isalpha() for char in word) for word in words) / max(len(words), 1)
        ),
        "language_profile": profile.name,
    }

    for n_value, _threshold in profile.top_n_grams:
        n_grams = get_n_grams(words, n_value)
        metrics[f"top_{n_value}gram_fraction"] = (
            find_top_duplicate(n_grams) / max(len(text), 1)
            if n_grams
            else 0.0
        )

    for n_value, _threshold in profile.dup_n_grams:
        metrics[f"duplicated_{n_value}gram_fraction"] = (
            find_all_duplicate(words, n_value) / max(len(text), 1)
        )

    metrics["line_punctuation_fraction"] = (
        sum(1 for line in non_empty_lines if line.endswith(tuple(TERMINAL_PUNCTUATION)))
        / len(non_empty_lines)
        if non_empty_lines
        else 0.0
    )

    split_lines = text.splitlines() or [text]
    metrics["bullet_line_ratio"] = sum(
        line.lstrip().startswith("•") or line.lstrip().startswith("-")
        for line in split_lines
    ) / len(split_lines)
    metrics["ending_ellipsis_line_ratio"] = sum(
        line.rstrip().endswith("...") or line.rstrip().endswith("…")
        for line in split_lines
    ) / len(split_lines)

    lower_words = {word.lower() for word in words}
    stopword_count = len(set(profile.stopwords).intersection(lower_words))
    indonesian_stopwords = indonesian_stopwords or (
        profile.stopwords if profile.name == "ind_Latn" else ()
    )
    indonesian_stopword_count = len(set(indonesian_stopwords).intersection(lower_words))
    metrics["stopword_count"] = stopword_count
    metrics["at_least_2_profile_stopwords_present"] = stopword_count >= 2
    metrics["indonesian_stopword_count"] = indonesian_stopword_count
    metrics["at_least_2_indonesian_stopwords_present"] = indonesian_stopword_count >= 2
    return metrics
