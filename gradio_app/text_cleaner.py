from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Sequence, Set

from .models import CleaningOptions


SMART_REPLACEMENTS = {
    "“": '"',
    "”": '"',
    "„": '"',
    "«": '"',
    "»": '"',
    "‹": '"',
    "›": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "–": "-",
    "—": "-",
    "−": "-",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "…": "...",
    "•": "-",
    "·": "-",
}

SMART_PUNCTUATION_RE = re.compile(r"[“”„«»‹›‘’‚‛–—−‐‑‒…•·]")
NBSP_RE = re.compile(r"\u00a0")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE)
BRACKET_REFERENCE_RE = re.compile(
    r"\[\s*(?:[0-9]+|[ivxlcdmIVXLCDM]+)(?:[\s,.;:-]*(?:[0-9]+|[ivxlcdmIVXLCDM]+))*\s*\]"
)
PAREN_FOOTNOTE_RE = re.compile(
    r"\(\s*(?:[0-9]+|[ivxlcdmIVXLCDM]+)(?:[\s,.;:-]*(?:[0-9]+|[ivxlcdmIVXLCDM]+))*\s*\)"
)
CAMEL_CASE_SPLIT_RE = re.compile(r"([a-z])([A-Z][a-z]+)")
MERGED_WORD_RE = re.compile(r"\b[a-zA-Z]{6,}\b")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
LEADING_SPACE_NEWLINE_RE = re.compile(r"[ \t]+\n")
TRAILING_SPACE_NEWLINE_RE = re.compile(r"\n[ \t]+")
EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")

FALLBACK_WORDS: Sequence[str] = (
    "a",
    "able",
    "about",
    "above",
    "after",
    "again",
    "against",
    "air",
    "all",
    "along",
    "also",
    "always",
    "among",
    "an",
    "and",
    "another",
    "any",
    "anything",
    "are",
    "around",
    "as",
    "ask",
    "at",
    "away",
    "back",
    "be",
    "because",
    "become",
    "been",
    "before",
    "began",
    "behind",
    "being",
    "below",
    "between",
    "both",
    "brought",
    "but",
    "by",
    "call",
    "called",
    "can",
    "cannot",
    "come",
    "could",
    "day",
    "did",
    "didn",
    "do",
    "does",
    "done",
    "down",
    "each",
    "end",
    "enough",
    "even",
    "ever",
    "every",
    "face",
    "fact",
    "far",
    "feel",
    "felt",
    "few",
    "find",
    "first",
    "for",
    "form",
    "found",
    "from",
    "gave",
    "get",
    "give",
    "given",
    "go",
    "good",
    "got",
    "great",
    "had",
    "half",
    "hand",
    "has",
    "have",
    "having",
    "he",
    "head",
    "hear",
    "heard",
    "help",
    "her",
    "here",
    "high",
    "him",
    "his",
    "home",
    "house",
    "how",
    "however",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "keep",
    "knew",
    "know",
    "known",
    "land",
    "large",
    "last",
    "later",
    "least",
    "leave",
    "left",
    "let",
    "life",
    "like",
    "little",
    "long",
    "look",
    "looked",
    "made",
    "make",
    "man",
    "many",
    "may",
    "mean",
    "men",
    "might",
    "mind",
    "moment",
    "more",
    "most",
    "mother",
    "much",
    "must",
    "near",
    "need",
    "never",
    "new",
    "night",
    "no",
    "not",
    "nothing",
    "now",
    "of",
    "off",
    "old",
    "on",
    "once",
    "one",
    "only",
    "open",
    "or",
    "other",
    "our",
    "out",
    "over",
    "own",
    "part",
    "people",
    "place",
    "point",
    "put",
    "right",
    "room",
    "said",
    "same",
    "saw",
    "say",
    "says",
    "see",
    "seem",
    "seemed",
    "shall",
    "she",
    "should",
    "side",
    "since",
    "small",
    "so",
    "some",
    "someone",
    "something",
    "soon",
    "still",
    "such",
    "take",
    "taken",
    "tell",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "thing",
    "think",
    "this",
    "those",
    "though",
    "thought",
    "three",
    "through",
    "time",
    "to",
    "told",
    "too",
    "took",
    "toward",
    "turn",
    "two",
    "under",
    "up",
    "upon",
    "us",
    "use",
    "very",
    "want",
    "was",
    "way",
    "we",
    "well",
    "went",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whole",
    "why",
    "will",
    "with",
    "within",
    "without",
    "word",
    "words",
    "work",
    "would",
    "year",
    "years",
    "yes",
    "you",
    "young",
    "your",
)

DICT_CANDIDATE_PATHS: Sequence[str] = (
    "/usr/share/dict/words",
    "/usr/share/dict/american-english",
    "/usr/share/dict/english",
    "/usr/share/dict/british-english",
)

MAX_LEXICON_SIZE = 60_000
_cached_lexicon: Set[str] | None = None


@dataclass
class DeterministicCleaningResult:
    text: str
    applied: List[str]


def _load_lexicon() -> Set[str]:
    global _cached_lexicon
    if _cached_lexicon is not None:
        return _cached_lexicon

    lexicon: Set[str] = set(word.lower() for word in FALLBACK_WORDS)
    for candidate in DICT_CANDIDATE_PATHS:
        if len(lexicon) >= MAX_LEXICON_SIZE:
            break
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    word = line.strip().lower()
                    if not word or not word.isalpha():
                        continue
                    lexicon.add(word)
                    if len(lexicon) >= MAX_LEXICON_SIZE:
                        break
        except OSError:
            continue
    _cached_lexicon = lexicon
    return lexicon


def _replace_smart_punctuation(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        char = match.group(0)
        return SMART_REPLACEMENTS.get(char, char)

    return NBSP_RE.sub(" ", SMART_PUNCTUATION_RE.sub(repl, text))


def _remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def _remove_references(text: str) -> str:
    text = BRACKET_REFERENCE_RE.sub(" ", text)
    return PAREN_FOOTNOTE_RE.sub(" ", text)


def _fix_hyphenation_artifacts(text: str) -> str:
    text = re.sub(r"(?<=[A-Za-z])-\s*\n\s*(?=[A-Za-z])", "", text)
    text = re.sub(r"(?<=[A-Za-z])\n(?=[A-Za-z])", " ", text)
    return text


def _split_camel_case(text: str) -> str:
    return CAMEL_CASE_SPLIT_RE.sub(r"\1 \2", text)


def _find_lexicon_split(original: str, lexicon: Set[str], depth: int = 0) -> List[str] | None:
    lower = original.lower()
    if lower in lexicon or depth > 3 or len(lower) < 6:
        return None
    min_part = 3
    for i in range(len(lower) - min_part, min_part - 1, -1):
        left_lower = lower[:i]
        right_lower = lower[i:]
        if left_lower not in lexicon:
            continue
        right_original = original[i:]
        if right_lower in lexicon:
            return [original[:i], right_original]
        deeper = _find_lexicon_split(right_original, lexicon, depth + 1)
        if deeper:
            return [original[:i], *deeper]
    return None


def _split_merged_words(text: str, lexicon: Set[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        if len(word) > 30:
            return word
        lower = word.lower()
        if lower in lexicon:
            return word
        split = _find_lexicon_split(word, lexicon)
        return " ".join(split) if split else word

    return MERGED_WORD_RE.sub(repl, text)


def _normalize_spacing(text: str) -> str:
    text = LEADING_SPACE_NEWLINE_RE.sub("\n", text)
    text = TRAILING_SPACE_NEWLINE_RE.sub("\n", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = EXCESS_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def apply_deterministic_cleaning(
    input_text: str,
    options: CleaningOptions,
    *,
    phase: str = "pre",
) -> DeterministicCleaningResult:
    text = input_text
    applied: List[str] = []

    if options.replace_smart_quotes:
        replaced = _replace_smart_punctuation(text)
        if replaced != text:
            applied.append("replaceSmartQuotes")
            text = replaced

    if options.remove_urls:
        stripped = _remove_urls(text)
        if stripped != text:
            applied.append("removeUrls")
            text = stripped

    if options.remove_footnotes and phase == "pre":
        scrubbed = _remove_references(text)
        if scrubbed != text:
            applied.append("removeFootnotes")
            text = scrubbed

    if options.fix_hyphenation:
        merged = _fix_hyphenation_artifacts(text)
        if merged != text:
            applied.append("fixHyphenation")
            text = merged

    if options.fix_ocr_errors:
        camel_split = _split_camel_case(text)
        if camel_split != text:
            applied.append("splitCamelCase")
            text = camel_split
        lexicon = _load_lexicon()
        merged_split = _split_merged_words(text, lexicon)
        if merged_split != text:
            applied.append("splitMergedWords")
            text = merged_split

    text = _normalize_spacing(text)
    return DeterministicCleaningResult(text=text, applied=applied)
