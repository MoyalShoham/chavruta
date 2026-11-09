"""Hebrew, Aramaic and English text: cleaning Sefaria markup and tokenising for keyword search."""

from __future__ import annotations

import html
import re
import unicodedata

_FOOTNOTE = re.compile(
    r"<sup[^>]*class=\"footnote-marker\"[^>]*>.*?</sup>\s*<i[^>]*class=\"footnote\"[^>]*>.*?</i>", re.DOTALL
)
_TAG = re.compile(r"<[^>]+>")
# Cantillation marks and vowel points; letters (05D0-05EA) are kept.
_MARKS = re.compile(r"[֑-ְ֯-ׇֽֿׁׂׅׄ]")
_PUNCT_HE = re.compile(r"[־׀׃׳״\"'`׳״]")
_TOKEN = re.compile(r"[א-ת]+|[a-z0-9]+")
_HEBREW_PREFIXES = "והבכלמש"


def clean(text: str) -> str:
    """Sefaria HTML to plain display text: footnotes and tags removed, entities decoded, spaces collapsed."""
    t = _FOOTNOTE.sub(" ", text)
    t = _TAG.sub(" ", t)
    t = html.unescape(t)
    return " ".join(t.split())


def strip_marks(text: str) -> str:
    """Removes niqqud and cantillation, so pointed and unpointed text match."""
    return _MARKS.sub("", unicodedata.normalize("NFC", text))


def is_mostly_hebrew(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    hebrew = sum(1 for ch in letters if "א" <= ch <= "ת")
    return hebrew / len(letters) > 0.6


def is_mostly_latin(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    latin = sum(1 for ch in letters if "a" <= ch.lower() <= "z")
    return latin / len(letters) > 0.8


def tokens(text: str) -> list[str]:
    """Keyword-search tokens. Hebrew words also yield a variant without a one-letter prefix (ו, ה, ב, ...),
    a cheap stand-in for morphology: "ובשכבך" matches "בשכבך"."""
    t = _PUNCT_HE.sub("", strip_marks(text).lower())
    out: list[str] = []
    for tok in _TOKEN.findall(t):
        out.append(tok)
        if len(tok) >= 4 and tok[0] in _HEBREW_PREFIXES and "א" <= tok[0] <= "ת":
            out.append(tok[1:])
    return out
