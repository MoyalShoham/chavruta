"""The corpus: which works, which openly licensed versions, and how to cut them into natural units.

A chunk is always a unit the text itself defines (a verse, a mishnah, a Talmud segment, one comment
of Rashi or Tosafot, a halakhah, a seif), never a fixed number of characters. Each chunk keeps its
reference, language, version and license.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from chavruta.sefaria import Section, is_open
from chavruta.text import clean, is_mostly_hebrew, is_mostly_latin


@dataclass(frozen=True)
class WorkSpec:
    title: str
    he_title: str
    category: str
    first_ref: str
    #: (Sefaria language, version title); only versions with an open license are kept.
    versions: tuple[tuple[str, str], ...]
    #: Stop after this section ref (inclusive), for works indexed only in part.
    last_ref: str | None = None
    #: Commentary on another work (Rashi, Tosafot): shown beside it in the sugya view.
    base_title: str | None = None


TORAH = [
    ("Genesis", "בראשית"),
    ("Exodus", "שמות"),
    ("Leviticus", "ויקרא"),
    ("Numbers", "במדבר"),
    ("Deuteronomy", "דברים"),
]
TANAKH_VERSIONS = (
    ("hebrew", "Tanach with Ta'amei Hamikra"),
    ("english", "The Holy Scriptures: A New Translation (JPS 1917)"),
)

CORPUS: list[WorkSpec] = [
    *[WorkSpec(t, he, "Tanakh", f"{t} 1", TANAKH_VERSIONS) for t, he in TORAH],
    WorkSpec(
        "Mishnah Berakhot",
        "משנה ברכות",
        "Mishnah",
        "Mishnah Berakhot 1",
        (("hebrew", "Torat Emet 357"), ("english", "Mishnah Yomit by Dr. Joshua Kulp")),
    ),
    WorkSpec(
        "Berakhot",
        "ברכות",
        "Talmud",
        "Berakhot 2a",
        (
            ("hebrew", "Wikisource Talmud Bavli"),
            ("english", "Tractate Berakot by A. Cohen, Cambridge University Press, 1921"),
        ),
    ),
    WorkSpec(
        "Rashi on Berakhot",
        'רש"י על ברכות',
        "Commentary",
        "Rashi on Berakhot 2a",
        (("hebrew", "Vilna Edition"),),
        base_title="Berakhot",
    ),
    WorkSpec(
        "Tosafot on Berakhot",
        "תוספות על ברכות",
        "Commentary",
        "Tosafot on Berakhot 2a",
        (("hebrew", "Vilna Edition"), ("english", "Tosafot, Translated by Jan Buckler.")),
        base_title="Berakhot",
    ),
    *[
        WorkSpec(
            f"Mishneh Torah, {name}",
            he,
            "Halakhah",
            f"Mishneh Torah, {name} 1",
            (
                ("hebrew", "Wikisource Mishneh Torah"),
                ("english", "The Mishneh Torah by Maimonides. trans. by Moses Hyamson, 1937-1949"),
            ),
        )
        for name, he in (
            ("Reading the Shema", "משנה תורה, הלכות קריאת שמע"),
            ("Prayer and the Priestly Blessing", "משנה תורה, הלכות תפילה וברכת כהנים"),
            ("Blessings", "משנה תורה, הלכות ברכות"),
        )
    ],
    WorkSpec(
        "Shulchan Arukh, Orach Chayim",
        "שולחן ערוך, אורח חיים",
        "Halakhah",
        "Shulchan Arukh, Orach Chayim 1",
        (("hebrew", "Torat Emet 363"),),
        last_ref="Shulchan Arukh, Orach Chayim 241",
    ),
]

BY_TITLE = {w.title: w for w in CORPUS}


@dataclass(frozen=True)
class Chunk:
    ref: str
    work: str
    category: str
    language: str  # "he" or "en"
    version: str
    license: str
    text: str
    section_ref: str
    position: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Skipped:
    work: str
    section_ref: str
    version: str
    reason: str
    language: str = ""
    license: str = ""


def _units(text: Any, prefix: str) -> Iterator[tuple[str, str]]:
    """Leaves of a Sefaria text array with their sub-references: ["a", ["b", "c"]] -> (prefix:1, a), ..."""
    if isinstance(text, str):
        yield prefix, text
        return
    if isinstance(text, list):
        for i, item in enumerate(text, 1):
            yield from _units(item, f"{prefix}:{i}")


def chunks_of(section: Section, work: WorkSpec, start_position: int) -> tuple[list[Chunk], list[Skipped]]:
    chunks: list[Chunk] = []
    skipped: list[Skipped] = []
    position = start_position
    for v in section.versions:
        language = "he" if v.language in ("he", "hebrew") else "en"
        if not is_open(v.license):
            skipped.append(
                Skipped(
                    work.title, section.ref, v.title, f"license {v.license} is not open", language, v.license
                )
            )
            continue
        for ref, raw in _units(v.text, section.ref):
            body = clean(raw)
            if not body:
                continue
            script_ok = is_mostly_hebrew(body) if language == "he" else is_mostly_latin(body)
            if not script_ok:
                skipped.append(
                    Skipped(
                        work.title,
                        ref,
                        v.title,
                        f"text is not in the expected script ({language})",
                        language,
                        v.license,
                    )
                )
                continue
            chunks.append(
                Chunk(
                    ref, work.title, work.category, language, v.title, v.license, body, section.ref, position
                )
            )
            position += 1
    return chunks, skipped


class SectionSource(Protocol):
    def section(self, ref: str, versions: list[tuple[str, str]]) -> Section: ...


def fetch_work(
    client: SectionSource, work: WorkSpec, max_sections: int = 400
) -> tuple[list[Chunk], list[Skipped]]:
    """All chunks of a work, walking its sections with Sefaria's 'next' pointer."""
    ref: str | None = work.first_ref
    chunks: list[Chunk] = []
    skipped: list[Skipped] = []
    seen = 0
    while ref and seen < max_sections:
        section = client.section(ref, list(work.versions))
        new, skip = chunks_of(section, work, len(chunks))
        chunks.extend(new)
        skipped.extend(skip)
        seen += 1
        if work.last_ref and section.ref == work.last_ref:
            break
        ref = section.next_ref
        # Commentaries' 'next' points at a segment ("Rashi on Berakhot 2b:1"); use its section.
        if ref and work.base_title and ref.count(":"):
            ref = ref.split(":")[0]
        if ref and not ref.startswith(work.title):
            break
    return chunks, skipped
