"""Finds the sources an answer names, so that a source the model did not retrieve can be flagged.

The API's citations prove that cited text came from a retrieved source. They do not stop the model
from also writing "as the Mishnah Berurah says", or "Berakhot 60b" when only 2a was retrieved,
from memory. This module catches such mentions at two levels:

- the work: a work no retrieved source belongs to ("the Zohar", "Rambam, Laws of Shabbat");
- the location: a place in a retrieved work that is not itself retrieved ("Berakhot 60b",
  "Shulchan Arukh, Orach Chayim 300:1", "דברים ו, ט").

A mention is a work name with a location, or a work whose name alone is a citation ("Rambam",
"Mishnah Berurah"). Hebrew locations must be valid Hebrew numerals, so "בשבת זו." is not a daf
reference. It is a heuristic, tuned for few false alarms; the README reports it as a check, not a
guarantee.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from chavruta.retrieval import matches
from chavruta.text import normalize

# ---- Hebrew numerals ------------------------------------------------------------------------------

_UNITS = {"א": 1, "ב": 2, "ג": 3, "ד": 4, "ה": 5, "ו": 6, "ז": 7, "ח": 8, "ט": 9}
_TENS = {"י": 10, "כ": 20, "ך": 20, "ל": 30, "מ": 40, "ם": 40, "נ": 50, "ן": 50, "ס": 60, "ע": 70,
         "פ": 80, "ף": 80, "צ": 90, "ץ": 90}  # fmt: skip
_HUNDREDS = {"ק": 100, "ר": 200, "ש": 300, "ת": 400}


def gematria(numeral: str) -> int | None:
    """The value of a Hebrew numeral, or None if it is not one. Strict: hundreds, then at most one
    tens letter, then at most one units letter; 15 and 16 are טו and טז, never יה and יו. So
    "זו", "כל" and "זה" are words, not numbers."""
    s = re.sub("[׳״\"']", "", numeral)
    if not s:
        return None
    if s in ("טו", "טז"):
        return 15 if s == "טו" else 16
    if s.endswith(("יה", "יו")):
        return None
    i, total = 0, 0
    while i < len(s) and s[i] == "ת":
        total += 400
        i += 1
    if i < len(s) and s[i] in "קרש":
        total += _HUNDREDS[s[i]]
        i += 1
    if i < len(s) and s[i] in _TENS:
        total += _TENS[s[i]]
        i += 1
    if i < len(s) and s[i] in _UNITS:
        total += _UNITS[s[i]]
        i += 1
    return total if i == len(s) and total > 0 else None


# ---- names ----------------------------------------------------------------------------------------

# canonical (Sefaria) title -> (English names, Hebrew names)
TANAKH = {
    "Genesis": (["Genesis", "Bereshit", "Bereishit"], ["בראשית"]),
    "Exodus": (["Exodus", "Shemot"], ["שמות"]),
    "Leviticus": (["Leviticus", "Vayikra"], ["ויקרא"]),
    "Numbers": (["Numbers", "Bamidbar"], ["במדבר"]),
    "Deuteronomy": (["Deuteronomy", "Devarim", "Deut\\."], ["דברים"]),
    "Joshua": (["Joshua"], ["יהושע"]),
    "Judges": (["Judges"], ["שופטים"]),
    "Samuel": (["I Samuel", "II Samuel", "Samuel"], ["שמואל"]),
    "Kings": (["I Kings", "II Kings", "Kings"], ["מלכים"]),
    "Isaiah": (["Isaiah"], ["ישעיהו", "ישעיה"]),
    "Jeremiah": (["Jeremiah"], ["ירמיהו", "ירמיה"]),
    "Ezekiel": (["Ezekiel"], ["יחזקאל"]),
    "Hosea": (["Hosea"], ["הושע"]),
    "Joel": (["Joel"], ["יואל"]),
    "Amos": (["Amos"], ["עמוס"]),
    "Obadiah": (["Obadiah"], ["עובדיה"]),
    "Jonah": (["Jonah"], ["יונה"]),
    "Micah": (["Micah"], ["מיכה"]),
    "Nahum": (["Nahum"], ["נחום"]),
    "Habakkuk": (["Habakkuk"], ["חבקוק"]),
    "Zephaniah": (["Zephaniah"], ["צפניה"]),
    "Haggai": (["Haggai"], ["חגי"]),
    "Zechariah": (["Zechariah"], ["זכריה"]),
    "Malachi": (["Malachi"], ["מלאכי"]),
    "Psalms": (["Psalms?", "Tehillim"], ["תהלים", "תהילים"]),
    "Proverbs": (["Proverbs", "Mishlei"], ["משלי"]),
    "Job": (["Job"], ["איוב"]),
    "Song of Songs": (["Song of Songs"], ["שיר השירים"]),
    "Ruth": (["Ruth"], ["רות"]),
    "Lamentations": (["Lamentations", "Eichah"], ["איכה"]),
    "Ecclesiastes": (["Ecclesiastes", "Kohelet"], ["קהלת"]),
    "Esther": (["Esther"], ["אסתר"]),
    "Daniel": (["Daniel"], ["דניאל"]),
    "Ezra": (["Ezra"], ["עזרא"]),
    "Nehemiah": (["Nehemiah"], ["נחמיה"]),
    "Chronicles": (["I Chronicles", "II Chronicles", "Chronicles"], ["דברי הימים"]),
}
TALMUD = {
    "Berakhot": "ברכות", "Shabbat": "שבת", "Eruvin": "עירובין", "Pesachim": "פסחים", "Yoma": "יומא",
    "Sukkah": "סוכה", "Beitzah": "ביצה", "Rosh Hashanah": "ראש השנה", "Taanit": "תענית",
    "Megillah": "מגילה", "Moed Katan": "מועד קטן", "Chagigah": "חגיגה", "Yevamot": "יבמות",
    "Ketubot": "כתובות", "Nedarim": "נדרים", "Nazir": "נזיר", "Sotah": "סוטה", "Gittin": "גיטין",
    "Kiddushin": "קידושין", "Bava Kamma": "בבא קמא", "Bava Metzia": "בבא מציעא",
    "Bava Batra": "בבא בתרא", "Sanhedrin": "סנהדרין", "Makkot": "מכות", "Shevuot": "שבועות",
    "Avodah Zarah": "עבודה זרה", "Horayot": "הוריות", "Zevachim": "זבחים", "Menachot": "מנחות",
    "Chullin": "חולין", "Bekhorot": "בכורות", "Arakhin": "ערכין", "Temurah": "תמורה",
    "Keritot": "כריתות", "Meilah": "מעילה", "Tamid": "תמיד", "Niddah": "נדה",
}  # fmt: skip
MISHNAH_ONLY = [
    "Avot", "Peah", "Demai", "Kilayim", "Sheviit", "Terumot", "Maasrot", "Maaser Sheni", "Challah",
    "Orlah", "Bikkurim", "Shekalim", "Eduyot", "Middot", "Kinnim", "Kelim", "Oholot", "Negaim",
    "Parah", "Tahorot", "Mikvaot", "Makhshirin", "Zavim", "Tevul Yom", "Yadayim", "Oktzin",
]  # fmt: skip
#: Rambam's books in the library, by the names an answer might use.
MT_BOOKS = {
    "Reading the Shema": ["Reading the Shema", "Keriat Shema", "Kriat Shema", "Shema"],
    "Prayer and the Priestly Blessing": ["Prayer and the Priestly Blessing", "Prayer", "Tefillah", "Tefilah"],
    "Blessings": ["Blessings", "Berakhot", "Brachot", "Berachot"],
}
# Names that are a citation by themselves.
NAMED = {
    "Rashi": (["Rashi"], ['רש"י', "רש״י"]),
    "Tosafot": (["Tosafot", "Tosafos", "Tosfot"], ["תוספות", "תוס'", "תוס׳"]),
    "Rambam": (["Rambam", "Maimonides", "Mishneh Torah"], ['רמב"ם', "רמב״ם", "משנה תורה"]),
    "Shulchan Arukh": (
        ["Shulchan Arukh", "Shulchan Aruch", "Shulḥan ʿArukh", "Rema", "Rama"],
        ["שולחן ערוך", "שלחן ערוך", 'שו"ע', "שו״ע", 'רמ"א', "רמ״א"],
    ),
    "Mishnah Berurah": (["Mishnah Berurah", "Mishna Berura"], ["משנה ברורה"]),
    "Ramban": (["Ramban", "Nachmanides"], ['רמב"ן', "רמב״ן"]),
    "Rif": (["Rif"], ['רי"ף', "רי״ף"]),
    "Rosh": (["the Rosh"], ['הרא"ש', "הרא״ש"]),
    "Tur": (["the Tur", "Arba'ah Turim"], ["הטור", "ארבעה טורים"]),
    "Beit Yosef": (["Beit Yosef", "Beis Yosef"], ["בית יוסף"]),
    "Magen Avraham": (["Magen Avraham"], ["מגן אברהם"]),
    "Taz": (["Taz", "Turei Zahav"], ['ט"ז', "ט״ז"]),
    "Aruch HaShulchan": (["Aruch HaShulchan", "Arukh HaShulchan"], ["ערוך השולחן"]),
    "Kitzur Shulchan Arukh": (["Kitzur Shulchan Arukh"], ["קיצור שולחן ערוך"]),
    "Zohar": (["Zohar"], ["זוהר"]),
    "Tosefta": (["Tosefta"], ["תוספתא"]),
    "Jerusalem Talmud": (["Yerushalmi", "Jerusalem Talmud"], ["ירושלמי"]),
    "Midrash": (
        ["Midrash Rabbah", "Bereshit Rabbah", "Genesis Rabbah", "Tanchuma", "Sifrei", "Sifra", "Mekhilta"],
        ["מדרש רבה", "בראשית רבה", "תנחומא", "מכילתא"],
    ),
    "Sefer HaChinukh": (["Sefer HaChinukh", "Chinuch"], ["ספר החינוך"]),
}
#: Which mention families each corpus work satisfies.
WORK_FAMILIES = {
    "Genesis": {"Genesis"},
    "Exodus": {"Exodus"},
    "Leviticus": {"Leviticus"},
    "Numbers": {"Numbers"},
    "Deuteronomy": {"Deuteronomy"},
    "Mishnah Berakhot": {"Berakhot"},
    "Berakhot": {"Berakhot"},
    "Rashi on Berakhot": {"Rashi", "Berakhot"},
    "Tosafot on Berakhot": {"Tosafot", "Berakhot"},
    "Mishneh Torah, Reading the Shema": {"Rambam"},
    "Mishneh Torah, Prayer and the Priestly Blessing": {"Rambam"},
    "Mishneh Torah, Blessings": {"Rambam"},
    "Shulchan Arukh, Orach Chayim": {"Shulchan Arukh"},
}

_HE_PREFIX = r"(?<![א-ת])[והבכלמש]{0,2}"
_NUM = r"[א-ת]{1,4}[׳״\"']?[א-ת]?"


@dataclass(frozen=True)
class Mention:
    family: str
    text: str
    start: int
    #: The place named, as a Sefaria-style reference, when there is one.
    ref: str | None = None


def _alt(names: Iterable[str]) -> str:
    return "|".join(sorted(names, key=len, reverse=True))


Parser = tuple[str, re.Pattern[str], str]  # family, pattern, kind of location


def _parsers() -> list[Parser]:
    out: list[Parser] = []
    for title, (en, he) in TANAKH.items():
        out.append((title, re.compile(rf"\b(?:{_alt(en)})\s+(?P<c>\d{{1,3}})(?::(?P<v>\d{{1,3}}))?\b"), "en"))
        he_loc = rf"\s+(?:פרק\s+)?(?P<c>{_NUM})(?:,\s*(?P<v>{_NUM}))?(?![א-ת])"
        out.append((title, re.compile(rf"{_HE_PREFIX}(?:{_alt(he)}){he_loc}"), "he"))
    for title, he_name in TALMUD.items():
        out.append((title, re.compile(rf"\b{title}\s+(?P<d>\d{{1,3}})(?P<a>[ab])\b"), "daf_en"))
        out.append((title, re.compile(rf"(?:מסכת|במסכת|בבלי)\s+{he_name}(?![א-ת])"), "name"))
        # A daf ends at its "." or ":"; a numeral right after it is chapter:halakha ("ברכות ח:יד").
        daf = rf"\s+(?:דף\s+)?(?P<d>{_NUM})\s*(?P<a>[.:])(?![\dא-ת])"
        out.append((title, re.compile(rf"{_HE_PREFIX}{he_name}{daf}"), "daf_he"))
    for title in [*TALMUD, *MISHNAH_ONLY]:
        loc = r"\s+(?P<c>\d{1,2})(?::(?P<v>\d{1,2}))?"
        out.append((title, re.compile(rf"\bMishnah\s+{title}{loc}"), "mishnah"))
    oc = r"(?:Orach Chayim|Orach Chaim|OC|O\.C\.)"
    loc = r"\s+(?P<c>\d{1,3})(?::(?P<v>\d{1,2}))?"
    out.append(("Shulchan Arukh", re.compile(rf"\b(?:Shulchan Arukh|Shulchan Aruch|SA),?\s+{oc}{loc}"), "sa"))
    out.append(("Shulchan Arukh", re.compile(rf"\b{oc}{loc}"), "sa"))
    he_oc = (
        rf"(?:או\"ח|או״ח|אורח חיים)\s+(?:סימן\s+)?(?P<c>{_NUM})(?:,?\s*(?:סעיף\s+)?(?P<v>{_NUM}))?(?![א-ת])"
    )
    out.append(("Shulchan Arukh", re.compile(he_oc), "sa_he"))
    mt = (
        r"\b(?:Rambam|Maimonides|Mishneh Torah)(?:'s)?,?\s+(?:Laws of|Hilchot|Hilkhot|Hilchos)\s+"
        r"(?P<book>[A-Z][A-Za-z']*(?:\s+(?:and\s+the\s+|the\s+)?[A-Z][A-Za-z']*)*)"
        r"(?:,?\s+(?P<c>\d{1,2})(?::(?P<v>\d{1,2}))?)?"
    )
    out.append(("Rambam", re.compile(mt), "mt"))
    for family, (en, he) in NAMED.items():
        out.append((family, re.compile(rf"\b(?:{_alt(en)})\b"), "name"))
        out.append((family, re.compile(rf"{_HE_PREFIX}(?:{_alt(he)})(?![א-ת])"), "name"))
    return out


_PARSERS = _parsers()


def _ref(family: str, kind: str, m: re.Match[str]) -> str | bool | None:
    """The reference a match names; None if it names only a work; False if it is not a citation
    after all (an invalid Hebrew numeral)."""
    g = m.groupdict()
    if kind == "name":
        return None
    if kind in ("en", "mishnah", "sa"):
        prefix = {"mishnah": f"Mishnah {family}", "sa": "Shulchan Arukh, Orach Chayim"}.get(kind, family)
        return f"{prefix} {g['c']}" + (f":{g['v']}" if g.get("v") else "")
    if kind == "daf_en":
        return f"{family} {g['d']}{g['a']}"
    if kind in ("he", "sa_he"):
        c = gematria(g["c"])
        v = gematria(g["v"]) if g.get("v") else None
        if c is None or (g.get("v") and v is None):
            return False
        prefix = "Shulchan Arukh, Orach Chayim" if kind == "sa_he" else family
        return f"{prefix} {c}" + (f":{v}" if v else "")
    if kind == "daf_he":
        d = gematria(g["d"])
        if d is None or d < 2:
            return False
        return f"{family} {d}{'a' if g['a'] == '.' else 'b'}"
    if kind == "mt":
        name = g["book"].strip()
        book = next((b for b, names in MT_BOOKS.items() if name in names), None)
        if book is None:
            return f"Mishneh Torah, {name}"  # a book the library does not have
        loc = (f" {g['c']}" if g.get("c") else "") + (f":{g['v']}" if g.get("v") else "")
        return f"Mishneh Torah, {book}{loc}"
    return None


def find_mentions(text: str) -> list[Mention]:
    found: dict[int, Mention] = {}
    for family, pattern, kind in _PARSERS:
        for m in pattern.finditer(text):
            ref = _ref(family, kind, m)
            if ref is False:
                continue
            mention = Mention(family, m.group(0).strip(), m.start(), ref if isinstance(ref, str) else None)
            current = found.get(m.start())
            # At one position keep the most specific reading: one with a location, then the longest.
            better = current is None or (
                (mention.ref is not None, len(mention.text)) > (current.ref is not None, len(current.text))
            )
            if better:
                found[m.start()] = mention
    # A mention inside a longer one ("Rambam" in "Rambam, Laws of...", "Orach Chayim 300" in
    # "Shulchan Arukh, Orach Chayim 300") is part of it.
    spans = list(found.values())

    def inside(x: Mention) -> bool:
        end = x.start + len(x.text)
        return any(
            y is not x and y.start <= x.start and end <= y.start + len(y.text) and len(y.text) > len(x.text)
            for y in spans
        )

    return sorted((x for x in spans if not inside(x)), key=lambda x: x.start)


#: The Rambam's works other than the Mishneh Torah, by their Sefaria titles.
RAMBAM_WORKS = ("Sefer HaMitzvot", "Eight Chapters", "Iggerot HaRambam", "Guide for the Perplexed",
                "Rambam on ", "Treatise on Resurrection")  # fmt: skip


def families_of(work: str) -> set[str]:
    """The mention families a work satisfies: a library work by the table above; a work fetched
    from Sefaria for one question by its title: "Rashi on Bava Batra" is Rashi and Bava Batra,
    "Pirkei Avot" is Avot, "Mishneh Torah, Sabbath" is the Rambam, "Bereshit Rabbah" a Midrash."""
    if work in WORK_FAMILIES:
        return WORK_FAMILIES[work]
    if work.startswith(RAMBAM_WORKS):
        return {work, "Rambam"}
    if " on " in work:  # a commentary: the commentator and the work commented on
        commentator, base = work.split(" on ", 1)
        return {commentator, *families_of(commentator), *families_of(base)}
    out = {work}
    for family, (en, _) in TANAKH.items():
        if work in (family, *en):
            out.add(family)
    if not work.startswith("Jerusalem Talmud"):  # its tractates are not the Bavli's
        out |= {t for t in [*TALMUD, *MISHNAH_ONLY] if work == t or work.endswith(" " + t)}
    for family, (en, _) in NAMED.items():
        if any(work == n or work.startswith((n + " ", n + ",")) for n in (family, *en)):
            out.add(family)
    return out


def _names(family: str) -> list[str]:
    """Every English and Hebrew name of a mention family that has them."""
    names = [family]
    if family in TANAKH:
        names += [*TANAKH[family][0], *TANAKH[family][1]]
    if family in TALMUD:
        names.append(TALMUD[family])
    if family in NAMED:
        names += [*NAMED[family][0], *NAMED[family][1]]
    return names


def named_in(text: str, ref: str) -> bool:
    """Whether a text names the work a reference belongs to, by any of its names: "בבבא מציעא דף נט
    עמוד ב" names "Bava Metzia 59b", "What does Rashi say on Numbers 13:2" names "Rashi on Numbers
    13:2" (both Rashi and Numbers). Lenient on purpose: it only confirms a place a planner already
    proposed, and never flags anything by itself."""
    work = re.sub(r"\s+[\d:ab-]+$", "", ref.strip())
    families = [f for f in families_of(work) if len(_names(f)) > 1]
    if not families:
        return False
    plain = normalize(text).lower()
    return all(any(normalize(n).lower() in plain for n in _names(f)) for f in families)


def unretrieved(
    answer_text: str,
    retrieved_works: set[str],
    retrieved_texts: list[str],
    retrieved_refs: Iterable[str] = (),
) -> list[Mention]:
    """Mentions of a work, or of a place in a work, that was not retrieved, and that no retrieved
    text itself quotes (a Gemara quoting a verse may be followed in naming that verse)."""
    allowed: set[str] = set()
    for w in retrieved_works:
        allowed |= families_of(w)
    refs = list(retrieved_refs)
    quoted = [m for t in retrieved_texts for m in find_mentions(t)]
    quoted_families = {m.family for m in quoted}
    quoted_refs = [m.ref for m in quoted if m.ref]
    out: list[Mention] = []
    for m in find_mentions(answer_text):
        unknown_work = m.family not in allowed and m.family not in quoted_families
        unknown_place = m.ref is not None and not _covered(m.ref, refs) and not _covered(m.ref, quoted_refs)
        if unknown_place and (comment := _commentary_on(answer_text, m)) and _covered(comment, refs):
            unknown_place = False
        if unknown_work or unknown_place:
            out.append(m)
    return out


_COMMENTATOR_BEFORE = {
    name: re.compile(rf"(?:{_alt([*NAMED[name][0], *NAMED[name][1]])})\s*(?:on\s+|\(|,)?\s*$")
    for name in ("Rashi", "Tosafot")
}


def _commentary_on(text: str, m: Mention) -> str | None:
    """A daf named right after Rashi or Tosafot ("Tosafot (Berakhot 18a)", "רש"י (ברכות טז.)") is
    that commentator's comment on the daf: "Tosafot on Berakhot 18a"."""
    if m.ref is None or m.family not in TALMUD:
        return None
    before = text[max(0, m.start - 20) : m.start]
    for name, pattern in _COMMENTATOR_BEFORE.items():
        if pattern.search(before):
            return f"{name} on {m.ref}"
    return None


def _covered(ref: str, retrieved: list[str]) -> bool:
    """A named place is covered by a retrieved reference at the same, a finer or a coarser level:
    "Berakhot 2a" by "Berakhot 2a:3", "Deuteronomy 6:4" by "Deuteronomy 6"."""
    return any(matches(r, ref) or matches(ref, r) for r in retrieved)
