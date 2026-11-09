"""Finds the sources an answer names, so that a source the model did not retrieve can be flagged.

The API's citations prove that cited text came from a retrieved source. They do not stop the model
from also writing "as the Mishnah Berurah says" from memory. This module catches such mentions.

A mention is a work name with a location ("Shabbat 21b", "תהלים קמה", "סימן פט"), or a work whose
name alone is a citation ("Rambam", "Mishnah Berurah", "Zohar"). Bare words that double as ordinary
words ("Shabbat", "ברכות", "שבת") count only with a location. It is a heuristic, tuned for few false
alarms; its misses are why the README reports it as a check, not a guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# family -> (English names, Hebrew names). A family groups names that refer to one work.
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
    "Hosea": (["Hosea"], ["הושע"]),
    "Amos": (["Amos"], ["עמוס"]),
    "Micah": (["Micah"], ["מיכה"]),
    "Malachi": (["Malachi"], ["מלאכי"]),
}
TRACTATES = [
    "Berakhot", "Shabbat", "Eruvin", "Pesachim", "Yoma", "Sukkah", "Beitzah", "Rosh Hashanah",
    "Taanit", "Megillah", "Moed Katan", "Chagigah", "Yevamot", "Ketubot", "Nedarim", "Nazir",
    "Sotah", "Gittin", "Kiddushin", "Bava Kamma", "Bava Metzia", "Bava Batra", "Sanhedrin",
    "Makkot", "Shevuot", "Avodah Zarah", "Horayot", "Zevachim", "Menachot", "Chullin", "Bekhorot",
    "Arakhin", "Temurah", "Keritot", "Meilah", "Tamid", "Niddah", "Avot", "Peah", "Demai",
]  # fmt: skip
TRACTATES_HE = [
    "ברכות", "שבת", "עירובין", "פסחים", "יומא", "סוכה", "ביצה", "ראש השנה", "תענית", "מגילה",
    "מועד קטן", "חגיגה", "יבמות", "כתובות", "נדרים", "נזיר", "סוטה", "גיטין", "קידושין",
    "בבא קמא", "בבא מציעא", "בבא בתרא", "סנהדרין", "מכות", "שבועות", "עבודה זרה", "הוריות",
    "זבחים", "מנחות", "חולין", "בכורות", "ערכין", "תמורה", "כריתות", "מעילה", "תמיד", "נדה", "אבות",
]  # fmt: skip
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

_EN_LOC = r"\s+\d{1,3}(?:[ab]\b|:\d{1,3}|\b)"
# Hebrew numerals are letters, so a location must look like one: "ו, ז" (chapter, verse), a label
# ("פרק ו", "סימן פט"), or a numeral with geresh/gershayim ("קמ\"ה"). "דברים רבים." is not a verse.
_HE_LOC = (
    r"\s+(?:[א-ת]{1,3},\s*[א-ת]{1,3}(?![א-ת])"
    r"|(?:פרק|סימן|דף)\s+[א-ת]{1,3}[׳״\"']?[א-ת]?(?![א-ת])"
    r"|[א-ת]{1,2}[׳״\"'][א-ת]?(?![א-ת]))"
)
_HE_DAF = r"\s+(?:דף\s+)?(?:[א-ת]{1,3}|[א-ת]{1,2}[׳״\"'][א-ת])\s*[.:](?!\d)"
# One- and two-letter prefixes (ה, ו, ב, כ, ל, מ, ש): "הרמב\"ם", "וכן במשנה ברורה".
_HE_PREFIX = r"(?<![א-ת])[והבכלמש]{0,2}"


@dataclass(frozen=True)
class Mention:
    family: str
    text: str
    start: int


def _alt(names: list[str]) -> str:
    return "|".join(sorted(names, key=len, reverse=True))


def _patterns() -> list[tuple[str, re.Pattern[str]]]:
    out: list[tuple[str, re.Pattern[str]]] = []
    for family, (en, he) in TANAKH.items():
        out.append((family, re.compile(rf"\b(?:{_alt(en)}){_EN_LOC}")))
        out.append((family, re.compile(rf"{_HE_PREFIX}(?:{_alt(he)}){_HE_LOC}")))
    for name in TRACTATES:
        out.append((name, re.compile(rf"\b{name}\s+\d{{1,3}}[ab]\b")))
        out.append((name, re.compile(rf"\bMishnah\s+{name}\s+\d{{1,2}}(?::\d{{1,2}})?")))
    for name, en_name in zip(TRACTATES_HE, TRACTATES, strict=False):
        out.append((en_name, re.compile(rf"(?:מסכת|במסכת|בבלי)\s+{name}(?![א-ת])")))
        out.append((en_name, re.compile(rf"{_HE_PREFIX}{name}{_HE_DAF}")))
    for family, (en, he) in NAMED.items():
        out.append((family, re.compile(rf"\b(?:{_alt(en)})\b")))
        out.append((family, re.compile(rf"{_HE_PREFIX}(?:{_alt(he)})(?![א-ת])")))
    return out


_PATTERNS = _patterns()


def find_mentions(text: str) -> list[Mention]:
    found: dict[tuple[str, int], Mention] = {}
    for family, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            found.setdefault((family, m.start()), Mention(family, m.group(0).strip(), m.start()))
    return sorted(found.values(), key=lambda m: m.start)


def unretrieved(answer_text: str, retrieved_works: set[str], retrieved_texts: list[str]) -> list[Mention]:
    """Mentions of works that no retrieved source belongs to, and that no retrieved text itself
    quotes (a Gemara quoting a verse may be followed in naming that verse)."""
    allowed: set[str] = set()
    for w in retrieved_works:
        allowed |= WORK_FAMILIES.get(w, {w})
    quoted = {m.family for t in retrieved_texts for m in find_mentions(t)}
    return [m for m in find_mentions(answer_text) if m.family not in allowed and m.family not in quoted]
