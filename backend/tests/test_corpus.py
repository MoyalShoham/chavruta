"""Text normalisation, the license policy, chunking and the Sefaria client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pytest

from chavruta.corpus import WorkSpec, chunks_of, fetch_work
from chavruta.sefaria import Section, SefariaClient, SefariaError, VersionText, is_open
from chavruta.text import clean, is_mostly_hebrew, is_mostly_latin, normalize, strip_marks, tokens

# ---- text ---------------------------------------------------------------------------------------


def test_clean_removes_footnotes_tags_and_entities() -> None:
    raw = (
        'In the beginning<sup class="footnote-marker">1</sup><i class="footnote">a note</i> God '
        "<b>created</b> &amp; made"
    )
    assert clean(raw) == "In the beginning God created & made"


def test_strip_marks_removes_niqqud_and_cantillation() -> None:
    assert strip_marks("בְּרֵאשִׁ֖ית בָּרָ֣א") == "בראשית ברא"


def test_maqaf_separates_words_and_gershayim_join_them() -> None:
    assert normalize("אֶת־י֥וֹם") == "את יום"
    assert normalize('נו"ן ר׳ עקיבא') == "נון ר עקיבא"
    assert tokens("זָכ֛וֹר אֶת־י֥וֹם הַשַּׁבָּ֖ת")[:4] == ["זכור", "את", "יום", "השבת"]


def test_tokens_add_a_variant_without_a_one_letter_prefix() -> None:
    assert "בשכבך" in tokens("ובשכבך")
    assert "שבת" in tokens("השבת")
    assert tokens("לא") == ["לא"]  # too short to strip


def test_tokens_lowercase_english() -> None:
    assert tokens("Hear, O Israel") == ["hear", "o", "israel"]


def test_script_detection() -> None:
    assert is_mostly_hebrew("שמע ישראל (Deut. 6:4)")
    assert not is_mostly_hebrew("Hear O Israel")
    assert is_mostly_latin("Hear O Israel")
    assert not is_mostly_latin("Слушай, Израиль")
    assert not is_mostly_latin("")


# ---- license policy and chunking ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "ok"),
    [
        ("Public Domain", True),
        ("CC0", True),
        ("CC-BY", True),
        ("CC-BY-SA", True),
        ("CC-BY-NC", False),
        ("CC-BY-NC-SA", False),
        ("Copyright: JPS, 1985", False),
        ("unknown", False),
        (None, False),
        ("", False),
    ],
)
def test_only_open_licenses_are_kept(name: str | None, ok: bool) -> None:
    assert is_open(name) is ok


WORK = WorkSpec("Berakhot", "ברכות", "Talmud", "Berakhot 2a", (("hebrew", "A"), ("english", "B")))


def _section(versions: list[VersionText], ref: str = "Berakhot 2a", next_ref: str | None = None) -> Section:
    return Section(ref, "ברכות ב.", next_ref, ["Daf", "Line"], versions)


def test_chunks_follow_the_texts_own_units_and_keep_license() -> None:
    he = VersionText("he", "A", "CC-BY-SA", ["מאימתי קורין", "<b>את שמע</b>"])
    en = VersionText("en", "B", "Public Domain", ["From when", "do we recite"])
    chunks, skipped = chunks_of(_section([he, en]), WORK, 0)
    assert [(c.ref, c.language, c.license) for c in chunks] == [
        ("Berakhot 2a:1", "he", "CC-BY-SA"),
        ("Berakhot 2a:2", "he", "CC-BY-SA"),
        ("Berakhot 2a:1", "en", "Public Domain"),
        ("Berakhot 2a:2", "en", "Public Domain"),
    ]
    assert chunks[1].text == "את שמע"
    assert skipped == []


def test_non_open_versions_are_skipped_with_a_reason() -> None:
    closed = VersionText("en", "Steinsaltz", "CC-BY-NC", ["From when"])
    chunks, skipped = chunks_of(_section([closed]), WORK, 0)
    assert chunks == []
    assert skipped[0].reason == "license CC-BY-NC is not open"


def test_text_in_the_wrong_script_is_skipped() -> None:
    """Sefaria has had Russian text tagged as an English version."""
    mislabelled = VersionText("en", "Community", "CC0", ["Слушай, Израиль"])
    chunks, skipped = chunks_of(_section([mislabelled]), WORK, 0)
    assert chunks == []
    assert "expected script" in skipped[0].reason


def test_nested_units_get_nested_refs_and_empty_ones_are_dropped() -> None:
    rashi = VersionText("he", "Vilna", "Public Domain", [["דיבור א", "דיבור ב"], [], ["דיבור ג"]])
    chunks, _ = chunks_of(_section([rashi], ref="Rashi on Berakhot 2a"), WORK, 10)
    assert [c.ref for c in chunks] == [
        "Rashi on Berakhot 2a:1:1",
        "Rashi on Berakhot 2a:1:2",
        "Rashi on Berakhot 2a:3:1",
    ]
    assert [c.position for c in chunks] == [10, 11, 12]


class FakeSefaria:
    def __init__(self, sections: dict[str, Section]) -> None:
        self.sections = sections
        self.requested: list[str] = []

    def section(self, ref: str, versions: list[tuple[str, str]]) -> Section:
        self.requested.append(ref)
        return self.sections[ref]


def _he(text: str) -> list[VersionText]:
    return [VersionText("he", "A", "Public Domain", [text])]


def test_fetch_work_walks_next_pointers_until_the_last_ref() -> None:
    spec = WorkSpec("SA", "שו״ע", "Halakhah", "SA 1", (("hebrew", "A"),), last_ref="SA 2")
    fake = FakeSefaria(
        {
            "SA 1": _section(_he("יתגבר כארי"), "SA 1", "SA 2"),
            "SA 2": _section(_he("ידיו"), "SA 2", "SA 3"),
            "SA 3": _section(_he("never fetched"), "SA 3", None),
        }
    )
    chunks, _ = fetch_work(fake, spec)
    assert fake.requested == ["SA 1", "SA 2"]
    assert [c.ref for c in chunks] == ["SA 1:1", "SA 2:1"]


def test_commentary_next_pointer_is_trimmed_to_its_section_and_stops_at_another_work() -> None:
    spec = WorkSpec(
        "Rashi on Berakhot",
        "רש״י",
        "Commentary",
        "Rashi on Berakhot 2a",
        (("hebrew", "A"),),
        None,
        "Berakhot",
    )
    fake = FakeSefaria(
        {
            "Rashi on Berakhot 2a": _section(_he("א"), "Rashi on Berakhot 2a", "Rashi on Berakhot 2b:1"),
            "Rashi on Berakhot 2b": _section(_he("ב"), "Rashi on Berakhot 2b", "Rashi on Shabbat 2a:1"),
        }
    )
    chunks, _ = fetch_work(fake, spec)
    assert fake.requested == ["Rashi on Berakhot 2a", "Rashi on Berakhot 2b"]
    assert len(chunks) == 2


# ---- Sefaria client -----------------------------------------------------------------------------


def _client(handler: Any, tmp: Path | None = None) -> SefariaClient:
    return SefariaClient(
        cache_dir=tmp, client=httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0
    )


V3 = {
    "ref": "Berakhot 2a",
    "heRef": "ברכות ב׳ א",
    "next": "Berakhot 2b",
    "sectionNames": ["Daf", "Line"],
    "versions": [
        {"language": "he", "versionTitle": "A", "license": "CC-BY-SA", "text": ["מאימתי"]},
        {"language": "en", "versionTitle": "B", "text": ["From when"]},
    ],
}


def test_section_reads_versions_and_their_licenses(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=V3)

    client = _client(handler, tmp_path)
    s = client.section("Berakhot 2a", [("hebrew", "A"), ("english", "B")])
    assert s.next_ref == "Berakhot 2b"
    assert [(v.title, v.license) for v in s.versions] == [("A", "CC-BY-SA"), ("B", "unknown")]
    assert "version=hebrew%7CA" in seen[0]
    # The second call comes from the disk cache.
    client.section("Berakhot 2a", [("hebrew", "A"), ("english", "B")])
    assert len(seen) == 1
    assert len(list(tmp_path.iterdir())) == 1


def test_retries_rate_limits_then_fails_loudly() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429 if calls["n"] < 2 else 200, json=V3)

    assert _client(handler).section("Berakhot 2a", []).ref == "Berakhot 2a"
    assert calls["n"] == 2

    with pytest.raises(SefariaError, match="404"):
        _client(lambda r: httpx.Response(404, json={})).section("Nowhere 1", [])
    with pytest.raises(SefariaError, match="no such ref"):
        _client(lambda r: httpx.Response(200, json={"error": "no such ref"})).section("Nowhere 1", [])


def test_links_and_calendars_are_parsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/links/" in request.url.path:
            return httpx.Response(
                200,
                json=[
                    {
                        "anchorRef": "Berakhot 2a:1",
                        "ref": "Rashi on Berakhot 2a:1:1",
                        "category": "Commentary",
                    },
                    {"anchorRef": "Berakhot 2a:1"},
                ],
            )
        return httpx.Response(
            200, json={"calendar_items": [{"title": {"en": "Daf Yomi"}, "ref": "Niddah 12"}, "junk"]}
        )

    client = _client(handler)
    links = client.links("Berakhot 2a")
    assert [(link.ref, link.category) for link in links] == [("Rashi on Berakhot 2a:1:1", "Commentary")]
    assert client.calendars("2026-11-09") == [{"title": {"en": "Daf Yomi"}, "ref": "Niddah 12"}]


def test_corpus_chunks_file_roundtrips(tmp_path: Path) -> None:
    he = VersionText("he", "A", "CC-BY-SA", ["מאימתי"])
    (chunk,), _ = chunks_of(_section([he]), WORK, 0)
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(chunk.as_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
    from chavruta.evaluation import load_chunks

    assert load_chunks(path) == [chunk]


@pytest.mark.parametrize(
    "handler",
    [
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused", request=r)),
        lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=r)),
        lambda r: httpx.Response(200, text="<html>challenge</html>"),
    ],
)
def test_network_and_format_failures_become_sefaria_errors(handler: Any) -> None:
    with pytest.raises(SefariaError):
        _client(handler).section("Berakhot 2a", [])


def test_error_responses_are_not_cached(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"error": "temporarily unavailable"})

    client = _client(handler, tmp_path)
    for _ in range(2):
        with pytest.raises(SefariaError):
            client.section("Berakhot 2a", [])
    assert calls["n"] == 2
    assert list(tmp_path.iterdir()) == []
