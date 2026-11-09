"""Sefaria lookups: planning, open licenses only, what is fetched when, how it reaches the answer,
and that a question the library answers is untouched."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pytest

import chavruta.answer as answer_module
from chavruta.answer import (
    FETCHED,
    LOOKUP_FAILED,
    NO_SOURCE,
    NO_SOURCE_LOOKED_UP,
    Chavruta,
    CorpusIndex,
    Settings,
    Source,
)
from chavruta.llm import MissingRecordingError
from chavruta.lookup import PLANNER_SYSTEM, Plan, SefariaLookup, parse_plan, plan_request, start_of
from chavruta.mentions import families_of, unretrieved
from chavruta.retrieval import Retriever
from chavruta.scope import Strictness
from chavruta.sefaria import SefariaClient, SefariaError, SefariaUnreachable
from tests.test_answer import ALL, cite, classification, text_response
from tests.test_retrieval import HashEmbedder, chunk

# ---- a small fake Sefaria ---------------------------------------------------------------------------

NC = "CC-BY-NC"
TALMUD = ["Talmud", "Bavli", "Seder Nezikin"]


def version(
    lang: str, title: str, license: str, text: Any = None, actual: str | None = None
) -> dict[str, Any]:
    family = {"he": "hebrew", "en": "english", "de": "german"}[actual or lang]
    v = {"language": lang, "languageFamilyName": family, "actualLanguage": actual or lang,
         "versionTitle": title, "license": license}  # fmt: skip
    return v if text is None else {**v, "text": text}


BB_HE = ["כותל חצר שנפל מחייבין אותו לבנות עד ארבע אמות", "אמר רבינא הב לי כמה דגדרי"]
BB_AVAILABLE = [
    version("he", "William Davidson Edition - Vocalized Aramaic", NC),
    version("en", "William Davidson Edition - English", NC),
    version("he", "Wikisource Talmud Bavli", "CC-BY-SA"),
    version("en", "Sefaria Community Translation", "CC0"),
    version("en", "Talmud Bavli. German trans. [de]", "Public Domain", actual="de"),
]
PSALM_119 = [f"פסוק {i}" for i in range(1, 177)]


def text_data(
    ref: str,
    section: str,
    index: str,
    categories: list[str],
    versions: list[dict[str, Any]],
    available: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "ref": ref,
        "sectionRef": section,
        "indexTitle": index,
        "categories": categories,
        "versions": versions,
        "available_versions": available or [{k: v for k, v in x.items() if k != "text"} for x in versions],
    }


def open_he(ref: str, section: str, index: str, categories: list[str], text: Any) -> dict[str, Any]:
    return text_data(ref, section, index, categories, [version("he", "Open", "Public Domain", text)])


class FakeSefaria:
    def __init__(self, unreachable: bool = False, broken: bool = False) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.unreachable = unreachable
        self.broken = broken

    def text(self, ref: str, versions: list[str]) -> dict[str, Any]:
        self.calls.append(("text", (ref, tuple(versions))))
        if self.unreachable:
            raise SefariaUnreachable("Sefaria unreachable: ConnectError")
        if self.broken:
            raise ValueError("damaged response")
        if ref == "Bava Batra 5a:1":
            return open_he(ref, "Bava Batra 5a", "Bava Batra", TALMUD, BB_HE[0])
        if ref == "Bava Batra 5a":
            if versions == ["hebrew", "english"]:  # the defaults: not openly licensed
                return text_data(ref, ref, "Bava Batra", TALMUD, [
                    version("he", "William Davidson Edition - Vocalized Aramaic", NC, ["מנוקד"]),
                    version("en", "William Davidson Edition - English", NC, ["english"]),
                ], BB_AVAILABLE)  # fmt: skip
            return text_data(ref, ref, "Bava Batra", TALMUD, [
                version("he", "Wikisource Talmud Bavli", "CC-BY-SA", BB_HE),
                version("en", "Sefaria Community Translation", "CC0", ["A courtyard wall that fell", ""]),
            ])  # fmt: skip
        if ref == "Rashi on Bava Batra 5a":
            return text_data(ref, ref + ":1", "Rashi on Bava Batra", ["Talmud", "Bavli", "Commentary"], [
                version("he", "Vilna Edition", "Public Domain", [["כותל - של אבנים"], ["הב לי - תן לי"]]),
            ])  # fmt: skip
        if ref == "Eruvin 13b:10":
            return open_he(ref, "Eruvin 13b", "Eruvin", ["Talmud", "Bavli"], "אלו ואלו דברי אלהים חיים")
        if ref in ("Psalms 119:105", "Psalms 119"):
            text = PSALM_119[104] if ref.endswith("105") else PSALM_119
            return open_he(ref, "Psalms 119", "Psalms", ["Tanakh", "Writings"], text)
        if ref == "Genesis 1-50":
            raise AssertionError("a range is narrowed before it is asked for")
        if ref == "Bava Kamma 2":  # Sefaria reads a daf number without an amud as both amudim
            return open_he("Bava Kamma 2a-2b", "Bava Kamma 2a-2b", "Bava Kamma", TALMUD, [["a"], ["b"]])
        if ref == "Bava Kamma 2a":
            return open_he(ref, ref, "Bava Kamma", TALMUD, ["ארבעה אבות נזיקין"])
        if ref == "Mishnah Brachot 1:1":  # an alias Sefaria knows; the library holds the place
            return open_he("Mishnah Berakhot 1:1", "Mishnah Berakhot 1", "Mishnah Berakhot", ["Mishnah"], "x")
        if ref == "Zohar 1:1a":
            return open_he(ref, "Zohar 1", "Zohar", ["Kabbalah"], "בריש הורמנותא דמלכא")
        if ref == "Berakhot":
            return text_data("Berakhot", "Berakhot", "Berakhot", ["Talmud", "Bavli"], [])
        return {"error": f"Could not find {ref}"}

    def search(self, query: str, size: int = 20) -> list[dict[str, Any]]:
        self.calls.append(("search", query))
        if self.unreachable:
            raise SefariaUnreachable("Sefaria unreachable: ConnectError")
        hits = {
            "אלו ואלו": [
                {"ref": "Jastrow, אלו", "version": "x", "lang": "he", "categories": ["Reference"]},
                {"ref": "Eruvin 13b:10", "version": "Wikisource", "lang": "he", "categories": []},
                {"ref": "Eruvin 13b:10", "version": "William Davidson", "lang": "he", "categories": []},
                {"ref": "Mishnah Berakhot 1:1", "version": "Wikisource", "lang": "he", "categories": []},
            ]
        }
        return hits.get(query, [])

    def texts_asked(self) -> list[str]:
        return [args[0] for kind, args in self.calls if kind == "text"]


class PlannerLLM:
    """Answers planner requests with a fixed plan, classifier requests with a classification and
    answer requests with fixed blocks."""

    def __init__(
        self,
        plan: dict[str, Any],
        cls: dict[str, Any] | None = None,
        blocks: list[dict[str, Any]] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.plan = plan
        self.cls = cls or classification()
        self.blocks = blocks or [{"type": "text", "text": "x"}]
        self.delay = delay
        self.requests: list[dict[str, Any]] = []

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        if request["system"] == PLANNER_SYSTEM:
            time.sleep(self.delay)
            return text_response(json.dumps(self.plan))
        if request["model"] == "claude-haiku-5-5":
            return text_response(json.dumps(self.cls))
        return {"model": "claude-opus-5-5", "stop_reason": "end_turn", "content": self.blocks, "usage": {}}

    @property
    def planner_requests(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["system"] == PLANNER_SYSTEM]

    @property
    def answer_requests(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["model"] == "claude-opus-5-5"]


def lookup(sefaria: FakeSefaria | None = None, **kw: Any) -> SefariaLookup:
    return SefariaLookup(sefaria or FakeSefaria(), PlannerLLM({"refs": [], "queries": []}), **kw)


def nothing(_: str) -> bool:
    return False


def in_berakhot(ref: str) -> bool:
    return ref.startswith(("Mishnah Berakhot", "Berakhot"))


# ---- the plan -----------------------------------------------------------------------------------------


def test_plan_request_is_a_small_structured_call_with_the_follow_up_context() -> None:
    req = plan_request("and Rashi?", context="What does Bava Batra 5a say?")
    assert req["model"] == "claude-haiku-5-5"
    assert req["output_config"]["format"]["type"] == "json_schema"
    content = req["messages"][0]["content"]
    assert content == "Earlier question: What does Bava Batra 5a say?\n\nFollow-up: and Rashi?"
    assert "do not answer" in req["system"].lower()


@pytest.mark.parametrize(
    "response",
    [
        {"stop_reason": "refusal", "content": []},
        {"stop_reason": "max_tokens", "content": [{"type": "text", "text": '{"refs": ["A'}]},
        text_response("not json"),
        text_response('{"refs": "Bava Batra 5a"}'),
    ],
)
def test_a_planner_that_fails_plans_nothing(response: dict[str, Any]) -> None:
    assert parse_plan(response) == Plan()


def test_a_plan_is_deduplicated_capped_and_never_carries_a_path() -> None:
    refs = ["A 1", "A 1", "../../sheets/12345", "B 2?x=1", "B 2", "C 3", "D 4"]
    planned = parse_plan(text_response(json.dumps({"refs": refs, "queries": [" q ", ""]})))
    assert planned == Plan(("A 1", "B 2", "C 3"), ("q",))


def test_a_range_is_narrowed_to_where_it_starts() -> None:
    assert start_of("Deuteronomy 6:4-9") == "Deuteronomy 6:4"
    assert start_of("Genesis 1-50") == "Genesis 1"
    assert start_of("Bava Batra 5a") == "Bava Batra 5a"


# ---- fetching -----------------------------------------------------------------------------------------


def test_a_named_daf_is_fetched_in_open_versions_with_rashi_and_never_in_closed_ones() -> None:
    s = FakeSefaria()
    found = lookup(s).fetch(Plan(("Bava Batra 5a",)), nothing, search=False)
    assert found.fetched_refs == ["Bava Batra 5a", "Rashi on Bava Batra 5a"]
    assert {(c.version, c.license) for c in found.chunks} == {
        ("Wikisource Talmud Bavli", "CC-BY-SA"),
        ("Sefaria Community Translation", "CC0"),
        ("Vilna Edition", "Public Domain"),
    }
    assert {s.license for s in found.skipped} == {NC}
    # The open versions are asked for by name, one per language; the German "translation" never is.
    asked = [v for kind, (ref, vs) in s.calls if kind == "text" and ref == "Bava Batra 5a" for v in vs]
    assert asked == [
        "hebrew",
        "english",
        "hebrew|Wikisource Talmud Bavli",
        "english|Sefaria Community Translation",
    ]
    daf = [c for c in found.chunks if c.work == "Bava Batra"]
    assert [(c.ref, c.language) for c in daf] == [
        ("Bava Batra 5a:1", "he"),
        ("Bava Batra 5a:2", "he"),
        ("Bava Batra 5a:1", "en"),
    ]
    assert {c.category for c in daf} == {"Talmud"}
    rashi = [c for c in found.chunks if c.work == "Rashi on Bava Batra"]
    assert rashi[0].ref == "Rashi on Bava Batra 5a:1:1" and rashi[0].category == "Commentary"
    assert not found.searched and found.errors == []  # a missing Tosafot is not an error


def test_two_planned_places_in_one_section_fetch_it_once() -> None:
    found = lookup().fetch(Plan(("Bava Batra 5a", "Bava Batra 5a:1")), nothing, search=False)
    assert found.fetched_refs == ["Bava Batra 5a", "Rashi on Bava Batra 5a"]
    assert len([c for c in found.chunks if c.ref == "Bava Batra 5a:1" and c.language == "he"]) == 1


def test_a_place_the_library_holds_is_not_fetched_even_under_another_name_or_as_a_range() -> None:
    s = FakeSefaria()
    plan = Plan(("Mishnah Berakhot 1:1-2", "Mishnah Brachot 1:1"))
    found = lookup(s).fetch(plan, in_berakhot, search=False)
    assert found.chunks == [] and found.fetched_refs == [] and not found.searched
    assert found.in_library == ["Mishnah Berakhot 1:1"]  # the alias, resolved by Sefaria
    assert s.texts_asked() == ["Mishnah Brachot 1:1"]  # the range was known locally


def test_search_skips_reference_works_and_library_passages_and_fetches_each_passage_once() -> None:
    s = FakeSefaria()
    found = lookup(s).fetch(Plan((), ("אלו ואלו",)), in_berakhot, search=True)
    assert found.searched and found.fetched_refs == ["Eruvin 13b:10"]
    assert [c.ref for c in found.chunks] == ["Eruvin 13b:10"]
    assert s.texts_asked() == ["Eruvin 13b:10"]


def test_a_named_place_that_cannot_be_fetched_is_an_error_and_search_still_runs() -> None:
    found = lookup().fetch(Plan(("Eruvin 999z",), ("אלו ואלו",)), in_berakhot, search=True)
    assert found.searched and found.fetched_refs == ["Eruvin 13b:10"]
    assert found.errors == ["Eruvin 999z: Could not find Eruvin 999z"]
    quiet = lookup().fetch(Plan(("Eruvin 999z",), ("אלו ואלו",)), in_berakhot, search=False)
    assert not quiet.searched and quiet.chunks == []


def test_a_whole_book_or_a_range_is_never_fetched_whole() -> None:
    found = lookup().fetch(Plan(("Berakhot",)), nothing, search=False)
    assert found.chunks == [] and "not one specific place" in found.errors[0]
    s = FakeSefaria()
    found = lookup(s).fetch(Plan(("Genesis 1-50", "Bava Kamma 2")), nothing, search=False)
    assert "Genesis 1-50" not in s.texts_asked() and "Genesis 1" in s.texts_asked()
    assert "Bava Kamma 2a" in found.fetched_refs  # "Bava Kamma 2" is read as 2a-2b: 2a is taken


def test_a_long_chapter_is_cut_around_the_verse_asked_about() -> None:
    found = lookup(max_segments=60).fetch(Plan(("Psalms 119:105",)), nothing, search=False)
    refs = [c.ref for c in found.chunks]
    assert "Psalms 119:105" in refs and len(refs) == 60
    assert refs[0] == "Psalms 119:75" and refs[-1] == "Psalms 119:134"


def test_the_passage_limit_and_the_deadline_bound_a_lookup() -> None:
    found = lookup(max_segments=200).fetch(Plan(("Psalms 119",)), nothing, search=False)
    assert len(found.chunks) == 176  # one version, under the limit of 240
    s = FakeSefaria()
    late = lookup(s, deadline=0.0).fetch(Plan(("Bava Batra 5a",), ("אלו ואלו",)), nothing, search=True)
    assert late.chunks == [] and s.calls == [] and "time limit" in late.errors[0]


def test_a_work_disabled_in_the_library_is_not_fetched_from_sefaria_either() -> None:
    found = lookup().fetch(Plan(("Zohar 1:1a",)), nothing, search=False, blocked=lambda w: w == "Zohar")
    assert found.chunks == [] and found.errors == ["Zohar 1:1a: Zohar is disabled in the library"]


def test_any_failure_inside_a_fetch_is_that_ref_s_error_not_the_answer_s() -> None:
    found = lookup(FakeSefaria(broken=True)).fetch(Plan(("Bava Batra 5a",)), nothing, search=False)
    assert found.chunks == [] and "ValueError: damaged response" in found.errors[0]


def test_sefaria_unreachable_stops_the_lookup_after_the_first_failures() -> None:
    s = FakeSefaria(unreachable=True)
    found = lookup(s).fetch(Plan(("Bava Batra 5a",), ("אלו ואלו",)), nothing, search=True)
    assert found.unreachable and found.chunks == [] and not found.searched
    assert [kind for kind, _ in s.calls] == ["text"]


def test_a_planner_without_a_recording_or_key_skips_the_lookup() -> None:
    class NoKey:
        def create(self, request: dict[str, Any]) -> dict[str, Any]:
            raise MissingRecordingError("no recording")

    s = FakeSefaria()
    found = SefariaLookup(s, NoKey()).find("Bava Batra 5a?", None, nothing, search=True)
    assert found.chunks == [] and s.calls == [] and not found.attempted


# ---- the Sefaria client -------------------------------------------------------------------------------


def _client(handler: Any, tmp: Path, **kw: Any) -> SefariaClient:
    transport = httpx.MockTransport(handler)
    return SefariaClient(cache_dir=tmp, client=httpx.Client(transport=transport), min_interval=0, **kw)


def test_text_of_a_version_that_is_not_openly_licensed_is_never_stored(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        data = text_data("Bava Batra 5a", "Bava Batra 5a", "Bava Batra", TALMUD, [
            version("he", "Wikisource", "CC-BY-SA", ["פתוח"]),
            version("en", "William Davidson Edition - English", NC, ["closed english text"]),
        ])  # fmt: skip
        return httpx.Response(200, json={**data, "huge": ["x"] * 1000})

    data = _client(handler, tmp_path).text("Bava Batra 5a", ["hebrew", "english"])
    stored = "".join(p.read_text(encoding="utf-8") for p in tmp_path.iterdir())
    assert "closed english text" not in stored and "closed english text" not in json.dumps(data)
    assert "פתוח" in stored and "huge" not in stored
    closed = next(v for v in data["versions"] if v["license"] == NC)
    assert closed["hasText"] is True and "text" not in closed
    assert not list(tmp_path.glob("*.tmp"))  # written atomically


def test_client_records_what_a_lookup_saw_and_replays_it_offline(tmp_path: Path) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        if request.method == "POST":
            if json.loads(request.content)["query"] == "broken":
                return httpx.Response(200, json={"error": "search is down"})
            hit = {"_source": {"ref": "Eruvin 13b:10", "version": "v", "lang": "he", "categories": [],
                               "exact": "long text " * 50}}  # fmt: skip
            return httpx.Response(200, json={"hits": {"hits": [hit]}})
        if "Zevulun" in str(request.url):
            return httpx.Response(404, json={"error": "Could not find title"})
        if "Berakhot" in str(request.url):
            return httpx.Response(
                200, content=b'{"ref": "Berakhot", "versions": [' + b'"x",' * 400 + b'"x"]}'
            )
        v = version("he", "W", "CC-BY-SA", ["a"])
        return httpx.Response(200, json=text_data("Eruvin 13b", "Eruvin 13b", "Eruvin", ["Talmud"], [v]))

    live = _client(handler, tmp_path, max_bytes=1000)
    assert live.search("q") == [{"ref": "Eruvin 13b:10", "version": "v", "lang": "he", "categories": []}]
    assert live.text("Eruvin 13b", ["hebrew", "english"])["ref"] == "Eruvin 13b"
    assert live.text("Mishnah Zevulun 1:1", ["hebrew"]) == {"error": "not found (404)"}
    assert live.text("Berakhot 200a", ["hebrew"]) == {"error": "response larger than 1 KB"}
    with pytest.raises(SefariaError, match="search failed"):
        live.search("broken")
    assert seen[0] == "POST /api/search-wrapper/es8" and seen[1].startswith("GET /api/v3/texts/Eruvin")

    def offline_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("an offline client never sends a request")

    replay = _client(offline_handler, tmp_path, offline=True)
    assert replay.search("q")[0]["ref"] == "Eruvin 13b:10"
    # An unknown place and an oversized one replay as what they were; a failed search was not kept.
    assert replay.text("Mishnah Zevulun 1:1", ["hebrew"]) == {"error": "not found (404)"}
    assert replay.text("Berakhot 200a", ["hebrew"]) == {"error": "response larger than 1 KB"}
    with pytest.raises(SefariaUnreachable, match="no recorded response"):
        replay.search("broken")


def test_a_damaged_cache_file_is_a_miss(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=open_he("Eruvin 13b", "Eruvin 13b", "Eruvin", ["Talmud"], ["a"]))

    client = _client(handler, tmp_path)
    client.text("Eruvin 13b", ["hebrew"])
    for p in tmp_path.iterdir():
        p.write_text('{"ref": "Eruv', encoding="utf-8")  # as a crash mid-write would have left it
    assert client.text("Eruvin 13b", ["hebrew"])["ref"] == "Eruvin 13b" and len(calls) == 2


def test_a_ref_cannot_reach_another_api_path(tmp_path: Path) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.raw_path.decode())
        return httpx.Response(200, json={"error": "Could not find"})

    _client(handler, tmp_path).text("../../sheets/12345", ["hebrew"])
    assert paths[0].startswith("/api/v3/texts/..%2F..%2Fsheets%2F12345")


# ---- in the answer --------------------------------------------------------------------------------------


class KeywordReranker:
    """Relevant if the passage shares a word of four letters or more with the question."""

    def scores(self, query: str, passages: list[str]) -> list[float]:
        words = {w for w in query.split() if len(w) >= 4}
        return [0.9 if words & set(p.split()) else 0.01 for p in passages]


def bot(
    llm: Any, sefaria: FakeSefaria | None = None, strictness: Strictness = Strictness.BALANCED
) -> Chavruta:
    e = HashEmbedder()
    r = Retriever(ALL, e, e.passages([c.text for c in ALL]), KeywordReranker())
    settings = Settings(strictness=strictness, relevance_threshold=0.05)
    return Chavruta(r, CorpusIndex(ALL), llm, settings, lookup=SefariaLookup(sefaria or FakeSefaria(), llm))


def plain_bot(llm: Any) -> Chavruta:
    e = HashEmbedder()
    r = Retriever(ALL, e, e.passages([c.text for c in ALL]), KeywordReranker())
    return Chavruta(r, CorpusIndex(ALL), llm, Settings(relevance_threshold=0.05))


def test_a_question_on_a_daf_outside_the_library_is_answered_from_it_with_a_notice() -> None:
    llm = PlannerLLM({"refs": ["Bava Batra 5a"], "queries": []}, blocks=[
        {"type": "text", "text": "The wall is rebuilt to four cubits.", "citations": [cite(0, BB_HE[0])]}
    ])  # fmt: skip
    a = bot(llm).ask("בבא בתרא דף ה עמוד א: כותל חצר שנפל")
    assert a.status.value == "answered"
    assert a.sources[0].ref == "Bava Batra 5a:1" and a.sources[0].fetched
    sent = llm.answer_requests[0]["messages"][-1]["content"][0]
    assert sent["type"] == "search_result" and sent["title"] == "Bava Batra 5a:1 (Bava Batra)"
    assert sent["source"] == "https://www.sefaria.org/Bava_Batra_5a.1"
    assert FETCHED in a.notices and not a.unretrieved
    assert a.lookup is not None and a.lookup["fetched"] == ["Bava Batra 5a", "Rashi on Bava Batra 5a"]


@pytest.mark.parametrize(
    "plan",
    [
        {"refs": ["Mishnah Berakhot 1:1"], "queries": ["קורין את שמע"]},
        {"refs": ["Mishnah Berakhot 1:1-2"], "queries": []},  # a range starting in the library
        {"refs": ["Mishnah Brachot 1:1"], "queries": []},  # an alias Sefaria resolves into it
    ],
)
def test_a_question_the_library_answers_sends_the_same_request_as_without_lookup(
    plan: dict[str, Any],
) -> None:
    """The library's recorded evaluation stays valid: planning a place the library holds changes
    nothing that reaches the answering model."""
    question = "מאימתי קורין את שמע בערבית"
    with_lookup = PlannerLLM(plan)
    a = bot(with_lookup).ask(question)
    without = PlannerLLM({"refs": [], "queries": []})
    plain_bot(without).ask(question)
    assert with_lookup.answer_requests == without.answer_requests
    assert FETCHED not in a.notices and not any(s.fetched for s in a.sources)


def test_a_place_both_hold_comes_from_the_library() -> None:
    b = plain_bot(PlannerLLM({"refs": [], "queries": []}))
    library = b.retrieve("שמע ישראל")
    duplicate = [chunk("Deuteronomy 6:4", "שמע ישראל יהוה אלהינו יהוה אחד"), chunk("Psalms 1:1", "אשרי האיש")]
    merged = b.merge("שמע ישראל", library, duplicate)
    assert [(s.ref, s.fetched) for s in merged if s.ref == "Deuteronomy 6:4"] == [("Deuteronomy 6:4", False)]


@pytest.mark.parametrize(
    ("cls", "strictness"),
    [
        (classification(scope="out_of_scope"), Strictness.BALANCED),
        (classification(scope="out_of_scope"), Strictness.LENIENT),
        (classification(manipulation=True), Strictness.BALANCED),
        (classification(sensitive=True, sensitive_area="mourning"), Strictness.BALANCED),
        (classification(scope="borderline"), Strictness.STRICT),
    ],
)
def test_refused_or_sensitive_questions_never_fetch_anything(
    cls: dict[str, Any], strictness: Strictness
) -> None:
    s = FakeSefaria()
    llm = PlannerLLM({"refs": ["Bava Batra 5a"], "queries": ["אלו ואלו"]}, cls=cls)
    a = bot(llm, s, strictness).ask("מאימתי קורין את שמע בערבית, ובבא בתרא ה.?")
    assert s.calls == [] and not any(src.fetched for src in a.sources)
    if cls["scope"] == "out_of_scope" or cls["manipulation"] or cls["sensitive"]:
        assert llm.planner_requests == []


def test_nothing_found_anywhere_says_so_and_still_does_not_answer_from_memory() -> None:
    llm = PlannerLLM({"refs": [], "queries": ["אין כזה"]})
    a = bot(llm).ask("Zohar elephants?")
    assert a.status.value == "no_source" and a.notices[-1] == NO_SOURCE_LOOKED_UP
    assert llm.answer_requests == []


def test_without_a_key_the_lookup_is_skipped_and_the_library_answer_stands() -> None:
    class Replay(PlannerLLM):
        def create(self, request: dict[str, Any]) -> dict[str, Any]:
            if request["system"] == PLANNER_SYSTEM:
                raise MissingRecordingError("no recording")
            return super().create(request)

    a = bot(Replay({"refs": [], "queries": []})).ask("Zohar elephants?")
    assert a.status.value == "no_source" and a.notices[-1] == NO_SOURCE


def test_a_slow_planner_is_not_waited_for(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(answer_module, "PLAN_TIMEOUT", 0.05)
    s = FakeSefaria()
    a = bot(PlannerLLM({"refs": ["Bava Batra 5a"], "queries": []}, delay=0.5), s).ask("Zohar elephants?")
    assert a.status.value == "no_source" and a.notices[-1] == NO_SOURCE and s.calls == []
    assert a.lookup is not None and a.lookup["errors"] == ["planner timed out: lookup skipped"]


def test_sefaria_down_is_reported_without_claiming_a_lookup_was_made() -> None:
    llm = PlannerLLM({"refs": ["Bava Batra 5a"], "queries": []})
    a = bot(llm, FakeSefaria(unreachable=True)).ask("Bava Batra 5a about a fallen wall?")
    assert a.status.value == "no_source"
    assert a.notices == [LOOKUP_FAILED, NO_SOURCE]


def test_sefaria_down_is_not_mentioned_when_the_library_answered_on_its_own() -> None:
    a = bot(PlannerLLM({"refs": [], "queries": []}), FakeSefaria(unreachable=True)).ask("מאימתי קורין את שמע")
    assert a.status.value == "answered" and LOOKUP_FAILED not in a.notices


def test_a_failing_ranking_keeps_the_library_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: Any, **kw: Any) -> list[Source]:
        raise RuntimeError("reranker out of memory")

    b = bot(PlannerLLM({"refs": ["Bava Batra 5a"], "queries": []}))
    monkeypatch.setattr(b, "merge", broken)
    a = b.ask("מאימתי קורין את שמע, ובבא בתרא ה.")
    assert a.status.value == "answered" and not any(s.fetched for s in a.sources)
    assert a.lookup is not None and "ranking failed" in a.lookup["errors"][-1]


def test_a_fetched_commentary_counts_for_the_mention_check() -> None:
    assert families_of("Rashi on Bava Batra") == {"Rashi", "Bava Batra"}
    assert "Avot" in families_of("Pirkei Avot") and "Midrash" in families_of("Bereshit Rabbah")
    assert "Berakhot" not in families_of("Jerusalem Talmud Berakhot")
    assert families_of("Berakhot") == {"Berakhot"}
    text = 'רש"י מסביר, ובבא בתרא ה. אומרת'
    assert unretrieved(text, {"Rashi on Bava Batra", "Bava Batra"}, [], ["Bava Batra 5a:1"]) == []
    flagged = unretrieved(text, {"Berakhot"}, [], ["Berakhot 2a:1"])
    assert [m.family for m in flagged] == ["Rashi", "Bava Batra"]


def test_library_membership_by_place_section_part_and_range() -> None:
    index = CorpusIndex(ALL)
    assert index.has("Mishnah Berakhot 1:1") and index.has("Mishnah Berakhot 1:1:2")
    assert index.has("Deuteronomy 6:4-9") and index.has("Deuteronomy 6:4")
    assert not index.has("Bava Batra 5a") and not index.has("Deuteronomy 6:5")
    assert NO_SOURCE.kind == NO_SOURCE_LOOKED_UP.kind == "no_source"


# ---- choices made on the lookup's dev split ---------------------------------------------------------


def test_a_keyword_alone_does_not_stop_the_study_of_a_text_but_its_notices_stay() -> None:
    s = FakeSefaria()
    llm = PlannerLLM({"refs": ["Bava Batra 5a"], "queries": []})
    a = bot(llm, s).ask("בבא בתרא דף ה עמוד א: כותל חצר שנפל, ופיקוח נפש")
    assert a.decision is not None and a.decision.sensitive_area == "danger_to_life"
    assert any(src.fetched for src in a.sources) and a.notices[0].kind == "safety"


def test_a_personal_question_with_a_sensitive_keyword_is_not_looked_up() -> None:
    s = FakeSefaria()
    llm = PlannerLLM({"refs": ["Bava Batra 5a"], "queries": []})
    bot(llm, s).ask("אבא שלי במצב של פיקוח נפש, מה אומרת בבא בתרא ה.?")
    assert s.calls == []


class FixedScore:
    def __init__(self, score: float) -> None:
        self.score = score

    def scores(self, query: str, passages: list[str]) -> list[float]:
        return [self.score] * len(passages)


def test_search_also_runs_when_the_library_s_best_source_is_weak() -> None:
    s = FakeSefaria()
    llm = PlannerLLM({"refs": [], "queries": ["אלו ואלו"]})
    e = HashEmbedder()
    weak = Retriever(ALL, e, e.passages([c.text for c in ALL]), FixedScore(0.2))
    Chavruta(weak, CorpusIndex(ALL), llm, Settings(), lookup=SefariaLookup(s, llm)).ask("שאלה כלשהי")
    assert ("search", "אלו ואלו") in s.calls
    s2 = FakeSefaria()
    strong = Retriever(ALL, e, e.passages([c.text for c in ALL]), FixedScore(0.6))
    Chavruta(strong, CorpusIndex(ALL), llm, Settings(), lookup=SefariaLookup(s2, llm)).ask("שאלה כלשהי")
    assert s2.calls == []


def test_fetched_passages_are_ranked_unvocalized_and_with_the_planner_s_phrases() -> None:
    class Literal:
        """Relevant only to an unvocalized passage containing a phrase from the query."""

        def scores(self, query: str, passages: list[str]) -> list[float]:
            return [0.9 if "דינא דמלכותא" in query and "דינא דמלכותא" in p else 0.01 for p in passages]

    e = HashEmbedder()
    b = Chavruta(Retriever(ALL, e, e.passages([c.text for c in ALL]), Literal()), CorpusIndex(ALL), None)  # type: ignore[arg-type]
    vocalized = chunk("Bava Kamma 113a:14", "וְהָאָמַר שְׁמוּאֵל דִּינָא דְמַלְכוּתָא דִּינָא")
    merged = b.merge("the law of the kingdom is the law", [], [vocalized], ["דינא דמלכותא דינא"])
    assert merged[0].ref == "Bava Kamma 113a:14" and merged[0].score == 0.9
    assert merged[0].texts[0].text == vocalized.text  # what is sent keeps its vowel points
    assert b.merge("the law of the kingdom is the law", [], [vocalized])[0].score == 0.01


def test_the_rambam_s_other_works_count_as_the_rambam() -> None:
    assert "Rambam" in families_of("Sefer HaMitzvot, Shorashim")
    assert "Rambam" in families_of("Eight Chapters")
    text = 'הרמב"ם מונה את המצוות'
    assert unretrieved(text, {"Sefer HaMitzvot, Shorashim"}, [], ["Sefer HaMitzvot, Shorashim 1:2"]) == []


def test_a_question_the_library_answers_well_is_not_looked_up_even_if_the_planner_guesses_elsewhere() -> None:
    """The planner may infer a place outside the library ("Jeremiah 12:1") or misspell one inside
    it ("Orach Chaim"); neither matters while the library's best source is strong and the question
    names nothing outside it."""
    s = FakeSefaria()
    llm = PlannerLLM({"refs": ["Bava Batra 5a", "Shulchan Arukh, Orach Chaim 4"], "queries": ["אלו ואלו"]})
    e = HashEmbedder()
    strong = Retriever(ALL, e, e.passages([c.text for c in ALL]), FixedScore(0.6))
    a = Chavruta(strong, CorpusIndex(ALL), llm, Settings(), lookup=SefariaLookup(s, llm)).ask("שאלה כלשהי")
    assert s.calls == [] and a.lookup is None and len(llm.planner_requests) == 1


def test_borderline_questions_are_not_planned() -> None:
    s = FakeSefaria()
    llm = PlannerLLM(
        {"refs": ["Bava Batra 5a"], "queries": ["אלו ואלו"]}, cls=classification(scope="borderline")
    )
    bot(llm, s).ask("מה ההבדל בין חסידים למתנגדים?")
    assert llm.planner_requests == [] and s.calls == []


def test_when_a_lookup_happens_library_places_the_planner_named_compete_on_equal_terms() -> None:
    """The mezuzah question: the library held the verse but retrieval missed it, and commentaries
    found by search quoted it. The planned library place is ranked with them, the same way."""

    class Verse:
        def scores(self, query: str, passages: list[str]) -> list[float]:
            return [0.8 if "ושננתם" in p else 0.3 if "ושננתם" in query else 0.01 for p in passages]

    e = HashEmbedder()
    b = Chavruta(Retriever(ALL, e, e.passages([c.text for c in ALL]), Verse()), CorpusIndex(ALL), None)  # type: ignore[arg-type]
    commentary = chunk("Ibn Ezra on Deuteronomy 6:7:1", "פירוש על הפסוק")
    library = b.retrieve("Genesis")  # a weak first ranking that misses Deuteronomy 6:7
    merged = b.merge(
        "teach them to your children", library, [commentary], ["ושננתם לבניך"], ["Deuteronomy 6:7"]
    )
    assert merged[0].ref == "Deuteronomy 6:7" and not merged[0].fetched and merged[0].score == 0.8


def test_a_book_the_question_names_is_recognised_in_any_of_its_names() -> None:
    from chavruta.mentions import named_in

    assert named_in("מה אומרת הגמרא בבבא מציעא דף נט עמוד ב?", "Bava Metzia 59b")
    assert named_in("What does Rashi say on Numbers 13:2?", "Rashi on Numbers 13:2")
    assert named_in('רמב"ם הלכות תשובה פרק ב', "Mishneh Torah, Repentance 2:2")
    assert not named_in("Where does Moses ask why the righteous suffer?", "Jeremiah 12:1")
    assert not named_in("מה הדין בשניים שמוצאים מציאה?", "Bava Metzia 2a")
    assert not named_in("What does Rashi say on it?", "Rashi on Numbers 13:2")  # the book is not named


def test_a_place_the_question_names_is_fetched_even_when_the_library_looks_strong_but_not_searched() -> None:
    """ "בבבא מציעא דף נט עמוד ב": the library had a strong-looking passage about the same story
    (Berakhot 19a), and the question still asked about Bava Metzia."""
    s = FakeSefaria()
    llm = PlannerLLM({"refs": ["Bava Batra 5a"], "queries": ["אלו ואלו"]})
    e = HashEmbedder()
    strong = Retriever(ALL, e, e.passages([c.text for c in ALL]), FixedScore(0.6))
    b = Chavruta(strong, CorpusIndex(ALL), llm, Settings(), lookup=SefariaLookup(s, llm))
    a = b.ask("מה אומרת הגמרא בבבא בתרא דף ה עמוד א על כותל שנפל?")
    assert any(src.fetched for src in a.sources)
    assert not any(kind == "search" for kind, _ in s.calls)


def test_a_named_place_the_library_turns_out_to_hold_leaves_a_strong_answer_untouched() -> None:
    question = "מה אומרת משנה ברכות על קריאת שמע בערבית, מאימתי קורין את שמע"
    with_lookup = PlannerLLM({"refs": ["Mishnah Brachot 1:1"], "queries": ["קורין את שמע"]})
    a = bot(with_lookup).ask(question)
    without = PlannerLLM({"refs": [], "queries": []})
    plain_bot(without).ask(question)
    assert with_lookup.answer_requests == without.answer_requests and not any(s.fetched for s in a.sources)
