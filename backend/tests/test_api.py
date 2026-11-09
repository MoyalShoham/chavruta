"""End to end through the HTTP API, on PostgreSQL with pgvector and a real corpus sample.

The model is scripted (no API calls); Hebcal and Sefaria answer from recorded responses."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx2
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker

from chavruta.answer import Chavruta
from chavruta.answer import Settings as AnswerSettings
from chavruta.api import CLIENT_HEADER, AppState, create_app
from chavruta.config import Settings
from chavruta.hebcal import Schedule
from chavruta.library import Library
from chavruta.llm import MissingRecordingError, Mode, RecordingLLM
from chavruta.sefaria import SefariaClient
from chavruta.services import CallLimit, Throttle, ensure_users
from tests.test_answer import FixedReranker, ScriptedLLM, cite, classification
from tests.test_retrieval import HashEmbedder

pytestmark = pytest.mark.db

IL = ZoneInfo("Asia/Jerusalem")
HEBCAL = json.loads((Path(__file__).parent / "fixtures" / "hebcal.json").read_text(encoding="utf-8"))
WEEKDAY = datetime(2027, 4, 20, 10, 0, tzinfo=IL)
PESACH = datetime(2027, 4, 22, 10, 0, tzinfo=IL)
H = {CLIENT_HEADER: "web"}
LEARNER_PW = "learner-password-1"
ADMIN_PW = "admin-password-12"


def sefaria_handler(request: httpx2.Request) -> httpx2.Response:
    if "/links/" in request.url.path:
        return httpx2.Response(
            200,
            json=[
                {"anchorRef": "Berakhot 2a:1", "ref": "Rashi on Berakhot 2a:1:1", "category": "Commentary"},
                {
                    "anchorRef": "Berakhot 2a:1",
                    "ref": "Leviticus 22:7",
                    "category": "Tanakh",
                    "type": "quotation",
                },
                {"anchorRef": "Berakhot 2a:1", "ref": "Jerusalem Talmud Berakhot 1:1", "category": "Talmud"},
            ],
        )
    if "calendars" in request.url.path:
        return httpx2.Response(
            200, json={"calendar_items": [{"title": {"en": "Daf Yomi"}, "ref": "Berakhot 2a"}]}
        )
    return httpx2.Response(404, json={})


def build(
    sessions: sessionmaker[Session],
    llm: Any,
    now: datetime = WEEKDAY,
    closing: str = "auto",
    per_hour: int = 1000,
    data_dir: Path | None = None,
) -> TestClient:
    settings = Settings(learner_password=LEARNER_PW, admin_password=ADMIN_PW, closing=closing)
    if data_dir is not None:
        settings.data_dir = data_dir
    with sessions() as s:
        ensure_users(s, LEARNER_PW, ADMIN_PW)
    library = Library(sessions, HashEmbedder(), FixedReranker())
    bot = Chavruta(library.retriever, library.index, llm, AnswerSettings(relevance_threshold=0.05))
    hebcal = httpx2.Client(
        transport=httpx2.MockTransport(lambda r: httpx2.Response(200, json=HEBCAL["pesach_5787"]))
    )
    state = AppState(
        settings=settings,
        sessions=sessions,
        library=library,
        llm=llm,
        bot=bot,
        schedule=Schedule(client=hebcal),
        sefaria=SefariaClient(
            client=httpx2.Client(transport=httpx2.MockTransport(sefaria_handler)), min_interval=0
        ),
        tz=IL,
        throttle=Throttle(),
        limits=CallLimit(per_hour, per_hour * 5),
        now=lambda: now,
        lock=threading.Lock(),
    )
    return TestClient(create_app(state))


def answer_llm() -> ScriptedLLM:
    return ScriptedLLM(
        classification(),
        [
            {
                "type": "text",
                "text": "The Mishnah opens with the time for the evening Shema.",
                "citations": [cite(0, "x")],
            }
        ],
    )


def sign_in(client: TestClient, user: str = "learner", pw: str = LEARNER_PW) -> None:
    r = client.post("/api/session", json={"username": user, "password": pw}, headers=H)
    assert r.status_code == 200, r.text


@pytest.fixture
def make(loaded: sessionmaker[Session]) -> Callable[..., TestClient]:
    def _make(llm: Any = None, **kw: Any) -> TestClient:
        return build(loaded, llm or answer_llm(), **kw)

    return _make


# ---- sessions ----------------------------------------------------------------------------------------


def test_sign_in_sign_out_revokes_the_session(make: Callable[..., TestClient]) -> None:
    client = make()
    assert client.get("/api/me").status_code == 401
    sign_in(client)
    assert client.get("/api/me").json()["role"] == "LEARNER"
    token = client.cookies.get("chavruta_session")
    assert client.delete("/api/session", headers=H).status_code == 200
    stale = make()
    stale.cookies.set("chavruta_session", token or "")
    assert stale.get("/api/me").status_code == 401  # the old cookie no longer works


def test_wrong_password_is_throttled_per_client_and_user(make: Callable[..., TestClient]) -> None:
    client = make()
    for _ in range(5):
        assert (
            client.post("/api/session", json={"username": "admin", "password": "nope"}, headers=H).status_code
            == 401
        )
    assert (
        client.post("/api/session", json={"username": "admin", "password": ADMIN_PW}, headers=H).status_code
        == 429
    )


def test_writes_need_the_csrf_header(make: Callable[..., TestClient]) -> None:
    client = make()
    sign_in(client)
    assert client.post("/api/notes", json={"ref": "Berakhot 2a:1", "body": "x"}).status_code == 403


def test_app_refuses_to_start_without_real_passwords() -> None:
    for bad in (None, "short", "change-me"):
        with pytest.raises(RuntimeError, match="LEARNER_PASSWORD"):
            Settings(learner_password=bad).password("learner")


def test_forwarded_for_is_ignored_unless_behind_the_bundled_proxy(make: Callable[..., TestClient]) -> None:
    """Rotating X-Forwarded-For must not reset the login throttle when nothing sets it for us."""
    client = make()
    for i in range(5):
        r = client.post(
            "/api/session",
            json={"username": "admin", "password": "nope"},
            headers={**H, "X-Forwarded-For": f"10.0.0.{i}"},
        )
        assert r.status_code == 401
    blocked = client.post(
        "/api/session",
        json={"username": "admin", "password": ADMIN_PW},
        headers={**H, "X-Forwarded-For": "10.9.9.9"},
    )
    assert blocked.status_code == 429


def test_model_calls_are_limited_per_account(make: Callable[..., TestClient]) -> None:
    client = make(per_hour=2)
    sign_in(client)
    for _ in range(2):
        assert client.post("/api/ask", json={"question": "שמע ישראל"}, headers=H).status_code == 200
    r = client.post("/api/ask", json={"question": "שמע ישראל"}, headers=H)
    assert r.status_code == 429
    assert "limit" in r.json()["detail"]


# ---- asking -------------------------------------------------------------------------------------------


def test_ask_returns_cited_sources_with_attribution_and_keeps_the_conversation(
    make: Callable[..., TestClient],
) -> None:
    client = make()
    sign_in(client)
    r = client.post("/api/ask", json={"question": "מאימתי קורין את שמע בערבית"}, headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "answered"
    cited = body["segments"][0]["citations"][0]
    source = body["sources"][cited["source_index"]]
    assert source["ref"] == cited["ref"]
    assert source["url"].startswith("https://www.sefaria.org/")
    assert all(t["license"] in ("Public Domain", "CC-BY", "CC-BY-SA") for t in source["texts"])
    conv = client.get(f"/api/conversations/{body['conversation_id']}").json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant"]
    assert conv["messages"][1]["answer"]["segments"][0]["text"].startswith("The Mishnah")


def test_out_of_scope_is_refused_with_the_scope_explained(make: Callable[..., TestClient]) -> None:
    llm = ScriptedLLM(classification(scope="out_of_scope", kind="other"))
    client = make(llm)
    sign_in(client)
    body = client.post("/api/ask", json={"question": "Give me a lasagna recipe"}, headers=H).json()
    assert body["status"] == "refused"
    assert body["notices"][-1]["kind"] == "refusal"
    assert llm.answer_requests == []


def test_chavruta_mode_on_a_passage(make: Callable[..., TestClient]) -> None:
    llm = answer_llm()
    client = make(llm)
    sign_in(client)
    first = client.post(
        "/api/ask",
        json={"question": "Let's learn", "mode": "chavruta", "anchor_ref": "Berakhot 2a:1"},
        headers=H,
    ).json()
    client.post(
        "/api/ask",
        json={"question": "Why evening first?", "conversation_id": first["conversation_id"]},
        headers=H,
    )
    second = llm.answer_requests[-1]
    assert "Chavruta mode" in second["system"]
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
    assert (
        client.post("/api/ask", json={"question": "x", "anchor_ref": "Nowhere 1:1"}, headers=H).status_code
        == 422
    )


def test_without_a_key_unrecorded_questions_get_a_clear_503(
    make: Callable[..., TestClient], tmp_path: Path
) -> None:
    client = make(RecordingLLM(tmp_path, Mode.REPLAY))
    sign_in(client)
    r = client.post("/api/ask", json={"question": "שמע ישראל"}, headers=H)
    assert r.status_code == 503
    assert "No API key" in r.json()["detail"]
    with pytest.raises(MissingRecordingError):
        RecordingLLM(tmp_path, Mode.REPLAY).create({"model": "m"})


def test_conversations_are_private(make: Callable[..., TestClient]) -> None:
    learner = make()
    sign_in(learner)
    cid = learner.post("/api/ask", json={"question": "שמע ישראל"}, headers=H).json()["conversation_id"]
    admin = make()
    sign_in(admin, "admin", ADMIN_PW)
    assert admin.get(f"/api/conversations/{cid}").status_code == 404
    assert (
        admin.post("/api/ask", json={"question": "q", "conversation_id": cid}, headers=H).status_code == 404
    )


# ---- texts, notes, cards, plans ---------------------------------------------------------------------


def test_sugya_view_aligns_commentaries_to_segments(make: Callable[..., TestClient]) -> None:
    client = make()
    sign_in(client)
    body = client.get("/api/sugya", params={"ref": "Berakhot 2a"}).json()
    first = body["segments"][0]
    assert first["ref"] == "Berakhot 2a:1"
    assert {t["language"] for t in first["texts"]} == {"he", "en"}
    works = {c["work"] for s in body["segments"] for c in s["commentary"]}
    assert works == {"Rashi on Berakhot", "Tosafot on Berakhot"}
    assert all(
        c["ref"].split(":")[1] == s["ref"].split(":")[1] for s in body["segments"] for c in s["commentary"]
    )
    assert {a["license"] for a in body["attribution"]} <= {"Public Domain", "CC-BY", "CC-BY-SA"}
    assert client.get("/api/sugya", params={"ref": "Shabbat 2a"}).status_code == 404
    assert client.get("/api/sugya", params={"ref": "<script>"}).status_code == 422


def test_links_are_grouped_and_marked_when_in_the_library(make: Callable[..., TestClient]) -> None:
    client = make()
    sign_in(client)
    body = client.get("/api/links", params={"ref": "Berakhot 2a"}).json()
    assert set(body["groups"]) == {"Commentary", "Tanakh", "Talmud"}
    assert body["groups"]["Commentary"][0]["in_library"] is True
    assert body["groups"]["Tanakh"][0]["in_library"] is False


def test_notes_crud_and_privacy(make: Callable[..., TestClient]) -> None:
    client = make()
    sign_in(client)
    note = client.post(
        "/api/notes", json={"ref": "Berakhot 2a:1", "body": "ערבית לפני שחרית"}, headers=H
    ).json()
    assert client.get("/api/sugya", params={"ref": "Berakhot 2a"}).json()["notes"][0]["id"] == note["id"]
    edited = client.put(
        f"/api/notes/{note['id']}", json={"ref": "Berakhot 2a:1", "body": "edited"}, headers=H
    )
    assert edited.json()["body"] == "edited"
    other = make()
    sign_in(other, "admin", ADMIN_PW)
    assert other.delete(f"/api/notes/{note['id']}", headers=H).status_code == 404
    assert client.delete(f"/api/notes/{note['id']}", headers=H).status_code == 204
    assert client.delete("/api/notes/99999999999", headers=H).status_code == 404


def test_cards_are_scheduled_by_sm2(make: Callable[..., TestClient]) -> None:
    client = make()
    sign_in(client)
    card = client.post(
        "/api/cards",
        json={"ref": "Mishnah Berakhot 1:1", "front": "Until when?", "back": "Midnight"},
        headers=H,
    ).json()
    assert card["id"] in [c["id"] for c in client.get("/api/cards", params={"due": True}).json()]
    reviewed = client.post(f"/api/cards/{card['id']}/review", json={"grade": 4}, headers=H).json()
    assert reviewed["interval"] == 1
    assert reviewed["due"] == "2027-04-21"
    assert card["id"] not in [c["id"] for c in client.get("/api/cards", params={"due": True}).json()]
    assert client.post(f"/api/cards/{card['id']}/review", json={"grade": 9}, headers=H).status_code == 422


def test_card_suggestions_carry_the_passage_reference_set_by_code(make: Callable[..., TestClient]) -> None:
    class CardLLM(ScriptedLLM):
        def create(self, request: dict[str, Any]) -> dict[str, Any]:
            cards = {
                "cards": [
                    {"front": "Who says until midnight?", "back": "The Sages"},
                    {"front": " ", "back": "x"},
                ]
            }
            return {"stop_reason": "end_turn", "content": [{"type": "text", "text": json.dumps(cards)}]}

    client = make(CardLLM(classification()))
    sign_in(client)
    cards = client.post("/api/cards/suggest", json={"unit": "Mishnah Berakhot 1:1"}, headers=H).json()
    assert cards == [
        {"ref": "Mishnah Berakhot 1:1", "front": "Who says until midnight?", "back": "The Sages"}
    ]


def test_work_plan_and_daf_yomi(make: Callable[..., TestClient]) -> None:
    client = make()
    sign_in(client)
    plan = client.post(
        "/api/plans",
        json={"kind": "work", "work_title": "Deuteronomy", "per_day": 1, "start_date": "2027-04-20"},
        headers=H,
    ).json()
    today = client.get(f"/api/plans/{plan['id']}/today").json()
    assert today["units"] == [{"ref": "Deuteronomy 6", "done": False}]
    client.post(f"/api/plans/{plan['id']}/progress", json={"unit": "Deuteronomy 6"}, headers=H)
    assert client.get(f"/api/plans/{plan['id']}/today").json()["done"] == 1
    daf = client.get("/api/daf-yomi").json()
    assert daf == {
        "date": "2027-04-20",
        "ref": "Berakhot 2a",
        "in_library": True,
        "url": "https://www.sefaria.org/Berakhot_2a",
    }
    bad = client.post("/api/plans", json={"kind": "work", "work_title": "Nowhere"}, headers=H)
    assert bad.status_code == 422


# ---- corpus management --------------------------------------------------------------------------------


def test_corpus_dashboard_is_admin_only_and_shows_licenses(make: Callable[..., TestClient]) -> None:
    learner = make()
    sign_in(learner)
    assert learner.get("/api/corpus").status_code == 403
    admin = make()
    sign_in(admin, "admin", ADMIN_PW)
    body = admin.get("/api/corpus").json()
    berakhot = next(w for w in body["works"] if w["title"] == "Berakhot")
    assert {(v["language"], v["license"], v["included"]) for v in berakhot["versions"]} == {
        ("he", "CC-BY-SA", True),
        ("en", "Public Domain", True),
    }
    assert berakhot["embedded"] == "28/28"


def test_disabling_a_work_removes_it_from_retrieval(make: Callable[..., TestClient]) -> None:
    admin = make()
    sign_in(admin, "admin", ADMIN_PW)
    work = next(w for w in admin.get("/api/corpus").json()["works"] if w["title"] == "Deuteronomy")
    before = admin.get("/api/corpus").json()["library"]["chunks"]
    r = admin.post(f"/api/corpus/works/{work['id']}/enabled", json={"enabled": False}, headers=H)
    try:
        assert r.json()["library"]["chunks"] == before - 50
        assert admin.get("/api/sugya", params={"ref": "Deuteronomy 6"}).status_code == 404
    finally:
        admin.post(f"/api/corpus/works/{work['id']}/enabled", json={"enabled": True}, headers=H)


def test_strictness_is_configurable_by_the_admin(make: Callable[..., TestClient]) -> None:
    admin = make()
    sign_in(admin, "admin", ADMIN_PW)
    assert admin.post("/api/corpus/strictness", json={"strictness": "strict"}, headers=H).json() == {
        "strictness": "strict"
    }
    assert admin.get("/api/corpus").json()["strictness"] == "strict"
    assert admin.post("/api/corpus/strictness", json={"strictness": "anything"}, headers=H).status_code == 422


# ---- Shabbat and Yom Tov -----------------------------------------------------------------------------


def test_closed_on_yom_tov_in_the_backend(make: Callable[..., TestClient]) -> None:
    client = make(now=PESACH)
    status = client.get("/api/status").json()
    assert status["closed"] is True
    assert status["reason"] == "Pesach I"
    assert client.get("/api/health").status_code == 200
    for method, path in (
        ("GET", "/api/me"),
        ("POST", "/api/session"),
        ("POST", "/api/ask"),
        ("GET", "/api/sugya?ref=Berakhot%202a"),
        ("GET", "/api/corpus"),
    ):
        r = client.request(method, path, headers=H, json={})
        assert r.status_code == 423, path
        assert r.json()["until"].startswith("2027-04-22T19:51")


def test_closing_can_be_forced_for_a_demo_but_not_disabled(make: Callable[..., TestClient]) -> None:
    client = make(closing="closed")
    assert client.get("/api/status").json()["closed"] is True
    with pytest.raises(ValueError, match="closing"):
        Settings(closing="open")


def test_database_password_comes_from_a_file(tmp_path: Path) -> None:
    secret = tmp_path / "db_password"
    secret.write_text("p@ss/word\n", encoding="utf-8")
    s = Settings(database_url="postgresql+psycopg://chavruta@db:5432/chavruta", db_password_file=str(secret))
    assert s.dsn() == "postgresql+psycopg://chavruta:p%40ss%2Fword@db:5432/chavruta"
    assert s.db_host == "db"
