"""HTTP API for the study dashboard.

- Sign-in sets an HttpOnly, SameSite=Strict cookie holding a random token; the server keeps its hash
  and revokes it on sign-out. State-changing requests must also send X-Chavruta-Client (CSRF).
- From candle lighting to havdalah every endpoint except /api/status and /api/health answers 423:
  the app is closed on Shabbat and Yom Tov in the backend, not only on screen.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from chavruta import services
from chavruta.answer import Chavruta
from chavruta.config import Settings
from chavruta.db import Role, User
from chavruta.hebcal import Schedule, Status
from chavruta.library import Library
from chavruta.llm import LLM, MissingRecordingError
from chavruta.scope import Strictness
from chavruta.sefaria import SefariaClient, SefariaError

COOKIE = "chavruta_session"
CLIENT_HEADER = "X-Chavruta-Client"
REF_PATTERN = r"^[A-Za-z0-9 ,.:'\-]{1,200}$"
OPEN_PATHS = ("/api/status", "/api/health")


class Login(BaseModel):
    username: str = Field(min_length=1, max_length=60)
    password: str = Field(min_length=1, max_length=200)


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = Field(default=None, ge=1, le=2**31 - 1)
    mode: str = Field(default="ask", pattern="^(ask|chavruta)$")
    anchor_ref: str | None = Field(default=None, pattern=REF_PATTERN)


class NoteIn(BaseModel):
    ref: str = Field(pattern=REF_PATTERN)
    body: str = Field(min_length=1, max_length=10_000)


class CardIn(BaseModel):
    ref: str = Field(pattern=REF_PATTERN)
    front: str = Field(min_length=1, max_length=1000)
    back: str = Field(min_length=1, max_length=2000)
    origin: str = Field(default="manual", pattern="^(manual|suggested)$")


class ReviewIn(BaseModel):
    grade: int = Field(ge=0, le=5)


class PlanIn(BaseModel):
    kind: str = Field(pattern="^(daf_yomi|work)$")
    work_title: str | None = Field(default=None, max_length=200)
    per_day: int = Field(default=1, ge=1, le=50)
    start_date: date | None = None


class ProgressIn(BaseModel):
    unit: str = Field(pattern=REF_PATTERN)


class EnabledIn(BaseModel):
    enabled: bool


class StrictnessIn(BaseModel):
    strictness: str = Field(pattern="^(strict|balanced|lenient)$")


@dataclass
class AppState:
    settings: Settings
    sessions: sessionmaker[Session]
    library: Library
    llm: LLM
    bot: Chavruta
    schedule: Schedule
    sefaria: SefariaClient
    tz: ZoneInfo
    throttle: services.Throttle
    limits: services.CallLimit
    now: Callable[[], datetime]
    lock: threading.Lock


class CsrfMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        unsafe = request.method not in ("GET", "HEAD", "OPTIONS")
        if unsafe and request.url.path.startswith("/api/") and not request.headers.get(CLIENT_HEADER):
            return JSONResponse({"detail": f"missing {CLIENT_HEADER} header"}, status_code=403)
        return await call_next(request)


def status_of(state: AppState) -> Status:
    if state.settings.closing == "closed":
        return Status(True, None, "Closed by configuration", "סגור לפי הגדרה", "config")
    return state.schedule.status(state.now())


def _status_json(st: Status) -> dict[str, Any]:
    return {
        "closed": st.closed,
        "until": st.until.isoformat() if st.until else None,
        "reason": st.reason,
        "reason_he": st.reason_he,
        "source": st.source,
    }


class ClosedMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if path.startswith("/api/") and path not in OPEN_PATHS:
            state: AppState = request.app.state.chavruta
            # The schedule may refresh from Hebcal: keep that blocking call off the event loop.
            st = await run_in_threadpool(status_of, state)
            if st.closed:
                return JSONResponse(
                    {"detail": "closed for Shabbat or Yom Tov", **_status_json(st)}, status_code=423
                )
        return await call_next(request)


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.chavruta
    return state


def get_session(request: Request) -> Iterator[Session]:
    with get_state(request).sessions() as s:
        yield s


Db = Annotated[Session, Depends(get_session)]
State = Annotated[AppState, Depends(get_state)]


def get_user(request: Request, s: Db) -> User:
    token = request.cookies.get(COOKIE)
    user = services.user_for(s, token) if token else None
    if user is None:
        raise HTTPException(401, "sign in first")
    return user


Me = Annotated[User, Depends(get_user)]


def guarded(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except services.NotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except services.ForbiddenError as e:
        raise HTTPException(403, str(e)) from e
    except services.InvalidError as e:
        raise HTTPException(422, str(e)) from e


def create_app(state: AppState) -> FastAPI:
    docs = state.settings.api_docs
    app = FastAPI(
        title="chavruta",
        docs_url="/api/docs" if docs else None,
        openapi_url="/api/openapi.json" if docs else None,
        redoc_url=None,
    )
    app.add_middleware(ClosedMiddleware)
    app.add_middleware(CsrfMiddleware)
    app.state.chavruta = state

    # ---- status and session --------------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/status")
    def status(st: State) -> dict[str, Any]:
        return _status_json(status_of(st))

    @app.post("/api/session")
    def login(body: Login, request: Request, response: Response, s: Db, st: State) -> dict[str, Any]:
        key = (client_address(request, st.settings.trust_proxy), body.username)
        if st.throttle.blocked(key):
            raise HTTPException(429, "too many failed attempts; wait a minute")
        result = services.login(s, body.username, body.password, st.settings.session_hours)
        if result is None:
            st.throttle.fail(key)
            raise HTTPException(401, "wrong username or password")
        st.throttle.clear(key)
        user, token = result
        response.set_cookie(
            COOKIE,
            token,
            max_age=st.settings.session_hours * 3600,
            httponly=True,
            samesite="strict",
            secure=st.settings.secure_cookies,
        )
        return _user(user)

    @app.delete("/api/session")
    def logout(request: Request, response: Response, s: Db) -> dict[str, str]:
        token = request.cookies.get(COOKIE)
        if token:
            services.logout(s, token)
        response.delete_cookie(COOKIE)
        return {"status": "signed out"}

    @app.get("/api/me")
    def me(user: Me) -> dict[str, Any]:
        return _user(user)

    # ---- asking ---------------------------------------------------------------------------------

    def spend(st: AppState, user: User) -> None:
        if not st.limits.take(user.id):
            raise HTTPException(429, "this account has reached its limit of model calls; try again later")

    @app.post("/api/ask")
    def ask(body: AskIn, user: Me, s: Db, st: State) -> dict[str, Any]:
        if body.anchor_ref and body.anchor_ref not in st.library.index.by_ref:
            raise HTTPException(422, "that passage is not in the library")
        spend(st, user)
        try:
            result: dict[str, Any] = guarded(
                lambda: services.ask(
                    s, st.bot, user, body.question, body.conversation_id, body.mode, body.anchor_ref
                )
            )
        except MissingRecordingError as e:
            raise HTTPException(
                503,
                "No API key is configured and this question has no recorded answer. "
                "Try one of the sample questions, or set ANTHROPIC_API_KEY and CHAVRUTA_LLM_MODE=live.",
            ) from e
        return result

    @app.get("/api/conversations")
    def conversations(user: Me, s: Db) -> list[dict[str, Any]]:
        return services.conversations(s, user)

    @app.get("/api/conversations/{conversation_id}")
    def conversation(
        conversation_id: Annotated[int, Path(ge=1, le=2**31 - 1)], user: Me, s: Db
    ) -> dict[str, Any]:
        result: dict[str, Any] = guarded(lambda: services.conversation(s, user, conversation_id))
        return result

    # ---- texts ----------------------------------------------------------------------------------

    @app.get("/api/works")
    def works(user: Me, s: Db, st: State) -> list[dict[str, Any]]:
        out = []
        for w in services.corpus_overview(s):
            if w["enabled"]:
                out.append(
                    {
                        "title": w["title"],
                        "he_title": w["he_title"],
                        "category": w["category"],
                        "sections": services.work_units(s, w["title"]),
                    }
                )
        return out

    @app.get("/api/sugya")
    def sugya(ref: Annotated[str, Query(pattern=REF_PATTERN)], user: Me, s: Db, st: State) -> dict[str, Any]:
        """A section with its translation, the commentaries on each segment, and the user's notes."""
        chunks = st.library.section(ref)
        if not chunks:
            raise HTTPException(404, "that section is not in the library")
        segments: dict[str, dict[str, Any]] = {}
        for c in sorted(chunks, key=lambda c: c.position):
            seg = segments.setdefault(c.ref, {"ref": c.ref, "texts": [], "commentary": []})
            seg["texts"].append(
                {"language": c.language, "text": c.text, "version": c.version, "license": c.license}
            )
        work = chunks[0].work
        for commentary in ("Rashi on", "Tosafot on"):
            for c in st.library.section(f"{commentary} {ref}"):
                base = f"{ref}:{c.ref.removeprefix(f'{commentary} {ref}:').split(':')[0]}"
                if base in segments:
                    segments[base]["commentary"].append(
                        {
                            "ref": c.ref,
                            "work": c.work,
                            "language": c.language,
                            "text": c.text,
                            "license": c.license,
                        }
                    )
        notes = services.notes(s, user, ref)
        units = services.work_units(s, work)
        i = units.index(ref) if ref in units else -1
        return {
            "ref": ref,
            "work": work,
            "segments": list(segments.values()),
            "notes": notes,
            "previous": units[i - 1] if i > 0 else None,
            "next": units[i + 1] if 0 <= i < len(units) - 1 else None,
            "attribution": _attribution(chunks),
        }

    @app.get("/api/links")
    def links(ref: Annotated[str, Query(pattern=REF_PATTERN)], user: Me, st: State) -> dict[str, Any]:
        """Connections of a passage from Sefaria, grouped by category, marked when in the library."""
        try:
            found = st.sefaria.links(ref)
        except SefariaError as e:
            raise HTTPException(502, "Sefaria could not be reached for links") from e
        groups: dict[str, list[dict[str, Any]]] = {}
        seen: set[str] = set()
        for link in found:
            if link.ref in seen:
                continue
            seen.add(link.ref)
            groups.setdefault(link.category or "Other", []).append(
                {
                    "ref": link.ref,
                    "anchor": link.anchor_ref,
                    "type": link.type,
                    "in_library": link.ref in st.library.index.by_ref
                    or link.ref.split(":")[0] in st.library.by_section,
                    "url": "https://www.sefaria.org/" + link.ref.replace(" ", "_").replace(":", "."),
                }
            )
        return {"ref": ref, "groups": groups, "total": len(seen)}

    # ---- notes ----------------------------------------------------------------------------------

    @app.get("/api/notes")
    def notes(
        user: Me, s: Db, ref: Annotated[str | None, Query(pattern=REF_PATTERN)] = None
    ) -> list[dict[str, Any]]:
        return services.notes(s, user, ref)

    @app.post("/api/notes", status_code=201)
    def add_note(body: NoteIn, user: Me, s: Db) -> dict[str, Any]:
        result: dict[str, Any] = guarded(lambda: services.save_note(s, user, body.ref, body.body))
        return result

    @app.put("/api/notes/{note_id}")
    def edit_note(note_id: int, body: NoteIn, user: Me, s: Db) -> dict[str, Any]:
        result: dict[str, Any] = guarded(
            lambda: services.save_note(s, user, body.ref, body.body, _id(note_id))
        )
        return result

    @app.delete("/api/notes/{note_id}", status_code=204)
    def remove_note(note_id: int, user: Me, s: Db) -> Response:
        guarded(lambda: services.delete_note(s, user, _id(note_id)))
        return Response(status_code=204)

    # ---- cards ----------------------------------------------------------------------------------

    @app.get("/api/cards")
    def cards(user: Me, s: Db, st: State, due: bool = False) -> list[dict[str, Any]]:
        return services.cards(s, user, st.now().astimezone(st.tz).date(), due)

    @app.post("/api/cards", status_code=201)
    def add_card(body: CardIn, user: Me, s: Db) -> dict[str, Any]:
        result: dict[str, Any] = guarded(
            lambda: services.add_card(s, user, body.ref, body.front, body.back, body.origin)
        )
        return result

    @app.post("/api/cards/suggest")
    def suggest(body: ProgressIn, user: Me, st: State) -> list[dict[str, str]]:
        spend(st, user)
        try:
            result: list[dict[str, str]] = guarded(
                lambda: services.suggest_cards(st.llm, st.library, body.unit)
            )
        except MissingRecordingError as e:
            raise HTTPException(503, "No API key is configured, so cards cannot be suggested.") from e
        return result

    @app.post("/api/cards/{card_id}/review")
    def review(card_id: int, body: ReviewIn, user: Me, s: Db, st: State) -> dict[str, Any]:
        today = st.now().astimezone(st.tz).date()
        result: dict[str, Any] = guarded(
            lambda: services.review_card(s, user, _id(card_id), body.grade, today)
        )
        return result

    @app.delete("/api/cards/{card_id}", status_code=204)
    def remove_card(card_id: int, user: Me, s: Db) -> Response:
        guarded(lambda: services.delete_card(s, user, _id(card_id)))
        return Response(status_code=204)

    # ---- study plans ----------------------------------------------------------------------------

    def daf_of(day: date) -> str | None:
        try:
            for item in state.sefaria.calendars(day.isoformat()):
                title = item.get("title") or {}
                if isinstance(title, dict) and title.get("en") == "Daf Yomi":
                    ref = item.get("ref")
                    return str(ref) if ref else None
        except SefariaError:
            return None
        return None

    @app.get("/api/daf-yomi")
    def daf_yomi(user: Me, st: State, day: date | None = None) -> dict[str, Any]:
        today = st.now().astimezone(st.tz).date()
        d = day or today
        if abs((d - today).days) > 400:  # each date is a cached Sefaria request: keep it bounded
            raise HTTPException(422, "choose a date within about a year of today")
        ref = daf_of(d)
        return {
            "date": d.isoformat(),
            "ref": ref,
            "in_library": bool(ref and ref in st.library.by_section),
            "url": "https://www.sefaria.org/" + ref.replace(" ", "_") if ref else None,
        }

    @app.get("/api/plans")
    def plans(user: Me, s: Db) -> list[dict[str, Any]]:
        return services.plans(s, user)

    @app.post("/api/plans", status_code=201)
    def add_plan(body: PlanIn, user: Me, s: Db, st: State) -> dict[str, Any]:
        start = body.start_date or st.now().astimezone(st.tz).date()
        result: dict[str, Any] = guarded(
            lambda: services.create_plan(s, user, body.kind, body.work_title, body.per_day, start)
        )
        return result

    @app.get("/api/plans/{plan_id}/today")
    def plan_today(plan_id: int, user: Me, s: Db, st: State) -> dict[str, Any]:
        today = st.now().astimezone(st.tz).date()
        result: dict[str, Any] = guarded(lambda: services.plan_today(s, user, _id(plan_id), today, daf_of))
        return result

    @app.post("/api/plans/{plan_id}/progress", status_code=204)
    def progress(plan_id: int, body: ProgressIn, user: Me, s: Db) -> Response:
        guarded(lambda: services.mark_done(s, user, _id(plan_id), body.unit))
        return Response(status_code=204)

    @app.delete("/api/plans/{plan_id}", status_code=204)
    def remove_plan(plan_id: int, user: Me, s: Db) -> Response:
        guarded(lambda: services.delete_plan(s, user, _id(plan_id)))
        return Response(status_code=204)

    # ---- corpus management (admin) --------------------------------------------------------------

    @app.get("/api/corpus")
    def corpus(user: Me, s: Db, st: State) -> dict[str, Any]:
        guarded(lambda: services.require_admin(user))
        return {
            "works": services.corpus_overview(s),
            "library": st.library.stats(),
            "strictness": st.bot.settings.strictness.value,
            "relevance_threshold": st.bot.settings.relevance_threshold,
            "llm_mode": st.settings.llm_mode.value,
        }

    @app.post("/api/corpus/works/{work_id}/enabled")
    def enable(work_id: int, body: EnabledIn, user: Me, s: Db, st: State) -> dict[str, Any]:
        guarded(lambda: services.require_admin(user))
        guarded(lambda: services.set_enabled(s, _id(work_id), body.enabled))
        with st.lock:
            st.library.reload()
            st.bot.retriever, st.bot.index = st.library.retriever, st.library.index
        return {"enabled": body.enabled, "library": st.library.stats()}

    @app.post("/api/corpus/strictness")
    def strictness(body: StrictnessIn, user: Me, st: State) -> dict[str, str]:
        guarded(lambda: services.require_admin(user))
        st.bot.settings = replace(st.bot.settings, strictness=Strictness(body.strictness))
        services.save_strictness(st.settings, body.strictness)
        return {"strictness": body.strictness}

    return app


def client_address(request: Request, trust_proxy: bool) -> str:
    """The browser's address. X-Forwarded-For is used only when configured behind the bundled nginx,
    which overwrites it with the connecting address; otherwise anyone could rotate it."""
    direct = request.client.host if request.client else "?"
    if not trust_proxy:
        return direct
    return request.headers.get("x-forwarded-for", "").split(",")[0].strip() or direct


def _id(value: int) -> int:
    if not 1 <= value <= 2**31 - 1:
        raise HTTPException(404, "not found")
    return value


def _user(u: User) -> dict[str, Any]:
    return {"id": u.id, "username": u.username, "display_name": u.display_name, "role": Role(u.role).value}


def _attribution(chunks: list[Any]) -> list[dict[str, str]]:
    seen: dict[tuple[str, str], dict[str, str]] = {}
    for c in chunks:
        seen.setdefault(
            (c.language, c.version), {"language": c.language, "version": c.version, "license": c.license}
        )
    return list(seen.values())
