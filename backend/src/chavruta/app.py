"""The ASGI app: `uvicorn --factory chavruta.app:make`."""

from __future__ import annotations

import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI

from chavruta.answer import Chavruta
from chavruta.answer import Settings as AnswerSettings
from chavruta.api import AppState, create_app
from chavruta.config import Settings
from chavruta.db import make_engine, make_sessions
from chavruta.embeddings import CHOSEN_MODEL, MODELS, CrossEncoderReranker, SentenceEmbedder
from chavruta.hebcal import Schedule
from chavruta.library import Library
from chavruta.llm import AnthropicLLM, Mode, RecordingLLM
from chavruta.net import install_socket_guard
from chavruta.sefaria import SefariaClient
from chavruta.services import Throttle, ensure_users


def make() -> FastAPI:
    settings = Settings()
    install_socket_guard(extra_hosts=[settings.db_host])
    settings.secret()  # refuse to start without a real secret
    sessions = make_sessions(make_engine(settings.database_url))
    with sessions() as s:
        ensure_users(s, settings.password("learner"), settings.password("admin"))
    embedder = SentenceEmbedder(MODELS[CHOSEN_MODEL])
    reranker = CrossEncoderReranker() if settings.reranker else None
    library = Library(sessions, embedder, reranker)
    llm = RecordingLLM(
        settings.recordings_dir,
        settings.llm_mode,
        AnthropicLLM() if settings.llm_mode != Mode.REPLAY else None,
    )
    tz = ZoneInfo(settings.timezone)
    bot = Chavruta(
        library.retriever,
        library.index,
        llm,
        AnswerSettings(strictness=settings.strictness, relevance_threshold=settings.relevance_threshold),
    )
    state = AppState(
        settings=settings,
        sessions=sessions,
        library=library,
        llm=llm,
        bot=bot,
        schedule=Schedule(settings.geonameid, settings.israel, settings.timezone),
        sefaria=SefariaClient(cache_dir=settings.data_dir / "sefaria-cache"),
        tz=tz,
        throttle=Throttle(),
        now=lambda: datetime.now(tz),
        lock=threading.Lock(),
    )
    return create_app(state)
