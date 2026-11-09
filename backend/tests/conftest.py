"""PostgreSQL with pgvector in a container, migrated with Alembic and loaded with a real corpus
sample. Tests marked `db` need Docker."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

FIXTURES = Path(__file__).parent / "fixtures"
BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def pg_url() -> Iterator[str]:
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer(
        "pgvector/pgvector:pg17", username="chavruta", password="test", dbname="chavruta"
    ) as pg:
        url = pg.get_connection_url().replace("postgresql+psycopg2", "postgresql+psycopg")
        if not url.startswith("postgresql+psycopg"):
            url = url.replace("postgresql://", "postgresql+psycopg://")
        from alembic import command
        from alembic.config import Config

        cfg = Config(str(BACKEND / "alembic.ini"))
        cfg.set_main_option("sqlalchemy.url", url)
        command.upgrade(cfg, "head")
        yield url


@pytest.fixture(scope="session")
def engine(pg_url: str) -> Iterator[Engine]:
    from chavruta.db import make_engine

    e = make_engine(pg_url)
    yield e
    e.dispose()


@pytest.fixture(scope="session")
def loaded(engine: Engine) -> sessionmaker[Session]:
    """The corpus sample, stored and embedded with the deterministic test embedder."""
    from chavruta.corpus import BY_TITLE
    from chavruta.db import make_sessions
    from chavruta.ingest import embed_missing, load_file, store_chunks
    from tests.test_retrieval import HashEmbedder

    sessions = make_sessions(engine)
    with sessions() as s:
        for work, chunks in load_file(FIXTURES / "chunks-sample.jsonl").items():
            store_chunks(s, BY_TITLE[work], chunks, [])
        s.commit()
        embed_missing(s, HashEmbedder())
    return sessions


@pytest.fixture(autouse=True)
def _no_real_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never call the real API, even on a machine that has a key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
