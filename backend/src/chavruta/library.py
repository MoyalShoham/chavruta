"""The library as the running app sees it: chunks of enabled works, BM25 in memory, vectors in
pgvector. Reloaded when the corpus changes (a work is enabled, disabled or re-synced)."""

from __future__ import annotations

import threading
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session, sessionmaker

from chavruta.answer import CorpusIndex
from chavruta.corpus import Chunk
from chavruta.db import ChunkRow, Version, Work
from chavruta.embeddings import CHOSEN_MODEL, EMBEDDING_DIM, Embedder
from chavruta.retrieval import CANDIDATES, Reranker, Retriever


class EmbeddingMismatchError(RuntimeError):
    pass


def check_embeddings(sessions: sessionmaker[Session]) -> None:
    """Refuses to run on vectors from another model, or a column of another dimension: mixed
    vectors would silently ruin semantic search."""
    with sessions() as s:
        models = set(
            s.scalars(
                select(ChunkRow.embedding_model).where(ChunkRow.embedding_model.is_not(None)).distinct()
            )
        )
        dim = s.scalar(
            text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
            )
        )
    if models - {CHOSEN_MODEL}:
        raise EmbeddingMismatchError(
            f"the library was embedded with {sorted(m for m in models if m)}, the app uses "
            f"{CHOSEN_MODEL}; re-embed it"
        )
    if dim is not None and dim != EMBEDDING_DIM:
        raise EmbeddingMismatchError(
            f"the embedding column has {dim} dimensions and {CHOSEN_MODEL} needs {EMBEDDING_DIM}: "
            "add a migration"
        )


class PgRetriever(Retriever):
    """Keyword side in memory (same BM25 as the evaluation); semantic side in pgvector (HNSW,
    cosine), which is approximate where the evaluation used exact search."""

    def __init__(
        self,
        chunks: list[Chunk],
        ids: list[int],
        sessions: sessionmaker[Session],
        embedder: Embedder | None,
        reranker: Reranker | None,
        rerank_top: int = CANDIDATES,
    ) -> None:
        super().__init__(chunks, embedder, None, reranker, rerank_top)
        self.index_of = {chunk_id: i for i, chunk_id in enumerate(ids)}
        self.sessions = sessions

    def semantic(self, query: str, n: int = CANDIDATES) -> list[int]:
        if self.embedder is None:
            return []
        q = self.embedder.queries([query])[0].tolist()
        with self.sessions() as s:
            s.execute(text("SET LOCAL hnsw.ef_search = 100"))
            rows = s.execute(
                select(ChunkRow.id)
                .join(Work, Work.id == ChunkRow.work_id)
                .where(Work.enabled, ChunkRow.embedding.is_not(None))
                .order_by(ChunkRow.embedding.cosine_distance(q))
                .limit(n)
            ).scalars()
            return [self.index_of[i] for i in rows if i in self.index_of]


class Library:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        embedder: Embedder | None,
        reranker: Reranker | None,
        rerank_top: int = CANDIDATES,
    ) -> None:
        self.sessions = sessions
        self.embedder = embedder
        self.reranker = reranker
        self.rerank_top = rerank_top
        self._lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        with self.sessions() as s:
            rows = s.execute(
                select(ChunkRow, Work, Version)
                .join(Work, Work.id == ChunkRow.work_id)
                .join(Version, Version.id == ChunkRow.version_id)
                .where(Work.enabled)
                .order_by(ChunkRow.id)
            ).all()
        chunks = [
            Chunk(
                c.ref, w.title, w.category, c.language, v.title, v.license, c.text, c.section_ref, c.position
            )
            for c, w, v in rows
        ]
        ids = [c.id for c, _, _ in rows]
        with self.sessions() as s:
            disabled = set(s.scalars(select(Work.title).where(Work.enabled.is_(False))))
        retriever = PgRetriever(chunks, ids, self.sessions, self.embedder, self.reranker, self.rerank_top)
        by_section: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_section.setdefault(c.section_ref, []).append(c)
        with self._lock:
            self.chunks = chunks
            self.retriever = retriever
            self.index = CorpusIndex(chunks)
            self.index.disabled = disabled
            self.by_section = by_section

    def section(self, section_ref: str) -> list[Chunk]:
        return self.by_section.get(section_ref, [])

    def stats(self) -> dict[str, Any]:
        return {"chunks": len(self.chunks), "refs": len(self.index.by_ref)}
