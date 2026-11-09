"""Hybrid retrieval: keyword search (BM25) and embeddings, fused with reciprocal rank fusion, then
optionally reranked by a cross-encoder.

Hebrew and Aramaic need the keyword side: exact phrases and names ("רבן גמליאל", "תכלת ללבן") are
what a learner remembers, and embeddings blur them. Embeddings carry English questions to Hebrew and
Aramaic sources. Results are distinct references: the Hebrew and English chunks of one verse count
once.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from chavruta.corpus import Chunk
from chavruta.embeddings import Embedder, Vectors
from chavruta.text import tokens

RRF_K = 60
CANDIDATES = 50


class Reranker(Protocol):
    def scores(self, query: str, passages: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder | None = None,
        vectors: Vectors | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.bm25 = BM25Okapi([tokens(c.text) or ["_"] for c in self.chunks])
        self.embedder = embedder
        self.vectors = vectors
        self.reranker = reranker
        if vectors is not None and len(vectors) != len(self.chunks):
            raise ValueError("one vector per chunk is required")

    def keyword(self, query: str, n: int = CANDIDATES) -> list[int]:
        q = tokens(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        order = np.argsort(-scores)[:n]
        return [int(i) for i in order if scores[i] > 0]

    def semantic(self, query: str, n: int = CANDIDATES) -> list[int]:
        if self.embedder is None or self.vectors is None:
            return []
        q = self.embedder.queries([query])[0]
        scores = self.vectors @ q
        return [int(i) for i in np.argsort(-scores)[:n]]

    def search(self, query: str, k: int = 10, mode: str = "hybrid", rerank: bool = True) -> list[Hit]:
        """Top-k distinct references. mode: "keyword", "semantic" or "hybrid"."""
        lists: list[list[int]] = []
        if mode in ("keyword", "hybrid"):
            lists.append(self.keyword(query))
        if mode in ("semantic", "hybrid"):
            lists.append(self.semantic(query))
        fused: dict[int, float] = {}
        for ranking in lists:
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        candidates = sorted(fused, key=lambda i: -fused[i])[:CANDIDATES]
        scored = [(i, fused[i]) for i in candidates]
        if rerank and self.reranker is not None and candidates:
            rr = self.reranker.scores(query, [self.chunks[i].text[:1500] for i in candidates])
            scored = sorted(zip(candidates, rr, strict=True), key=lambda x: -x[1])
        hits: list[Hit] = []
        seen: set[str] = set()
        for i, s in scored:
            ref = self.chunks[i].ref
            if ref in seen:
                continue
            seen.add(ref)
            hits.append(Hit(self.chunks[i], s))
            if len(hits) == k:
                break
        return hits


def matches(retrieved_ref: str, gold_ref: str) -> bool:
    """A retrieved segment counts for a gold reference at the same or a coarser level:
    "Berakhot 2a:3" matches "Berakhot 2a"; "Genesis 1:1" does not match "Genesis 1:10"."""
    return retrieved_ref == gold_ref or retrieved_ref.startswith(gold_ref + ":")
