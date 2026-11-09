"""Multilingual embedding models, compared on the retrieval test set before one was chosen.

Model weights are downloaded at build time; at run time the process is network-locked (net.py) and
loads them from the local cache only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    dim: int
    query_prefix: str = ""
    passage_prefix: str = ""
    #: Weight file pattern to download. Safetensors wherever the authors publish it; bge-m3 has only
    #: a PyTorch checkpoint, which torch loads with weights_only (no code execution).
    weights: str = "*.safetensors"


MODELS: dict[str, ModelSpec] = {
    "e5-small": ModelSpec("e5-small", "intfloat/multilingual-e5-small", 384, "query: ", "passage: "),
    "e5-base": ModelSpec("e5-base", "intfloat/multilingual-e5-base", 768, "query: ", "passage: "),
    "bge-m3": ModelSpec("bge-m3", "BAAI/bge-m3", 1024, weights="pytorch_model.bin"),
    "minilm": ModelSpec("minilm", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384),
}
#: The model the app uses, chosen on the dev split (docs/decisions.md).
CHOSEN_MODEL = "e5-small"
EMBEDDING_DIM = MODELS[CHOSEN_MODEL].dim
RERANKER = "BAAI/bge-reranker-v2-m3"


class Embedder(Protocol):
    key: str

    def queries(self, texts: list[str]) -> Vectors: ...

    def passages(self, texts: list[str]) -> Vectors: ...


class SentenceEmbedder:
    """A sentence-transformers model with the prefixes its authors require; outputs unit vectors."""

    def __init__(self, spec: ModelSpec, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.key = spec.key
        self.spec = spec
        # Weights come from the local cache only; the backend never downloads at run time.
        self.model: Any = SentenceTransformer(spec.name, device=device or _device(), local_files_only=True)

    def _encode(self, texts: list[str], prefix: str) -> Vectors:
        out = self.model.encode(
            [prefix + t for t in texts],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(out, dtype=np.float32)

    def queries(self, texts: list[str]) -> Vectors:
        return self._encode(texts, self.spec.query_prefix)

    def passages(self, texts: list[str]) -> Vectors:
        return self._encode(texts, self.spec.passage_prefix)


class CrossEncoderReranker:
    def __init__(self, name: str = RERANKER, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        device = device or _device()
        # Weights come from the local cache only; the backend never downloads at run time.
        self.model: Any = CrossEncoder(name, device=device, max_length=512, local_files_only=True)
        if device != "cpu":
            self.model.model.half()  # half the memory on a GPU; scores agree to about 1e-3

    def scores(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        raw = self.model.predict([(query, p) for p in passages], batch_size=16, show_progress_bar=False)
        return [float(x) for x in raw]


def _device() -> str:
    if os.environ.get("CHAVRUTA_DEVICE"):
        return os.environ["CHAVRUTA_DEVICE"]
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


def save_vectors(directory: Path, refs: list[str], vectors: Vectors) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "vectors.npy", vectors.astype(np.float16))
    (directory / "ids.json").write_text(json.dumps(refs), encoding="utf-8")


def load_vectors(directory: Path) -> tuple[list[str], Vectors]:
    refs = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
    return refs, np.load(directory / "vectors.npy").astype(np.float32)
