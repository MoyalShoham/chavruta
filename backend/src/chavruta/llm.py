"""Claude calls, recorded once and replayed everywhere.

A recording is keyed by the SHA-256 of the complete request (model, system prompt, every message
and source text, output settings), so any change to the prompt, the retrieved sources or the
question makes a new key. Replay never reuses a stale answer: a missing key fails loudly.

The recording stores the request next to the response, so the evaluation can check citation
faithfulness from the recordings alone, without the corpus or the models (that is what CI does).
"""

from __future__ import annotations

import hashlib
import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

ANSWER_MODEL = "claude-opus-5-5"
CLASSIFIER_MODEL = "claude-haiku-5-5"
#: The faithfulness judge is a different model from the one that wrote the answers.
JUDGE_MODEL = "claude-sonnet-5-5"


class Mode(StrEnum):
    LIVE = "live"  # call the API, store nothing
    RECORD = "record"  # replay if recorded, otherwise call the API and store the response
    REPLAY = "replay"  # recordings only; never call the API


class MissingRecordingError(LookupError):
    pass


class LLM(Protocol):
    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        """Messages API request in, response JSON out."""
        ...


class AnthropicLLM:
    """At most two attempts of 120 s each: 240 s, inside nginx's 300 s."""

    def __init__(self, timeout: float = 120.0) -> None:
        import anthropic

        from chavruta.net import http_client

        # The SDK's HTTP client goes through the allowlist transport too.
        self.client = anthropic.Anthropic(
            http_client=http_client(timeout=timeout),
            max_retries=1,
            timeout=timeout,
        )

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        response = self.client.messages.create(**request)
        dumped: dict[str, Any] = response.model_dump(mode="json")
        return dumped


def request_key(request: dict[str, Any]) -> str:
    canonical = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:32]


class RecordingLLM:
    def __init__(self, directory: Path, mode: Mode, inner: LLM | None = None) -> None:
        if mode != Mode.REPLAY and inner is None:
            raise ValueError(f"{mode} mode needs an API client")
        self.directory = directory
        self.mode = mode
        self.inner = inner
        self.calls = 0  # API calls actually made

    def path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def create(self, request: dict[str, Any]) -> dict[str, Any]:
        key = request_key(request)
        if self.mode != Mode.LIVE:
            p = self.path(key)
            if p.exists():
                stored: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))["response"]
                return stored
            if self.mode == Mode.REPLAY:
                raise MissingRecordingError(
                    f"no recorded response for request {key}; record it with CHAVRUTA_LLM_MODE=record"
                )
        assert self.inner is not None
        response = self.inner.create(request)
        self.calls += 1
        if self.mode == Mode.RECORD:
            p = self.path(key)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(
                    {"key": key, "request": request, "response": response}, ensure_ascii=False, indent=1
                ),
                encoding="utf-8",
            )
        return response


def from_env(directory: Path) -> RecordingLLM:
    mode = Mode(os.environ.get("CHAVRUTA_LLM_MODE", "replay"))
    inner = AnthropicLLM() if mode != Mode.REPLAY else None
    return RecordingLLM(directory, mode, inner)


def iter_recordings(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*/*.json"))]
