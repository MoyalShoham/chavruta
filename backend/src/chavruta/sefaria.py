"""Sefaria API client: texts by version, licenses, links, calendars.

Only openly licensed versions are ever stored: the license comes from the API for each version and
is kept on every chunk so attribution can be shown. Responses are cached on disk so a re-index does
not refetch unchanged sections.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2 as httpx

from chavruta.net import http_client

BASE = "https://www.sefaria.org/api"
OPEN_LICENSES = frozenset({"Public Domain", "CC0", "CC-BY", "CC-BY-SA"})


class SefariaError(RuntimeError):
    pass


def is_open(license_name: str | None) -> bool:
    return (license_name or "").strip() in OPEN_LICENSES


@dataclass(frozen=True)
class VersionText:
    language: str  # "he" or "en"
    title: str
    license: str
    text: Any  # nested lists of strings, depth = number of section levels below the requested ref
    source: str | None = None


@dataclass(frozen=True)
class Section:
    ref: str
    he_ref: str
    next_ref: str | None
    section_names: list[str]
    versions: list[VersionText] = field(default_factory=list)


@dataclass(frozen=True)
class Link:
    anchor_ref: str
    ref: str
    category: str
    type: str


class SefariaClient:
    def __init__(
        self, cache_dir: Path | None = None, client: httpx.Client | None = None, min_interval: float = 0.25
    ) -> None:
        self.cache_dir = cache_dir
        self.client = client or http_client()
        self.min_interval = min_interval
        self._last = 0.0

    def _get(self, path: str, params: list[tuple[str, str]] | None = None) -> Any:
        query = urllib.parse.urlencode(params or [])
        url = f"{BASE}/{path}" + (f"?{query}" if query else "")
        cache = None
        if self.cache_dir is not None:
            cache = self.cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")
            if cache.exists():
                return json.loads(cache.read_text(encoding="utf-8"))
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            try:
                response = self.client.get(url)
            except httpx.HTTPError as e:  # connection refused, timeouts, TLS: not OSErrors in httpx
                raise SefariaError(f"Sefaria unreachable: {type(e).__name__}") from e
            self._last = time.monotonic()
            if response.status_code in (429, 502, 503, 504) and attempt < 2:
                time.sleep(2**attempt)
                continue
            if response.status_code != 200:
                raise SefariaError(f"{response.status_code} for {url}")
            try:
                data = response.json()  # a 200 HTML page (a challenge, a proxy) is not JSON
            except ValueError as e:
                raise SefariaError(f"not JSON from {url}") from e
            if cache is not None and not (isinstance(data, dict) and "error" in data):
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        raise SefariaError(f"gave up on {url}")  # pragma: no cover

    def versions(self, index_title: str) -> list[dict[str, Any]]:
        data = self._get(f"texts/versions/{urllib.parse.quote(index_title)}")
        if not isinstance(data, list):
            raise SefariaError(f"unexpected versions response for {index_title}")
        return data

    def section(self, ref: str, versions: list[tuple[str, str]]) -> Section:
        """Text of one section in the given (language, versionTitle) pairs; language is 'hebrew'/'english'."""
        params = [("version", f"{lang}|{title}") for lang, title in versions]
        data = self._get(f"v3/texts/{urllib.parse.quote(ref)}", params)
        if "error" in data:
            raise SefariaError(str(data["error"]))
        found = [
            VersionText(
                language=v.get("language", ""),
                title=v.get("versionTitle", ""),
                license=v.get("license") or "unknown",
                text=v.get("text"),
                source=v.get("versionSource"),
            )
            for v in data.get("versions", [])
        ]
        return Section(
            ref=data.get("ref", ref),
            he_ref=data.get("heRef", ""),
            next_ref=data.get("next"),
            section_names=list(data.get("sectionNames") or []),
            versions=found,
        )

    def links(self, ref: str) -> list[Link]:
        data = self._get(f"links/{urllib.parse.quote(ref)}", [("with_text", "0")])
        out = []
        for item in data if isinstance(data, list) else []:
            anchor = item.get("anchorRef")
            other = item.get("ref")
            if anchor and other:
                out.append(Link(anchor, other, item.get("category", ""), item.get("type", "")))
        return out

    def calendars(self, day: str) -> list[dict[str, Any]]:
        """Learning schedules (Daf Yomi, Parashah, ...) for a date YYYY-MM-DD."""
        y, m, d = day.split("-")
        data = self._get("calendars", [("year", y), ("month", str(int(m))), ("day", str(int(d)))])
        items = data.get("calendar_items", []) if isinstance(data, dict) else []
        return [i for i in items if isinstance(i, dict)]
