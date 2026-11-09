"""Sefaria API client: texts by version, licenses, links, search, calendars.

Only openly licensed versions are ever stored: the license comes from the API for each version and
is kept on every chunk so attribution can be shown. Responses are cached on disk so a re-index does
not refetch unchanged sections.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.parse
from collections.abc import Callable
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
        self,
        cache_dir: Path | None = None,
        client: httpx.Client | None = None,
        min_interval: float = 0.25,
        offline: bool = False,
        max_bytes: int = 1_500_000,
    ) -> None:
        """`offline`: lookups answer only from the cache (recorded responses); a miss is an error.
        `max_bytes`: a lookup response larger than this (a whole tractate, say) is refused."""
        self.cache_dir = cache_dir
        self.client = client or http_client()
        self.min_interval = min_interval
        self.offline = offline
        self.max_bytes = max_bytes
        self.requests = 0  # requests actually sent
        self._last = 0.0
        self._lock = threading.Lock()

    def _get(self, path: str, params: list[tuple[str, str]] | None = None) -> Any:
        query = urllib.parse.urlencode(params or [])
        url = f"{BASE}/{path}" + (f"?{query}" if query else "")
        cache = None
        if self.cache_dir is not None:
            cache = self.cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")
            if cache.exists():
                return json.loads(cache.read_text(encoding="utf-8"))
        for attempt in range(3):
            self._space()
            try:
                response = self.client.get(url)
            except httpx.HTTPError as e:  # connection refused, timeouts, TLS: not OSErrors in httpx
                raise SefariaError(f"Sefaria unreachable: {type(e).__name__}") from e
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
                _write(cache, data)
            return data
        raise SefariaError(f"gave up on {url}")  # pragma: no cover

    def _space(self) -> None:
        """Requests may overlap, but start at least min_interval apart."""
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests += 1

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

    # ---- lookups for one question (see lookup.py) ----------------------------------------------------
    #
    # Unlike ingest, a lookup must replay exactly from its recordings, so what it caches is what the
    # code saw: an unknown ref (Sefaria's {"error"} or a 404) and an oversized response are cached
    # as errors too. A transient failure (no connection, 5xx, a search error) is never cached. The
    # text of a version that is not openly licensed is never cached at all, only whether it had any.

    def _lookup(
        self,
        path: str,
        params: list[tuple[str, str]],
        body: dict[str, Any] | None,
        trim: Callable[[Any], Any],
        cache_errors: bool,
    ) -> Any:
        query = urllib.parse.urlencode(params)
        url = f"{BASE}/{path}" + (f"?{query}" if query else "")
        key = url if body is None else url + " " + json.dumps(body, ensure_ascii=False, sort_keys=True)
        cache = None
        if self.cache_dir is not None:
            cache = self.cache_dir / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except FileNotFoundError:
                pass
            except ValueError:
                cache.unlink(missing_ok=True)  # a damaged file is a miss, and is replaced
        if self.offline:
            raise SefariaUnreachable(f"no recorded response for {key}")
        for attempt in range(3):
            self._space()
            try:
                with self.client.stream("GET" if body is None else "POST", url, json=body) as response:
                    if response.status_code in (429, 502, 503, 504) and attempt < 2:
                        time.sleep(2**attempt)
                        continue
                    raw, too_large = b"", False
                    if response.status_code == 200:
                        for chunk in response.iter_bytes():
                            raw += chunk
                            if len(raw) > self.max_bytes:
                                too_large = True
                                break
            except httpx.HTTPError as e:
                raise SefariaUnreachable(f"Sefaria unreachable: {type(e).__name__}") from e
            if response.status_code == 404:
                data: Any = {"error": "not found (404)"}
            elif response.status_code != 200:
                raise SefariaError(f"{response.status_code} for {url}")
            elif too_large:
                data = {"error": f"response larger than {self.max_bytes // 1000} KB"}
            else:
                try:
                    data = trim(json.loads(raw))
                except ValueError as e:
                    raise SefariaError(f"not JSON from {url}") from e
            error = isinstance(data, dict) and "error" in data
            if cache is not None and (cache_errors or not error):
                _write(cache, data)
            return data
        raise SefariaError(f"gave up on {url}")  # pragma: no cover

    def text(self, ref: str, versions: list[str]) -> dict[str, Any]:
        """A ref in the given versions ("hebrew", "english" for the defaults, or "hebrew|Title"),
        with the list of all versions and their licenses. An unknown ref gives {"error": ...}."""
        data = self._lookup(
            f"v3/texts/{urllib.parse.quote(ref, safe='')}",
            [("version", v) for v in versions],
            None,
            _trim_text,
            cache_errors=True,
        )
        if not isinstance(data, dict):
            raise SefariaError(f"unexpected text response for {ref}")
        return data

    def search(self, query: str, size: int = 20) -> list[dict[str, Any]]:
        """Full-text search over the whole Sefaria library: hits (ref, version, lang, categories),
        best first; one hit per version, so a ref can appear several times."""
        body = {
            "query": query,
            "type": "text",
            "field": "naive_lemmatizer",
            "size": size,
            "slop": 10,
            "source_proj": True,
            "filters": [],
            "filter_fields": [],
            "aggs": [],
            "sort_method": "score",
            "sort_fields": ["pagesheetrank"],
            "sort_score_missing": 0.04,
            "sort_reverse": False,
        }
        data = self._lookup("search-wrapper/es8", [], body, _trim_search, cache_errors=False)
        if not isinstance(data, list):
            raise SefariaError(f"search failed for {query!r}: {data}")
        return data


class SefariaUnreachable(SefariaError):  # noqa: N818 - reads as a condition, like the others
    """No connection to Sefaria (or, replaying, no recording): stop asking for this question."""


def _write(path: Path, data: Any) -> None:
    """Atomically: a reader never sees half a file, even with lookups running in parallel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def has_text(text: Any) -> bool:
    if isinstance(text, str):
        return bool(text.strip())
    if isinstance(text, list):
        return any(has_text(t) for t in text)
    return False


_VERSION_FIELDS = (
    "language", "languageFamilyName", "actualLanguage", "versionTitle", "license", "versionSource",
)  # fmt: skip
_TEXT_FIELDS = (
    "ref", "heRef", "sectionRef", "indexTitle", "heIndexTitle", "categories", "next", "sectionNames", "error",
)  # fmt: skip


def _trim_version(v: dict[str, Any]) -> dict[str, Any]:
    out = {k: v.get(k) for k in _VERSION_FIELDS}
    if is_open(v.get("license")):
        out["text"] = v.get("text")
    else:
        out["hasText"] = has_text(v.get("text"))  # never the text itself
    return out


def _trim_text(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    out = {k: data[k] for k in _TEXT_FIELDS if k in data}
    if "error" not in data:
        out["versions"] = [_trim_version(v) for v in data.get("versions", [])]
        out["available_versions"] = [
            {k: v.get(k) for k in _VERSION_FIELDS} for v in data.get("available_versions", [])
        ]
    return out


def _trim_search(data: Any) -> Any:
    if not isinstance(data, dict) or not isinstance(data.get("hits"), dict):
        return {"error": f"unexpected search response: {str(data)[:200]}"}
    return [
        {k: h.get("_source", {}).get(k) for k in ("ref", "version", "lang", "categories")}
        for h in data["hits"].get("hits", [])
        if h.get("_source", {}).get("ref")
    ]
