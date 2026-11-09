"""Shabbat and Yom Tov times from Hebcal, and whether the app is closed right now.

The app is closed from candle lighting until havdalah, for Shabbat and for Yom Tov, at the
configured location. The backend enforces it (every study endpoint answers 423), not only the UI.

If Hebcal cannot be reached and no saved schedule covers today, the app does not guess the exact
times. It falls back to a conservative window, Friday 12:00 to Saturday 23:59 local time, and says
so. A Yom Tov on a weekday is then missed; the saved schedule (fetched weeks ahead, kept across
restarts) makes that rare.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx2 as httpx

from chavruta.net import http_client

BASE = "https://www.hebcal.com/hebcal"
JERUSALEM = 281184


@dataclass(frozen=True)
class Event:
    kind: str  # "candles" | "havdalah" | "yomtov"
    at: datetime | None  # candles, havdalah
    day: date
    title: str
    hebrew: str


@dataclass(frozen=True)
class Period:
    start: datetime
    end: datetime
    reason: str
    reason_he: str


@dataclass(frozen=True)
class Status:
    closed: bool
    until: datetime | None
    reason: str | None
    reason_he: str | None
    source: str  # "hebcal" | "fallback"


def parse_events(data: dict[str, Any]) -> list[Event]:
    events: list[Event] = []
    for item in data.get("items", []):
        category = item.get("category")
        raw = str(item.get("date", ""))
        if category in ("candles", "havdalah") and "T" in raw:
            at = datetime.fromisoformat(raw)
            events.append(Event(category, at, at.date(), item.get("title", ""), item.get("hebrew", "")))
        elif category == "holiday" and item.get("yomtov"):
            day = date.fromisoformat(raw[:10])
            events.append(Event("yomtov", None, day, item.get("title", ""), item.get("hebrew", "")))
    return events


def closed_periods(events: Sequence[Event]) -> list[Period]:
    """Each candle lighting opens a period that the next havdalah ends. Consecutive days (Yom Tov
    next to Shabbat) chain into one period, because the second candle lighting falls inside it."""
    timed = sorted(((e.at, e.kind) for e in events if e.at is not None), key=lambda x: x[0])
    yomtov = {e.day: e for e in events if e.kind == "yomtov"}
    periods: list[Period] = []
    for i, (start, kind) in enumerate(timed):
        if kind != "candles":
            continue
        if periods and periods[-1].start <= start < periods[-1].end:
            continue
        end = next((t for t, k in timed[i + 1 :] if k == "havdalah"), None)
        if end is None:
            continue
        days = [start.date() + timedelta(days=d) for d in range((end.date() - start.date()).days + 1)]
        holiday = next((yomtov[d] for d in days if d in yomtov), None)
        if holiday is not None:
            periods.append(Period(start, end, holiday.title, holiday.hebrew))
        else:
            periods.append(Period(start, end, "Shabbat", "שבת"))
    return periods


def fallback_status(now: datetime) -> Status:
    """Friday 12:00 to Saturday 23:59 local time, used only when no schedule is available."""
    weekday = now.weekday()  # Monday 0 ... Friday 4, Saturday 5
    friday_noon = datetime.combine(now.date() - timedelta(days=(weekday - 4) % 7), time(12), now.tzinfo)
    end = friday_noon + timedelta(days=1, hours=11, minutes=59)
    if friday_noon <= now <= end:
        return Status(True, end, "Shabbat (times unavailable)", "שבת (הזמנים אינם זמינים)", "fallback")
    return Status(False, None, None, None, "fallback")


class Schedule:
    """Closed periods for one location, refreshed at most every `refresh` interval.

    - A failed refresh backs off for `retry` before trying again, so an outage does not make every
      request wait on Hebcal.
    - The fetch runs outside the lock: one request refreshes, the others use what is known.
    - The last good schedule is saved to `cache_file`, so a restart during an outage still knows
      the real times (they cover weeks ahead) instead of falling back to Friday and Saturday.
    """

    def __init__(
        self,
        geonameid: int = JERUSALEM,
        israel: bool = True,
        tz: str = "Asia/Jerusalem",
        client: httpx.Client | None = None,
        refresh: timedelta = timedelta(hours=6),
        retry: timedelta = timedelta(minutes=5),
        days_ahead: int = 45,
        cache_file: Path | None = None,
    ) -> None:
        self.geonameid = geonameid
        self.israel = israel
        self.tz = ZoneInfo(tz)
        self.client = client or http_client(timeout=10)
        self.refresh = refresh
        self.retry = retry
        self.days_ahead = days_ahead
        self.cache_file = cache_file
        self._periods: list[Period] = self._load()
        self._fetched_at: datetime | None = None
        self._next_try: datetime | None = None
        self._fetching = False
        self._lock = threading.Lock()

    def fetch(self, start: date) -> list[Period]:
        params = {
            "v": "1",
            "cfg": "json",
            "maj": "on",
            "c": "on",
            "M": "on",
            "s": "off",
            "geo": "geoname",
            "geonameid": str(self.geonameid),
            "i": "on" if self.israel else "off",
            "start": (start - timedelta(days=3)).isoformat(),
            "end": (start + timedelta(days=self.days_ahead)).isoformat(),
        }
        response = self.client.get(BASE, params=params)
        response.raise_for_status()
        return closed_periods(parse_events(response.json()))

    def status(self, now: datetime | None = None) -> Status:
        now = now or datetime.now(self.tz)
        with self._lock:
            stale = self._fetched_at is None or now - self._fetched_at > self.refresh
            due = self._next_try is None or now >= self._next_try
            refresh = stale and due and not self._fetching
            if refresh:
                self._fetching = True
        if refresh:
            try:
                periods = self.fetch(now.date())
            except (httpx.HTTPError, ValueError, PermissionError):
                with self._lock:
                    self._next_try = now + self.retry
                    self._fetching = False
            else:
                with self._lock:
                    self._periods, self._fetched_at, self._next_try = periods, now, None
                    self._fetching = False
                self._save(periods)
        with self._lock:
            periods = list(self._periods)
        if not periods or periods[-1].end < now:
            return fallback_status(now.astimezone(self.tz))
        return status_at(periods, now)

    def _save(self, periods: Sequence[Period]) -> None:
        if self.cache_file is None:
            return
        data = [
            {
                "start": p.start.isoformat(),
                "end": p.end.isoformat(),
                "reason": p.reason,
                "reason_he": p.reason_he,
            }
            for p in periods
        ]
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass  # the cache is a convenience; the in-memory schedule still applies

    def _load(self) -> list[Period]:
        if self.cache_file is None or not self.cache_file.exists():
            return []
        try:
            data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            return [
                Period(
                    datetime.fromisoformat(d["start"]),
                    datetime.fromisoformat(d["end"]),
                    d["reason"],
                    d["reason_he"],
                )
                for d in data
            ]
        except (OSError, ValueError, KeyError):
            return []


def status_at(periods: Sequence[Period], now: datetime) -> Status:
    for p in periods:
        if p.start <= now < p.end:
            return Status(True, p.end, p.reason, p.reason_he, "hebcal")
    return Status(False, None, None, None, "hebcal")
