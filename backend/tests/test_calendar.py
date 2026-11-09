"""Shabbat and Yom Tov closing times (from recorded Hebcal responses) and SM-2 scheduling."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx2 as httpx
import pytest

from chavruta.hebcal import Schedule, closed_periods, fallback_status, parse_events, status_at
from chavruta.srs import CardState, is_due, review

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "hebcal.json").read_text(encoding="utf-8"))
IL = ZoneInfo("Asia/Jerusalem")


def at(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=IL)


def test_rosh_hashana_next_to_shabbat_is_one_closed_period() -> None:
    """Rosh Hashana 5788 begins Friday evening: Shabbat, then the second day, then havdalah Sunday."""
    periods = closed_periods(parse_events(FIXTURE["rosh_hashana_5788"]))
    first = periods[0]
    assert first.reason == "Rosh Hashana 5788"
    assert first.start == at("2027-10-01T17:44")
    assert first.end == at("2027-10-03T18:58")
    assert status_at(periods, at("2027-10-02T21:00")).closed  # Saturday night, second day of Yom Tov
    assert [p.reason for p in periods] == ["Rosh Hashana 5788", "Shabbat", "Yom Kippur"]


def test_pesach_first_day_then_shabbat_chol_hamoed() -> None:
    periods = closed_periods(parse_events(FIXTURE["pesach_5787"]))
    assert [(p.reason, p.reason_he) for p in periods[:2]] == [("Pesach I", "פסח א׳"), ("Shabbat", "שבת")]
    weekday_chol_hamoed = at("2027-04-23T12:00")
    assert not status_at(periods, weekday_chol_hamoed).closed


def test_boundaries_candle_lighting_closes_and_havdalah_reopens() -> None:
    periods = closed_periods(parse_events(FIXTURE["pesach_5787"]))
    p = periods[0]
    assert not status_at(periods, p.start - timedelta(minutes=1)).closed
    s = status_at(periods, p.start)
    assert s.closed and s.until == p.end and s.source == "hebcal"
    assert not status_at(periods, p.end).closed


def test_fallback_closes_friday_noon_to_saturday_night() -> None:
    assert not fallback_status(at("2026-11-13T11:59")).closed  # Friday morning
    s = fallback_status(at("2026-11-13T12:00"))
    assert s.closed and s.source == "fallback"
    assert fallback_status(at("2026-11-14T23:00")).closed  # Saturday night
    assert not fallback_status(at("2026-11-15T00:00")).closed  # Sunday
    assert not fallback_status(at("2026-11-11T15:00")).closed  # Wednesday


def _schedule(handler: Any) -> Schedule:
    return Schedule(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_schedule_uses_hebcal_and_caches() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=FIXTURE["pesach_5787"])

    s = _schedule(handler)
    assert s.status(at("2027-04-22T10:00")).reason == "Pesach I"
    assert s.status(at("2027-04-22T15:00")).closed
    assert len(calls) == 1  # within the refresh interval: served from the cache
    assert not s.status(at("2027-04-22T20:00")).closed
    assert len(calls) == 2
    params = calls[0].url.params
    assert params["geonameid"] == "281184" and params["c"] == "on" and params["i"] == "on"


def test_schedule_falls_back_when_hebcal_is_down_and_nothing_is_cached() -> None:
    s = _schedule(lambda r: httpx.Response(503))
    assert s.status(at("2026-11-14T10:00")).source == "fallback"


def test_schedule_keeps_cached_times_when_a_refresh_fails() -> None:
    state = {"up": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if state["up"]:
            return httpx.Response(200, json=FIXTURE["pesach_5787"])
        return httpx.Response(503)

    s = _schedule(handler)
    s.status(at("2027-04-20T10:00"))
    state["up"] = False
    later = s.status(at("2027-04-22T10:00"))  # refresh is due and fails: cached periods still apply
    assert later.closed and later.source == "hebcal"


# ---- SM-2 -----------------------------------------------------------------------------------------

TODAY = date(2026, 11, 9)


def test_good_answers_grow_intervals_1_6_then_by_ease() -> None:
    s = CardState()
    intervals = []
    for _ in range(4):
        s = review(s, 4, TODAY)
        intervals.append(s.interval)
    assert intervals == [1, 6, 15, 38]
    assert s.ease == pytest.approx(2.5)
    assert s.due == TODAY + timedelta(days=38)


def test_a_lapse_resets_the_card_and_lowers_ease() -> None:
    s = review(review(review(CardState(), 5, TODAY), 5, TODAY), 1, TODAY)
    assert (s.repetitions, s.interval, s.lapses) == (0, 1, 1)
    assert s.ease < 2.7


def test_ease_never_drops_below_the_floor() -> None:
    s = CardState()
    for _ in range(20):
        s = review(s, 3, TODAY)
    assert s.ease == pytest.approx(1.3)


def test_due_and_invalid_grades() -> None:
    assert is_due(CardState(), TODAY)
    s = review(CardState(), 4, TODAY)
    assert not is_due(s, TODAY) and is_due(s, TODAY + timedelta(days=1))
    with pytest.raises(ValueError, match="grade"):
        review(CardState(), 7, TODAY)


def test_an_outage_backs_off_instead_of_refetching_on_every_request() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503)

    s = _schedule(handler)
    for minute in range(3):
        s.status(at(f"2026-11-14T10:0{minute}"))
    assert calls["n"] == 1  # retried only after the back-off
    s.status(at("2026-11-14T10:06"))
    assert calls["n"] == 2


def test_a_restart_during_an_outage_keeps_the_saved_times(tmp_path: Path) -> None:
    cache = tmp_path / "schedule.json"
    up = Schedule(
        client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=FIXTURE["pesach_5787"]))
        ),
        cache_file=cache,
    )
    up.status(at("2027-04-20T10:00"))
    down = Schedule(
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))), cache_file=cache
    )
    s = down.status(at("2027-04-22T10:00"))  # Pesach I, a weekday: the fallback alone would miss it
    assert s.closed
    assert s.reason == "Pesach I"
