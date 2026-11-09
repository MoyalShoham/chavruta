"""Spaced repetition (SM-2) for review cards.

Grades: 1 again, 3 hard, 4 good, 5 easy (SM-2's 0-5 scale; below 3 is a lapse). Intervals are whole
days and due dates are calendar dates, so a review late at night does not shift the schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta

MIN_EASE = 1.3
GRADES = (1, 3, 4, 5)


@dataclass(frozen=True)
class CardState:
    ease: float = 2.5
    interval: int = 0
    repetitions: int = 0
    due: date | None = None
    lapses: int = 0


def review(state: CardState, grade: int, today: date) -> CardState:
    if grade not in range(6):
        raise ValueError("grade must be 0-5")
    ease = max(MIN_EASE, state.ease + 0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02))
    if grade < 3:
        return replace(
            state,
            ease=ease,
            interval=1,
            repetitions=0,
            due=today + timedelta(days=1),
            lapses=state.lapses + 1,
        )
    repetitions = state.repetitions + 1
    if repetitions == 1:
        interval = 1
    elif repetitions == 2:
        interval = 6
    else:
        interval = max(state.interval + 1, round(state.interval * ease))
    return replace(
        state, ease=ease, interval=interval, repetitions=repetitions, due=today + timedelta(days=interval)
    )


def is_due(state: CardState, today: date) -> bool:
    return state.due is None or state.due <= today
