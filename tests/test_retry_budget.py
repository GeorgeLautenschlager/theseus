"""The retry budget, as arithmetic (#32 Task 2). Offline and clock-free."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from theseus.surrogates.retry import RetryBudget, backoff_delay, is_too_old


def test_defaults_are_the_issue_39_numbers() -> None:
    budget = RetryBudget()
    assert budget.max_attempts == 5
    assert budget.base_seconds == 2.0
    assert budget.multiplier == 3.0
    assert budget.jitter == 0.25
    assert budget.ceiling_seconds == 120.0
    assert budget.max_age == timedelta(hours=6)


def test_unjittered_sequence_matches_issue_39() -> None:
    budget = RetryBudget(jitter=0.0)
    assert [backoff_delay(a, budget) for a in range(1, 7)] == [
        2.0,
        6.0,
        18.0,
        54.0,
        120.0,
        120.0,
    ]


def test_ceiling_clamps_far_out_attempts() -> None:
    budget = RetryBudget(jitter=0.0)
    assert backoff_delay(20, budget) == budget.ceiling_seconds


def test_jitter_stays_inside_its_band() -> None:
    budget = RetryBudget(jitter=0.25)
    raw = 6.0  # attempt 2, well below the ceiling
    for r in (0.0, 1.0):
        d = backoff_delay(2, budget, random_fn=lambda r=r: r)
        assert 0.75 * raw <= d <= 1.25 * raw


def test_jitter_never_exceeds_the_ceiling() -> None:
    """The obvious clamp-then-jitter implementation fails exactly this."""
    budget = RetryBudget(jitter=0.25)
    assert backoff_delay(6, budget, random_fn=lambda: 1.0) <= budget.ceiling_seconds


def test_delay_is_never_negative() -> None:
    budget = RetryBudget(jitter=5.0)
    for r in (0.0, 1.0):
        assert backoff_delay(1, budget, random_fn=lambda r=r: r) >= 0.0


def test_inside_max_age_is_not_too_old() -> None:
    budget = RetryBudget()
    now = datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc)
    oldest = now - budget.max_age
    assert not is_too_old(oldest, now, budget)  # boundary: > not >=
    assert is_too_old(oldest - timedelta(seconds=1), now, budget)


def test_naive_timestamp_is_treated_as_host_local() -> None:
    budget = RetryBudget()
    now = datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc)
    naive = now.replace(tzinfo=None) - timedelta(seconds=1)
    assert isinstance(is_too_old(naive, now, budget), bool)


def test_a_naive_timestamp_is_read_as_host_local_not_as_utc():
    """`astimezone()`, matching `StimulusLog._aware` and `replication_events._utc_span`.

    Stamping UTC on a naive value instead is a different claim, and west of Greenwich a
    wrong one: it makes an event look hours older than it is. Against a six-hour budget
    that abandons batches with hours of life left, so this is data loss, not a style point.
    """
    now = datetime(2026, 9, 7, 16, 0, tzinfo=timezone.utc)
    three_hours_old = (now - timedelta(hours=3)).astimezone().replace(tzinfo=None)

    assert is_too_old(three_hours_old, now, RetryBudget()) is False
