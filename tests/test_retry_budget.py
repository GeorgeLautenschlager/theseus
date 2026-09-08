"""The retry budget, as arithmetic (#32 Task 2). Offline and clock-free."""

from __future__ import annotations

import contextlib
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from theseus.surrogates.retry import RetryBudget, backoff_delay, is_too_old


@contextlib.contextmanager
def _local_zone(tz_name: str):
    """Force the process's local timezone for the duration.

    `datetime.astimezone()` with no argument reads the process zone, which on CI is usually
    UTC — precisely the case where reading a naive value as UTC and reading it as local give
    the same answer, so a naive-as-UTC bug becomes invisible. `TZ` + `tzset()` is the only
    thing that actually moves what `astimezone()` believes.
    """
    previous = os.environ.get("TZ")
    os.environ["TZ"] = tz_name
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = previous
        time.tzset()


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
    """At the maximum legal jitter the low end of the band is exactly zero, so the clamp is
    load-bearing rather than decorative. Anything above 1.0 is now refused at construction
    (see the validation test), which is why this pins 1.0 rather than an absurd value."""
    budget = RetryBudget(jitter=1.0)
    for r in (0.0, 0.5, 1.0):
        assert backoff_delay(1, budget, random_fn=lambda r=r: r) >= 0.0


def test_inside_max_age_is_not_too_old() -> None:
    budget = RetryBudget()
    now = datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc)
    oldest = now - budget.max_age
    assert not is_too_old(oldest, now, budget)  # boundary: > not >=
    assert is_too_old(oldest - timedelta(seconds=1), now, budget)


def test_a_naive_timestamp_does_not_raise() -> None:
    """Comparing naive against aware raises a TypeError naming neither value, and a
    surrogate must not die of that. This only pins "no exception" — that the *reading* is
    host-local is pinned below, where it can actually fail."""
    budget = RetryBudget()
    now = datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc)
    naive = now.replace(tzinfo=None) - timedelta(seconds=1)
    assert is_too_old(naive, now, budget) is False


def test_a_naive_timestamp_is_read_as_host_local_not_as_utc():
    """`astimezone()`, matching `StimulusLog._aware` and `replication_events._utc_span`.

    Stamping UTC on a naive value instead is a different claim, and west of Greenwich a
    wrong one: it makes an event look hours older than it is. Against a six-hour budget
    that abandons batches with hours of life left, so this is data loss, not a style point.

    On a machine whose local zone IS UTC — an ordinary CI container — reading naive as UTC
    and reading it as host-local give the same answer, so a test that relies on the host's
    own offset stops catching the bug there and says nothing about it. This forces a
    non-UTC zone for the duration instead.
    """
    now = datetime(2026, 9, 7, 16, 0, tzinfo=timezone.utc)

    # A zone WEST of UTC specifically. East of it, stamping UTC makes the event look like
    # it happened in the future — which is not "too old" either, so both readings answer
    # False and the bug hides. Only a negative offset makes the wrong reading look *older*.
    with _local_zone("Etc/GMT+8"):  # UTC-8, no DST
        # Genuinely 3h old. Read as host-local that is 3h and inside the 6h budget; stamped
        # UTC it reads as 11h and the batch is abandoned with hours of life left.
        three_hours_old = (now - timedelta(hours=3)).astimezone().replace(tzinfo=None)

        assert is_too_old(three_hours_old, now, RetryBudget()) is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"max_attempts": -1},
        {"base_seconds": -1.0},
        {"multiplier": 0.5},
        {"jitter": 1.5},
        {"jitter": -0.1},
        {"ceiling_seconds": -1.0},
        {"max_age": timedelta(0)},
        {"max_age": timedelta(seconds=-1)},
    ],
)
def test_a_budget_that_would_silently_do_nothing_is_rejected(kwargs):
    """`max_attempts=0` is the one that matters: the drain's retry loop is a `range` over
    it, so the body never runs — no send, no abandon, no cursor advance, no `stopped_on`.
    The drain then reports a clean pass having shipped nothing, forever. That is the stall
    this issue closes, arriving through configuration instead of through a host."""
    with pytest.raises(ValueError):
        RetryBudget(**kwargs)


def test_the_default_budget_still_constructs():
    assert RetryBudget().max_attempts == 5
