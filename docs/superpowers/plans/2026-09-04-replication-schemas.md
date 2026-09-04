# Replication Schemas, Read-Time Ordering and High-Water Marks

> **For agentic workers:** REQUIRED SUB-SKILL: Use steward:steward-local-sdd to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **⚠️ The task sections below are the plan as written *before* implementation, and the source
> they embed is superseded.** Five review rounds changed both new modules: `replication_events.py`
> gained `_check_origin` / `_check_range` / `_utc_span` / `_clean_reason`, a `MAX_REASON_CHARS`
> bound, a `DeclaredReason` literal and per-field type checks; `high_water.py` gained a
> `threading.Lock` and a `_committed_seq` guard on recovery. **The shipped modules are the source
> of truth** — read them, not this. The sections are kept unedited so the gap between intent and
> outcome stays legible, and every change is recorded with its reasoning in
> `.steward/runs/27-28-29/runlog.md`.

**Goal:** Land the three issues that depend only on #26 — the gap/rejection event vocabulary (#27), the Assembler's read-time chronological ordering (#28), and the host's per-origin high-water marks (#29).

**Architecture:** Three independent pieces, each touching a different file. #27 and #29 are new flat modules under `src/theseus/`; #28 is a surgical change to `ContextAssembler`'s output ordering that leaves its budget maths untouched. Nothing here transports, emits, or applies anything — #27 is vocabulary, #29 is state, #28 is a sort. The ingress that uses all three is #30.

**Tech Stack:** Python 3.12, Poetry (in-project `.venv`), pytest. Run everything with `poetry run`.

**Source issues:** GitHub #27, #28, #29.
**Spec:** `docs/superpowers/specs/2026-09-04-surrogate-replication-protocol.md` — § Gap markers (#27), § "Append, do not interleave" (#28), § Dedupe (#29).
**Builds on:** #26, merged as `4eee1ed`. `StimulusEvent` already carries `origin`, `seq`, `ts` (the spec's `event_ts`) and `appended_ts`; `StimulusLog` already allocates a monotonic seq per origin and accepts replicated appends via `append(..., origin=, seq=)`.

---

## File Structure

- **Create: `src/theseus/replication_events.py`** (#27) — the shared vocabulary for holes in a
  stream. Pure constructors + validation returning `content` dicts. No I/O, no imports from
  anything but the standard library.
- **Create: `src/theseus/high_water.py`** (#29) — `HighWaterMarks`, derived from the log at
  construction. Imports `StimulusLog` to read; nothing imports it back.
- **Modify: `src/theseus/context_assembler.py`** (#28) — one module-level sort key and two
  call sites. `_fit_to_budget`'s accumulation, clamping and at-least-one guarantee are
  untouched; only the order it emits changes.
- **Create: `tests/test_replication_events.py`**, **`tests/test_high_water.py`**; **modify
  `tests/test_context_assembler.py`**.

## Decisions locked in for this plan

Settled — implement as written, do not re-derive:

| Decision | Value |
|---|---|
| #27 constructors return | a `content` dict, not a `StimulusEvent`. The caller owns actor/ts and calls `append`; this module does no I/O. |
| #27 event type names | module constants `GAP = "stimulus.gap"`, `BATCH_REJECTED = "replication.batch_rejected"` |
| Datetimes inside `content` | normalised with `.astimezone(timezone.utc).isoformat()`, matching `StimulusEvent.to_json`. `content` must be JSON-native. |
| Naive datetimes | accepted and treated as host-local, exactly as `StimulusEvent.to_json` already does. Forking that behaviour in one new module would be worse than the whole-codebase follow-up already tracked. |
| #29 recovery strategy | **derive from the log**, no sidecar. The issue's recommended option — a sidecar that disagrees with the log after a crash means dropping real events or double-appending them. |
| `high_water` on an unseen origin | `None`, deliberately distinct from `0`. Seqs start at 1 (from #26) precisely so `None` stays available to mean "nothing has ever arrived". |
| `HighWaterMarks` vs `StimulusLog._recover_next_seq` | kept separate, not refactored into one. Documented in the class docstring — they answer different questions and diverge the moment an ingress advances a mark on commit. |
| New public exports | **none in this branch.** `src/theseus/__init__.py` is unchanged. Nothing composes these until #30 wires an ingress, and `__all__` is a compatibility commitment for tag-pinned consumers. Export the whole set together when #30 gives them a caller. |

---

### Task 1 (#27): the gap and rejection vocabulary

**Files:**
- Create: `src/theseus/replication_events.py`
- Test: `tests/test_replication_events.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_replication_events.py`:

```python
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from theseus.replication_events import (
    BATCH_REJECTED,
    DECLARED_REASONS,
    GAP,
    INFERRED_REASON,
    batch_rejected,
    declared_gap,
    inferred_gap,
)
from theseus.stimulus_log import StimulusEvent

SPAN_START = datetime(2026, 9, 4, 16, 2, tzinfo=timezone.utc)
SPAN_END = datetime(2026, 9, 4, 16, 40, tzinfo=timezone.utc)


def test_a_declared_gap_carries_the_range_the_surrogate_abandoned():
    content = declared_gap(
        origin="kitchen-surrogate",
        from_seq=12,
        to_seq=48,
        reason="link_down",
        span_start=SPAN_START,
        span_end=SPAN_END,
    )

    assert content == {
        "origin": "kitchen-surrogate",
        "from_seq": 12,
        "to_seq": 48,
        "reason": "link_down",
        "span_start": "2026-09-04T16:02:00+00:00",
        "span_end": "2026-09-04T16:40:00+00:00",
        "declared": True,
    }


def test_an_inferred_gap_is_marked_undeclared_and_reasonless():
    """A seq jump with no marker: the surrogate died mid-buffer, or something is broken.
    Same hole as a declared gap, different diagnosis — so the host says which it is."""
    content = inferred_gap(
        origin="android-01",
        from_seq=5,
        to_seq=9,
        span_start=SPAN_START,
        span_end=SPAN_END,
    )

    assert content["declared"] is False
    assert content["reason"] == INFERRED_REASON


@pytest.mark.parametrize("reason", DECLARED_REASONS)
def test_every_declared_reason_is_accepted(reason):
    content = declared_gap(
        origin="kitchen-surrogate",
        from_seq=1,
        to_seq=2,
        reason=reason,
        span_start=SPAN_START,
        span_end=SPAN_END,
    )

    assert content["reason"] == reason


def test_an_unknown_reason_raises_rather_than_serialising():
    with pytest.raises(ValueError):
        declared_gap(
            origin="kitchen-surrogate",
            from_seq=1,
            to_seq=2,
            reason="gremlins",
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_the_inferred_reason_is_not_declarable():
    """`inferred` is what the host writes when nobody declared anything. A surrogate
    claiming it would erase the one distinction these events exist to carry."""
    with pytest.raises(ValueError):
        declared_gap(
            origin="kitchen-surrogate",
            from_seq=1,
            to_seq=2,
            reason=INFERRED_REASON,
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_an_inverted_seq_range_raises():
    with pytest.raises(ValueError):
        inferred_gap(
            origin="android-01",
            from_seq=9,
            to_seq=5,
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_a_seq_below_one_raises():
    """Seqs start at 1, so 0 in a range is a bug, not a boundary."""
    with pytest.raises(ValueError):
        inferred_gap(
            origin="android-01",
            from_seq=0,
            to_seq=5,
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_an_inverted_span_raises():
    with pytest.raises(ValueError):
        inferred_gap(
            origin="android-01",
            from_seq=1,
            to_seq=2,
            span_start=SPAN_END,
            span_end=SPAN_START,
        )


def test_a_single_event_range_is_legal():
    """A one-event hole is a hole."""
    content = inferred_gap(
        origin="android-01",
        from_seq=7,
        to_seq=7,
        span_start=SPAN_START,
        span_end=SPAN_START,
    )

    assert (content["from_seq"], content["to_seq"]) == (7, 7)


def test_a_batch_rejection_records_what_the_host_said():
    content = batch_rejected(
        origin="kitchen-surrogate",
        from_seq=100,
        to_seq=120,
        status=413,
        reason="batch exceeds max byte size",
    )

    assert content == {
        "origin": "kitchen-surrogate",
        "from_seq": 100,
        "to_seq": 120,
        "status": 413,
        "reason": "batch exceeds max byte size",
    }


def test_a_non_4xx_rejection_status_raises():
    """This event exists for the permanently-unacceptable class. A 5xx is retried, not
    rejected, and recording one here would make the tape claim a batch was abandoned
    when the surrogate is still trying to send it."""
    with pytest.raises(ValueError):
        batch_rejected(
            origin="kitchen-surrogate",
            from_seq=1,
            to_seq=2,
            status=503,
            reason="upstream down",
        )


def test_an_empty_origin_raises():
    with pytest.raises(ValueError):
        inferred_gap(
            origin="",
            from_seq=1,
            to_seq=2,
            span_start=SPAN_START,
            span_end=SPAN_END,
        )


def test_a_gap_round_trips_through_the_event_envelope():
    """These are ordinary events on the tape — the whole point is that a hole is
    something the agent reads, not an error channel beside the log."""
    event = StimulusEvent(
        id="01ABCDEFGHJKMNPQRSTVWXYZ0",
        ts=SPAN_END,
        actor="kitchen-surrogate",
        type=GAP,
        content=declared_gap(
            origin="kitchen-surrogate",
            from_seq=12,
            to_seq=48,
            reason="retry_exhausted",
            span_start=SPAN_START,
            span_end=SPAN_END,
        ),
    )

    assert StimulusEvent.from_json(event.to_json()) == event


def test_a_rejection_round_trips_through_the_event_envelope():
    event = StimulusEvent(
        id="01ABCDEFGHJKMNPQRSTVWXYZ1",
        ts=SPAN_END,
        actor="kitchen-surrogate",
        type=BATCH_REJECTED,
        content=batch_rejected(
            origin="kitchen-surrogate",
            from_seq=100,
            to_seq=120,
            status=400,
            reason="malformed line 3",
        ),
    )

    assert StimulusEvent.from_json(event.to_json()) == event


def test_spans_are_normalised_to_utc():
    """A gap is evidence about time, so it is stored in the one zone every node agrees on."""
    tokyo = timezone(timedelta(hours=9))
    content = inferred_gap(
        origin="android-01",
        from_seq=1,
        to_seq=2,
        span_start=datetime(2026, 9, 5, 1, 2, tzinfo=tokyo),
        span_end=datetime(2026, 9, 5, 1, 40, tzinfo=tokyo),
    )

    assert content["span_start"] == "2026-09-04T16:02:00+00:00"
    assert content["span_end"] == "2026-09-04T16:40:00+00:00"


def test_content_is_json_native():
    """`content` goes through `json.dumps` inside `to_json`; a datetime left in it would
    raise there rather than here, a long way from the mistake."""
    content = declared_gap(
        origin="kitchen-surrogate",
        from_seq=1,
        to_seq=2,
        reason="storage_pressure",
        span_start=SPAN_START,
        span_end=SPAN_END,
    )

    assert json.loads(json.dumps(content)) == content
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/test_replication_events.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'theseus.replication_events'`

- [ ] **Step 3: Write the implementation**

Create `src/theseus/replication_events.py`:

```python
"""Vocabulary for the holes in a replicated stream.

Gaps are normal operation on this protocol, not errors — a surrogate may be an edge device
kilometres from the nearest tower, and the host is enriched by as much as it manages to
deliver and unbothered by the rest. But the protocol keeps a distinction worth preserving:

- A **declared** gap carries a marker the surrogate emitted with the abandoned range in
  hand — "I wasn't observing from 16:02 to 16:40, the link was down." The agent can reason
  about it.
- An **inferred** gap is a seq jump the host noticed with no marker to explain it: the
  surrogate died mid-buffer, or something is genuinely broken.

Same hole, different diagnosis. Neither is rejected; both go on the tape as ordinary
events, because a hole the agent can read about is worth more than a silence it cannot
account for.

This module is the shared vocabulary only — constructors and validation. Nothing here
emits, appends or transports anything; the ingress and the replicator own that. What it
does own is refusing to build a malformed marker: an invalid reason or an inverted range
raises here, at the mistake, rather than serialising into the log where it is permanent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Event types. These strings are wire protocol — a surrogate and a host built from
# different releases still have to agree on them.
GAP = "stimulus.gap"
BATCH_REJECTED = "replication.batch_rejected"

# Why a surrogate abandoned a range, in its own words.
DECLARED_REASONS = ("link_down", "retry_exhausted", "storage_pressure")

# What the host writes when it sees a jump nobody declared. Not declarable by a surrogate:
# a surrogate claiming it would erase the distinction these events exist to carry.
INFERRED_REASON = "inferred"


def declared_gap(
    *,
    origin: str,
    from_seq: int,
    to_seq: int,
    reason: str,
    span_start: datetime,
    span_end: datetime,
) -> dict[str, Any]:
    """Content for a `stimulus.gap` the surrogate declared about its own stream.

    The surrogate always knows what it dropped — `seq` is assigned at local write time, so
    eviction or abandonment happens with the range in hand.
    """
    if reason not in DECLARED_REASONS:
        raise ValueError(
            f"unknown declared reason {reason!r}; expected one of {DECLARED_REASONS}. "
            f"{INFERRED_REASON!r} is host-minted — use inferred_gap()."
        )
    return {
        "origin": _origin(origin),
        **_seq_range(from_seq, to_seq),
        "reason": reason,
        **_span(span_start, span_end),
        "declared": True,
    }


def inferred_gap(
    *,
    origin: str,
    from_seq: int,
    to_seq: int,
    span_start: datetime,
    span_end: datetime,
) -> dict[str, Any]:
    """Content for a `stimulus.gap` the host minted because nobody declared one.

    `declared: False` is the whole payload of this event: it says the hole was diagnosed,
    not reported, and that whatever is on the other end may be in trouble.
    """
    return {
        "origin": _origin(origin),
        **_seq_range(from_seq, to_seq),
        "reason": INFERRED_REASON,
        **_span(span_start, span_end),
        "declared": False,
    }


def batch_rejected(
    *,
    origin: str,
    from_seq: int,
    to_seq: int,
    status: int,
    reason: str,
) -> dict[str, Any]:
    """Content for a `replication.batch_rejected`, emitted locally by the surrogate.

    The `4xx` class exists so one malformed or oversized batch cannot wedge the channel
    forever. Recording the rejection makes the resulting hole visible on the tape instead
    of a silence. A `5xx` is retried rather than rejected, so it is not a valid status
    here — writing one would claim a batch was abandoned while the surrogate is still
    trying to send it.
    """
    if not 400 <= status < 500:
        raise ValueError(
            f"status must be 4xx — permanently unacceptable — but got {status!r}"
        )
    return {
        "origin": _origin(origin),
        **_seq_range(from_seq, to_seq),
        "status": status,
        "reason": reason,
    }


def _origin(origin: str) -> str:
    if not origin:
        raise ValueError("origin must be a non-empty name")
    return origin


def _seq_range(from_seq: int, to_seq: int) -> dict[str, int]:
    """The abandoned range, inclusive. A single-event hole is `from_seq == to_seq`."""
    if from_seq < 1:
        raise ValueError(f"from_seq must be 1 or greater (got {from_seq!r})")
    if to_seq < from_seq:
        raise ValueError(
            f"inverted seq range: from_seq {from_seq} is above to_seq {to_seq}"
        )
    return {"from_seq": from_seq, "to_seq": to_seq}


def _span(span_start: datetime, span_end: datetime) -> dict[str, str]:
    """The wall-clock span the hole covers, normalised to UTC.

    `content` is serialised by `json.dumps` inside `StimulusEvent.to_json`, which cannot
    encode a datetime — so these become ISO strings here rather than failing at write
    time, a long way from the mistake.
    """
    if span_end < span_start:
        raise ValueError(
            f"inverted span: span_start {span_start} is after span_end {span_end}"
        )
    return {
        "span_start": span_start.astimezone(timezone.utc).isoformat(),
        "span_end": span_end.astimezone(timezone.utc).isoformat(),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `poetry run pytest tests/test_replication_events.py -q`
Expected: PASS.

- [ ] **Step 5: Run the offline suite**

Run: `poetry run pytest -q tests/ --ignore=tests/test_fact_retention.py --ignore=tests/e2e`
Expected: PASS, 0 failures. (`tests/test_fact_retention.py` and `tests/e2e` need a live LLM
endpoint; they are excluded from every test command in this plan. Never run the bare
`tests/` directory — it hangs.)

- [ ] **Step 6: Commit**

```bash
git add src/theseus/replication_events.py tests/test_replication_events.py
git commit -m "Add gap marker and batch rejection event schemas"
```

---

### Task 2 (#28): the Assembler orders its window by `event_ts`

**Files:**
- Modify: `src/theseus/context_assembler.py` (`assemble_context`, `_fit_to_budget`, plus one new module-level function)
- Test: `tests/test_context_assembler.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_context_assembler.py`. It already imports `json` and `StimulusLog`; add
`from datetime import datetime, timedelta, timezone` to the imports at the top.

```python
class TestChronologicalOrdering:
    """The spec's load-bearing decision: the host appends a replicated batch in *arrival*
    order and never interleaves it into history. Chronology is a read-time concern, and
    this is where it is paid for."""

    def test_a_late_batch_interleaves_in_the_window_but_not_in_the_log(self, tmp_path):
        log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
        base = datetime(2026, 9, 4, 16, 0, tzinfo=timezone.utc)
        log.append(actor="george", type="exchange", content={"message": "first"},
                   ts=base)
        log.append(actor="george", type="exchange", content={"message": "third"},
                   ts=base + timedelta(minutes=40))
        # A surrogate comes back online and ships an hour-old observation.
        log.append(
            actor="kitchen", type="observation", content={"message": "second"},
            ts=base + timedelta(minutes=20), origin="kitchen-surrogate", seq=1,
        )

        assembled = ContextAssembler(stimulus_log=log, window_size=50).assemble_context()

        window = [json.loads(line)["content"]["message"]
                  for line in assembled.recent_events.splitlines()]
        assert window == ["first", "second", "third"]
        # The log itself is untouched: append-only, arrival-ordered.
        assert [e.content["message"] for e in log.read_all()] == [
            "first", "third", "second"
        ]

    def test_identical_event_ts_comes_out_in_a_stable_deterministic_order(self, tmp_path):
        """Two surrogates stamping the same millisecond must not swap places between two
        assemblies of the same window."""
        log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
        same = datetime(2026, 9, 4, 16, 0, tzinfo=timezone.utc)
        for i in range(5):
            log.append(actor="george", type="exchange", content={"message": f"msg {i}"},
                       ts=same)

        assembler = ContextAssembler(stimulus_log=log, window_size=50)
        first = assembler.assemble_context().recent_events
        second = assembler.assemble_context().recent_events

        assert first == second
        messages = [json.loads(line)["content"]["message"]
                    for line in first.splitlines()]
        assert messages == [f"msg {i}" for i in range(5)]

    def test_a_window_with_no_skew_is_unchanged(self, tmp_path):
        """The common case — one producer, arrival order already chronological — must
        come out byte-identical to what it was before ordering existed."""
        log = fill_log(tmp_path, 5)

        assembled = ContextAssembler(stimulus_log=log, window_size=50).assemble_context()

        assert assembled.recent_events == "\n".join(
            e.to_json() for e in log.read_all()
        )

    def test_the_unbudgeted_path_is_ordered_too(self, tmp_path):
        """`token_budget=None` with no declared context skips `_fit_to_budget` entirely —
        it needs the same ordering, or the guarantee depends on which budget is in force."""
        log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
        base = datetime(2026, 9, 4, 16, 0, tzinfo=timezone.utc)
        log.append(actor="george", type="exchange", content={"message": "later"},
                   ts=base + timedelta(minutes=40))
        log.append(actor="george", type="exchange", content={"message": "earlier"},
                   ts=base)

        assembled = ContextAssembler(
            stimulus_log=log, window_size=50, token_budget=None
        ).assemble_context()

        messages = [json.loads(line)["content"]["message"]
                    for line in assembled.recent_events.splitlines()]
        assert messages == ["earlier", "later"]

    def test_selection_is_still_by_arrival_so_the_newest_events_are_kept(self, tmp_path):
        """Ordering the output must not turn into ordering the *selection*: `window_size`
        means the most recently arrived events, which is what keeps selection cheap on a
        long log."""
        log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
        base = datetime(2026, 9, 4, 16, 0, tzinfo=timezone.utc)
        for i in range(5):
            log.append(actor="george", type="exchange", content={"message": f"old {i}"},
                       ts=base + timedelta(minutes=i))
        # Arrives last, but is the oldest thing in the window by its own clock.
        log.append(actor="kitchen", type="observation", content={"message": "backfill"},
                   ts=base - timedelta(hours=1), origin="kitchen-surrogate", seq=1)

        assembled = ContextAssembler(stimulus_log=log, window_size=2).assemble_context()

        messages = [json.loads(line)["content"]["message"]
                    for line in assembled.recent_events.splitlines()]
        assert messages == ["backfill", "old 4"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/test_context_assembler.py -q -k Chronological`
Expected: FAIL — the interleaving and unbudgeted tests assert an order the code does not
produce yet.

- [ ] **Step 3: Write the implementation**

In `src/theseus/context_assembler.py`, add this module-level function immediately above the
`ContextAssembler` class definition:

```python
def _chronological(event: StimulusEvent) -> tuple[datetime, datetime, str]:
    """Sort key for the emitted window: when the event *happened*, by its producer's clock.

    The log is append-only and arrival-ordered — the host appends a replicated batch where
    it lands and does not interleave it into history. Chronology is a read-time concern,
    and this is the read. Without it, a surrogate that replicates an hour of backlog hands
    the model stale observation *after* the things that happened since.

    Ties break on arrival and then on id, so the order is total and deterministic: two
    surrogates stamping the same millisecond must not swap places between two assemblies of
    the same window.
    """
    return (event.ts, event.appended_ts, event.id)
```

That module does **not** currently import `datetime` — its imports are `copy`, `dataclass`/
`replace`, `Any`, and `StimulusEvent`/`StimulusLog`. Add:

```python
from datetime import datetime
```

alongside the existing `import copy` block, keeping the standard-library imports together and
above the `theseus` import.

In `assemble_context`, sort the unbudgeted branch:

```python
        if budget is None:
            lines = [event.to_json() for event in sorted(events, key=_chronological)]
        else:
            lines = self._fit_to_budget(events, budget)
```

Leave the `events = self.stimulus_log.read_all()[-self.window_size:]` line exactly as it is —
selection stays by arrival order.

In `_fit_to_budget`, keep each event beside its serialized line so the accumulated set can be
sorted, and replace the final `kept.reverse()` with the sort. The accumulation loop, the
per-event clamp and the at-least-one guarantee are unchanged:

```python
    def _fit_to_budget(self, events: list[StimulusEvent], budget: float) -> list[str]:
        """Take events newest-first until the budget runs out, then emit them in the order
        they happened.

        Always returns at least one event. A window that overshoots the budget is
        recoverable — the backend truncates, or errors, and the next pass is calibrated —
        whereas an empty one asks the model to decide with no stimulus at all. The
        per-event clamp is what makes that guarantee affordable: without it the one event
        we promise to emit could itself be a whole file.

        Selection is newest-first by *arrival* and emission is by `event_ts`; the two are
        deliberately different. Arrival is what "recent" means and what keeps selection
        cheap on a long log, but chronology is what the model has to read.
        """
        max_event_chars = self._max_event_chars(budget)
        kept: list[tuple[StimulusEvent, str]] = []
        used = 0.0

        for event in reversed(events):
            line = self._serialize(event, max_event_chars)
            cost = len(line) / self.chars_per_token
            if kept and used + cost > budget:
                break
            kept.append((event, line))
            used += cost

        kept.sort(key=lambda pair: _chronological(pair[0]))
        return [line for _, line in kept]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `poetry run pytest tests/test_context_assembler.py -q`
Expected: PASS — the new class plus every pre-existing test in the file.

- [ ] **Step 5: Run the offline suite**

Run: `poetry run pytest -q tests/ --ignore=tests/test_fact_retention.py --ignore=tests/e2e`
Expected: PASS, 0 failures. `tests/test_context_assembler.py` is the one to watch — the budget
maths must be untouched.

- [ ] **Step 6: Commit**

```bash
git add src/theseus/context_assembler.py tests/test_context_assembler.py
git commit -m "Order the assembled window by event_ts, not arrival"
```

---

### Task 3 (#29): per-origin high-water marks

**Files:**
- Create: `src/theseus/high_water.py`
- Test: `tests/test_high_water.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_high_water.py`:

```python
from __future__ import annotations

from theseus.high_water import HighWaterMarks
from theseus.stimulus_log import StimulusLog


def make_log(tmp_path) -> StimulusLog:
    return StimulusLog(path=tmp_path / "stimulus_log.jsonl")


def replicate(log: StimulusLog, origin: str, seq: int) -> None:
    log.append(
        actor="sensor", type="observation", content={}, origin=origin, seq=seq
    )


def test_marks_are_recovered_from_several_origins_interleaved(tmp_path):
    log = make_log(tmp_path)
    replicate(log, "kitchen-surrogate", 1)
    replicate(log, "android-01", 7)
    replicate(log, "kitchen-surrogate", 2)
    replicate(log, "android-01", 5)  # out of order on the wire; the mark is a maximum

    marks = HighWaterMarks(log)

    assert marks.high_water("kitchen-surrogate") == 2
    assert marks.high_water("android-01") == 7


def test_an_origin_never_seen_returns_none(tmp_path):
    """None is not 0. Seqs start at 1, so 0 would claim a seq had been seen; an origin
    whose first event is still in flight has no mark at all."""
    marks = HighWaterMarks(make_log(tmp_path))

    assert marks.high_water("kitchen-surrogate") is None


def test_a_restart_mid_stream_recovers_the_same_marks(tmp_path):
    path = tmp_path / "stimulus_log.jsonl"
    log = StimulusLog(path=path)
    replicate(log, "kitchen-surrogate", 4)
    before = HighWaterMarks(log).high_water("kitchen-surrogate")

    after = HighWaterMarks(StimulusLog(path=path)).high_water("kitchen-surrogate")

    assert before == after == 4


def test_local_host_events_do_not_pollute_a_surrogates_mark(tmp_path):
    """The host's own log entries carry its own origin and its own seq counter. If those
    leaked into a surrogate's mark, dedupe would discard real events as duplicates."""
    log = make_log(tmp_path)
    replicate(log, "kitchen-surrogate", 2)
    for _ in range(9):
        log.append(actor="george", type="exchange", content={})

    marks = HighWaterMarks(log)

    assert marks.high_water("kitchen-surrogate") == 2
    assert marks.high_water(log.origin) == 9


def test_advance_records_a_commit(tmp_path):
    marks = HighWaterMarks(make_log(tmp_path))

    marks.advance("kitchen-surrogate", 12)

    assert marks.high_water("kitchen-surrogate") == 12


def test_a_mark_never_moves_backwards(tmp_path):
    """A batch straddling the mark commits only its tail and a duplicate commits nothing,
    so the mark is a maximum rather than a last-write."""
    marks = HighWaterMarks(make_log(tmp_path))
    marks.advance("kitchen-surrogate", 12)

    marks.advance("kitchen-surrogate", 5)

    assert marks.high_water("kitchen-surrogate") == 12


def test_pre_envelope_lines_leave_no_mark(tmp_path):
    """A log written before the envelope existed has no seqs at all. Those lines are
    history, not deliveries, and must not invent a mark for the origin reading them."""
    path = tmp_path / "stimulus_log.jsonl"
    path.write_text(
        '{"id":"01ABCDEFGHJKMNPQRSTVWXYZ0","ts":"2026-01-01T12:00:00+00:00",'
        '"actor":"user","type":"chat_message","content":{}}\n',
        encoding="utf-8",
    )
    log = StimulusLog(path=path)

    marks = HighWaterMarks(log)

    assert marks.high_water(log.origin) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/test_high_water.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'theseus.high_water'`

- [ ] **Step 3: Write the implementation**

Create `src/theseus/high_water.py`:

```python
"""Per-origin high-water marks — the host's memory of what it has already accepted.

Duplicate suppression is the host's whole defence against double-appending a retried batch,
and it hinges on one number per origin: the highest `seq` committed so far. That number has
to survive a restart, because the surrogate's retry does not care that the host bounced.

The marks are **derived from the log**, not persisted beside it. The log is the bedrock; a
sidecar checkpoint can disagree with it after a crash, and disagreeing here means either
silently dropping real events or double-appending them — precisely the two failures this
state exists to prevent. The cost is one boot-time pass over the log, which the process
already makes elsewhere. If that pass ever becomes too slow, the fallback is a sidecar that
is only ever a *hint*: always reconciled forward against the log, never trusted past it.

Tracking marks is all this does. Deciding what to do with a batch that sits below, straddles
or jumps past a mark is the ingress's job.
"""

from __future__ import annotations

from theseus.stimulus_log import StimulusLog


class HighWaterMarks:
    """Highest committed `seq` per origin, recovered from the log at construction.

    Deliberately separate from `StimulusLog`'s own seq allocator, though both begin by
    scanning the same file. The allocator answers "what should I issue next, for myself";
    these marks answer "what have I already accepted, from everyone" — and the two diverge
    the moment an ingress advances a mark on commit without appending anything of its own.
    Folding them together would couple the writer's counter to the reader's dedupe state.
    """

    def __init__(self, log: StimulusLog) -> None:
        self._marks: dict[str, int] = {}
        for event in log.read_all():
            # Lines written before the envelope existed carry no seq. They are history,
            # not deliveries, and must not invent a mark.
            if event.seq is not None:
                self.advance(event.origin, event.seq)

    def high_water(self, origin: str) -> int | None:
        """Highest seq committed for `origin`, or `None` if nothing has ever arrived from it.

        `None` is distinct from `0`: seqs start at 1, so `0` would claim a seq had been
        seen, while an origin whose first event is still in flight has no mark at all.
        """
        return self._marks.get(origin)

    def advance(self, origin: str, seq: int) -> None:
        """Record that `seq` from `origin` is committed.

        Never moves a mark backwards. A batch straddling the mark commits only the events
        above it and a duplicate commits nothing, so the mark is a maximum rather than a
        last-write.

        `seq` is not validated here: a mark advances *on commit*, so the only path that can
        produce one has already been through `StimulusLog.append`, which rejects a seq below
        1. Validating untrusted input is the ingress's job, at the door.
        """
        current = self._marks.get(origin)
        if current is None or seq > current:
            self._marks[origin] = seq
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `poetry run pytest tests/test_high_water.py -q`
Expected: PASS.

- [ ] **Step 5: Run the offline suite**

Run: `poetry run pytest -q tests/ --ignore=tests/test_fact_retention.py --ignore=tests/e2e`
Expected: PASS, 0 failures.

- [ ] **Step 6: Commit**

```bash
git add src/theseus/high_water.py tests/test_high_water.py
git commit -m "Add per-origin high-water marks derived from the log"
```

---

## Acceptance

**#27**
- [x] Constructors for declared gap, inferred gap, and batch rejection — Task 1
- [x] Invalid reason or inverted seq range raises rather than serialising — Task 1
- [x] Round-trips through `StimulusEvent.to_json` / `from_json` — Task 1

**#28**
- [x] A batch arriving late with old `event_ts` interleaves correctly in the window, with the log file itself unchanged — Task 2
- [x] Events with identical `event_ts` come out in a stable, deterministic order — Task 2
- [x] Existing `tests/test_context_assembler.py` stays green — budget maths untouched — Task 2
- [x] Window with no `event_ts` skew is byte-identical to today's output — Task 2

**#29**
- [x] Correct marks recovered from a log containing several origins interleaved — Task 3
- [x] An origin never seen returns `None` (distinct from seq 0) — Task 3
- [x] Restart mid-stream recovers the same marks it had before — Task 3
- [x] Locally-produced host events don't pollute a surrogate's mark — Task 3

---

## Decisions escalated to the repo owner

Two questions surfaced in review that are protocol calls rather than implementation ones. Both are
**pinned by a test rather than fixed**, so changing the answer later shows up as a failing test
rather than a silent shift.

### 1. What the budget should drop when producers are skewed (#28)

`_fit_to_budget` drops the earliest-*arrived* while emitting by `event_ts`. Once a surrogate is
attached, a backfill can survive a cut that removes events which happened *after* it, leaving a
hole in the middle of a window that reads as continuous — the same invisible-hole failure #27's
gap markers exist to prevent, manufactured by the assembler.

Dropping by chronology instead removes the class entirely, at the cost of discarding a
just-delivered backlog first under budget pressure. That is a real trade, not an oversight, and
#28's text explicitly scopes budget behaviour out ("keep the budget behaviour exactly as it is —
only the output ordering changes").

`tests/test_context_assembler.py::test_a_truncated_window_can_have_a_hole_in_its_chronology` is the
reproduction case: four events, both policies' output named side by side in the comment. It fails
if the drop order changes.

### 2. What `link_down` describes (#27)

The spec's own example — "I wasn't observing from 16:02 to 16:40, the link was down" — reads as an
*observation* outage, in which no seqs were ever allocated and there is therefore no range to
report. But `declared_gap` requires a range, and the spec assigns buffered-then-dropped to
`retry_exhausted` and eviction to `storage_pressure`.

Read as: **a range that was buffered and then given up on.** The link being down never stops a
surrogate observing or allocating seqs, so a `link_down` gap always has a range. Chosen because
making a required field optional later is backward-compatible while the reverse is not — the more
reversible default under genuine ambiguity.

**If `link_down` was meant to describe an observation outage with no seqs, that is a different
schema and it should change before #32**, which is the first issue that emits one.

## Follow-ups this branch created

| Item | Where it belongs |
|---|---|
| Read-side validation of gap/rejection content arriving over the wire — must re-apply `replication_events`' rules rather than invent a second definition | #30 |
| `floor(origin) -> int` on `HighWaterMarks`, so the three dedupe branches compare without an `is None` at each site | #30 |
| A convention for what first contact above seq 1 means — gap needing an inferred marker, or a surrogate's first contact. Either answer writes a permanent event to the tape | #30 |
| `span_start` for a host-minted inferred gap: it is the `ts` of the last event previously committed from that origin, which neither `HighWaterMarks` nor the log index provides today | #30 |
| Origin canonicalisation at the door — case and whitespace currently fork a mark | #30 |
| Whether `advance` should be bound to the log (via `subscribe`, or a `commit(log, events)`) rather than to caller discipline | #30's design |
| Pacing a backfill drain, or capping any one origin's share of a context window | #30 |
| Streaming `iter_events()` on `StimulusLog` to replace `read_all()`. Measured at a million events: 63s and 1.4 GB. Note `ContextAssembler` calls `read_all()` every cognitive turn, so it bites there first. A reverse scan is **unsound** — `test_marks_are_recovered_from_several_origins_interleaved` proves it | its own issue |
| Naive-datetime handling codebase-wide. Contained at the parse boundary now; `StimulusEvent(ts=<naive>)` in memory is still naive | its own issue |
| `pytest-timeout`, so a hang-shaped regression fails red instead of stalling CI | its own issue |
