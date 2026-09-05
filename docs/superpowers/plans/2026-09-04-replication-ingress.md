# Replication Ingress, Replicator and the Abandon Rule

> **For agentic workers:** REQUIRED SUB-SKILL: Use steward:steward-local-sdd to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the host side of the upstream path (#30), the surrogate side's happy path (#31), and the retry/abandon rule that keeps one undeliverable batch from wedging the channel (#32).

**Architecture:** #30 is four pieces that stack: an all-or-nothing batch append on the log, a pure batch parser that owns the `4xx` class, a pure dedupe planner over the high-water marks, and the FastAPI endpoint that wires those three together behind a coalesced orient trigger. #31 and #32 are the mirror image on the surrogate, behind a `StimulusTransport` interface that is also the test seam. Nothing here reasons; the surrogate has no agency and the host applies rules, not judgement.

**Tech Stack:** Python 3.12, Poetry (in-project `.venv`), pytest, FastAPI + uvicorn (already core dependencies). Run everything with `poetry run`.

**Source issues:** GitHub #30, #31, #32.
**Spec:** `docs/superpowers/specs/2026-09-04-surrogate-replication-protocol.md` — § Upstream: stimulus replication (#30, #31), § Abandon rule (#32).
**Builds on:** #26, #27, #28, #29, all merged as `3c514d5`.

---

## What already exists, and must be used rather than rebuilt

| Piece | Where | What it gives this plan |
|---|---|---|
| `StimulusEvent` envelope | `stimulus_log.py` | `origin`, `seq`, `ts` (`event_ts`), `appended_ts`; `from_json` parses a wire line and coerces naive timestamps to aware |
| Replicated append | `StimulusLog.append(..., origin=, seq=)` | rejects a foreign origin without a seq, an own-origin append carrying a seq, an empty origin, and `seq < 1` — all before the lock, so a rejected append leaves nothing on disk |
| Gap vocabulary | `replication_events.py` | `GAP`, `BATCH_REJECTED`, `declared_gap`, `inferred_gap`, `batch_rejected`, `MAX_REASON_CHARS`. **Write-side validation only** — its module docstring says the ingress must re-apply the same rules to anything off the wire |
| Dedupe state | `HighWaterMarks` | `high_water(origin) -> int | None`, `advance(origin, seq)`; a snapshot per log, recovered at construction, and it rejects a malformed seq found on disk |
| Read-time ordering | `ContextAssembler` | already sorts the emitted window by `event_ts`, so a replicated backlog interleaves correctly without further work |
| Cycle gate | `OODACore.try_orient` (skip-on-contention), `OODACore.orient_and_wait` (wait-on-contention) | neither coalesces; see Task 2 |
| FastAPI + background-thread template | `web_chat_ui_observer.py` | the shape to follow: own `app`, endpoints return immediately, work happens on a background thread |

## Decisions locked in for this plan

Settled — implement as written, do not re-derive:

| Decision | Value |
|---|---|
| `append_many` signature | takes `Iterable[StimulusEvent]` — the events as parsed off the wire — and returns the newly-minted ones. The ingress already has parsed events; a second parallel spec type would be churn. |
| `append_many` re-mints `id` and `appended_ts` | yes, exactly as `append` does. Identity across nodes is `(origin, seq)`, never `id`. |
| Intra-batch id ordering | **fixed here.** A batch lands in one millisecond, and `older_batch` bisects a list of ids expecting them sorted. `append_many` mints a monotonic run rather than N independent random suffixes. This is the follow-up #29's review asked to be folded into `append_many` because it already holds the lock. |
| `append_many` is replication-shaped | every event must carry a foreign origin and a seq. A local batch append has no caller and is not built. |
| Batch parsing and dedupe are **pure** | Task 2 and Task 3 are functions over data, no I/O, no FastAPI. The endpoint is the only place that touches the log. This is what makes the `4xx` and dedupe cases testable without a server. |
| First contact above seq 1 | **is a gap.** An origin with no mark whose first batch starts at seq 5 means seqs 1–4 never arrived, so the host mints an inferred gap for `1..4`. Rationale: the alternative — treating first contact as "no gap" — silently discards the information that four events are missing, and the protocol's whole posture is that a hole the agent can read about beats a silence. Escalated to George in the plan's decisions section; reversible, since it only changes whether one marker event is written. |
| `span_start` for a host-minted inferred gap | the `appended_ts` of the last event this host committed from that origin, or the batch's own first `event_ts` when there is none. `HighWaterMarks` does not carry timestamps and is not being widened to; the ingress reads it off the log at mint time. |
| Coalesced trigger lives in the ingress | not on the core. It needs a worker thread with a lifecycle, which is ingress-shaped; and it must work in front of both an `Autocore` (where the callback is `wake`, idempotent) and an `OODACore` (where it is `orient_and_wait`, which would otherwise stack threads). |
| New public exports | **none until the series is composable.** `src/theseus/__init__.py` stays unchanged, same as the last branch. |

---

## File Structure

- **Modify: `src/theseus/stimulus_log.py`** (Task 1) — `append_many`, plus a `_id_run` helper minting a monotonic id sequence.
- **Create: `src/theseus/replication_batch.py`** (Task 2) — pure parsing and limit enforcement. Owns the `4xx` class: what makes a batch permanently unacceptable.
- **Create: `src/theseus/replication_dedupe.py`** (Task 3) — pure planner: given the marks and a parsed batch, what to append and what gap to mint. No I/O.
- **Create: `src/theseus/replication_ingress.py`** (Task 4) — the FastAPI app, the coalesced trigger, and the wiring.
- **Tests:** `tests/test_replication_batch.py`, `tests/test_replication_dedupe.py`, `tests/test_replication_ingress.py`; additions to `tests/test_stimulus_log.py`.

Tasks 5+ (#31 and #32) are planned once #30 has landed — the transport interface should be shaped by what the ingress actually turned out to need, not guessed at now. That is a deliberate departure from planning everything up front: the last branch's plan embedded module source that five fix rounds made stale, and this avoids repeating it at larger scale.

---

### Task 1: `StimulusLog.append_many` — one batch, one fsync, ordered ids

**Files:**
- Modify: `src/theseus/stimulus_log.py`
- Test: `tests/test_stimulus_log.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stimulus_log.py`, at the end of the "Origin and seq allocation" section:

```python
def _replicated(seq: int, message: str = "hi", origin: str = "kitchen-surrogate"):
    """One event shaped as it arrives off the wire: the producer's id and appended_ts are
    placeholders, because this log re-mints both."""
    return StimulusEvent(
        id="01PRODUCERSIDWILLBEDROPPED",
        ts=datetime(2026, 9, 4, 16, 0, tzinfo=timezone.utc) + timedelta(seconds=seq),
        actor="sensor",
        type="observation",
        content={"message": message},
        origin=origin,
        seq=seq,
    )


def test_append_many_writes_the_whole_batch(tmp_path):
    log = make_log(tmp_path)

    appended = log.append_many([_replicated(1), _replicated(2), _replicated(3)])

    assert [e.seq for e in appended] == [1, 2, 3]
    assert [e.seq for e in log.read_all()] == [1, 2, 3]
    assert {e.origin for e in log.read_all()} == {"kitchen-surrogate"}


def test_append_many_re_mints_id_and_appended_ts(tmp_path):
    """Identity across nodes is (origin, seq), never id. The producer's id is its own."""
    log = make_log(tmp_path)

    (appended,) = log.append_many([_replicated(1)])

    assert appended.id != "01PRODUCERSIDWILLBEDROPPED"
    assert appended.appended_ts > appended.ts
    assert appended.ts == datetime(2026, 9, 4, 16, 0, 1, tzinfo=timezone.utc)


def test_append_many_ids_are_strictly_increasing(tmp_path):
    """A whole batch lands inside one millisecond, and `older_batch` bisects a list of ids
    expecting it sorted. Independent random suffixes would collide with that; a batch
    mints a monotonic run instead."""
    log = make_log(tmp_path)

    appended = log.append_many([_replicated(n) for n in range(1, 51)])

    ids = [e.id for e in appended]
    assert ids == sorted(ids)
    assert len(set(ids)) == 50


def test_append_many_ids_sort_above_everything_already_on_the_log(tmp_path):
    log = make_log(tmp_path)
    earlier = log.append(actor="george", type="exchange", content={})

    appended = log.append_many([_replicated(1), _replicated(2)])

    assert earlier.id < appended[0].id


def test_append_many_is_one_fsync_for_the_whole_batch(tmp_path, monkeypatch):
    """All-or-nothing application: the spec forbids a partial state the surrogate would
    have to reason about. One open, one write, one fsync."""
    log = make_log(tmp_path)
    fsyncs = []
    real_fsync = os.fsync
    monkeypatch.setattr(
        os, "fsync", lambda fd: (fsyncs.append(fd), real_fsync(fd))[1]
    )

    log.append_many([_replicated(n) for n in range(1, 11)])

    assert len(fsyncs) == 1


def test_append_many_notifies_listeners_for_every_event(tmp_path):
    log = make_log(tmp_path)
    seen = []
    log.subscribe(seen.append)

    appended = log.append_many([_replicated(1), _replicated(2)])

    assert seen == appended


def test_append_many_notifies_after_the_whole_batch_is_durable(tmp_path):
    """A listener must never see event 1 while event 2 of the same batch could still be
    lost — the batch is the unit that is all-or-nothing."""
    log = make_log(tmp_path)
    on_disk = []
    log.subscribe(lambda event: on_disk.append(len(log.read_all())))

    log.append_many([_replicated(1), _replicated(2)])

    assert on_disk == [2, 2]


def test_append_many_rejects_a_batch_before_writing_any_of_it(tmp_path):
    """Validation ahead of the lock, exactly as `append` does it: a rejected batch leaves
    nothing on disk, so the surrogate has no partial state to reason about."""
    log = make_log(tmp_path)

    with pytest.raises(ValueError):
        log.append_many([_replicated(1), _replicated(2, origin=log.origin)])

    assert log.read_all() == []


def test_append_many_rejects_a_batch_spanning_two_origins(tmp_path):
    """The spec's batch is a contiguous seq range from exactly one origin. Two origins in
    one batch would make the all-or-nothing guarantee span two dedupe streams."""
    log = make_log(tmp_path)

    with pytest.raises(ValueError, match="exactly one origin"):
        log.append_many([_replicated(1), _replicated(1, origin="android-01")])


def test_append_many_of_nothing_is_a_no_op(tmp_path):
    log = make_log(tmp_path)

    assert log.append_many([]) == []
    assert log.read_all() == []


def test_append_many_does_not_disturb_the_local_counter(tmp_path):
    log = make_log(tmp_path)
    log.append(actor="george", type="exchange", content={})

    log.append_many([_replicated(90), _replicated(91)])

    assert log.append(actor="george", type="exchange", content={}).seq == 2
```

That file already imports `timedelta`, but **not** `os` — add `import os` to its standard-library
imports, above `import threading`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/test_stimulus_log.py -q`
Expected: FAIL — `AttributeError: 'StimulusLog' object has no attribute 'append_many'`

- [ ] **Step 3: Write the implementation**

In `src/theseus/stimulus_log.py`, add this helper immediately below `new_id`:

```python
def _id_run(ms: int, count: int) -> list[str]:
    """`count` ids for one instant, strictly increasing.

    `new_id` gives each id an independent random suffix, which is fine at human rates
    where the millisecond prefix does the ordering. A replicated batch is not that: fifty
    events land in one write, share a millisecond, and are then distinguished only by
    chance — while `older_batch` in the debug pagination bisects a list of ids and needs it
    sorted ascending.

    So a batch draws one random base and walks it. The base is drawn below `2**80 - count`
    so the walk cannot wrap, which would put the run out of order at exactly the moment it
    matters.
    """
    base = int.from_bytes(os.urandom(10), "big") % ((1 << 80) - count)
    return [_b32(ms, 10) + _b32(base + i, 16) for i in range(count)]
```

Add `append_many` immediately after `append`:

```python
    def append_many(self, events: Iterable[StimulusEvent]) -> list[StimulusEvent]:
        """Append a replicated batch, all of it or none of it.

        The spec forbids a partial state the surrogate would have to reason about, so the
        whole batch goes down under one `open`/`write`/`fsync`. That also makes a crash
        mid-batch harmless in a way a per-event loop is not: the only line that can tear is
        the last one, which `read_all` already recovers from, while an interior tear — the
        one case it raises on — becomes unreachable.

        Takes events as parsed off the wire and returns the ones actually written. `id` and
        `appended_ts` are re-minted here, exactly as in `append`: identity across nodes is
        `(origin, seq)`, never `id`. `ts` is the producer's own and is left alone.

        Replication-shaped: every event must carry a foreign origin and a seq, and the whole
        batch must come from one origin — the spec's batch is a contiguous seq range from
        exactly one producer, and a batch spanning two would make one all-or-nothing write
        span two dedupe streams. Validation happens before the lock, so a rejected batch
        leaves nothing behind.
        """
        events = list(events)
        if not events:
            return []

        origins = {event.origin for event in events}
        if len(origins) != 1:
            raise ValueError(
                f"a batch must come from exactly one origin, got {sorted(origins)}"
            )
        for event in events:
            self._check_replicated(event.origin, event.seq)

        with self._append_lock:
            appended_ts = datetime.now(timezone.utc)
            ids = _id_run(int(appended_ts.timestamp() * 1000), len(events))
            minted = [
                StimulusEvent(
                    id=id,
                    ts=event.ts,
                    actor=event.actor,
                    type=event.type,
                    content=event.content,
                    origin=event.origin,
                    seq=event.seq,
                    appended_ts=appended_ts,
                )
                for id, event in zip(ids, events)
            ]
            with open(self.path, "a", encoding="utf-8") as f:
                f.write("".join(event.to_json() + "\n" for event in minted))
                f.flush()
                os.fsync(f.fileno())

        # Outside the lock, and only once the whole batch is durable: a listener must never
        # see the first event of a batch while the last could still be lost.
        for event in minted:
            self._notify(event)
        return minted
```

`append`'s validation must now be shared rather than duplicated. In `append`, replace this exact
block:

```python
        origin = self.origin if origin is None else origin
        if not origin:
            raise ValueError("origin must be a non-empty name")
        if origin == self.origin:
            if seq is not None:
                raise ValueError(
                    f"{origin!r} is this log's own origin and it allocates those seqs "
                    f"itself. A replicated append must arrive under its producer's own "
                    f"origin name — two producers sharing one name break the per-origin "
                    f"monotonicity that duplicate suppression depends on."
                )
        elif seq is None:
            raise ValueError(
                f"a replicated append (origin {origin!r}) must carry the seq its "
                f"producer assigned"
            )
        elif seq < 1:
            raise ValueError(
                f"seq must be 1 or greater (got {seq!r}); starting at 1 keeps 0 below "
                f"every real seq, as a safe comparison floor for a reader tracking what "
                f"it has accepted"
            )
```

with:

```python
        origin = self.origin if origin is None else origin
        # A local append is the one shape that carries no seq, because the allocator below
        # supplies it. Everything else is somebody else's event and goes through the
        # replicated contract.
        if not (origin == self.origin and seq is None):
            self._check_replicated(origin, seq)
```

and add the extracted method immediately above `append`:

```python
    def _check_replicated(self, origin: str | None, seq: int | None) -> None:
        """The contract for an event this log did not produce: `origin` and `seq` are
        supplied together, the origin is somebody else's, and the seq is a real one.

        Shared by `append` and `append_many` so the two cannot drift — an ingress that
        could get a batch past a check a single append would have caught is exactly the
        hole this protocol's dedupe depends on not existing.
        """
        if not origin:
            raise ValueError("origin must be a non-empty name")
        if origin == self.origin:
            raise ValueError(
                f"{origin!r} is this log's own origin and it allocates those seqs "
                f"itself. A replicated append must arrive under its producer's own "
                f"origin name — two producers sharing one name break the per-origin "
                f"monotonicity that duplicate suppression depends on."
            )
        if seq is None:
            raise ValueError(
                f"a replicated append (origin {origin!r}) must carry the seq its "
                f"producer assigned"
            )
        if isinstance(seq, bool) or not isinstance(seq, int):
            raise ValueError(f"seq must be an integer (got {seq!r})")
        if seq < 1:
            raise ValueError(
                f"seq must be 1 or greater (got {seq!r}); starting at 1 keeps 0 below "
                f"every real seq, as a safe comparison floor for a reader tracking what "
                f"it has accepted"
            )
```

Note this adds the `isinstance` check to `append` too, which it did not have — the same class
of hole `HighWaterMarks._committed_seq` closed on the recovery side.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `poetry run pytest tests/test_stimulus_log.py -q`
Expected: PASS — the new tests plus every pre-existing one. The existing rejection tests
(`test_a_replicated_append_must_carry_a_seq`, `test_an_own_origin_append_may_not_carry_a_seq`,
`test_an_empty_origin_is_rejected`, `test_a_replicated_seq_below_one_is_rejected`) exercise the
extracted method and must stay green unmodified.

- [ ] **Step 5: Run the offline suite**

Run: `poetry run pytest -q tests/ --ignore=tests/test_fact_retention.py --ignore=tests/e2e`
Expected: 0 failures. (`tests/test_fact_retention.py` and `tests/e2e` need a live LLM endpoint;
they are excluded from every command in this plan. Never run the bare `tests/` directory — it
hangs.)

- [ ] **Step 6: Commit**

```bash
git add src/theseus/stimulus_log.py tests/test_stimulus_log.py
git commit -m "Add append_many: one batch, one fsync, ordered ids"
```

---

### Task 2 (#30): the batch parser — what makes a batch permanently unacceptable

**Files:**
- Create: `src/theseus/replication_batch.py`
- Test: `tests/test_replication_batch.py`

This task owns the protocol's `4xx` class and nothing else. It is a pure module: it turns a
request body into events, or refuses. No I/O, no FastAPI, no log, no marks.

**Why the `4xx` class exists, from the brief:** "The `4xx` class exists so one malformed or
oversized batch cannot wedge the channel forever." A `4xx` tells the surrogate *do not retry* —
it advances its cursor, emits `replication.batch_rejected`, and moves on. So anything answered
with `4xx` must genuinely be unfixable by retrying, and anything transient must **not** land here.

**One deliberate deviation from the brief's literal wording, ledgered.** The brief calls a batch
"a contiguous `seq` range from exactly one `origin`". This task enforces **strictly ascending**,
not contiguous. A surrogate that evicted seqs under storage pressure holds a buffer with real
holes in it; requiring contiguity would make that legal buffer unshippable and would answer it
with a `4xx`, which tells the surrogate to abandon data it still has. Ascending is what the
ordering and the dedupe straddle actually need. Recorded in the plan's decisions section.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_replication_batch.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from theseus.replication_batch import (
    DEFAULT_MAX_BATCH_BYTES,
    DEFAULT_MAX_BATCH_EVENTS,
    BatchRejected,
    parse_batch,
)


def line(n: int, origin: str = "kitchen-surrogate", **overrides) -> str:
    """The first parameter is `n`, not `seq`, so a test can override `seq` by keyword
    without colliding with the positional argument."""
    fields = {
        "id": f"01PRODUCERID{n:014d}",
        "ts": "2026-09-04T16:00:00+00:00",
        "actor": "sensor",
        "type": "observation",
        "content": {"n": n},
        "origin": origin,
        "seq": n,
    }
    fields.update(overrides)
    import json

    return json.dumps(fields)


def body(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def test_a_normal_batch_parses_into_events():
    events = parse_batch(body(line(1), line(2), line(3)))

    assert [e.seq for e in events] == [1, 2, 3]
    assert {e.origin for e in events} == {"kitchen-surrogate"}
    assert events[0].ts == datetime(2026, 9, 4, 16, 0, tzinfo=timezone.utc)


def test_a_trailing_newline_is_optional():
    assert len(parse_batch(line(1) + "\n" + line(2))) == 2


def test_blank_lines_are_ignored():
    """A JSONL producer that ends with a blank line has not malformed anything."""
    assert len(parse_batch(line(1) + "\n\n" + line(2) + "\n\n")) == 2


def test_an_empty_body_is_rejected():
    """Nothing to commit is not the same as a batch, and answering 2xx would advance the
    surrogate's cursor past events it never sent."""
    with pytest.raises(BatchRejected) as caught:
        parse_batch("   \n\n")

    assert caught.value.status == 400


def test_a_line_that_is_not_json_is_rejected():
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1), "{not json", line(3)))

    assert caught.value.status == 400
    assert "line 2" in caught.value.reason


def test_a_line_missing_a_required_field_is_rejected():
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1).replace('"actor": "sensor", ', "")))

    assert caught.value.status == 400


def test_a_line_without_a_seq_is_rejected():
    """Read-side validation: `replication_events` validates only what this node builds, so
    the ingress re-applies the rules to anything off the wire."""
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1, seq=None)))

    assert caught.value.status == 400
    assert "seq" in caught.value.reason


@pytest.mark.parametrize("seq", ["7", 0, -3, True, 1.5])
def test_a_seq_that_is_not_a_positive_integer_is_rejected(seq):
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1, seq=seq)))

    assert caught.value.status == 400


@pytest.mark.parametrize("origin", [None, "", "   ", 42])
def test_a_line_without_a_real_origin_is_rejected(origin):
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1, origin=origin)))

    assert caught.value.status == 400


def test_a_batch_spanning_two_origins_is_rejected():
    """The brief's batch comes from exactly one producer. Two would make one
    all-or-nothing write span two dedupe streams."""
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1), line(2, origin="android-01")))

    assert caught.value.status == 400
    assert "one origin" in caught.value.reason


def test_seqs_must_ascend():
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(2), line(1)))

    assert caught.value.status == 400
    assert "ascending" in caught.value.reason


def test_a_repeated_seq_in_one_batch_is_rejected():
    """Two events claiming one seq makes the high-water mark ambiguous about which was
    committed."""
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1), line(1)))

    assert caught.value.status == 400


def test_a_gap_inside_a_batch_is_accepted():
    """Deliberately *not* contiguous. A surrogate that evicted seqs under storage pressure
    holds a buffer with real holes; rejecting it would tell the surrogate to abandon data it
    still has, which is the opposite of what the abandon rule is for."""
    events = parse_batch(body(line(1), line(2), line(90)))

    assert [e.seq for e in events] == [1, 2, 90]


def test_too_many_events_is_rejected():
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(*[line(n) for n in range(1, 5)]), max_events=3)

    assert caught.value.status == 413
    assert "3" in caught.value.reason


def test_too_many_bytes_is_rejected():
    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1), line(2)), max_bytes=50)

    assert caught.value.status == 413


def test_the_byte_limit_is_measured_on_the_encoded_body():
    """A limit that counted characters would let a multi-byte payload through at several
    times the size the host meant to accept."""
    fat = parse_batch  # alias for readability
    with pytest.raises(BatchRejected):
        fat(body(line(1, content={"m": "é" * 200})), max_bytes=300)


def test_limits_default_to_the_module_constants():
    assert DEFAULT_MAX_BATCH_EVENTS > 0
    assert DEFAULT_MAX_BATCH_BYTES > 0
    assert len(parse_batch(body(line(1)))) == 1


def test_a_rejection_carries_a_reason_short_enough_to_record():
    """The surrogate copies this into a `replication.batch_rejected` event, which is bounded
    at `MAX_REASON_CHARS` and lands on a permanent tape."""
    from theseus.replication_events import MAX_REASON_CHARS

    with pytest.raises(BatchRejected) as caught:
        parse_batch(body(line(1), "{not json"))

    assert 0 < len(caught.value.reason) <= MAX_REASON_CHARS


def test_bytes_and_str_bodies_behave_the_same():
    text = body(line(1), line(2))

    assert [e.seq for e in parse_batch(text)] == [
        e.seq for e in parse_batch(text.encode("utf-8"))
    ]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/test_replication_batch.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'theseus.replication_batch'`

- [ ] **Step 3: Write the implementation**

Create `src/theseus/replication_batch.py`:

```python
"""Turning a replication request body into events, or refusing it.

This module owns the protocol's `4xx` class, and that class has a specific meaning: a `4xx`
tells the surrogate *do not retry*. It advances its cursor past the batch, emits a
`replication.batch_rejected` marker so the hole is visible on its tape, and moves on. So
everything answered here must be genuinely unfixable by sending it again — malformed,
oversized, or the wrong shape — and anything merely transient must never reach this module.

The class exists so one poison batch cannot wedge the channel forever. That is the whole of
its job: without it, a surrogate retries an unacceptable batch until its budget runs out
while everything behind it waits.

Validation here is the read-side counterpart to `replication_events`, whose constructors
validate only what *this* node builds. A remote surrogate goes through none of those, so the
rules are re-applied to anything arriving over the wire rather than a second definition of
well-formed being invented at the endpoint.

Pure: no I/O, no log, no marks, no FastAPI. The endpoint is the only thing that touches those.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from theseus.replication_events import MAX_REASON_CHARS
from theseus.stimulus_log import StimulusEvent

# A batch is bounded twice, because the two limits fail differently: a count keeps one
# commit from stalling the cognitive loop, and a byte size keeps a single event with a
# hundred-megabyte payload from doing the same with one line. Both are configurable; these
# are the defaults a LAN surrogate can rely on.
DEFAULT_MAX_BATCH_EVENTS = 500
DEFAULT_MAX_BATCH_BYTES = 4 * 1024 * 1024


class BatchRejected(Exception):
    """A batch that will never be acceptable, however many times it is sent.

    Carries the status the endpoint should answer with and the reason the surrogate will
    copy onto its own tape — so the reason is written for that reader, and bounded to the
    length that tape will keep.
    """

    def __init__(self, status: int, reason: str) -> None:
        self.status = status
        self.reason = reason[:MAX_REASON_CHARS]
        super().__init__(f"{status}: {self.reason}")


def parse_batch(
    body: str | bytes,
    *,
    max_events: int = DEFAULT_MAX_BATCH_EVENTS,
    max_bytes: int = DEFAULT_MAX_BATCH_BYTES,
) -> list[StimulusEvent]:
    """Parse a JSONL replication body into events, or raise `BatchRejected`.

    Accepts one event per line, ascending by `seq`, all from one origin. Blank lines are
    ignored — a producer that ends with a newline has malformed nothing.

    Seqs must **ascend**, but need not be contiguous. The brief describes a batch as a
    contiguous range, and a well-behaved surrogate sends one; but a surrogate that evicted
    events under storage pressure holds a buffer with real holes in it, and answering that
    with a `4xx` would tell it to abandon data it still has — the opposite of what the
    abandon rule is for. Ascending is what ordering and the dedupe straddle actually need.
    """
    raw = body if isinstance(body, bytes) else body.encode("utf-8")
    if len(raw) > max_bytes:
        raise BatchRejected(
            413, f"batch is {len(raw)} bytes, over the {max_bytes} byte limit"
        )

    lines = [
        (number, stripped)
        for number, text in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1)
        if (stripped := text.strip())
    ]
    if not lines:
        raise BatchRejected(400, "batch is empty")
    if len(lines) > max_events:
        raise BatchRejected(
            413,
            f"batch carries {len(lines)} events, over the {max_events} event limit",
        )

    events = [_parse_line(number, text) for number, text in lines]
    _check_one_origin(events)
    _check_ascending(events)
    return events


def _parse_line(number: int, text: str) -> StimulusEvent:
    """Validate the wire values *before* building an event from them.

    This ordering is the point of the module, not an accident. `StimulusEvent.from_json`
    is generous by design — it coerces an absent or empty `origin` to the reading log's own
    origin, because that is the right reading for a line this node wrote before the envelope
    existed. Applied to an untrusted body it would turn a surrogate's missing origin into
    the *host's* own name, which is precisely the collision that puts two numbering
    authorities on one origin. So the raw values are checked first, and only then parsed.
    """
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BatchRejected(400, f"line {number} is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise BatchRejected(400, f"line {number} is not a JSON object")

    _check_origin(number, raw.get("origin"))
    _check_seq(number, raw.get("seq"))

    try:
        return StimulusEvent.from_json(text)
    except (KeyError, ValueError, TypeError) as exc:
        raise BatchRejected(400, f"line {number} is not a usable event: {exc}") from exc


def _check_origin(number: int, origin: Any) -> None:
    if not isinstance(origin, str) or not origin.strip():
        raise BatchRejected(
            400, f"line {number} has no usable origin (got {origin!r})"
        )


def _check_seq(number: int, seq: Any) -> None:
    """`bool` is an `int` in Python, and `true` on the wire would sail past a `< 1` guard
    and then compare as 1 against a high-water mark."""
    if seq is None:
        raise BatchRejected(400, f"line {number} carries no seq")
    if isinstance(seq, bool) or not isinstance(seq, int):
        raise BatchRejected(400, f"line {number} has a non-integer seq ({seq!r})")
    if seq < 1:
        raise BatchRejected(400, f"line {number} has seq {seq}; seqs start at 1")


def _check_one_origin(events: Iterable[StimulusEvent]) -> None:
    origins = sorted({event.origin for event in events})
    if len(origins) != 1:
        raise BatchRejected(
            400, f"a batch must come from one origin, got {origins}"
        )


def _check_ascending(events: list[StimulusEvent]) -> None:
    for earlier, later in zip(events, events[1:]):
        if later.seq <= earlier.seq:
            raise BatchRejected(
                400,
                f"seqs must be ascending; {later.seq} follows {earlier.seq}",
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `poetry run pytest tests/test_replication_batch.py -q`
Expected: PASS.

- [ ] **Step 5: Run the offline suite**

Run: `poetry run pytest -q tests/ --ignore=tests/test_fact_retention.py --ignore=tests/e2e`
Expected: 0 failures.

- [ ] **Step 6: Commit**

```bash
git add src/theseus/replication_batch.py tests/test_replication_batch.py
git commit -m "Add the replication batch parser and its 4xx class"
```

---

### Tasks 3–4 (#30) and Tasks 5+ (#31, #32)

Written as each predecessor lands. The dedupe planner and the endpoint depend on the shapes
Tasks 1 and 2 settle, and this plan deliberately does not embed source for modules that review
rounds will change — the previous branch's plan had to be marked superseded for exactly that
reason.

---

### Task 2 fix round (#30): close the two poison-batch holes, and settle where the one-origin rule lives

Two reviewers reached opposite conclusions about `append_many`'s one-origin check. This task
resolves that, and closes the review findings against `replication_batch.py`.

#### The origin rule: where it lives, and in what form

Task 1's reviewer said remove it from `append_many` (it blocks the legitimate mixed write, and
`parse_batch` already guards the wire). Task 2's reviewer said keep it (one all-or-nothing fsync
must not span two dedupe streams, and a future non-wire caller — the #31 surrogate side, a replay
tool, a test — routes around `parse_batch` entirely). The fix round removed it outright, which
takes the first reviewer's side without answering the second's objection.

The spec settles it. Line 161: *"Batch: a contiguous `seq` range from exactly one `origin`."* That
is a **wire** rule, and `parse_batch` owns it — including "contiguous", which the parser has
deliberately relaxed to "ascending" for a reason it documents. But `append_many` is not handed the
wire batch; it is handed the wire batch **plus the host's own gap marker**. So the storage layer's
invariant is that same rule as it looks after the host adds its marker:

> **At most one origin other than this log's own, per batch.**

That permits exactly the case both reviewers agreed must work, keeps a two-line tripwire for
callers that never touch the wire, and restates none of the ordering rules that genuinely belong at
the door. Two foreign origins in one `append_many` is a caller bug, not a wire condition, and the
storage layer naming both origins is a perfectly good explanation of a caller bug.

#### The two Criticals, measured

Both were reproduced against `a6e6332` before this task was written.

**1. `RecursionError` escapes `parse_batch`.** Only `json.JSONDecodeError` is caught, and
`json.loads` recurses:

```
depth   1000 (   2185 bytes, 0.05% of limit): accepted
depth   5000 (  10185 bytes, 0.24% of limit): accepted
depth  20000 (  40185 bytes, 0.96% of limit): !!! UNHANDLED RecursionError
depth 100000 ( 200185 bytes, 4.77% of limit): !!! UNHANDLED RecursionError
```

40 KB — one percent of the byte limit — escapes as an unhandled exception. Task 4 would answer
`500`, the surrogate reads `5xx` as transient, and it retries that batch forever while everything
behind it waits. That is precisely the poison batch the `4xx` class exists to prevent.

**2. `splitlines()` splits well-formed lines.** `StimulusEvent.to_json` uses
`ensure_ascii=False`, so a Theseus surrogate emits U+2028 / U+2029 / U+0085 raw inside string
values, and `str.splitlines()` breaks on all three. Verified end to end, building the line with
Theseus's own serialiser:

```
to_json emits U+2028 raw: True
U+2028 in content: BatchRejected 400: line 1 is not JSON: Unterminated string starting at ...
U+0085 in content: BatchRejected 400: line 1 is not JSON: Unterminated string starting at ...
```

A `400` means *do not retry*. So any transcript containing a line separator — pasted text, a
Windows-authored file, an LLM's own output — makes the host permanently discard a well-formed
batch. Silent, permanent, content-triggered data loss. `split("\n")` is the only correct split for
JSONL, because `\n` is the only separator the format defines.

#### The Importants, also measured

```
seq = 10**100:  ACCEPTED     -> poisons that origin's high-water mark forever
content = null: ACCEPTED     -> content is typed dict[str, Any]
type = null:    ACCEPTED
id = {}:        ACCEPTED
naive ts:       ACCEPTED     -> coerced to the *host's local zone* (EDT), not UTC
invalid utf-8:  ACCEPTED     -> the replacement character written to the tape
```

The naive-`ts` case is worse than "silently coerced": `_aware` attaches the **host's** zone, which
is the right reading for an old local line and the wrong one for a surrogate in another zone — it
shifts the event by hours in the Assembler's chronological sort. The wire format is fully
specified, so a `ts` without an offset is a producer bug and is unfixable by retrying.

`errors="replace"` is the same shape of mistake: a body that is not UTF-8 will never become UTF-8
by being sent again, and substituting the replacement character writes corruption onto an
append-only tape that nothing later can distinguish from content.

#### Changes

**a. `src/theseus/replication_events.py` — promote `_clean_reason` to `clean_reason`.**

Rename the function and its call sites within that module. Nothing else changes. It is promoted
because a second module now needs the identical rule: `BatchRejected.reason` is copied verbatim
onto the surrogate's tape by a `replication.batch_rejected` marker, so it is the same value class,
and it must be bounded the same way — marked where it was cut, so a reader can tell "the host said
exactly this" from "the host said this and more".

**b. `src/theseus/stimulus_log.py` — restore the narrowed invariant in `append_many`.**

Keep everything `9a7fb11` did (the mixed batch, `_next_local_seq`, `_write_durably`). Add the
foreign-origin check **after** the per-event validation loop and before the lock:

```python
        # At most one origin other than this log's own. The spec's wire batch is one
        # origin's range, and `parse_batch` enforces that at the door; what reaches here is
        # that batch plus, sometimes, this host's own gap marker explaining a hole in it. So
        # the rule the *storage* layer holds is the wire rule as it looks after the marker
        # is added. Two foreign origins in one write is a caller that never went through the
        # door — the #31 surrogate side, a replay tool, a test — putting one all-or-nothing
        # fsync across two dedupe streams, which is a bug in that caller, not a wire
        # condition. It costs a set comprehension over a list already in hand.
        #
        # This runs after the loop above, so every origin here is a non-empty string and the
        # sort cannot raise TypeError on a mixed None.
        foreign = {event.origin for event in events if event.origin != self.origin}
        if len(foreign) > 1:
            raise ValueError(
                f"a batch may carry at most one origin beside this log's own "
                f"({self.origin!r}), got {sorted(foreign)}"
            )
```

**c. `src/theseus/replication_batch.py` — the parser.** Replace the file's body from the imports
down with the source below. The module docstring stays as it is.

```python
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from theseus.replication_events import clean_reason
from theseus.stimulus_log import StimulusEvent

# A batch is bounded twice, because the two limits fail differently: a count keeps one
# commit from stalling the cognitive loop, and a byte size keeps a single event with a
# hundred-megabyte payload from doing the same with one line. Both are configurable; these
# are the defaults a LAN surrogate can rely on.
DEFAULT_MAX_BATCH_EVENTS = 500
DEFAULT_MAX_BATCH_BYTES = 4 * 1024 * 1024

# A seq is a counter, and a counter that arrives as 10**100 is not a counter — it is a mark
# that no real event can ever exceed, so accepting it discards that origin's whole future.
# int64 is the ceiling every store, wire format and database this could pass through shares,
# and it is beyond any producer that increments once per event.
MAX_SEQ = 2**63 - 1

# The envelope fields that must be present, non-empty strings on the wire. `origin` and
# `seq` are checked separately, with reasons of their own.
_REQUIRED_STRINGS = ("id", "actor", "type")


class BatchRejected(Exception):
    """A batch that will never be acceptable, however many times it is sent.

    Carries the status the endpoint should answer with and the reason the surrogate will
    copy onto its own tape — so the reason is written for that reader, and bounded by the
    same rule `replication_events` bounds a declared reason with, marked where it was cut.

    The status is checked because this exception *is* the protocol's "do not retry" signal:
    raising it with a `5xx` would tell the surrogate to abandon a batch it should have
    retried, and there is no later layer that could catch the mistake.
    """

    def __init__(self, status: int, reason: str) -> None:
        if (
            not isinstance(status, int)
            or isinstance(status, bool)
            or not 400 <= status < 500
        ):
            raise ValueError(
                f"BatchRejected is the 4xx class; {status!r} is not a 4xx status"
            )
        self.status = status
        self.reason = clean_reason(reason)
        super().__init__(f"{status}: {self.reason}")


def parse_batch(
    body: str | bytes,
    *,
    max_events: int = DEFAULT_MAX_BATCH_EVENTS,
    max_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    host_origin: str | None = None,
) -> list[StimulusEvent]:
    """Parse a JSONL replication body into events, or raise `BatchRejected`.

    Accepts one event per line, ascending by `seq`, all from one origin. Blank lines are
    ignored — a producer that ends with a newline has malformed nothing.

    Seqs must **ascend**, but need not be contiguous. The brief describes a batch as a
    contiguous range, and a well-behaved surrogate sends one; but a surrogate that evicted
    events under storage pressure holds a buffer with real holes in it, and answering that
    with a `4xx` would tell it to abandon data it still has — the opposite of what the
    abandon rule is for. Ascending is what ordering and the dedupe straddle actually need.

    `host_origin`, when given, is this log's own origin name, and a batch claiming it is
    rejected here. A surrogate misconfigured with the host's name puts two numbering
    authorities on one seq stream, which is unfixable by retrying and so belongs in the
    `4xx` class — without this it would surface further down as an unhandled `ValueError`
    out of `append_many`, which the endpoint would answer as a `5xx` and the surrogate would
    retry forever.
    """
    raw = body if isinstance(body, bytes) else body.encode("utf-8")
    if len(raw) > max_bytes:
        raise BatchRejected(
            413, f"batch is {len(raw)} bytes, over the {max_bytes} byte limit"
        )

    # Strict, not `errors="replace"`. A body that is not UTF-8 will not become UTF-8 by
    # being sent again, and substituting the replacement character writes corruption onto an
    # append-only tape that nothing downstream can tell apart from content the producer
    # meant.
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BatchRejected(
            400, f"batch is not valid UTF-8 (byte {exc.start}: {exc.reason})"
        ) from exc

    # `split("\n")`, never `splitlines()`. JSONL defines exactly one separator, and
    # `splitlines` invents six more: `StimulusEvent.to_json` serialises with
    # `ensure_ascii=False`, so a Theseus surrogate puts U+2028, U+2029 and U+0085 on the
    # wire raw inside string values, and splitting on those tears a well-formed line into
    # two invalid halves. The answer would be a `400` — do not retry — so any transcript
    # containing a line separator would be discarded permanently, silently, and only for
    # certain content.
    lines = [
        (number, stripped)
        for number, line in enumerate(text.split("\n"), 1)
        if (stripped := line.strip())
    ]
    if not lines:
        raise BatchRejected(400, "batch is empty")
    if len(lines) > max_events:
        raise BatchRejected(
            413,
            f"batch carries {len(lines)} events, over the {max_events} event limit",
        )

    events = [_parse_line(number, text) for number, text in lines]
    _check_one_origin(events, host_origin)
    _check_ascending(events)
    return events


def _parse_line(number: int, text: str) -> StimulusEvent:
    """Validate the wire values *before* building an event from them.

    This ordering is the point of the module, not an accident. `StimulusEvent.from_json`
    is generous by design — it coerces an absent or empty `origin` to the reading log's own
    origin, because that is the right reading for a line this node wrote before the envelope
    existed. Applied to an untrusted body it would turn a surrogate's missing origin into
    the *host's* own name, which is precisely the collision that puts two numbering
    authorities on one origin. So the raw values are checked first, and only then parsed.
    """
    raw = _loads(number, text)
    if not isinstance(raw, dict):
        raise BatchRejected(400, f"line {number} is not a JSON object")

    _check_origin(number, raw.get("origin"))
    _check_seq(number, raw.get("seq"))
    _check_envelope(number, raw)

    try:
        return StimulusEvent.from_json(text)
    except (KeyError, ValueError, TypeError, RecursionError) as exc:
        raise BatchRejected(400, f"line {number} is not a usable event: {exc}") from exc


def _loads(number: int, text: str) -> Any:
    """`json.loads` with its recursion made part of the `4xx` class.

    `json.loads` recurses once per level of nesting, so a deeply nested body raises
    `RecursionError` — which is not a `ValueError` and so is not a `JSONDecodeError`.
    Measured against the unguarded parser: 20,000 levels is 40 KB, one percent of the byte
    limit, and escaped as an unhandled exception. The endpoint would answer `500`, the
    surrogate would read that as transient, and it would resend that batch forever while
    every event behind it waited. A batch too deep to parse is as permanently unacceptable
    as one that is not JSON at all, and belongs in the same class.

    The stack has already unwound to this frame by the time the handler runs, so building
    the rejection here is safe.
    """
    try:
        return json.loads(text)
    except RecursionError as exc:
        raise BatchRejected(400, f"line {number} nests too deeply to parse") from exc
    except ValueError as exc:  # JSONDecodeError is a ValueError
        raise BatchRejected(400, f"line {number} is not JSON: {exc}") from exc


def _check_origin(number: int, origin: Any) -> None:
    if not isinstance(origin, str) or not origin.strip():
        raise BatchRejected(
            400, f"line {number} has no usable origin (got {origin!r})"
        )


def _check_seq(number: int, seq: Any) -> None:
    """`bool` is an `int` in Python, and `true` on the wire would sail past a `< 1` guard
    and then compare as 1 against a high-water mark."""
    if seq is None:
        raise BatchRejected(400, f"line {number} carries no seq")
    if isinstance(seq, bool) or not isinstance(seq, int):
        raise BatchRejected(400, f"line {number} has a non-integer seq ({seq!r})")
    if seq < 1:
        raise BatchRejected(400, f"line {number} has seq {seq}; seqs start at 1")
    if seq > MAX_SEQ:
        raise BatchRejected(
            400,
            f"line {number} has seq {seq}, above the {MAX_SEQ} ceiling; a mark that high "
            f"would discard everything that origin ever sends afterwards",
        )


def _check_envelope(number: int, raw: dict[str, Any]) -> None:
    """The rest of the envelope, which `from_json` would take on trust.

    `from_json` indexes these straight out of the parsed dict, so `"type": null` or
    `"id": {}` becomes an event with a `None` type or a dict id, appended to the tape and
    read back by everything downstream. None of that is fixable by resending, so it is a
    `4xx` and not something to discover later in the Assembler.
    """
    for field in _REQUIRED_STRINGS:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise BatchRejected(
                400, f"line {number} has no usable {field} (got {value!r})"
            )
    content = raw.get("content")
    if not isinstance(content, dict):
        raise BatchRejected(
            400, f"line {number} has a non-object content (got {content!r})"
        )
    _check_ts(number, raw.get("ts"))


def _check_ts(number: int, ts: Any) -> None:
    """A producer timestamp, with the offset the wire format requires.

    A naive `ts` is not merely imprecise here. `stimulus_log._aware` attaches the *host's*
    zone to it, which is the right reading for an old local line and the wrong one for a
    surrogate in another zone — the event lands hours from where it belongs in the
    Assembler's chronological sort, silently. The wire format is fully specified, so a `ts`
    without an offset is a producer bug and no amount of resending fixes it.
    """
    if not isinstance(ts, str):
        raise BatchRejected(400, f"line {number} has no usable ts (got {ts!r})")
    try:
        parsed = datetime.fromisoformat(ts)
    except ValueError as exc:
        raise BatchRejected(400, f"line {number} has an unparseable ts ({ts!r})") from exc
    if parsed.tzinfo is None:
        raise BatchRejected(
            400,
            f"line {number} has a ts with no UTC offset ({ts!r}); a naive timestamp would "
            f"be read in the host's zone, not the producer's",
        )


def _check_one_origin(events: list[StimulusEvent], host_origin: str | None) -> None:
    origins = sorted({event.origin for event in events})
    if len(origins) != 1:
        raise BatchRejected(
            400, f"a batch must come from one origin, got {origins}"
        )
    if host_origin is not None and origins[0] == host_origin:
        raise BatchRejected(
            400,
            f"batch claims this host's own origin ({host_origin!r}); a surrogate must send "
            f"under its own name, or two producers number one seq stream",
        )


def _check_ascending(events: list[StimulusEvent]) -> None:
    for earlier, later in zip(events, events[1:]):
        if later.seq <= earlier.seq:
            raise BatchRejected(
                400,
                f"seqs must be ascending; {later.seq} follows {earlier.seq}",
            )
```

#### Tests

**In `tests/test_replication_batch.py`:** hoist the `import json` out of the `line` helper up to
the module's imports. Add `from theseus.replication_events import MAX_REASON_CHARS` and
`from theseus.stimulus_log import StimulusEvent`. Keep every existing test. Add:

```python
def test_a_line_deep_enough_to_exhaust_the_stack_is_a_4xx_not_a_crash():
    """20,000 levels is 40 KB — one percent of the byte limit — and `json.loads` recurses
    once per level. `RecursionError` is not a `ValueError`, so it is not a `JSONDecodeError`
    and the obvious handler misses it. Escaping here means the endpoint answers 500, the
    surrogate reads that as transient, and it resends this batch forever."""
    depth = 20_000
    body = line(1).replace('"content": {"n": 1}', '"content": ' + "[" * depth + "]" * depth)
    assert len(body) < DEFAULT_MAX_BATCH_BYTES

    with pytest.raises(BatchRejected) as excinfo:
        parse_batch(body)

    assert excinfo.value.status == 400
    assert "nests too deeply" in excinfo.value.reason


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
def test_a_line_separator_inside_a_value_does_not_split_the_line(separator):
    """Not hypothetical: the line is built by the same `to_json` a Theseus surrogate runs.
    It serialises with `ensure_ascii=False`, so these three reach the wire raw, and
    `str.splitlines()` breaks on all three where `split("\\n")` does not. Tearing one line
    into two invalid halves answers 400 — do not retry — so a transcript containing a line
    separator would be discarded permanently, silently, and only for that content.

    The body must come from `to_json`, **not** from this module's `line()` helper:
    `json.dumps` defaults to `ensure_ascii=True` and would escape the separator, leaving
    nothing for `splitlines` to split — and the test would then pass against its own
    reverted fix, which is worse than not having it."""
    message = f"before{separator}after"
    event = StimulusEvent(
        id="01PRODUCERID00000000000001",
        ts=datetime(2026, 9, 4, 16, 0, tzinfo=timezone.utc),
        actor="george",
        type="exchange",
        content={"message": message},
        origin="kitchen-surrogate",
        seq=1,
    )
    wire = event.to_json()
    assert separator in wire  # the precondition: raw on the wire, not escaped
    assert len(wire.splitlines()) == 2  # and `splitlines` would indeed tear it

    events = parse_batch(wire + "\n")

    assert len(events) == 1
    assert events[0].content == {"message": message}


def test_a_body_that_is_not_utf8_is_rejected_rather_than_repaired():
    """`errors="replace"` would write the replacement character onto an append-only tape,
    where nothing downstream can tell it from content the producer meant. Bytes that are not
    UTF-8 will not become UTF-8 by being resent."""
    body = line(1).encode("utf-8").replace(b"sensor", b"sen\xffor")

    with pytest.raises(BatchRejected) as excinfo:
        parse_batch(body)

    assert excinfo.value.status == 400
    assert "UTF-8" in excinfo.value.reason


def test_a_seq_above_the_ceiling_is_rejected():
    """A mark of 10**100 is not a counter; accepting it discards that origin's entire
    future, because nothing it ever sends again clears the high-water mark."""
    with pytest.raises(BatchRejected, match="ceiling"):
        parse_batch(line(1, seq=10**100))


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"content": None}, "non-object content"),
        ({"content": "just a string"}, "non-object content"),
        ({"type": None}, "no usable type"),
        ({"type": ""}, "no usable type"),
        ({"id": {}}, "no usable id"),
        ({"actor": 7}, "no usable actor"),
        ({"ts": "not a timestamp"}, "unparseable ts"),
        ({"ts": None}, "no usable ts"),
    ],
)
def test_the_rest_of_the_envelope_is_checked_too(overrides, expected):
    """`from_json` indexes these straight out of the parsed dict, so without a check they
    reach the tape as a `None` type or a dict id."""
    with pytest.raises(BatchRejected, match=expected):
        parse_batch(line(1, **overrides))


def test_a_naive_ts_is_rejected_rather_than_read_in_the_hosts_zone():
    """`_aware` attaches the *host's* zone to a naive timestamp — right for an old local
    line, wrong for a surrogate elsewhere, and it moves the event hours from where it
    belongs in the Assembler's sort."""
    with pytest.raises(BatchRejected, match="no UTC offset"):
        parse_batch(line(1, ts="2026-09-04T16:00:00"))


def test_a_batch_claiming_the_hosts_own_origin_is_a_4xx():
    """Without this it surfaces as a ValueError out of `append_many`, which the endpoint
    answers as a 5xx and the surrogate retries forever — a misconfiguration no retry fixes."""
    with pytest.raises(BatchRejected) as excinfo:
        parse_batch(line(1, origin="local"), host_origin="local")

    assert excinfo.value.status == 400
    assert "own origin" in excinfo.value.reason


def test_host_origin_is_not_checked_unless_it_is_given():
    assert len(parse_batch(line(1, origin="local"))) == 1


def test_batch_rejected_refuses_a_status_outside_the_4xx_class():
    """This exception *is* the do-not-retry signal. Raising it with a 5xx would tell the
    surrogate to abandon a batch it should have retried, and nothing downstream could
    catch it."""
    for status in (200, 500, 503):
        with pytest.raises(ValueError, match="4xx"):
            BatchRejected(status, "whatever")


def test_batch_rejected_marks_a_truncated_reason():
    """The reason is copied verbatim onto the surrogate's tape. A reader that cannot tell
    "the host said exactly this" from "the host said this and more" is being misled."""
    rejected = BatchRejected(400, "x" * 5000)

    assert len(rejected.reason) == MAX_REASON_CHARS
    assert rejected.reason.endswith("…")


def test_batch_rejected_refuses_an_empty_reason():
    with pytest.raises(ValueError):
        BatchRejected(400, "   ")
```

**In `tests/test_stimulus_log.py`:** add `import io` to the imports, and these two tests.

```python
def test_append_many_rejects_two_foreign_origins(tmp_path):
    """One all-or-nothing fsync must not span two dedupe streams. `parse_batch` enforces
    the wire's one-origin rule at the door, but a caller that never goes through the door —
    the surrogate side, a replay tool, a test — would put both streams in one write."""
    log = make_log(tmp_path)

    with pytest.raises(ValueError, match="at most one origin"):
        log.append_many([_replicated(1), _replicated(1, origin="android-01")])

    assert log.read_all() == []


def test_append_many_rolls_back_a_write_that_dies_part_way(tmp_path, monkeypatch):
    """The failure this guards is not a crash — a crash can only tear the final line, which
    `read_all` drops. It is an exception mid-write, ENOSPC being the realistic one: the
    prefix is committed, the file ends mid-line, and the *next* successful append
    concatenates onto that stump and turns it into an interior corrupt record. `read_all`
    raises on those by design and forever, and `HighWaterMarks` derives itself by reading
    the whole log — so the agent never boots again."""
    log = make_log(tmp_path)
    log.append(actor="george", type="exchange", content={"message": "first"})
    committed = log.path.stat().st_size

    real_open = open

    class TornWriter(io.TextIOWrapper):
        """Commits a prefix, then dies the way ENOSPC does."""

        def write(self, s):
            super().write(s[: len(s) // 3])
            super().flush()
            raise OSError(28, "No space left on device")

    def failing_open(path, mode="r", *args, **kwargs):
        if "a" in mode:
            return TornWriter(real_open(path, "ab"), encoding="utf-8")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_open)
    with pytest.raises(OSError):
        log.append_many([_replicated(1), _replicated(2), _replicated(3)])
    monkeypatch.undo()

    assert log.path.stat().st_size == committed
    assert len(log.read_all()) == 1
    log.append(actor="george", type="exchange", content={"message": "later"})
    assert len(log.read_all()) == 2  # not a permanently unreadable log
```

#### Verification, and the mutation gate

`make test` is not the gate. Every new test must be shown to **fail** against the defect it
describes. Run each revert below, record the exact pytest summary line for the reverted run **and**
for the restored run, and put both in the report:

| Revert | Tests that must fail |
|---|---|
| `_loads` back to a plain `json.loads` in a `try/except json.JSONDecodeError` | the deep-nesting test |
| `split("\n")` back to `splitlines()` | the three separator cases |
| `decode("utf-8")` back to `decode("utf-8", errors="replace")` | the not-UTF-8 test |
| drop the `MAX_SEQ` branch | the ceiling test |
| drop the `_check_envelope` call | the eight envelope cases + the naive-ts test |
| drop the `host_origin` branch | the host-origin test |
| drop the status check in `BatchRejected.__init__` | the 4xx-class test |
| `clean_reason(reason)` back to `reason[:MAX_REASON_CHARS]` | the truncation-marker test + the empty-reason test |
| drop the `foreign` check in `append_many` | `test_append_many_rejects_two_foreign_origins` |
| drop the `try/except` in `_write_durably` | `test_append_many_rolls_back_a_write_that_dies_part_way` |

Restore the fix after each revert. **A test that still passes against its own reverted fix is not a
test** — stop and report that rather than proceeding.

Then the whole offline suite:

```
poetry run pytest -q tests/ --ignore=tests/test_fact_retention.py --ignore=tests/e2e
```

Baseline before this task: **461 passed**.
