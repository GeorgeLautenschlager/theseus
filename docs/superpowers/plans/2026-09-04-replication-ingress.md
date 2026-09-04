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
