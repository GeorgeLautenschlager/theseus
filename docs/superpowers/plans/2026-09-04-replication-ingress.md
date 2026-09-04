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

### Tasks 2–4 (#30) and Tasks 5+ (#31, #32)

Written after Task 1 lands. The parser, the dedupe planner and the endpoint each depend on the
exact shape `append_many` settles, and this plan deliberately does not embed source for modules
that review rounds will change — the previous branch's plan had to be marked superseded for
exactly that reason.
