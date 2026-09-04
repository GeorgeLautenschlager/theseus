# Surrogate Event Envelope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `StimulusEvent` the replication envelope the surrogate protocol needs — `origin`, `seq`, and both timestamps — and teach `StimulusLog` to allocate seq per origin.

**Architecture:** Two changes in one file. `StimulusEvent` (a frozen slots dataclass) gains three fields with backward-compatible defaults, and its `from_json` learns to read pre-change log lines. `StimulusLog` gains a configured `origin`, a per-origin seq allocator recovered from the log file itself (not a sidecar counter), and an `append` that mints `appended_ts` locally while letting a replicated append carry the `seq` its producer assigned. The ULID `id` moves from being minted off `ts` to being minted off `appended_ts`, so id order stays arrival order even when a replicated event carries an hours-old `event_ts`.

**Tech Stack:** Python 3.12, Poetry (in-project `.venv`), pytest. Run everything with `poetry run`.

**Source issue:** GitHub #26 — "Surrogate replication: extend StimulusEvent with origin, seq and dual timestamps".
**Spec:** `docs/superpowers/specs/2026-09-04-surrogate-replication-protocol.md` § Event envelope.

---

## File Structure

- **Modify: `src/theseus/stimulus_log.py`** — the only production file that changes. It already
  holds both `StimulusEvent` and `StimulusLog`; they change together, so they stay together.
- **Modify: `tests/test_stimulus_log.py`** — all new tests go here alongside the existing
  listener tests. No new test file: this is one module's behaviour.

Nothing else changes. `ContextAssembler` ordering by `event_ts`, the host ingress endpoint, gap
markers, and the high-water-mark store are **later issues (#27–#34) and explicitly out of scope**.

**Do not cut a release between Task 1 and Task 2.** After Task 1, `StimulusLog.append` still
constructs events without an origin, so every line it writes carries an explicit
`"origin":"local"` on the wire. Task 2's `from_json(..., default_origin=self.origin)` backfills
only when the key is *absent* — an explicit `"local"` wins — so a log later configured as
`origin="kitchen-surrogate"` would read its Task-1-era history back as `local`, and
`_recover_next_seq` (which filters on `event.origin == self.origin`) would skip all of it. The
window exists only for the intermediate commit; both tasks ship in one PR, and `make release`
runs from `main` after merge, so this cannot reach a deployed agent. It is recorded here because
the hazard is invisible from either task read alone.

## Naming decisions locked in for this plan

These are settled — implement them as written, do not re-derive:

| Decision | Value |
|---|---|
| Module-level default origin constant | `DEFAULT_ORIGIN = "local"` |
| `content` is **not** renamed to the spec's `payload` | keep `content`; note the alias in the docstring |
| Existing `ts` field is **not** renamed | `ts` *is* the spec's `event_ts`; say so in a comment |
| First seq issued by a fresh log | `1` (so a high-water mark of `0` means "nothing seen yet") |
| Seq counter recovery | lazy, on first local append; scanned from the log file |
| `appended_ts` on direct construction | defaults to `ts` via `__post_init__` |
| Replicated append with a foreign origin and no `seq` | `ValueError` |

---

### Task 1: `StimulusEvent` carries the envelope

**Files:**
- Modify: `src/theseus/stimulus_log.py:54-90` (the `StimulusEvent` dataclass, `to_json`, `from_json`)
- Test: `tests/test_stimulus_log.py`

- [ ] **Step 1: Write the failing tests**

Extend the imports at the top of `tests/test_stimulus_log.py` — the file already imports
`threading` and `StimulusLog`, so add only what Task 1 actually uses:

```python
from datetime import datetime, timezone

from theseus.stimulus_log import DEFAULT_ORIGIN, StimulusEvent, StimulusLog
```

Then append these tests to the end of the file:

```python
def _event(**overrides) -> StimulusEvent:
    fields = dict(
        id="01ABCDEFGHJKMNPQRSTVWXYZ0",
        ts=datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        actor="user",
        type="chat_message",
        content={"message": "hi"},
    )
    fields.update(overrides)
    return StimulusEvent(**fields)


def test_to_json_round_trips_every_envelope_field():
    event = _event(
        origin="kitchen-surrogate",
        seq=7,
        appended_ts=datetime(2026, 1, 1, 13, 30, tzinfo=timezone.utc),
    )

    assert StimulusEvent.from_json(event.to_json()) == event


def test_a_pre_change_log_line_parses_with_the_documented_defaults():
    """Every line written before this change lacks the envelope. They stay readable in
    place — no migration script — so the defaults are part of the contract."""
    legacy = (
        '{"id":"01ABCDEFGHJKMNPQRSTVWXYZ0","ts":"2026-01-01T12:00:00+00:00",'
        '"actor":"user","type":"chat_message","content":{"message":"hi"}}'
    )

    event = StimulusEvent.from_json(legacy, default_origin="kitchen-surrogate")

    assert event.origin == "kitchen-surrogate"
    assert event.seq is None
    assert event.appended_ts == event.ts


def test_from_json_falls_back_to_the_module_default_origin():
    legacy = (
        '{"id":"01ABCDEFGHJKMNPQRSTVWXYZ0","ts":"2026-01-01T12:00:00+00:00",'
        '"actor":"user","type":"chat_message","content":{}}'
    )

    assert StimulusEvent.from_json(legacy).origin == DEFAULT_ORIGIN


def test_appended_ts_defaults_to_event_ts_when_not_supplied():
    """An event that was never appended by a log still has a usable arrival timestamp,
    so downstream ordering never has to special-case None."""
    event = _event()

    assert event.appended_ts == event.ts


def test_envelope_fields_are_optional_so_existing_construction_sites_still_work():
    event = _event()

    assert event.origin == DEFAULT_ORIGIN
    assert event.seq is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/test_stimulus_log.py -q`
Expected: FAIL — `ImportError: cannot import name 'DEFAULT_ORIGIN' from 'theseus.stimulus_log'`

- [ ] **Step 3: Write the implementation**

In `src/theseus/stimulus_log.py`, add the constant just above the `# --- Event ---` banner:

```python
# The origin a log stamps on its own events when none is configured. `origin` answers
# *where* an event entered the system (`kitchen-surrogate`, `android-01`, `webchat`);
# `actor` answers *who* produced it. They stay separate: the same mind reaches the agent
# through several channels, and collapsing them makes the agent either believe in two
# users or unable to decide which mouth to answer from.
DEFAULT_ORIGIN = "local"
```

Replace the whole `StimulusEvent` class with:

```python
@dataclass(frozen=True, slots=True)
class StimulusEvent:
    """One thing that happened, plus the envelope that lets it be replicated.

    `content` is the surrogate protocol's `payload` under its original name — the alias
    is documented rather than renamed, because renaming it churns every module for no gain.
    Likewise `ts` is the protocol's `event_ts`.

    The two timestamps are both retained, and they answer different questions:
    `appended_ts` is authoritative for *log order*, `ts` for *meaning*. The gap between
    them is observable clock skew — a surrogate drifting 900ms shows up in the trace
    instead of silently scrambling the agent's sense of before-and-after.
    """

    id: str
    ts: datetime         # event_ts: when it happened, by the producer's clock
    actor: str           # who/what produced it ("george", "tam", "env", "sensor")
    type: str            # "exchange" | "capture" | "observation" | ...
    content: dict[str, Any]  # type-specific payload; e.g. {"prompt":..,"response":..}
    origin: str = DEFAULT_ORIGIN  # where it entered the system; assigned by the producer
    seq: int | None = None        # monotonic per origin. Not contiguous — gaps are legal.
    appended_ts: datetime | None = None  # when it landed on this log

    def __post_init__(self) -> None:
        # An event that no log has appended yet still needs an arrival timestamp, so
        # ordering code never has to special-case None.
        if self.appended_ts is None:
            object.__setattr__(self, "appended_ts", self.ts)

    def to_json(self) -> str:
        return json.dumps(
            {
                "id": self.id,
                "ts": self.ts.astimezone(timezone.utc).isoformat(),
                "actor": self.actor,
                "type": self.type,
                "content": self.content,
                "origin": self.origin,
                "seq": self.seq,
                "appended_ts": self.appended_ts.astimezone(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(
        cls, line: str, *, default_origin: str = DEFAULT_ORIGIN
    ) -> "StimulusEvent":
        """Parse one log line. Lines written before the envelope existed are still valid:
        `origin` falls back to `default_origin` (the reading log's own origin), `seq` to
        None, and `appended_ts` to `ts`. Old lines stay readable in place — there is no
        migration."""
        d = json.loads(line)
        ts = datetime.fromisoformat(d["ts"])
        appended_ts = d.get("appended_ts")
        return cls(
            id=d["id"],
            ts=ts,
            actor=d["actor"],
            type=d["type"],
            content=d["content"],
            origin=d.get("origin") or default_origin,
            seq=d.get("seq"),
            appended_ts=datetime.fromisoformat(appended_ts) if appended_ts else ts,
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `poetry run pytest tests/test_stimulus_log.py -q`
Expected: PASS, all tests in the file.

- [ ] **Step 5: Run the offline suite to verify nothing regressed**

Run: `poetry run pytest -q tests/ --ignore=tests/test_fact_retention.py --ignore=tests/e2e`
Expected: PASS — 325 existing tests plus the new ones, 0 failures.

(`tests/test_fact_retention.py` and `tests/e2e` need a live LLM endpoint; they are excluded
from every test command in this plan. Never run the bare `tests/` directory — it hangs.)

- [ ] **Step 6: Commit**

```bash
git add src/theseus/stimulus_log.py tests/test_stimulus_log.py
git commit -m "Add origin, seq and appended_ts to StimulusEvent"
```

---

### Task 2: `StimulusLog` allocates seq per origin

**Files:**
- Modify: `src/theseus/stimulus_log.py` (the `StimulusLog` class: `__init__`, `append`, `read_all`)
- Test: `tests/test_stimulus_log.py`

- [ ] **Step 1: Write the failing tests**

Task 2's tests need three more imports. Extend the top of `tests/test_stimulus_log.py` again
so it reads:

```python
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from theseus.stimulus_log import DEFAULT_ORIGIN, StimulusEvent, StimulusLog, new_id
```

Then append these tests:

```python
def test_local_appends_get_a_monotonic_seq_starting_at_one(tmp_path):
    log = make_log(tmp_path)

    events = [
        log.append(actor="user", type="chat_message", content={"n": n}) for n in range(3)
    ]

    assert [e.seq for e in events] == [1, 2, 3]
    assert {e.origin for e in events} == {DEFAULT_ORIGIN}


def test_seq_is_monotonic_across_a_restart(tmp_path):
    """The log file is the only durable state. A sidecar counter that disagreed with it
    after a crash would either drop real events or issue the same seq twice."""
    path = tmp_path / "stimulus_log.jsonl"
    first = StimulusLog(path=path)
    first.append(actor="user", type="chat_message", content={})
    first.append(actor="user", type="chat_message", content={})

    reopened = StimulusLog(path=path)
    event = reopened.append(actor="user", type="chat_message", content={})

    assert event.seq == 3


def test_the_log_stamps_its_own_origin_on_local_appends(tmp_path):
    log = StimulusLog(path=tmp_path / "stimulus_log.jsonl", origin="kitchen-surrogate")

    event = log.append(actor="user", type="chat_message", content={})

    assert event.origin == "kitchen-surrogate"


def test_id_order_matches_append_order_when_event_ts_is_backdated(tmp_path):
    """A replicated event can carry a skewed or hours-old event_ts. The id is minted from
    appended_ts so it never sorts into the middle of the log — read_range, the debug tail
    cursor and most_recent_page all read id order as arrival order."""
    log = make_log(tmp_path)
    backdated = datetime.now(timezone.utc) - timedelta(hours=1)

    old = log.append(actor="user", type="chat_message", content={"n": 1}, ts=backdated)
    time.sleep(0.002)  # ULIDs are millisecond-resolution; keep the two ids distinguishable
    new = log.append(actor="user", type="chat_message", content={"n": 2})

    assert old.id < new.id
    # Minted from arrival, not from the backdated event clock: an id minted an hour ago
    # would sort below this one and land in the middle of the log.
    assert old.id > new_id(int(backdated.timestamp() * 1000))
    assert log.read_range(old.id, new.id) == [old, new]


def test_appended_ts_is_minted_by_the_log_not_taken_from_the_caller(tmp_path):
    log = make_log(tmp_path)
    backdated = datetime.now(timezone.utc) - timedelta(hours=1)

    event = log.append(actor="user", type="chat_message", content={}, ts=backdated)

    assert event.ts == backdated
    assert event.appended_ts > event.ts


def test_a_replicated_append_keeps_the_origin_and_seq_its_producer_assigned(tmp_path):
    log = make_log(tmp_path)

    event = log.append(
        actor="user",
        type="chat_message",
        content={},
        origin="kitchen-surrogate",
        seq=41,
    )

    assert (event.origin, event.seq) == ("kitchen-surrogate", 41)
    assert log.read_all() == [event]


def test_a_replicated_append_must_carry_a_seq(tmp_path):
    """This log can only allocate for its own origin — inventing a seq for someone else's
    would collide with the one the producer already assigned."""
    log = make_log(tmp_path)

    with pytest.raises(ValueError):
        log.append(
            actor="user", type="chat_message", content={}, origin="kitchen-surrogate"
        )


def test_a_replicated_seq_does_not_disturb_the_local_counter(tmp_path):
    log = make_log(tmp_path)
    log.append(actor="user", type="chat_message", content={})
    log.append(
        actor="user", type="chat_message", content={}, origin="android-01", seq=900
    )

    event = log.append(actor="user", type="chat_message", content={})

    assert event.seq == 2


def test_seq_recovery_ignores_other_origins(tmp_path):
    """Recovery counts only this log's own origin — a surrogate's seq 900 must not push
    the host's own counter into the nine-hundreds."""
    path = tmp_path / "stimulus_log.jsonl"
    first = StimulusLog(path=path)
    first.append(actor="user", type="chat_message", content={})
    first.append(
        actor="user", type="chat_message", content={}, origin="android-01", seq=900
    )

    event = StimulusLog(path=path).append(actor="user", type="chat_message", content={})

    assert event.seq == 2


def test_legacy_lines_read_back_with_the_logs_own_origin(tmp_path):
    path = tmp_path / "stimulus_log.jsonl"
    path.write_text(
        '{"id":"01ABCDEFGHJKMNPQRSTVWXYZ0","ts":"2026-01-01T12:00:00+00:00",'
        '"actor":"user","type":"chat_message","content":{}}\n',
        encoding="utf-8",
    )
    log = StimulusLog(path=path, origin="kitchen-surrogate")

    (event,) = log.read_all()

    assert event.origin == "kitchen-surrogate"
    assert event.seq is None
    assert event.appended_ts == event.ts


def test_concurrent_appends_never_reuse_a_seq(tmp_path):
    log = make_log(tmp_path)
    events: list[StimulusEvent] = []
    barrier = threading.Barrier(8)

    def appender() -> None:
        barrier.wait()
        events.append(log.append(actor="user", type="chat_message", content={}))

    threads = [threading.Thread(target=appender) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(e.seq for e in events) == [1, 2, 3, 4, 5, 6, 7, 8]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `poetry run pytest tests/test_stimulus_log.py -q`
Expected: FAIL — `TypeError: StimulusLog.__init__() got an unexpected keyword argument 'origin'`
and assertion failures on `seq is None`.

- [ ] **Step 3: Write the implementation**

In `src/theseus/stimulus_log.py`, replace `StimulusLog.__init__` with:

```python
    def __init__(
        self, path: str | os.PathLike[str], origin: str = DEFAULT_ORIGIN
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.origin = origin
        self._listeners: list[Callable[[StimulusEvent], None]] = []
        self._listener_lock = threading.Lock()
        self._append_lock = threading.Lock()
        self._next_seq: int | None = None  # recovered from the log on first local append
```

Add this method just above `append`:

```python
    def _recover_next_seq(self) -> int:
        """The seq this log should issue next for its own origin, read back off the file.

        The log is the only durable state, so the counter is derived from it rather than
        kept in a sidecar: a sidecar that disagreed with the file after a crash would
        either skip real events or issue the same seq twice. Seqs start at 1, which
        leaves 0 free to mean "nothing seen yet" for a reader's high-water mark.
        """
        highest = 0
        for event in self.read_all():
            if event.origin == self.origin and event.seq is not None:
                highest = max(highest, event.seq)
        return highest + 1
```

Replace `append` with:

```python
    def append(
        self,
        actor: str,
        type: str,
        content: dict[str, Any],
        ts: datetime | None = None,
        *,
        origin: str | None = None,
        seq: int | None = None,
    ) -> StimulusEvent:
        """Append one event and notify listeners.

        `ts` is the event's own clock — when it happened. `appended_ts` is always minted
        here, and the id is minted from it, so id order stays arrival order however far a
        producer's clock has drifted.

        A local append (`origin` omitted) gets the next seq for this log's own origin. A
        replicated append carries the origin *and* the seq its producer already assigned;
        this log cannot allocate one on a producer's behalf without colliding with it.
        """
        ts = ts or datetime.now(timezone.utc)
        origin = self.origin if origin is None else origin
        if seq is None and origin != self.origin:
            raise ValueError(
                f"a replicated append (origin {origin!r}) must carry the seq its "
                f"producer assigned"
            )

        with self._append_lock:
            if seq is None:
                if self._next_seq is None:
                    self._next_seq = self._recover_next_seq()
                seq = self._next_seq
                self._next_seq = seq + 1
            elif origin == self.origin and self._next_seq is not None:
                # Someone replayed one of our own events with an explicit seq; never
                # hand that number out again.
                self._next_seq = max(self._next_seq, seq + 1)

            appended_ts = datetime.now(timezone.utc)
            event = StimulusEvent(
                id=new_id(int(appended_ts.timestamp() * 1000)),
                ts=ts,
                actor=actor,
                type=type,
                content=content,
                origin=origin,
                seq=seq,
                appended_ts=appended_ts,
            )
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(event.to_json() + "\n")
                f.flush()
                os.fsync(f.fileno())

        # Outside the lock: a listener is free to append, and holding the lock across a
        # callback would deadlock it.
        self._notify(event)
        return event
```

In `read_all`, pass this log's origin down as the fallback for pre-change lines:

```python
                events.append(
                    StimulusEvent.from_json(stripped, default_origin=self.origin)
                )
```

Finally, extend the `StimulusLog` class docstring with a paragraph after the existing text:

```
    A log has an `origin` — the name of the place its own events enter the system. It
    allocates a monotonic `seq` per origin, recovered from the file on first append, and
    accepts replicated events that carry the origin and seq their producer assigned.
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `poetry run pytest tests/test_stimulus_log.py -q`
Expected: PASS.

- [ ] **Step 5: Run the offline suite to verify nothing regressed**

Run: `poetry run pytest -q tests/ --ignore=tests/test_fact_retention.py --ignore=tests/e2e`
Expected: PASS, 0 failures. Pay attention to `tests/test_debug_pagination.py`,
`tests/test_debug_row_rendering.py`, `tests/test_core_concurrency.py` and
`tests/test_context_assembler.py` — they are the ones that touch event construction,
concurrent appends and id ordering.

- [ ] **Step 6: Commit**

```bash
git add src/theseus/stimulus_log.py tests/test_stimulus_log.py
git commit -m "Give StimulusLog an origin and a per-origin seq allocator"
```

---

## Acceptance (from issue #26)

- [ ] `to_json` / `from_json` round-trips all fields — Task 1
- [ ] A pre-change log line parses with the documented defaults — Task 1
- [ ] `seq` is monotonic per origin across a process restart — Task 2
- [ ] ULID ordering still matches append order when `event_ts` is backdated by an hour — Task 2
- [ ] Existing `tests/test_stimulus_log.py` stays green — both tasks, Step 5
