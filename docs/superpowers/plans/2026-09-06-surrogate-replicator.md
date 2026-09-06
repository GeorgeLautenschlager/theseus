# Surrogate-Side Replicator (#31)

> **For agentic workers:** REQUIRED SUB-SKILL: Use steward:steward-local-sdd to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **How this plan is written.** It specifies **contracts and the claim each test must prove** — not
> module source and not test bodies. Blueberry writes both the tests and the implementation; the
> frontier writes the contracts and reviews the diff. A fenced block containing a target module's
> body, or a full test function, means the skill has been inverted. The signature stubs are
> interface, not implementation: they pin names and types so a later task cannot invent a different
> spelling than an earlier one.

**Goal:** The surrogate half of the upstream path — read the local log from the acked cursor, chunk
it into limit-sized batches, POST them one at a time in `seq` order, advance the cursor on `2xx`.
The happy path only; retry and abandonment are #32.

**Architecture:** Four pieces behind one seam. `AckedCursor` is durable state — how far the host has
acknowledged, persisted beside the log. `StimulusTransport` is the protocol the replicator sends
through, and the test seam: the offline suite drives fakes, never a live server. `Replicator` is the
drain loop that reads, chunks and ships. `HttpTransport` is the Phase 1 implementation, and its test
drives the **real** `ReplicationIngress` router in-process, so the two halves of the protocol are
proven to agree on the wire rather than each agreeing with its own fixture.

**Tech Stack:** Python 3.12, Poetry (in-project `.venv`), pytest, httpx/`TestClient`. Run everything
with `env -u VIRTUAL_ENV poetry run ...` — the shell exports a `VIRTUAL_ENV` pointing elsewhere.

**Source issue:** GitHub #31.
**Spec:** `docs/superpowers/specs/2026-09-04-surrogate-replication-protocol.md` § "Upstream: stimulus
replication", § "Delivery posture".
**Builds on:** #26 (envelope), #27 (gap vocabulary), #30 (`replication_batch.py`'s limits and the
`parse_batch` contract this must satisfy), merged as `896795c`.

---

## File Structure

- **Create: `src/theseus/surrogates/cursor.py`** — `AckedCursor`. Durable, tiny, no knowledge of
  transports or batching.
- **Create: `src/theseus/surrogates/transport.py`** — `StimulusTransport` protocol and
  `TransportResult`. Pure interface; imports nothing from this package.
- **Create: `src/theseus/surrogates/replicator.py`** — `Replicator` and the pure `chunk_events`
  helper. The only module that knows the log, the cursor and the transport at once.
- **Create: `src/theseus/surrogates/http_transport.py`** — `HttpTransport`, the Phase 1
  implementation. The only module here that imports `httpx`.
- **Create:** `tests/test_acked_cursor.py`, `tests/test_chunk_events.py`,
  `tests/test_replicator.py`, `tests/test_replication_round_trip.py`.
- **Modify: `src/theseus/__init__.py`** — export `Replicator` (the composition entry point) and
  `StimulusTransport` (the interface a deployment implements). Not the internals.

## Decisions locked in for this plan

| Decision | Value |
|---|---|
| Cursor persistence | A **sidecar file beside the log**, written after each `2xx`. George chose this. The safety argument is the ingress's own: **behind is recoverable, ahead is not.** A cursor that is behind re-sends a batch the host dedupes to a `2xx`; a cursor that is ahead skips events permanently. So it is written only *after* an ack, never before or during, and a crash in between costs exactly one re-sent batch. |
| Cursor file shape | JSON, one object, `{"origin": ..., "acked_seq": ...}`, written by **atomic replace** (write a temp file in the same directory, `fsync`, `os.replace`). A half-written cursor read back as a lower number is merely a re-send; read back as garbage must not crash the surrogate — see the recovery rule below. |
| Cursor recovery | A missing file means "nothing acked yet" — `None`, deliberately distinct from `0`, matching `HighWaterMarks.high_water`. A file that is **unreadable, malformed, or names a different origin** is treated as `None` and the surrogate re-sends from the start, because the alternative — refusing to start — takes a surrogate offline over state whose only failure mode is redundant work the host already dedupes. Log it loudly; do not raise. |
| What gets replicated | **Only events under the log's own origin.** A surrogate's log will also carry host-origin events once the command channel (#34) exists, and shipping those back would put two numbering authorities on one seq stream — the failure `parse_batch`'s `host_origin` guard already rejects at the door. |
| Ordering | By `seq`, ascending. The log is arrival-ordered and a surrogate is the sole writer of its own origin, so `seq` order and file order coincide — but the replicator sorts by `seq` anyway, because that is the property the host's dedupe actually depends on and it should not rest on a coincidence. |
| Batch limits | Imported from `theseus.replication_batch` (`DEFAULT_MAX_BATCH_EVENTS`, `DEFAULT_MAX_BATCH_BYTES`), not redeclared. Two constants for one protocol limit is how a surrogate and a host come to disagree about what fits. Constructor-overridable for tests. |
| Byte limit is measured on the wire body | `max_bytes` bounds the **serialised body including newlines**, because that is what the host measures (`parse_batch` checks `len(raw)`). A chunker that counted only event payloads would build batches the host rejects at exactly the boundary. |
| One request in flight | Enforced with a lock inside `Replicator`, not left to "callers should only use one thread". #30 shipped a deadlock precisely because a concurrency rule lived in a docstring instead of the code. |
| Transport failures | For this issue a transport **raises** on network failure and returns a `TransportResult` for anything the server actually answered. The replicator does not catch it: retry, backoff and abandonment are #32, and swallowing the exception here would silently make this the place that decides them. |
| Known stall, deliberately not fixed here | A single event whose serialised line exceeds `max_bytes` becomes a lone batch the host will always `413`. The replicator stops draining and the cursor does not advance — it stalls. That is the honest behaviour for an issue with no abandon rule; **#32 closes it**. Name it in the code and test the stall, so the follow-up has something to change rather than something to discover. |

---

### Task 1: `AckedCursor` — how far the host has acknowledged

**Files:** Create `src/theseus/surrogates/cursor.py`; test `tests/test_acked_cursor.py`

**Contract:**

```python
class AckedCursor:
    """The highest seq this surrogate's host has acknowledged, persisted beside the log."""

    def __init__(self, path: str | os.PathLike[str], origin: str) -> None: ...

    @property
    def acked_seq(self) -> int | None:
        """Highest acked seq, or None if nothing has ever been acknowledged."""

    def advance(self, seq: int) -> None:
        """Record an ack, durably. Never moves backwards."""
```

Behaviour to satisfy:

1. Loads from the file at construction; a missing file is `None`.
2. `advance` writes by **atomic replace** — temp file in the same directory, flush, `os.fsync`,
   `os.replace` — so a crash mid-write leaves the previous value, never a truncated one.
3. `advance` never lowers the value, including under concurrent callers. Guard it with a lock.
4. A file that is unreadable, malformed, not JSON, or names a **different origin** loads as `None`
   without raising.
5. `advance` rejects a seq below 1, matching `StimulusLog`'s rule.

**Tests to write:**

| Test | Must prove |
|---|---|
| a fresh cursor has acked nothing | `acked_seq is None` with no file present — and `None`, not `0`. |
| an advanced cursor survives a reload | Advance, construct a second `AckedCursor` on the same path, read the value back. |
| the cursor never moves backwards | `advance(10)` then `advance(4)` leaves 10. |
| concurrent advances land on the maximum | Several threads advancing different values; the final value is the highest, and the file agrees with the in-memory value. |
| a crash mid-write leaves the previous value | Simulate by making the replace step fail (monkeypatch `os.replace` to raise) after a successful earlier advance: the file still holds the earlier value and is readable. |
| a corrupt cursor file loads as None rather than raising | Write garbage to the path; construction succeeds and `acked_seq is None`. |
| a cursor naming a different origin is ignored | A file whose `origin` is some other surrogate loads as `None` — a cursor is per-origin and applying one stream's position to another skips real events. |
| a seq below 1 is rejected | `advance(0)` raises `ValueError`. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run them; expect `ModuleNotFoundError`.
- [ ] **Step 3:** Implement. **Step 4:** Run them, all green.
- [ ] **Step 5:** Commit — `git commit -m "Remember how far the host has acknowledged"`

---

### Task 2: `chunk_events` and the transport seam

**Files:** Create `src/theseus/surrogates/transport.py` and the `chunk_events` helper in
`src/theseus/surrogates/replicator.py`; test `tests/test_chunk_events.py`

**Contract:**

```python
# transport.py
@dataclass(frozen=True, slots=True)
class TransportResult:
    status: int


class StimulusTransport(Protocol):
    """Where a batch goes. The seam that makes the transport swappable and the suite offline.

    `send` returns a `TransportResult` for anything the far end actually answered, and
    **raises** when it could not be reached at all. The replicator does not catch that:
    deciding what a failure means is #32's job, not the transport's and not the loop's.
    """

    def send(self, body: str) -> TransportResult: ...


# replicator.py
def chunk_events(
    events: Sequence[StimulusEvent],
    *,
    max_events: int,
    max_bytes: int,
) -> list[list[StimulusEvent]]:
    """Split into batches no larger than either limit, preserving order."""
```

Behaviour to satisfy:

1. Order is preserved; concatenating the batches returns the input exactly.
2. A batch is closed when adding the next event would exceed **either** limit.
3. `max_bytes` is measured on the **serialised body** — each event's `to_json()` plus the `"\n"`
   that separates it — because that is what `parse_batch` measures.
4. An event whose own serialised line exceeds `max_bytes` is emitted as a **lone batch** rather
   than dropped or silently merged. See the known-stall decision above.
5. An empty input yields no batches.

**Tests to write:**

| Test | Must prove |
|---|---|
| a small backlog is one batch | Under both limits, one batch, order preserved. |
| the count limit closes a batch | With `max_events=2` and 5 events: batches of 2, 2, 1. |
| the byte limit closes a batch | Limits set so bytes bind before count; every batch's serialised body is within `max_bytes`, and it took more batches than the count limit alone would have. |
| whichever limit binds first is the one that binds | One case where count binds and bytes are slack, one where bytes bind and count is slack — asserted in the same test or two, but both directions covered. |
| every emitted batch fits what the host accepts | For each batch, `len(("\n".join(e.to_json() for e in batch) + "\n").encode())` is `<= max_bytes`. This is the assertion that catches a chunker measuring the wrong thing. |
| no event is lost or duplicated across batches | Flattening the batches gives back the input list, same order, same length. |
| a single oversized event becomes a lone batch | Its line alone exceeds `max_bytes`; it is emitted alone, not dropped, not merged with a neighbour. |
| an empty input yields no batches | `chunk_events([])` is `[]`. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; expect import errors.
- [ ] **Step 3:** Implement. **Step 4:** Run them, all green.
- [ ] **Step 5:** Commit — `git commit -m "Chunk a backlog by whichever limit binds first"`

---

### Task 3: `Replicator` — the drain loop

**Files:** Create the rest of `src/theseus/surrogates/replicator.py`; test `tests/test_replicator.py`

**Contract:**

```python
@dataclass(frozen=True, slots=True)
class DrainResult:
    batches_sent: int
    events_sent: int
    acked_seq: int | None      # the cursor after this drain
    stopped_on: int | None     # the non-2xx status that ended it, or None if it drained fully


class Replicator:
    def __init__(
        self,
        log: StimulusLog,
        transport: StimulusTransport,
        cursor: AckedCursor,
        *,
        max_events: int = DEFAULT_MAX_BATCH_EVENTS,
        max_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    ) -> None: ...

    def drain(self) -> DrainResult:
        """Ship everything above the cursor, one batch at a time, in seq order."""
```

Behaviour to satisfy:

1. Reads the log, keeps **only events under the log's own origin** whose `seq` is above
   `cursor.acked_seq`, sorts by `seq`, chunks, and sends.
2. Sends **one batch at a time**, in order, with no inter-batch delay. No pipelining.
3. Advances the cursor after **each** `2xx`, not once at the end — a drain interrupted half way
   must not re-send the batches already acknowledged.
4. On a non-`2xx`, **stops immediately** and does not send the remaining batches. Retry is #32.
5. A transport exception propagates; the cursor reflects the batches acked before it.
6. `drain()` holds a lock for its duration, so two threads calling it never put two requests in
   flight.
7. Nothing to send is a no-op returning zeroes, with no call to the transport at all.

**Tests to write** — driven by a fake transport recording the bodies it was given:

| Test | Must prove |
|---|---|
| a backlog larger than the limits drains in seq order across several batches | More events than `max_events`; the transport sees several bodies, and the seqs across them concatenate to the full ascending run with nothing missing or repeated. |
| the cursor advances only on 2xx | A transport answering `200` then `500`: the cursor sits at the last event of the **first** batch, and `stopped_on == 500`. |
| a non-2xx stops the drain immediately | With three batches pending and the transport failing the second, the transport is called exactly **twice** — the third batch is never sent. |
| never more than one request in flight | A fake transport that records concurrent entries and sleeps; several threads call `drain()` at once; max observed concurrency is 1. Bound every wait. |
| only the surrogate's own origin is shipped | A log holding both own-origin events and events replicated in under another origin: the bodies contain only the former. |
| already-acked events are not re-sent | Drain, then append more and drain again: the second call's bodies contain only the new seqs. |
| an empty backlog never touches the transport | `drain()` with nothing pending returns zeroes and the transport records no calls. |
| a drain resumes from the persisted cursor after a restart | Drain, build a **new** `Replicator` and `AckedCursor` over the same paths, append more, drain: only the new events ship. |
| an interrupted drain keeps the batches it did ack | Transport raises on the second batch; the exception propagates and the cursor holds the first batch's last seq. |
| a batch the host will always reject stalls the drain | A single event over `max_bytes` against a transport answering `413`: the cursor does not advance and `stopped_on == 413`. **This pins the known stall #32 must close** — it is a specification of current behaviour, not an endorsement. |
| the interface swap works | A **second**, differently-implemented fake transport (e.g. one built on a Protocol-conforming class rather than the first fake) drives the same `Replicator` unchanged. This is the issue's acceptance box for design-for-deletion. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; expect import errors.
- [ ] **Step 3:** Implement. **Step 4:** Run them, all green.
- [ ] **Step 5:** Commit — `git commit -m "Drain the backlog one batch at a time"`

---

### Task 4: `HttpTransport`, and proving both halves agree

**Files:** Create `src/theseus/surrogates/http_transport.py`; modify `src/theseus/__init__.py`;
test `tests/test_replication_round_trip.py`

**Contract:**

```python
class HttpTransport:
    """POSTs a batch to a host's `/replicate`. The Phase 1 `StimulusTransport`."""

    def __init__(self, url: str, *, timeout: float = 30.0, client: Any | None = None) -> None: ...

    def send(self, body: str) -> TransportResult: ...
```

Behaviour to satisfy:

1. POSTs the body with `content-type: application/json` and returns `TransportResult(status=...)`
   for whatever the server answered, including `4xx` and `5xx`.
2. Raises on a network-level failure — it does not turn an unreachable host into a status code,
   because `5xx` and "never arrived" mean different things to #32's retry rule.
3. Accepts an injected client, so a test can hand it one bound to an in-process ASGI app.

**The round-trip test is the point of this task.** Stand up a real `ReplicationIngress` on a host
`StimulusLog`, mount its router in a `FastAPI` app, point an `HttpTransport` at that app in-process,
and drive a real `Replicator` over a separate surrogate `StimulusLog`. Nothing is faked but the
network.

| Test | Must prove |
|---|---|
| a surrogate's backlog arrives on the host, in order | Append N events to the surrogate log (N larger than the batch limits so it takes several batches); drain; the host's log holds all N under the surrogate's origin, ascending by seq, with the surrogate's own `ts` values intact. |
| the surrogate's cursor ends where the host's mark ends | After the drain, `cursor.acked_seq` equals the host's `HighWaterMarks.high_water(origin)`. The two halves agree on what was delivered. |
| a re-drain after a lost ack is a no-op on the host | Force the cursor back (construct one pointing lower) and drain again: the host answers `2xx`, appends nothing, and its log is byte-identical. This is the lost-ack path end to end. |
| a gap the surrogate declares survives the round trip | Append a `stimulus.gap` built by `replication_events.declared_gap` to the surrogate log, drain, and find it on the host with `declared` still `True` and its range intact — the host re-validates markers at ingress, so this proves the surrogate builds ones that survive that check. |
| an oversized batch is refused end to end | A surrogate whose `max_bytes` exceeds the host's sends a batch the host `413`s; the surrogate's cursor does not advance. Proves the two limits are enforced independently and the surrogate does not assume its own view. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; expect import errors.
- [ ] **Step 3:** Implement `HttpTransport`.
- [ ] **Step 4:** Export `Replicator` and `StimulusTransport` from `src/theseus/__init__.py` and
      `__all__` — the composition entry point and the interface a deployment implements.
      `AckedCursor`, `HttpTransport` and `chunk_events` stay importable by path.
- [ ] **Step 5:** Run the whole offline suite —
      `env -u VIRTUAL_ENV poetry run pytest -q tests/ --ignore=tests/test_fact_retention.py --ignore=tests/e2e`.
      All green, with no pre-existing test file modified.
- [ ] **Step 6:** Commit — `git commit -m "POST a batch, and prove both halves agree on the wire"`

---

## Acceptance mapping

| Issue #31 criterion | Test |
|---|---|
| Backlog larger than the batch limits drains in `seq` order across several batches | Task 3 "a backlog larger than the limits drains in seq order"; Task 4 round trip |
| Never more than one request in flight | Task 3 "never more than one request in flight" |
| Cursor advances only on `2xx` | Task 3 "the cursor advances only on 2xx", "a non-2xx stops the drain immediately" |
| A batch is capped by whichever limit binds first | Task 2 "whichever limit binds first is the one that binds", "every emitted batch fits what the host accepts" |
| Interface swap: a second fake transport works unchanged | Task 3 "the interface swap works" |

## Explicitly not in this plan

- **Retry, backoff, abandonment (#32).** Including the known stall on an over-limit event, which is
  tested here as current behaviour so #32 has something concrete to change.
- **Buffer retention and eviction (#33).** The cursor is not reconciled against a log that lost its
  tail, because nothing evicts yet.
- **When to flush.** The protocol allows a batch at any time; salience-based flush policy is a
  surrogate-local decision specified separately.
- **The command channel (#34, #35)** — the other direction.
- **Auth (#40).** Phase 1 is a LAN.
