# Retry, Backoff and the Abandon Rule (#32)

> **For agentic workers:** REQUIRED SUB-SKILL: Use steward:steward-local-sdd to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **How this plan is written.** It specifies **contracts and the claim each test must prove** — not
> module source and not test bodies. Blueberry writes both the tests and the implementation; the
> frontier writes the contracts and reviews the diff. A fenced block containing a target module's
> body, or a full test function, means the skill has been inverted. The signature stubs are
> interface, not implementation: they pin names across tasks.

**Goal:** Complete #31's happy path with response handling and the abandon rule, so one
undeliverable batch can no longer block the channel while the buffer grows behind it.

**Architecture:** Two new pure pieces and one rewritten loop. `Clock` is the injected time seam that
keeps the suite offline and instant. `RetryBudget` plus two pure functions hold the arithmetic —
what the next backoff delay is, and whether a batch is too old to be worth sending — so the policy
is testable without a transport or a wall clock. `Replicator.drain` then grows the branch per
response class. Markers the surrogate emits about its own stream are appended to its own log, which
means they replicate to the host later as ordinary events; nothing here sends anything special.

**Tech Stack:** Python 3.12, Poetry, pytest. Run with `env -u VIRTUAL_ENV poetry run ...`.

**Source issue:** GitHub #32. **Budget decided in:** #39, recorded in the spec's Open questions
section under **Resolved (2026-09-07)** — read that before starting.
**Builds on:** #27 (`replication_events`), #31 (`Replicator`, `AckedCursor`, `StimulusTransport`),
merged as `20edaba`.

---

## File Structure

- **Create: `src/theseus/surrogates/clock.py`** — `Clock` protocol and `SystemClock`. Standard
  library only.
- **Create: `src/theseus/surrogates/retry.py`** — `RetryBudget`, `backoff_delay`, `is_too_old`.
  Pure arithmetic; imports nothing from this package.
- **Modify: `src/theseus/surrogates/transport.py`** — `TransportResult` gains `reason`.
- **Modify: `src/theseus/surrogates/http_transport.py`** — read the host's reason off the response.
- **Modify: `src/theseus/surrogates/replicator.py`** — the response branch and the abandon rule.
- **Modify: `src/theseus/__init__.py`** — export `RetryBudget` (a deployment tunes it).
- **Tests:** create `tests/test_retry_budget.py`; extend `tests/test_replicator.py` and
  `tests/test_replication_round_trip.py`.

## Decisions locked in for this plan

| Decision | Value |
|---|---|
| The budget | From #39: `max_attempts=5`; backoff base `2.0s`, multiplier `3.0`, jitter `±25%`, ceiling `120.0s`; `max_age=6h`. All fields of one `RetryBudget` dataclass with these as defaults. |
| Max age is measured from | The batch's **oldest `event_ts`** — `min(e.ts for e in batch)`, not `batch[0].ts`. Events are sorted by `seq`, and `ts` need not ascend with `seq` when a producer's clock drifts. |
| Age is checked | **Before** a delivery attempt, and again before each retry. A batch that ages out mid-retry is abandoned rather than retried into staleness. |
| Only a host that answered may spend budget | Decided in #39. `transport.send` returning a status means the host replied: `5xx` spends an attempt. `transport.send` **raising** means nothing was reachable: the drain **stops cleanly**, abandons nothing, spends nothing, and the next drain retries the same range. This supersedes #31's "the exception propagates" — #31 said deciding what a failure means is #32's job, and this is that decision. |
| `4xx` handling | No retry, not once. Append `replication.batch_rejected` to the surrogate's own log, advance the cursor past the batch, continue to the next batch. A `4xx` is permanent; retrying it is how one poison batch wedges a channel. |
| Retry exhaustion | Append `stimulus.gap` with reason `retry_exhausted` covering the abandoned range, advance the cursor past it, continue to the next batch. Both exhaustion paths — attempts and age — produce the same outcome. |
| Where markers go | Appended to the **surrogate's own log** under its own origin with a freshly allocated seq, exactly like any other event it produces. They then replicate to the host on a later drain, where they are "just another event on the tape" per the spec. Nothing is sent out-of-band. |
| Known consequence: the host may also infer the same hole | A marker's seq is above the range it describes, so on a large backlog it lands in a *later* batch than the one that first reveals the hole to the host. The host mints its own `inferred` marker at that point, and the surrogate's `declared` one arrives afterwards. Both sit on the host's tape describing one range with different `reason`s. This is acceptable and already anticipated: `replication_events` documents `reason` as authoritative, and declared-vs-inferred is a diagnosis rather than a conflict. It cannot be avoided without violating ascending-seq. **Test it so it is a known property, not a surprise.** |
| The oversized-single-event stall | Closed here. Such a batch earns a `413`, which is a `4xx`, so it is rejected and stepped over like any other permanently-unacceptable batch. #31 pinned the stall as current behaviour; that test is **updated, not deleted** — it now asserts the drain steps past. |
| Jitter | Applied as a multiplier in `[1 - jitter, 1 + jitter]` on the computed delay, drawn from an injected `random` function so tests are deterministic. Jitter never pushes a delay above the ceiling. |
| Clock injection | `Clock` has `now()` and `sleep(seconds)`. `SystemClock` is the default. Tests inject a fake that records sleeps and advances `now` without real time passing — this is the issue's "backoff is driven by an injected clock" acceptance box. |

---

### Task 1: the clock seam and the host's reason

**Files:** create `src/theseus/surrogates/clock.py`; modify `src/theseus/surrogates/transport.py`
and `src/theseus/surrogates/http_transport.py`; test `tests/test_replicator.py` (a small addition).

**Contract:**

```python
# clock.py
class Clock(Protocol):
    """Time, injected — so backoff is testable without waiting for it."""
    def now(self) -> datetime: ...
    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    """The real one. `now()` is timezone-aware UTC, because every ts in this system is."""


# transport.py
@dataclass(frozen=True, slots=True)
class TransportResult:
    status: int
    reason: str = ""   # the host's own words, when it gave any
```

Behaviour to satisfy:

1. `SystemClock.now()` returns a **timezone-aware UTC** datetime. A naive one would compare
   wrongly against event timestamps, which are all aware.
2. `TransportResult.reason` defaults to `""`, so every existing construction still type-checks and
   every existing test still passes untouched.
3. `HttpTransport.send` populates `reason` from the response when the host provided one. The
   ingress answers a rejection with a JSON body carrying a `reason` field; read it defensively —
   a body that is not JSON, or carries no such field, yields `""` rather than raising. A transport
   that raises while parsing an error response turns a clean `4xx` into an unreachable host.
4. `reason` is bounded before it is stored — `replication_events.MAX_REASON_CHARS` already bounds
   it on write, but this is a remote party's words and the transport should not carry an unbounded
   string around either.

**Tests to write:**

| Test | Must prove |
|---|---|
| the system clock is timezone-aware UTC | `SystemClock().now().tzinfo` is not `None` and the offset is zero. |
| a result carries the host's reason | Constructing `TransportResult(status=400, reason="...")` round-trips it, and the default is `""`. |
| the transport reads the reason from a rejection body | Against an app returning `{"reason": "batch is 9000 bytes, over the ..."}` with a `4xx`, `send` returns that text. |
| a non-JSON error body does not raise | An app answering `4xx` with `text/plain` yields `status` set and `reason == ""`. A transport that raises here would look like an unreachable host. |
| an over-long reason is bounded | A host answering with a very long reason yields one no longer than `MAX_REASON_CHARS`. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; expect import errors.
- [ ] **Step 3:** Implement. **Step 4:** Run them and the full offline suite (674 passing before).
- [ ] **Step 5:** Commit — `git commit -m "Give the transport a clock and the host's own words"`

---

### Task 2: the retry budget, as arithmetic

**Files:** create `src/theseus/surrogates/retry.py`; test `tests/test_retry_budget.py`

**Contract:**

```python
@dataclass(frozen=True, slots=True)
class RetryBudget:
    """How hard to try, and how stale is too stale. Values decided in #39."""
    max_attempts: int = 5
    base_seconds: float = 2.0
    multiplier: float = 3.0
    jitter: float = 0.25
    ceiling_seconds: float = 120.0
    max_age: timedelta = timedelta(hours=6)


def backoff_delay(
    attempt: int,                      # 1 for the first retry
    budget: RetryBudget,
    *,
    random_fn: Callable[[], float] = random.random,
) -> float:
    """Seconds to wait before `attempt`, jittered, never above the ceiling."""


def is_too_old(oldest_event_ts: datetime, now: datetime, budget: RetryBudget) -> bool:
    """Whether a batch's oldest event has aged past the budget."""
```

Behaviour to satisfy:

1. Un-jittered the sequence is `base * multiplier ** (attempt - 1)`, clamped at `ceiling_seconds`:
   with the defaults, `2, 6, 18, 54, 120, 120, ...`.
2. Jitter scales the delay by a factor in `[1 - jitter, 1 + jitter]`, drawn from `random_fn`.
   **The result is still clamped to the ceiling** — jitter must not push a delay above it.
3. A delay is never negative and never below zero even with `jitter >= 1`.
4. `is_too_old` compares against `budget.max_age`; a batch exactly at the boundary is **not** too
   old (`>` not `>=`), matching how every other limit in this protocol is written.
5. `is_too_old` tolerates a naive `oldest_event_ts` by treating it as host-local, the way
   `StimulusEvent.to_json` and `replication_events._utc_span` already do. Comparing naive against
   aware raises a `TypeError` naming neither value, and a surrogate must not die of that.

**Tests to write:**

| Test | Must prove |
|---|---|
| the un-jittered sequence matches #39 | With `jitter=0`, attempts 1-6 give `2, 6, 18, 54, 120, 120`. |
| the ceiling clamps | A far-out attempt returns exactly `ceiling_seconds`, not something enormous. |
| jitter stays inside its band | Over `random_fn` returning 0.0 and 1.0, the delay is within `[0.75x, 1.25x]` of the un-jittered value. |
| jitter never exceeds the ceiling | At an attempt already at the ceiling, `random_fn` returning 1.0 still yields `<= ceiling_seconds`. This is the one an obvious implementation gets wrong. |
| a delay is never negative | Even with an absurd `jitter`, the result is `>= 0`. |
| the defaults are #39's numbers | `RetryBudget()` has `max_attempts=5`, `base=2.0`, `multiplier=3.0`, `jitter=0.25`, `ceiling=120.0`, `max_age=6h`. Pinned because they are a decision with a written rationale, not a guess. |
| a batch inside the age is not too old | Exactly at `max_age` is **not** too old; one second past is. |
| a naive timestamp does not raise | `is_too_old` with a naive `oldest_event_ts` returns a bool rather than raising `TypeError`. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; expect `ModuleNotFoundError`.
- [ ] **Step 3:** Implement. **Step 4:** Run them and the full suite.
- [ ] **Step 5:** Commit — `git commit -m "Hold the retry budget as arithmetic, not as a loop"`

---

### Task 3: response handling — reject, retry, and stop when nothing answers

**Files:** modify `src/theseus/surrogates/replicator.py`; test `tests/test_replicator.py`

**Contract:** `Replicator.__init__` gains three keyword arguments, all defaulted so every existing
construction still works:

```python
        budget: RetryBudget = RetryBudget(),
        clock: Clock = SystemClock(),        # a module-level singleton is fine; do not build per call
        random_fn: Callable[[], float] = random.random,
```

`DrainResult` gains:

```python
    rejected_batches: int = 0     # 4xx: stepped over, batch_rejected recorded
    unreachable: bool = False     # the drain stopped because nothing answered
```

The per-batch branch becomes:

- **`2xx`** — advance the cursor, next batch. Unchanged.
- **`4xx`** — do not retry. Append a `replication.batch_rejected` event to the surrogate's own log
  (`replication_events.batch_rejected`, carrying the batch's `from_seq`/`to_seq`, the status, and
  the host's `reason`), advance the cursor past the batch, count it, continue to the next batch.
- **`5xx`** — retry the **same** body. Between attempts, sleep `backoff_delay(...)` on the injected
  clock. Bounded by `budget.max_attempts` total attempts for that batch.
- **`3xx` or anything else non-2xx** — treat as it is treated today: stop the drain, do not retry,
  do not abandon. It is neither a permanent rejection nor a transient failure, and guessing is
  worse than stopping.
- **`transport.send` raises** — the link is down. Catch it, set `unreachable=True`, stop the drain,
  abandon nothing, and return. Nothing is counted as attempted beyond what was actually sent.

Retry exhaustion is Task 4; for this task, a batch that exhausts its attempts stops the drain the
way a non-2xx does today. Do not implement abandonment yet.

**Tests to write:**

| Test | Must prove |
|---|---|
| a 5xx then a 2xx retries the same range | The transport receives the **same body twice**, the host-visible seq run has no duplicates, and the cursor ends past the batch. |
| retrying sleeps on the injected clock | The fake clock records the delays; no real time passes. Assert the recorded sequence matches `backoff_delay` for attempts 1..n. |
| a 4xx is never retried | The transport is called **exactly once** for that batch. |
| a 4xx records a batch_rejected on the surrogate's own log | The event is on the surrogate log, type `replication.batch_rejected`, with the batch's range, the status, and the host's reason in its content. |
| a 4xx steps over and keeps draining | With three batches and the middle one rejected, the third is still sent and the cursor ends past all three. |
| an unreachable host stops the drain and abandons nothing | A transport that raises: `unreachable is True`, the cursor has not moved past the failed batch, no marker was appended, and a second drain re-sends the same range. |
| an unreachable host spends no attempts | The transport is called **once**, not `max_attempts` times — a link that is down must not burn budget. |
| a 3xx still stops without retrying or abandoning | From #31, unchanged: `stopped_on == 302`, cursor unmoved, and now also no marker appended. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; confirm each fails for the right reason.
- [ ] **Step 3:** Implement. **Step 4:** Run them and the full suite.
- [ ] **Step 5:** Commit — `git commit -m "Branch on what the host actually said"`

---

### Task 4: the abandon rule

**Files:** modify `src/theseus/surrogates/replicator.py`; test `tests/test_replicator.py`

**Contract:** `DrainResult` gains `abandoned_batches: int = 0`.

Two paths reach abandonment, and both do the same three things — append a `stimulus.gap` with
reason `retry_exhausted` covering the batch's `from_seq`..`to_seq`, advance the cursor past the
batch, and continue to the **next** batch:

1. **Age.** Before attempting a batch — and again before each retry — if
   `is_too_old(min(e.ts for e in batch), clock.now(), budget)` then abandon it **without sending**.
   A batch that ages out has nothing to gain from an attempt.
2. **Attempts.** A batch that has spent `budget.max_attempts` attempts against `5xx` responses.

The gap's `span_start` / `span_end` are the batch's own oldest and newest `event_ts` — the surrogate
knows exactly what it dropped, which is the whole reason a declared gap is worth more than an
inferred one.

**Tests to write:**

| Test | Must prove |
|---|---|
| attempt exhaustion abandons and moves on | A transport answering `500` forever for batch 1 then `200`: exactly `max_attempts` calls for batch 1, a `retry_exhausted` gap on the surrogate log covering batch 1's range, the cursor past it, and batch 2 delivered. |
| age exhaustion abandons without sending | A batch whose oldest event is older than `max_age` under the fake clock: the transport is **never called** for it, the gap is recorded, and the next batch drains. |
| age exhaustion with attempts remaining | The same, with attempts untouched — the two bounds are independent, and this is the issue's third acceptance box. |
| a batch that ages out mid-retry is abandoned | `5xx`, then the fake clock advances past `max_age` during backoff: the next check abandons rather than retrying. |
| the gap's range and span are the batch's own | `from_seq`/`to_seq` are the batch's first and last seq; `span_start`/`span_end` are its oldest and newest `event_ts`. |
| the gap is declared, not inferred | `reason == "retry_exhausted"`, `declared is True` — this is the surrogate's own account, and the distinction is the point. |
| an oversized single event is now stepped over, not stalled | **Update #31's stall test rather than deleting it.** The lone over-limit batch earns a `413`, is rejected, and the drain continues past it. |
| abandonment does not lose the events behind it | With batch 1 abandoned and batches 2-3 fine, every seq in 2-3 reaches the transport. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; confirm each fails for the right reason.
- [ ] **Step 3:** Implement. **Step 4:** Run them and the full suite.
- [ ] **Step 5:** Commit — `git commit -m "Abandon a batch rather than let it hold the channel"`

---

### Task 5: end to end against the real host

**Files:** modify `tests/test_replication_round_trip.py`; modify `src/theseus/__init__.py`

Extend the existing round-trip file — a real `Replicator` over a real `HttpTransport` against a
real `ReplicationIngress`, nothing faked but the socket.

| Test | Must prove |
|---|---|
| a 5xx then success delivers exactly once | An ingress wrapped so its first call answers `500`: after the drain the host holds each seq exactly once. This is the issue's first acceptance box, and the one that catches a retry that double-appends. |
| a rejected batch's marker reaches the host | Force a `4xx` (a surrogate `max_bytes` above the host's), drain, then drain again: the `replication.batch_rejected` the surrogate recorded is itself replicated and lands on the host's log. |
| an abandoned range shows up as a declared gap on the host | Abandon a batch, drain the rest, and find the surrogate's `retry_exhausted` marker on the host with `declared is True` and the right range. |
| the host may also infer the same hole, and that is understood | Following the above: the host's tape may hold **both** its own `inferred` marker and the surrogate's `declared` one for one range, because the declared marker's seq sits above the range it describes and so arrives in a later batch. Assert both are present and that `reason` tells them apart. Named in the decisions table; pinned here so it is a known property. |

- [ ] **Step 1:** Write the failing tests. **Step 2:** Run; confirm they fail for the right reason.
- [ ] **Step 3:** Make them pass.
- [ ] **Step 4:** Export `RetryBudget` from `src/theseus/__init__.py` and `__all__` — a deployment
      tunes it, so it belongs on the curated surface. `Clock`, `SystemClock`, `backoff_delay` and
      `is_too_old` stay importable by path.
- [ ] **Step 5:** Run the full offline suite. **Step 6:** Commit — `git commit -m "Prove the abandon rule end to end"`

---

## Acceptance mapping

| Issue #32 criterion | Test |
|---|---|
| `5xx` then `2xx`: same range retried, no duplicate events on the host | Task 3 "a 5xx then a 2xx retries the same range"; Task 5 "delivers exactly once" |
| Retry exhaustion by attempt count: abandoned, `retry_exhausted` gap with the correct range, next batch drains | Task 4 "attempt exhaustion abandons and moves on" |
| Retry exhaustion by age, attempts still remaining | Task 4 "age exhaustion with attempts remaining" |
| `4xx`: no retry, cursor advances, `batch_rejected` emitted | Task 3, the three `4xx` rows |
| Backoff driven by an injected clock — suite stays offline and fast | Task 3 "retrying sleeps on the injected clock"; the whole suite must stay under its current runtime |

## Explicitly not in this plan

- **Buffer retention and storage-pressure eviction (#33).** Age here decides what is worth
  *sending*; what is worth *keeping* is that issue.
- **Backpressure and `429` (#38).** It lands on top of this: sustained `429`s should burn retry
  budget and produce declared gaps. `429` is a `4xx`, so today it is treated as a permanent
  rejection — **name that in the code**, because #38 will change it and should find a signpost.
- **The command channel (#34, #35)** and **auth (#40).**
