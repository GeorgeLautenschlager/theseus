# Surrogate Replication: Acceptance Scenario Suite (#36)

> **For agentic workers:** REQUIRED SUB-SKILL: Use steward:steward-local-sdd to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **How this plan is written.** This is a **test-writing** plan: the tests *are* the deliverable,
> and there is no production code to add. It specifies, per scenario, the **setup, the claim the
> test must prove, and the exact assertions** — not the test function bodies. A fenced block
> holding a complete test function means the skill has been inverted; the frontier writes the
> claims, Blueberry writes the test bodies. Names in the contracts (helper names, test names) are
> interface: they pin what later tasks and the acceptance criteria refer to.

**Goal:** One canonical, offline, spec-traceable acceptance suite — twelve tests named after the
spec's twelve numbered scenarios — proving the surrogate-replication series (#26–#35) composes
across the host/surrogate boundary.

**Architecture:** A single new file, `tests/test_acceptance_scenarios.py`, holding exactly twelve
tests, one per spec scenario, each citing its scenario number in its docstring. The upstream
scenarios (1–9, 12) drive a real `Replicator` → `HttpTransport` → `ReplicationIngress` over an
in-process `TestClient` (no socket), with an injected `FakeClock` so retry/backoff and the
one-hour backlog cost no wall-clock time. The two command-channel scenarios (10, 11) need the
real-socket `serve` fixture, because Starlette's `TestClient` cannot stream an infinite SSE
endpoint. All fakes are injected; nothing sleeps on the wall clock.

**Tech Stack:** Python 3.12, Poetry, pytest, FastAPI (`TestClient` + a threaded uvicorn `serve`
fixture), httpx. Run with `env -u VIRTUAL_ENV poetry run ...` (this shell exports theseus's own
venv; unset it so Poetry resolves the project's).

**Source issue:** GitHub #36. **Spec:** `docs/superpowers/specs/2026-09-04-surrogate-replication-protocol.md`
§ "Acceptance scenarios" (the twelve numbered items), read together with § "Event envelope",
§ "Delivery posture", § "Gap markers", and § "Upstream: stimulus replication".
**Builds on (all merged on `main`):** #26–#35 — `StimulusLog`/`BufferedStimulusLog`,
`ReplicationIngress`, `HighWaterMarks`, `Replicator`/`AckedCursor`/`HttpTransport`,
`replication_events` (gap markers + `batch_rejected`), `RetryBudget`, `Clock`, `CommandFeed`,
`SseCommandChannel`, `CommandExecutor`/`command_reports`.

---

## Read these first (the harness already exists — mirror it, do not reinvent)

The three existing round-trip test files already establish every harness shape this suite needs.
**Read all three before writing anything**, and mirror their fixtures/helpers (define local
helpers in the new file in the same spirit — do not import private `_`-prefixed helpers across
test modules):

- `tests/test_replication_round_trip.py` — the **upstream** rig. Note its local helpers `_rig`,
  `_drain`, `_tick`, `_survivor_seqs`, `_host_events`, `_pressure_rig`, `_flaky`, and its
  `FakeClock` (records `.sleeps`, advances `.now()`), and that it uses `fastapi.testclient.TestClient`
  as the `HttpTransport.client` seam — in-process ASGI, no socket. The two logs carry **different
  origins** (`HOST="local"`, `SURROGATE="kitchen"`): the host rejects a batch claiming its own
  origin, so they must differ. `BASE = datetime.now(tz=timezone.utc).replace(microsecond=0)` is
  chosen recent enough that no event trips the six-hour age abandonment.
- `tests/test_command_round_trip.py` — the **downstream** command rig over the real `serve`
  socket: `CommandFeed(log, **FAST).build_app()`, `SseCommandChannel(...)`, `AckedCursor`,
  the threaded consumer with deadline-bounded waits (`FAST`, `FAST_BUDGET`). Scenario 10 lives here.
- `tests/test_command_report_round_trip.py` — the **report** rig: `CommandExecutor` on the
  surrogate side, `Replicator.drain()` shipping the report home, `ReplicationIngress` mounted on
  the same app. Scenario 11 lives here.

`serve` is a session fixture in `tests/conftest.py` (threaded uvicorn on an ephemeral port).

## File Structure

- **Create: `tests/test_acceptance_scenarios.py`** — the whole deliverable. Twelve tests, named
  after the scenarios (see each task for the exact name). At the top: a module docstring stating
  this is the spec's acceptance suite (§ Acceptance scenarios), the two-rig split, and that every
  test is offline. Below it, the local harness helpers (Task 1 builds them; later tasks reuse
  them). No source files are modified; if a scenario cannot be expressed without a source change,
  that is a finding to **report** (`DONE_WITH_CONCERNS`/`NEEDS_CONTEXT`), not a change to make.
- **No other files.** In particular, do **not** edit the three existing round-trip files, do not
  move helpers into `conftest.py` (keep the suite self-contained, matching the repo's
  one-rig-per-file convention), and do not touch `src/`.

## Decisions locked in for this plan

| Decision | Value |
|---|---|
| Deliberate overlap with existing round-trip tests | Scenarios 1–11 are each *already* exercised somewhere (upstream ones in `test_replication_round_trip.py`, 10 in `test_command_round_trip.py`, 11 in `test_command_report_round_trip.py`). #36 is **not** dead weight: it is the spec-traceability document of record — twelve tests named 1:1 with the spec, in one file, each proven to fail-on-revert. The per-issue tests stay; this suite reads as the acceptance checklist. Do not delete or "dedupe" the existing files. |
| Two rigs, by necessity | Upstream scenarios (1–9, 12) use in-process `TestClient` (no socket, fast). Scenarios 10 & 11 use the real-socket `serve` fixture, because `TestClient` cannot stream an infinite SSE endpoint (documented in `conftest.py`). The issue's "FastAPI `TestClient`" guidance holds for the upstream majority; the SSE two are the stated exception. |
| Injected clocks, never `sleep` | Every retry/backoff/age/one-hour path uses a `FakeClock` (mirror the one in `test_replication_round_trip.py`). Assert against the clock's recorded `.sleeps` and its advancing `.now()`, never against wall-clock elapsed. This is what keeps scenarios 7, 9, 12 fast (issue: "don't sleep"). |
| Different origins | Host and surrogate logs always carry distinct `origin`s. Cross-log identity is `(origin, seq)`; ids are re-minted on the host's append, so never assert `id` equality across the two logs — assert on `(origin, seq)`, `type`, and `content`. |
| Dual timestamps | `StimulusEvent` carries `ts` (event_ts, the producer's clock — meaning) and `appended_ts` (minted by the host log on append — order). Skew (scenario 12) is `appended_ts - ts`. The host mints `appended_ts` from `datetime.now(timezone.utc)` at append, so scenario 12 asserts the skew within a generous tolerance band, not to the millisecond. |
| Fail-on-revert is the gate's job | Acceptance criterion "each test fails for the right reason when its feature is reverted" is satisfied per-task by steward-local-sdd's **mutation gate** — one mutation per scenario, proving the test is non-vacuous. Each task's checklist ends by handing the gate rows to the controller. |
| Test naming | `test_scenario_NN_<slug>` where `<slug>` describes the behaviour (e.g. `test_scenario_01_normal_batch_appends_in_arrival_order`). The `NN` keeps the file ordered and greppable against the spec; the slug keeps it readable. Each docstring opens with "Spec acceptance scenario NN: <verbatim scenario line>". |

---

### Task 1: suite scaffolding + upstream rig + scenarios 1–3

**Files:** create `tests/test_acceptance_scenarios.py`

**Build the shared upstream harness** (local helpers, mirroring `test_replication_round_trip.py`):
a `FakeClock` (advancing `now()`, recording `sleeps`), a rig builder that wires a surrogate
`StimulusLog` (or `BufferedStimulusLog` where a task needs pressure) + `ReplicationIngress` on a
host `StimulusLog` + `HighWaterMarks`, exposed over a `TestClient`, plus a `_drain(...)` that runs
one `Replicator.drain()` with an injected clock/budget/`max_events`, and small helpers to read the
host's replicated events and seqs. Constants `HOST`/`SURROGATE` with distinct origins and a
`BASE` timestamp recent enough not to trip age abandonment.

**Scenarios (claims, not bodies):**

- **`test_scenario_01_normal_batch_appends_in_arrival_order`** — Spec #1. A surrogate emits a
  batch whose **arrival order differs from `event_ts` order** (e.g. events appended to the
  surrogate log in seq order 1,2,3 but with `ts` values out of chronological order). After one
  drain: the **host log** holds them in **arrival (seq) order** — the host appends, does not
  interleave by `event_ts` — and each event's `ts` (event_ts) survived the trip unchanged. Then
  prove the reader's half: **sorting the host's replicated events by `ts` recovers the surrogate's
  chronological order** ("the Assembler window sorts by `event_ts`"). Assert both: `[e.seq for e in
  host_events] == [1,2,3]` (arrival order) **and** `[e.seq for e in sorted(host_events, key=ts)]`
  equals the chronological order implied by the `ts` values (different from `[1,2,3]`), so the two
  orderings are demonstrably distinct.
- **`test_scenario_02_duplicate_batch_appends_nothing`** — Spec #2. Drain a batch to the host,
  then submit the **same batch again** (same origin/seq range). The second call returns a **2xx**
  and the host log gains **zero** new events (count unchanged; the deduping is the high-water
  mark). Assert host event count before == after the duplicate, and the response status is 2xx.
- **`test_scenario_03_lost_ack_retry_no_duplicate`** — Spec #3. Model a **lost ack**: the host
  appends the batch but the surrogate never sees the 2xx (mirror `_flaky`/the "5xx then success"
  pattern, or a transport that drops the first response then succeeds). On retry the host returns
  2xx and the host log holds the events **exactly once** — no duplicates from the replay. Assert
  the replicated seqs are the input seqs with no repeats, and the cursor ends at the last seq.

- [ ] Write the failing tests for scenarios 1–3 (module scaffolding + upstream rig + the three tests).
- [ ] Run `env -u VIRTUAL_ENV poetry run pytest tests/test_acceptance_scenarios.py -v`; expected: pass.
- [ ] Hand the controller three gate rows (one per scenario) — e.g. scenario 1: break the host's
      arrival-order append (sort by `ts` on ingest) → test 1 fails; scenario 2: disable the
      high-water dedupe → test 2 fails; scenario 3: make the retry re-append → test 3 fails.
- [ ] Commit.

---

### Task 2: gap scenarios 4–5

**Files:** modify `tests/test_acceptance_scenarios.py`

Reuse Task 1's rig. These prove the two diagnoses of the same hole (spec § Gap markers).

**Scenarios (claims, not bodies):**

- **`test_scenario_04_declared_gap_advances_high_water_past_the_hole`** — Spec #4. The surrogate
  **abandons a range** and emits a `stimulus.gap` marker (`declared_gap(...)`, a declared reason
  such as `link_down`), then continues with later events. After the drain: the host log contains
  **both** the `stimulus.gap` marker **and** the subsequent events, appended without error, and the
  host's **high-water mark has advanced past the hole** to the last delivered seq. Assert: a
  `GAP` event is present on the host with the abandoned `from_seq`/`to_seq` and `declared == True`;
  the post-hole events are present; the drain result's / marks' `high_water` equals the last seq.
- **`test_scenario_05_inferred_gap_is_recorded_by_the_host`** — Spec #5. The surrogate ships a
  batch whose `seq` **jumps with no gap marker** (a hole the surrogate did not declare). The host
  **appends the events** (a gap is not an error) **and records an inferred-gap** for the missing
  range. Assert: the delivered events are on the host log, and the host reports the inferred hole —
  via the ingress response body's `inferred_gaps` (`{"from_seq":…, "to_seq":…}`) and/or a
  host-minted `GAP` event with `reason == "inferred"` / `declared == False` (mirror
  `test_the_host_may_also_infer_the_same_hole...` in `test_replication_round_trip.py` for the exact
  surface). The inferred range matches the jump.

- [ ] Write the failing tests for scenarios 4–5.
- [ ] Run `env -u VIRTUAL_ENV poetry run pytest tests/test_acceptance_scenarios.py -v`; expected: pass.
- [ ] Hand the controller gate rows: scenario 4 — suppress the high-water advance past the marker →
      test 4 fails; scenario 5 — suppress inferred-gap detection on the ingress → test 5 fails.
- [ ] Commit.

---

### Task 3: rejection, retry-exhaustion, storage-pressure — scenarios 6–8

**Files:** modify `tests/test_acceptance_scenarios.py`

Scenario 8 needs a `BufferedStimulusLog` with a tight `BufferPolicy` (mirror `_pressure_rig`).

**Scenarios (claims, not bodies):**

- **`test_scenario_06_oversized_or_malformed_batch_is_rejected_and_marked`** — Spec #6. A batch the
  host must refuse (oversized, past the host's `max_bytes`, or malformed) draws a **4xx**. The
  surrogate **advances past it** (does not wedge) and emits a `replication.batch_rejected` event
  carrying the rejected range and the host's 4xx `status`. Assert: the host response is 4xx and
  appends nothing for that range; a `BATCH_REJECTED` event exists on the surrogate log with the
  right `from_seq`/`to_seq`/`status`; the surrogate's cursor moved past the rejected range.
  (Mirror `test_an_oversized_batch_is_refused_end_to_end` and
  `test_a_rejected_batchs_marker_reaches_the_host`.)
- **`test_scenario_07_retry_exhaustion_abandons_and_drains_the_rest`** — Spec #7. With an injected
  `FakeClock` and a small `RetryBudget` (e.g. `max_attempts=2`), a batch that keeps failing (5xx
  or unreachable) is **abandoned after the budget** — the drain steps over it, appends a
  `retry_exhausted` declared gap, and **drains subsequent batches** rather than blocking on it.
  Assert: `DrainResult.abandoned_batches >= 1`, the later batch's events reached the host, and the
  abandonment cost **no wall-clock time** (`FakeClock.sleeps` are bounded/recorded, not real).
  (Mirror `_abandon_first_batch_rig` / `test_an_abandoned_range_shows_up_as_a_declared_gap`.)
- **`test_scenario_08_storage_pressure_evicts_declares_and_keeps_observing`** — Spec #8. A
  `BufferedStimulusLog` under a tight `BufferPolicy`: appending past `max_bytes` **evicts the
  oldest** buffered events, the buffer **writes a `storage_pressure` gap** for the evicted range,
  and **observation never pauses** (the append that triggered eviction still lands). Assert: the
  oldest events are gone from the surrogate buffer; a `GAP` with `reason == "storage_pressure"`
  covers the evicted range; the triggering event is present; and (composition) after a drain the
  host sees the survivors and the `storage_pressure` marker. (Mirror the eviction tests in
  `test_replication_round_trip.py`.)

- [ ] Write the failing tests for scenarios 6–8.
- [ ] Run `env -u VIRTUAL_ENV poetry run pytest tests/test_acceptance_scenarios.py -v`; expected: pass.
- [ ] Hand the controller gate rows: 6 — accept the oversized batch (skip the size check) → test 6
      fails; 7 — make the drain block on the abandoned batch (never step over) → test 7 fails; 8 —
      skip the `storage_pressure` marker on eviction → test 8 fails.
- [ ] Commit.

---

### Task 4: the one-hour backlog — scenario 9

**Files:** modify `tests/test_acceptance_scenarios.py`

This is the scenario the issue calls out specifically: **a real hour of simulated backlog, not
three events** — the chunking maths and the final high-water mark are the point. No wall-clock time.

**Scenario (claims, not bodies):**

- **`test_scenario_09_hour_offline_then_drains_backlog_in_chunks`** — Spec #9. Using a `FakeClock`,
  the surrogate buffers a backlog whose `event_ts` values **span a full simulated hour** and whose
  **count forces many chunked batches** (e.g. 600 events at 6s spacing over 3600s, with a small
  `max_events` such as 50 → **12 batches**). After the surrogate reconnects and drains: the host
  received **every** event, in seq order, across the expected **number of batches** (count the
  batches with a counting transport wrapper, or assert `ceil(N / max_events)`); the final
  **high-water mark equals the last seq** (`N`), and the surrogate's cursor equals `N`. Assert the
  full seq set arrived (`len == N`, contiguous), the batch count matches the chunking maths, and no
  real time was spent (`FakeClock` drove any backoff). Keep `N` modest enough that the test stays
  well under a second (the issue requires `make test` not get noticeably slower).

- [ ] Write the failing test for scenario 9.
- [ ] Run `env -u VIRTUAL_ENV poetry run pytest tests/test_acceptance_scenarios.py::test_scenario_09_hour_offline_then_drains_backlog_in_chunks -v`; expected: pass.
- [ ] Time it: `env -u VIRTUAL_ENV poetry run pytest tests/test_acceptance_scenarios.py::test_scenario_09_hour_offline_then_drains_backlog_in_chunks --durations=1`; confirm sub-second.
- [ ] Hand the controller a gate row: force `max_events` to `N` (one batch) → the batch-count
      assertion fails, proving the chunking claim is real.
- [ ] Commit.

---

### Task 5: clock skew — scenario 12

**Files:** modify `tests/test_acceptance_scenarios.py`

**Scenario (claims, not bodies):**

- **`test_scenario_12_clock_skew_is_derivable_from_both_timestamps`** — Spec #12. The surrogate's
  clock is **skewed ~900ms** behind the host's: it stamps events' `ts` (event_ts) from a
  `FakeClock` set to `host_now - 900ms`. The host mints `appended_ts` on append from its own clock.
  After the drain, each host event has **both** `ts` and `appended_ts` present (non-`None`), and
  **the skew is derivable**: `appended_ts - ts` falls within a generous band around 900ms (e.g.
  `900ms <= skew < 900ms + slack`, slack covering the tiny real append elapsed). Assert both fields
  present on the replicated events and the derived skew is in-band — proving the trace preserves
  observable skew rather than silently scrambling before-and-after (spec § Event envelope).

- [ ] Write the failing test for scenario 12.
- [ ] Run `env -u VIRTUAL_ENV poetry run pytest tests/test_acceptance_scenarios.py::test_scenario_12_clock_skew_is_derivable_from_both_timestamps -v`; expected: pass.
- [ ] Hand the controller a gate row: drop `event_ts` (overwrite `ts` with `appended_ts` on the
      host, collapsing skew to ~0) → the in-band skew assertion fails.
- [ ] Commit.

---

### Task 6: the command channel — scenarios 10–11 (real socket), then close the suite

**Files:** modify `tests/test_acceptance_scenarios.py`

These two need the real-socket `serve` fixture (SSE cannot stream through `TestClient`). Mirror
`test_command_round_trip.py` (scenario 10) and `test_command_report_round_trip.py` (scenario 11):
`CommandFeed` + `SseCommandChannel` + `AckedCursor` downstream, `CommandExecutor` +
`Replicator.drain()` + `ReplicationIngress` for the report's trip home, threaded consumer with
deadline-bounded waits (`FAST`, `FAST_BUDGET`) — never a bare sleep.

**Scenarios (claims, not bodies):**

- **`test_scenario_10_command_issued_during_downtime_is_delivered_on_reconnect`** — Spec #10. The
  host issues a `command.say` while **no** channel is connected (surrogate "down"). The events sit
  on the host log. A fresh `SseCommandChannel` resuming from its `AckedCursor` **reconnects and
  receives every queued command in seq order**, and the cursor advances to the last. Assert the
  consumed commands' seqs equal the issued seqs, in order. (Mirror
  `test_command_issued_during_downtime_is_delivered_on_reconnect`.)
- **`test_scenario_11_muted_command_produces_failed_report_on_host_log`** — Spec #11. The host
  issues one `command.say`; the surrogate runs a `CommandExecutor` whose injected renderer returns
  `Failed("output muted")`. The executor appends one `command_report.failed` to the surrogate log;
  `Replicator.drain()` ships it, and the **host log** (through the ingress) holds that
  `command_report.failed`, under the surrogate origin, correlated to the issued command by
  `(command_origin, command_seq)`, with `reason == "output muted"` — arriving via the **ordinary
  replication path**, no report-specific route. (Mirror
  `test_muted_command_produces_failed_report_on_host_log`.)

**Close the suite:**

- [ ] Write the failing tests for scenarios 10–11.
- [ ] Run `env -u VIRTUAL_ENV poetry run pytest tests/test_acceptance_scenarios.py -v`; expected:
      **12 tests, all pass**, named after the twelve scenarios.
- [ ] Run the full offline suite and confirm it stays offline and fast:
      `env -u VIRTUAL_ENV poetry run pytest -q tests/ --ignore=tests/test_fact_retention.py`;
      expected: pass, with no noticeable slowdown (spot-check `--durations=10` — the new file's
      tests are all sub-second; scenario 9 is the slowest and still well under a second).
- [ ] Hand the controller gate rows: 10 — make the channel ignore the resume cursor (replay from 0
      or serve nothing) → test 10 fails; 11 — type the report under `command.` instead of
      `command_report.` (so it is filtered / not replicated as a report) → test 11 fails.
- [ ] Commit.

---

## Self-review notes (author)

- **Spec coverage.** Each of the twelve acceptance scenarios maps to exactly one named test:
  Task 1 → 1,2,3; Task 2 → 4,5; Task 3 → 6,7,8; Task 4 → 9; Task 5 → 12; Task 6 → 10,11. The
  file's twelve tests are the acceptance checklist (issue criterion 1).
- **Offline & fast (issue criterion 2).** Upstream scenarios use in-process `TestClient`; the two
  SSE scenarios use the existing threaded `serve` fixture (already in the suite's runtime). Every
  time-dependent path is on an injected `FakeClock`; scenario 9 is bounded to a modest event count
  and spot-checked sub-second. No new live-endpoint dependency; `test_fact_retention.py` stays the
  only excluded file.
- **Fail-on-revert (issue criterion 3).** Every task ends by handing the controller one mutation
  per scenario; steward-local-sdd's mutation gate mechanically proves each test fails for the right
  reason when its feature is reverted, and records the table in the runlog.
- **No source embedded / no test bodies.** Only setup, claims, assertions, test names, and helper
  names appear above — Blueberry writes the test bodies, mirroring the three existing round-trip
  files named in "Read these first".
- **No production code.** This is a test-only issue; if a scenario cannot be proven without a
  source change, that is a decomposition finding to report, not a change to make in this plan.
