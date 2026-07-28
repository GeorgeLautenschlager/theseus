# TimeObserver + Core Concurrency Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `TimeObserver` that periodically nudges the Core to orient on new stimuli, and make the Core a single-cycle-at-a-time shared resource with two explicit contention policies (skip / wait) that every Observer routes through.

**Architecture:** `OODACore` gains one `threading.Lock` guarding cognitive-cycle execution, exposed through two gated entry points — `try_orient()` (non-blocking, returns whether a cycle ran) and `orient_and_wait()` (blocking). The existing `orient()` stays un-gated and re-entrant (Act recurses into it), so the gate lives only at the two new boundaries. Observers pick a policy by which method they are wired to: `TimeObserver` → `try_orient` (skip), `TerminalChatObserver` and `WebChatUIObserver` → `orient_and_wait` (wait). `threading.Lock` (not `asyncio.Lock`) because it is held from plain OS threads in every case — the web observer already runs `orient()` on a per-message background thread.

**Tech Stack:** Python 3.12, `threading`, pytest, Poetry (`poetry run pytest`). Offline suite via `make test`.

---

## Grounding: brief vs. real code

The brief (`BRIEF-time-observer-core-lock.md`) was written **without repo access** (its own preserved chat transcript shows the repo wasn't reachable), so its names and one of its decisions don't match the code. This plan uses the real code and records the deltas here.

**Naming corrections (brief → actual):**
- `CognitiveCore` / `Core` → **`OODACore`** in `src/theseus/ooda_core.py`; cycle entry point is `orient()`.
- `ChatObserver` → **`TerminalChatObserver`** in `src/theseus/chat_observer.py`.
- `WebChatObserver` → **`WebChatUIObserver`** in `src/theseus/web_chat_ui_observer.py`.

**Deviations from the brief (approved by George, 2026-07-27):**
- **D8 / Q3 superseded.** The brief prescribes an async `await asyncio.sleep` poll-loop for the web observer on the theory that it acquires the lock inside a coroutine. It does not: `WebChatUIObserver._handle_chat_submit` already spawns a daemon thread (`_run_core`, `web_chat_ui_observer.py:137`) that runs `orient()`, so the event loop is never on the cycle path. The web observer therefore uses a **blocking acquire on that existing background thread** — identical to `TerminalChatObserver` — which trivially satisfies acceptance criterion #4. There is no poll interval and no async retry loop.
- **Q1 refined (correctness, not taste).** The brief's default heuristic is "any new StimulusLog entry triggers an attempt." Taken literally this self-triggers forever: the Core writes its own `decision` and `tool_result` events (actor = the Core's name) into the same log, so each cycle's own output would wake the next tick. `TimeObserver` therefore fires only on new entries authored by an actor **other than the Core itself**. This is the minimal correct form of Q1.
- **TimeObserver home:** ships as a reusable class + tests in Theseus only. No agent wiring for it (Alty stays terminal-only; consumers opt in). The retrofit of the *existing* terminal observer onto `orient_and_wait` still lands in Alty as the reference wiring.

**Defaults carried unchanged:** D1 (60s interval), D2 (in-memory checkpoint), D3–D7, D9; Q2 (plain thread + `Event().wait(interval)`); Q4 (indefinite blocking wait).

## File structure

- **Modify** `src/theseus/ooda_core.py` — add `threading` import, one `threading.Lock`, `try_orient()`, `orient_and_wait()`; one docstring note on `orient()`.
- **Create** `src/theseus/time_observer.py` — `TimeObserver` (thread-based, skip-on-contention).
- **Modify** `src/theseus/__init__.py` — export `TimeObserver`.
- **Modify** `src/theseus/agents/alty_mcgee.py` — retrofit terminal observer wiring to `core.orient_and_wait`.
- **Modify** `src/theseus/web_chat_ui_observer.py` — docstring only (record the gate + wiring; no behavioural change).
- **Create** `tests/test_core_concurrency.py` — mutual-exclusion + policy behaviour.
- **Create** `tests/test_time_observer.py` — heuristic, checkpoint, lifecycle.
- **Create** `tests/test_web_chat_observer.py` — the submit handler never blocks on the cycle (AC#4).

---

### Task 1: Core cycle gate + `try_orient()` (skip-on-contention)

**Files:**
- Modify: `src/theseus/ooda_core.py`
- Test: `tests/test_core_concurrency.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_core_concurrency.py`:

```python
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from theseus.mono_memory import MonoMemory
from theseus.ooda_core import OODACore
from theseus.stimulus_log import StimulusLog
from theseus.tools.tool import AssistantTurn


def make_core(tmp_path, provider):
    log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    return OODACore(
        name="Tam",
        constitution="You are Tam.",
        persona="Direct.",
        context_assembler=MonoMemory(stimulus_log=log),
        model_providers=[provider],
        tools={},
        stimulus_log=log,
    )


def gated_provider(started: threading.Event, release: threading.Event):
    """One-turn provider. The FIRST cycle parks inside complete_with_tools until
    `release` fires; later cycles return immediately. Lets a cycle be held 'in flight'
    while a second observer attempts to enter."""
    provider = MagicMock()
    provider.is_available.return_value = True
    calls = {"n": 0}
    lock = threading.Lock()

    def complete(messages, tools):
        with lock:
            calls["n"] += 1
            first = calls["n"] == 1
        if first:
            started.set()
            release.wait(timeout=5)
        return AssistantTurn(text="done", tool_calls=())

    provider.complete_with_tools.side_effect = complete
    return provider


def test_try_orient_skips_while_a_cycle_is_in_flight(tmp_path):
    started, release = threading.Event(), threading.Event()
    provider = gated_provider(started, release)
    core = make_core(tmp_path, provider)

    holder = threading.Thread(target=core.try_orient)
    holder.start()
    assert started.wait(2), "first cycle never entered the provider"

    # Skip-on-contention: non-blocking, returns False, no exception, no second cycle.
    assert core.try_orient() is False

    release.set()
    holder.join(2)
    assert provider.complete_with_tools.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_core_concurrency.py::test_try_orient_skips_while_a_cycle_is_in_flight -v`
Expected: FAIL with `AttributeError: 'OODACore' object has no attribute 'try_orient'`

- [ ] **Step 3: Add the gate and `try_orient()`**

In `src/theseus/ooda_core.py`, add the import near the top (after `from __future__ import annotations`):

```python
import threading
```

In `OODACore.__init__`, immediately after `self.max_loops = max_loops`, add:

```python
        # One gate guarding cognitive-cycle execution, shared across every Observer
        # regardless of concurrency model, so exactly one cycle runs at a time across
        # the whole process. threading.Lock (not asyncio.Lock) because it is held from
        # plain OS threads in every case: TerminalChatObserver's stdin thread, and
        # WebChatUIObserver's per-message background thread.
        self._cycle_lock = threading.Lock()
```

Add this method to `OODACore` (place it directly above `def orient(self):`):

```python
    def try_orient(self) -> bool:
        """Skip-on-contention entry point. Non-blocking acquire of the cycle gate: run
        one cognitive cycle iff no other cycle is in flight, and report whether one
        actually ran. A False return is the ordinary outcome of contention, never an
        error. Observers that must not block (TimeObserver) call this."""
        if not self._cycle_lock.acquire(blocking=False):
            return False
        try:
            self.orient()
            return True
        finally:
            self._cycle_lock.release()
```

Update the `orient()` docstring (`"""Callback to be invoked by chat UI"""`) to:

```python
        """Run one cognitive cycle. Un-gated on purpose: it assumes the caller already
        holds the cycle gate (via try_orient / orient_and_wait) or is a single-threaded
        test. Act() re-enters this method directly to process tool results, so it must
        never take the gate itself. Observers must NOT call orient() directly — they go
        through try_orient() or orient_and_wait()."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_core_concurrency.py::test_try_orient_skips_while_a_cycle_is_in_flight -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/theseus/ooda_core.py tests/test_core_concurrency.py
git commit -m "feat(core): add cycle gate and non-blocking try_orient()"
```

---

### Task 2: Core `orient_and_wait()` (wait-on-contention)

**Files:**
- Modify: `src/theseus/ooda_core.py`
- Test: `tests/test_core_concurrency.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_core_concurrency.py`:

```python
def test_orient_and_wait_blocks_then_runs_after_the_in_flight_cycle(tmp_path):
    started, release = threading.Event(), threading.Event()
    provider = gated_provider(started, release)
    core = make_core(tmp_path, provider)

    holder = threading.Thread(target=core.try_orient)
    holder.start()
    assert started.wait(2), "first cycle never entered the provider"

    waiter = threading.Thread(target=core.orient_and_wait)
    waiter.start()
    time.sleep(0.1)  # give the waiter a chance to (wrongly) proceed
    # It must be parked on the gate, not inside a second cycle.
    assert provider.complete_with_tools.call_count == 1

    release.set()
    holder.join(2)
    waiter.join(2)
    # Once the gate freed, the waiter ran its cycle: input is never dropped.
    assert provider.complete_with_tools.call_count == 2


def test_never_two_cycles_at_once(tmp_path):
    peak = {"cur": 0, "max": 0}
    guard = threading.Lock()
    provider = MagicMock()
    provider.is_available.return_value = True

    def complete(messages, tools):
        with guard:
            peak["cur"] += 1
            peak["max"] = max(peak["max"], peak["cur"])
        time.sleep(0.02)
        with guard:
            peak["cur"] -= 1
        return AssistantTurn(text="x", tool_calls=())

    provider.complete_with_tools.side_effect = complete
    core = make_core(tmp_path, provider)

    threads = [threading.Thread(target=core.orient_and_wait) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)

    assert peak["max"] == 1
    assert provider.complete_with_tools.call_count == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_core_concurrency.py -v -k "orient_and_wait or never_two"`
Expected: FAIL with `AttributeError: 'OODACore' object has no attribute 'orient_and_wait'`

- [ ] **Step 3: Add `orient_and_wait()`**

In `src/theseus/ooda_core.py`, add this method directly below `try_orient()`:

```python
    def orient_and_wait(self) -> None:
        """Wait-on-contention entry point. Blocking acquire of the cycle gate: wait for
        any in-flight cycle to finish, then run one. Safe only from a thread with
        nothing else to do while it waits — TerminalChatObserver's dedicated stdin
        thread, or WebChatUIObserver's per-message background thread. Never call this
        from a coroutine on an event loop."""
        self._cycle_lock.acquire()
        try:
            self.orient()
        finally:
            self._cycle_lock.release()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_core_concurrency.py -v`
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add src/theseus/ooda_core.py tests/test_core_concurrency.py
git commit -m "feat(core): add blocking orient_and_wait() and mutual-exclusion test"
```

---

### Task 3: `TimeObserver` — construction and checkpoint init

**Files:**
- Create: `src/theseus/time_observer.py`
- Test: `tests/test_time_observer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_time_observer.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock

from theseus.stimulus_log import StimulusLog
from theseus.time_observer import TimeObserver


def make(tmp_path, interval=60.0):
    log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    try_orient = MagicMock(return_value=True)
    obs = TimeObserver(log, try_orient, self_actor="Tam", interval_seconds=interval)
    return obs, log, try_orient


def test_start_checkpoints_past_existing_backlog(tmp_path):
    obs, log, try_orient = make(tmp_path)
    log.append(actor="user", type="chat_message", content={"message": "old"})

    obs.start()          # checkpoint moves to the current tip (the backlog entry)
    obs.stop(timeout=1)
    obs._tick()          # nothing newer than the backlog

    try_orient.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_time_observer.py::test_start_checkpoints_past_existing_backlog -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'theseus.time_observer'`

- [ ] **Step 3: Create `TimeObserver`**

Create `src/theseus/time_observer.py`:

```python
"""TimeObserver — wakes on an interval and nudges the Core to orient on new stimuli.

Unlike TerminalChatObserver and WebChatUIObserver, which react to an external event
(a keystroke, an HTTP request), TimeObserver reacts to the passage of time: it wakes
every `interval_seconds`, checks whether anything new has landed in the StimulusLog
since it last looked, and — if so — asks the Core to run a cognitive cycle.

Concurrency policy: skip-on-contention (brief D6). It calls the Core's non-blocking
`try_orient`; if a cycle is already in flight the attempt is a silent no-op until the
next wake. Safe because the StimulusLog is durable — whichever cycle runs next
assembles its context from the whole log and sees everything that accumulated.

Checkpoint (brief D2): in memory only, initialised to the log's tip at `start()`, so a
fresh run reacts to what arrives *during* the run, not to prior-run backlog. No
cross-restart persistence.

Self-authored events are ignored (brief Q1, corrected): the Core writes its own
`decision` and `tool_result` events into the same log, so counting those as "new"
would make every cycle's own output re-trigger the next wake forever. Only entries
from an actor other than the Core count as a stimulus worth waking for.
"""

from __future__ import annotations

import threading
from typing import Callable

from theseus.stimulus_log import StimulusLog


class TimeObserver:
    def __init__(
        self,
        stimulus_log: StimulusLog,
        try_orient: Callable[[], bool],
        self_actor: str,
        interval_seconds: float = 60.0,
    ):
        """
        Args:
            stimulus_log: the log to watch for new entries.
            try_orient: the Core's non-blocking entry point (OODACore.try_orient).
            self_actor: the Core's own actor name (OODACore.name). Entries this actor
                writes are ignored when deciding whether to wake the Core.
            interval_seconds: seconds between wakes. Defaults to 60 (brief D1).
        """
        self.stimulus_log = stimulus_log
        self.try_orient = try_orient
        self.self_actor = self_actor
        self.interval_seconds = interval_seconds
        self._checkpoint: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Initialise the checkpoint to the current tip and spawn the wake loop on a
        daemon thread. One TimeObserver runs one thread — don't call start() twice."""
        events = self.stimulus_log.read_all()
        self._checkpoint = events[-1].id if events else None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        """Signal the wake loop to exit and best-effort join it. Interrupts the interval
        immediately via the Event, so shutdown doesn't wait out a full interval."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        # Event().wait(interval) is both the sleep and the interruptible shutdown
        # signal: returns True the instant stop() fires, False on timeout (a wake).
        while not self._stop.wait(self.interval_seconds):
            self._tick()

    def _tick(self) -> None:
        """One wake: fire the Core iff a new, externally-authored entry has appeared
        since the last checkpoint, then advance the checkpoint to the current tip so the
        same entries never count twice. The Core's own output from the cycle this may
        trigger lands beyond the checkpoint but is excluded by the self_actor filter."""
        events = self.stimulus_log.read_all()
        if not events:
            return
        checkpoint = self._checkpoint
        has_new_external = any(
            (checkpoint is None or event.id > checkpoint)
            and event.actor != self.self_actor
            for event in events
        )
        self._checkpoint = events[-1].id
        if has_new_external:
            self.try_orient()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_time_observer.py::test_start_checkpoints_past_existing_backlog -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/theseus/time_observer.py tests/test_time_observer.py
git commit -m "feat(time-observer): add TimeObserver with in-memory checkpoint"
```

---

### Task 4: `TimeObserver` heuristic — fire, ignore self, fire once

**Files:**
- Test: `tests/test_time_observer.py`
- (No production change — these lock in `_tick` behaviour written in Task 3.)

- [ ] **Step 1: Write the characterization tests**

These lock in the `_tick` contract implemented in Task 3, so they pass on arrival rather than failing first. Append to `tests/test_time_observer.py`:

```python
def test_new_external_event_triggers_orient(tmp_path):
    obs, log, try_orient = make(tmp_path)
    obs.start()
    obs.stop(timeout=1)  # checkpoint = empty-log tip (None)

    log.append(actor="user", type="chat_message", content={"message": "hi"})
    obs._tick()

    try_orient.assert_called_once_with()


def test_core_authored_events_do_not_trigger(tmp_path):
    obs, log, try_orient = make(tmp_path)
    obs.start()
    obs.stop(timeout=1)

    # The agent's own decision output is not a stimulus to wake for.
    log.append(actor="Tam", type="decision", content={"text": "thinking"})
    obs._tick()

    try_orient.assert_not_called()


def test_same_event_triggers_only_once(tmp_path):
    obs, log, try_orient = make(tmp_path)
    obs.start()
    obs.stop(timeout=1)

    log.append(actor="user", type="chat_message", content={"message": "hi"})
    obs._tick()
    obs._tick()  # second wake, nothing new since the checkpoint advanced

    assert try_orient.call_count == 1
```

- [ ] **Step 2: Run tests to verify they pass immediately**

Run: `poetry run pytest tests/test_time_observer.py -v -k "external or core_authored or only_once"`
Expected: PASS (these assert the `_tick` behaviour from Task 3).

Note: these are characterization tests for Task 3's implementation, so they pass without new production code. If any fails, fix `_tick` in `src/theseus/time_observer.py` to match the docstring contract, not the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_time_observer.py
git commit -m "test(time-observer): lock in heuristic — fire, ignore self, fire once"
```

---

### Task 5: `TimeObserver` thread lifecycle — prompt stop

**Files:**
- Test: `tests/test_time_observer.py`

- [ ] **Step 1: Write the lifecycle test**

Append to `tests/test_time_observer.py` (add `import time` to the top of the file with the other imports):

```python
def test_stop_interrupts_the_interval_promptly(tmp_path):
    # A 1000s interval would never wake on its own; stop() must not wait it out.
    obs, log, try_orient = make(tmp_path, interval=1000.0)
    obs.start()

    t0 = time.monotonic()
    obs.stop(timeout=2)
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0
    assert obs._thread is not None and not obs._thread.is_alive()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `poetry run pytest tests/test_time_observer.py::test_stop_interrupts_the_interval_promptly -v`
Expected: PASS (the `Event().wait` loop from Task 3 already supports interruptible shutdown).

If it hangs or fails, verify `_run` uses `while not self._stop.wait(self.interval_seconds)` and `stop()` calls `self._stop.set()`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_time_observer.py
git commit -m "test(time-observer): stop() interrupts the interval promptly"
```

---

### Task 6: Export `TimeObserver` from the package

**Files:**
- Modify: `src/theseus/__init__.py`

- [ ] **Step 1: Add the import**

In `src/theseus/__init__.py`, add after the `from theseus.stimulus_log import ...` line:

```python
from theseus.time_observer import TimeObserver
```

- [ ] **Step 2: Add to `__all__`**

In the `__all__` list in `src/theseus/__init__.py`, add `"TimeObserver"` after `"StimulusLog"`:

```python
    "StimulusLog",
    "TimeObserver",
```

- [ ] **Step 3: Verify the import resolves**

Run: `poetry run python -c "from theseus import TimeObserver; print(TimeObserver.__name__)"`
Expected: prints `TimeObserver`

- [ ] **Step 4: Commit**

```bash
git add src/theseus/__init__.py
git commit -m "feat: export TimeObserver as public API"
```

---

### Task 7: Retrofit chat observers onto the gate (wiring + web AC#4)

Both chat observers already accept an injected orient callback, so the retrofit is: wire them to `orient_and_wait`, and prove the web path never blocks its event loop.

**Files:**
- Modify: `src/theseus/agents/alty_mcgee.py`
- Modify: `src/theseus/web_chat_ui_observer.py` (docstring only)
- Test: `tests/test_web_chat_observer.py`

- [ ] **Step 1: Write the failing web test**

Create `tests/test_web_chat_observer.py`:

```python
from __future__ import annotations

import threading

from theseus.stimulus_log import StimulusLog
from theseus.web_chat_ui_observer import WebChatUIObserver


def test_submit_handler_never_blocks_on_the_cognitive_cycle(tmp_path):
    """AC#4: the event-loop-facing path must return without waiting for orient. The
    cycle runs on the per-message background thread, so even a callback parked (as it
    would be while a wait-on-contention acquire blocks) does not stall the handler the
    HTTP route awaits."""
    gate = threading.Event()
    ran = threading.Event()

    def parked_callback():
        gate.wait(timeout=2)  # stand-in for a blocking gate acquire
        ran.set()

    log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    obs = WebChatUIObserver(orient_chat_message_callback=parked_callback, stimulus_log=log)

    obs._handle_chat_submit("hi")  # must return promptly, cycle still parked
    assert not ran.is_set(), "handler waited for the cycle instead of backgrounding it"

    gate.set()
    assert ran.wait(2), "cycle never ran on the background thread"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `poetry run pytest tests/test_web_chat_observer.py -v`
Expected: PASS — this characterizes the existing `_handle_chat_submit` → background-thread behaviour that makes a blocking acquire safe. If it fails, the web observer's threading model changed and the blocking-acquire decision must be revisited before proceeding.

- [ ] **Step 3: Retrofit Alty's terminal wiring**

In `src/theseus/agents/alty_mcgee.py`, change the `TerminalChatObserver` wiring (currently `orient_chat_message_callback=self.core.orient`):

```python
        self.chat_observer = TerminalChatObserver(
            stimulus_log=stimulus_log,
            orient_chat_message_callback=self.core.orient_and_wait
        )
```

- [ ] **Step 4: Update the web observer docstring**

In `src/theseus/web_chat_ui_observer.py`, replace the paragraph in the class docstring that begins `Only one \`orient\` call is ever in flight at a time:` (lines ~55–59) with:

```python
    Only one cognitive cycle runs at a time, now enforced by the Core's cycle gate
    rather than by the disabled composer alone. Wire this observer's callback to
    `OODACore.orient_and_wait` (wait-on-contention): the callback runs on the
    per-message background `_run_core` thread, so its blocking acquire waits out any
    in-flight cycle without ever touching the event loop.
```

- [ ] **Step 5: Run the full offline suite**

Run: `make test`
Expected: PASS — no regression in existing single-observer behaviour (AC#6), plus the new concurrency, time-observer, and web-observer tests.

- [ ] **Step 6: Commit**

```bash
git add src/theseus/agents/alty_mcgee.py src/theseus/web_chat_ui_observer.py tests/test_web_chat_observer.py
git commit -m "feat(observers): route chat observers through orient_and_wait; prove web path non-blocking"
```

---

## Acceptance criteria → coverage map

- **AC#1** (TimeObserver fires on schedule when the heuristic says to): `test_new_external_event_triggers_orient` + `test_stop_interrupts_the_interval_promptly` (interval loop). No live end-to-end timing test — the interval loop and the heuristic are covered separately, deliberately (offline suite).
- **AC#2** (in-flight cycle → TimeObserver wake is a no-op): `test_try_orient_skips_while_a_cycle_is_in_flight` + TimeObserver routes through `try_orient`.
- **AC#3** (chat attempt eventually runs, input never dropped): `test_orient_and_wait_blocks_then_runs_after_the_in_flight_cycle`; the terminal observer appends the stimulus *before* calling the callback, so the input is logged regardless.
- **AC#4** (web attempt runs without blocking the event loop): `test_submit_handler_never_blocks_on_the_cognitive_cycle` + the background-thread design.
- **AC#5** (two observers at once → one cycle at a time, loser per policy): `test_never_two_cycles_at_once` (mutual exclusion) + `test_try_orient_skips...` (skip loser) + `test_orient_and_wait_blocks...` (wait loser).
- **AC#6** (no single-observer regression): `make test` stays green; `orient()` is unchanged and un-gated.

## Consumer follow-up (out of this repo)

After releasing a Theseus tag with this change, consumer agents (e.g. Tam) that wire their observers to `core.orient` must switch to `core.orient_and_wait`, and any consumer that wants periodic orientation constructs a `TimeObserver(log, core.try_orient, self_actor=core.name)` and calls `.start()`. Not part of this plan.
