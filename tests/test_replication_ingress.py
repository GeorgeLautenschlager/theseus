from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from theseus.high_water import HighWaterMarks
from theseus.replication_events import GAP
from theseus.replication_ingress import (
    CoalescingTrigger,
    ReentrantIngest,
    ReplicationIngress,
)
from theseus.stimulus_log import StimulusLog

# Every wait in this file is bounded. A threading test that can hang is a threading test
# that will hang in CI at the worst moment, and a timeout that fires is a failure with a
# name rather than a build that never finishes.
TIMEOUT = 5.0


class Recorder:
    """A callback that records its calls and can be held open on demand.

    `entered` fires when a call begins; `gate` holds the call there until released. That
    is what makes the coalescing test deterministic rather than a race against a sleep.
    """

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.threads: list[int] = []
        self.entered = threading.Event()
        self.gate = threading.Event()
        self.gate.set()
        self._lock = threading.Lock()

    def __call__(self) -> None:
        with self._lock:
            self.calls.append(len(self.calls) + 1)
            self.threads.append(threading.get_ident())
        self.entered.set()
        assert self.gate.wait(TIMEOUT), "callback was held open past the timeout"

    def wait_for_calls(self, count: int) -> bool:
        deadline = time.monotonic() + TIMEOUT
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.calls) >= count:
                    return True
            time.sleep(0.005)
        return False


@pytest.fixture
def recorder():
    return Recorder()


@pytest.fixture
def trigger(recorder):
    trigger = CoalescingTrigger(recorder)
    trigger.start()
    yield trigger
    trigger.stop()


def test_a_request_produces_a_callback(trigger, recorder):
    trigger.request()

    assert recorder.wait_for_calls(1), "no callback within the timeout"


def test_the_callback_does_not_run_on_the_requesting_thread(trigger, recorder):
    """`request()` is called from a FastAPI handler. Running a cognitive cycle there would
    block the request the surrogate is waiting on, and on an event loop it would block
    every other request too."""
    trigger.request()
    assert recorder.wait_for_calls(1)

    assert recorder.threads[0] != threading.get_ident()


def test_a_burst_of_requests_collapses_into_one_further_callback(trigger, recorder):
    """The point of the class. A surrogate draining a backlog sends batches with no
    inter-batch delay; fifty requests must not become fifty cognitive cycles.

    Deterministic by construction: the first callback is held open, the burst arrives while
    it is held, and `stop()` joins the worker — so the count is taken when no further call
    is possible, not after a sleep and a hope."""
    recorder.gate.clear()
    trigger.request()
    assert recorder.entered.wait(TIMEOUT), "first callback never started"

    for _ in range(50):
        trigger.request()
    recorder.gate.set()

    assert recorder.wait_for_calls(2)
    trigger.stop()
    assert len(recorder.calls) == 2, f"50 requests became {len(recorder.calls)} callbacks"


def test_a_request_during_a_callback_is_not_lost(trigger, recorder):
    """The flag is cleared before the callback, not after. Clearing after would wipe a
    request that arrived mid-call — leaving a batch on the tape that nothing looks at until
    something else happens to arrive."""
    recorder.gate.clear()
    trigger.request()
    assert recorder.entered.wait(TIMEOUT)

    trigger.request()  # arrives while the first call is still running
    recorder.gate.set()

    assert recorder.wait_for_calls(2), "the request made during the callback was lost"


def test_a_callback_that_raises_does_not_kill_the_worker():
    """A core that raised — model endpoint down, bad tool result — must not take the
    ingress's thread with it. The events are committed either way, and the next arrival
    gets a fresh attempt."""
    calls = []
    boom = threading.Event()

    def failing():
        calls.append(1)
        boom.set()
        raise RuntimeError("the model endpoint is down")

    trigger = CoalescingTrigger(failing)
    trigger.start()
    try:
        trigger.request()
        assert boom.wait(TIMEOUT)

        boom.clear()
        trigger.request()
        assert boom.wait(TIMEOUT), "the worker died on the first exception"
    finally:
        trigger.stop()

    assert len(calls) == 2


def test_stop_waits_for_a_callback_already_running(recorder):
    """`stop` joins. A shutdown that returned while a cognitive cycle was still writing
    would race the process teardown against the log."""
    trigger = CoalescingTrigger(recorder)
    trigger.start()
    recorder.gate.clear()
    trigger.request()
    assert recorder.entered.wait(TIMEOUT)

    done = threading.Event()
    threading.Thread(target=lambda: (trigger.stop(), done.set()), daemon=True).start()
    assert not done.wait(0.1), "stop returned while the callback was still running"

    recorder.gate.set()
    assert done.wait(TIMEOUT), "stop did not return after the callback finished"


def test_stop_is_safe_before_start_and_twice():
    trigger = CoalescingTrigger(lambda: None)

    trigger.stop()
    trigger.start()
    trigger.stop()
    trigger.stop()


def test_start_twice_does_not_run_two_workers(recorder):
    """Two workers would defeat the whole purpose: each takes the flag and calls, so a
    burst becomes two bursts and `stop` joins only the one it last recorded, leaking the
    other for the life of the process.

    Asserted structurally, by counting live threads, rather than by counting callbacks. A
    callback-counting version of this test *passes against the bug*: `Thread.start()`
    returns before the new worker reaches its first `wait()`, so the second worker usually
    misses the first `set()` entirely and the totals come out looking correct. Measured —
    it was written that way first, and the mutation gate caught it. What is actually wrong
    with two workers is that there are two, so that is what this counts.
    """
    trigger = CoalescingTrigger(recorder, name="trigger-under-test")
    trigger.start()
    trigger.start()
    try:
        live = [t for t in threading.enumerate() if t.name == "trigger-under-test"]
        assert len(live) == 1, f"{len(live)} workers running after two starts"
    finally:
        trigger.stop()

    assert [t for t in threading.enumerate() if t.name == "trigger-under-test"] == []


def test_no_callback_without_a_request(trigger, recorder):
    """The worker waits; it does not poll and it does not fire on start."""
    assert not recorder.entered.wait(0.2)
    assert recorder.calls == []


# --- The ingress ----------------------------------------------------------------
SURROGATE = "kitchen-surrogate"


def wire(seq: int, *, origin: str = SURROGATE, **overrides) -> str:
    fields = {
        "id": f"01PRODUCERID{seq:014d}",
        "ts": f"2026-09-04T16:00:{seq % 60:02d}+00:00",
        "actor": "sensor",
        "type": "observation",
        "content": {"n": seq},
        "origin": origin,
        "seq": seq,
    }
    fields.update(overrides)
    return json.dumps(fields)


def batch(*seqs: int, origin: str = SURROGATE) -> str:
    return "\n".join(wire(seq, origin=origin) for seq in seqs) + "\n"


@pytest.fixture
def log(tmp_path):
    return StimulusLog(path=tmp_path / "stimulus_log.jsonl")


@pytest.fixture
def ingress(log):
    return ReplicationIngress(log, HighWaterMarks(log))


def seqs_on(log, origin=SURROGATE):
    return [e.seq for e in log.read_all() if e.origin == origin]


# --- The three dedupe cases, end to end -----------------------------------------
def test_a_new_batch_is_committed(ingress, log):
    result = ingress.ingest(batch(1, 2, 3))

    assert result.status == 200
    assert result.appended == 3
    assert result.high_water == 3
    assert seqs_on(log) == [1, 2, 3]


def test_a_duplicate_batch_commits_nothing_and_is_still_a_2xx(ingress, log):
    """A lost ack — the host committed, the response died in flight, the surrogate resent.
    Its job on retry is to stop worrying, not to find out it was wrong."""
    ingress.ingest(batch(1, 2, 3))

    result = ingress.ingest(batch(1, 2, 3))

    assert result.status == 200
    assert result.duplicate
    assert result.appended == 0
    assert result.high_water == 3
    assert seqs_on(log) == [1, 2, 3]


def test_a_straddling_batch_commits_only_the_tail(ingress, log):
    ingress.ingest(batch(1, 2, 3))

    result = ingress.ingest(batch(2, 3, 4, 5))

    assert result.appended == 2
    assert seqs_on(log) == [1, 2, 3, 4, 5]
    assert result.high_water == 5


def test_a_jump_is_committed_with_a_marker_and_never_rejected(ingress, log):
    ingress.ingest(batch(1))

    result = ingress.ingest(batch(5, 6))

    assert result.status == 200
    assert result.inferred_holes == ((2, 4),)
    assert result.appended == 3  # the marker, then the two events
    assert seqs_on(log) == [1, 5, 6]

    marker = [e for e in log.read_all() if e.type == GAP][0]
    assert marker.origin == log.origin
    assert marker.content["origin"] == SURROGATE
    assert (marker.content["from_seq"], marker.content["to_seq"]) == (2, 4)
    assert marker.content["declared"] is False


def test_the_marker_and_its_events_land_in_one_write(ingress, log):
    """One `append_many`, one fsync — so the marker and the events it explains share an
    arrival instant. Two writes would let a crash commit the hole and lose the
    explanation."""
    ingress.ingest(batch(5, 6))

    events = log.read_all()
    assert [e.type for e in events] == [GAP, "observation", "observation"]
    assert len({e.appended_ts for e in events}) == 1


# --- The lower bound of an inferred span ----------------------------------------
def _marker(log, about: str = SURROGATE):
    return [e for e in log.read_all() if e.type == GAP and e.content["origin"] == about][0]


def test_a_second_batchs_inferred_gap_spans_from_the_first_batches_last_event(ingress, log):
    """The host knows where this origin's stream last arrived: the `ts` of its last
    committed event. The gap's span starts there — a lower bound the host actually has,
    not a guess."""
    ingress.ingest(batch(1, 2))
    ingress.ingest(batch(5))

    events = {e.seq: e for e in log.read_all() if e.origin == SURROGATE}
    marker = _marker(log)

    assert marker.content["span_start"] == events[2].ts.isoformat()
    assert marker.content["span_end"] == events[5].ts.isoformat()
    assert marker.content["span_start"] != marker.content["span_end"]


def test_a_first_batchs_inferred_gap_stays_zero_width(ingress, log):
    """Nothing committed from the origin means no lower bound is known. The zero-width span
    says exactly that — the remembered clock must not fabricate a bound where there is
    none."""
    ingress.ingest(batch(5))

    marker = _marker(log)

    assert marker.content["span_start"] == marker.content["span_end"]


def test_the_remembered_clock_is_per_origin(ingress, log):
    """One clock per origin, not one per ingress. A hole in beta's stream spans from beta's
    own last event — not from whatever alpha committed most recently."""
    ingress.ingest(batch(1, origin="beta"))
    ingress.ingest(batch(1, 2, origin="alpha"))  # the latest commit overall is alpha's

    ingress.ingest(batch(5, origin="beta"))

    beta = {e.seq: e for e in log.read_all() if e.origin == "beta"}

    assert _marker(log, about="beta").content["span_start"] == beta[1].ts.isoformat()


def test_a_stale_partial_retry_does_not_move_the_remembered_clock(ingress, log):
    """A lost-ack retry that re-sends only the head of an already-committed batch carries
    older timestamps than the mark. If a duplicate rewrote the remembered clock, the next
    gap's span would start from seq 1's ts instead of seq 3's.

    The old version of this test was vacuous: `wire()` derives `ts` from `seq`, so a full
    duplicate carries identical timestamps and the question had no observable answer — it
    passed even with the clock written outside the `to_append` guard. A stale partial retry
    is what makes the two implementations observably different."""
    ingress.ingest(batch(1, 2, 3))
    ingress.ingest(batch(1))  # wholly below the mark: appends nothing

    ingress.ingest(batch(9))  # hole (4, 8)

    events = {e.seq: e for e in log.read_all() if e.origin == SURROGATE}

    marker = _marker(log)
    assert (marker.content["from_seq"], marker.content["to_seq"]) == (4, 8)
    assert marker.content["span_start"] == events[3].ts.isoformat()


def test_two_ingresses_over_one_log_share_the_remembered_clock(log):
    """The lower bound of an inferred span is per-log state, not per-ingress. Commit
    through ingress A, then drive a gap through ingress B: the minted span must start from
    what A committed — not from nothing."""
    marks = HighWaterMarks(log)
    first = ReplicationIngress(log, marks)
    second = ReplicationIngress(log, marks)

    first.ingest(batch(1, 2))
    second.ingest(batch(5))

    events = {e.seq: e for e in log.read_all() if e.origin == SURROGATE}

    assert _marker(log).content["span_start"] == events[2].ts.isoformat()


def test_a_host_minted_marker_is_not_the_origins_clock(ingress, log):
    """A marker's own `ts` is the host's clock — when the hole was noticed. Remembering it
    as the origin's clock would make every later span start from a moment on the host,
    not a moment in the origin's stream."""
    ingress.ingest(batch(1, 2))
    ingress.ingest(batch(5, 6))  # mints the (3, 4) gap, appends 5 and 6

    ingress.ingest(batch(9))  # hole (7, 8): its span must start at seq 6's ts

    events = {e.seq: e for e in log.read_all() if e.origin == SURROGATE}
    markers = [e for e in log.read_all() if e.type == GAP]

    assert (markers[-1].content["from_seq"], markers[-1].content["to_seq"]) == (7, 8)
    assert markers[-1].content["span_start"] == events[6].ts.isoformat()


# --- Rejection ------------------------------------------------------------------
def test_a_malformed_batch_is_a_4xx_and_commits_nothing(ingress, log):
    result = ingress.ingest("{not json\n")

    assert result.status == 400
    assert result.reason
    assert log.read_all() == []


def test_an_oversized_batch_carries_its_own_status(log):
    """413 rather than 400. Both are 4xx and both mean do not retry, but the surrogate
    copies the reason onto its own tape, and "too many events" tells an operator to lower a
    limit where "malformed" would send them looking for a bug."""
    ingress = ReplicationIngress(log, HighWaterMarks(log), max_events=2)

    result = ingress.ingest(batch(1, 2, 3))

    assert result.status == 413
    assert log.read_all() == []


def test_a_rejection_does_not_move_the_mark(ingress, log):
    ingress.ingest(batch(1, 2))

    ingress.ingest("{not json\n")

    assert ingress.ingest(batch(1, 2)).high_water == 2


def test_a_batch_claiming_the_hosts_own_origin_is_a_4xx(ingress, log):
    """Not a 5xx. Without the parser's guard this surfaced as a ValueError out of
    `append_many`, which the endpoint would answer as transient and the surrogate would
    retry forever over a misconfiguration no retry can fix."""
    result = ingress.ingest(batch(1, origin=log.origin))

    assert result.status == 400
    assert log.read_all() == []


# --- The mark is derived from the tape, not remembered beside it ----------------
def test_the_mark_survives_a_restart(ingress, log):
    """`HighWaterMarks` is a snapshot recovered from the log. If the ingress ever advanced
    a mark without writing, or wrote without advancing, a fresh instance would disagree —
    and disagreeing means either dropping real events or double-appending them."""
    ingress.ingest(batch(1, 2, 3))
    ingress.ingest(batch(7, 8))

    recovered = HighWaterMarks(log)

    assert recovered.high_water(SURROGATE) == 8


# --- The trigger ----------------------------------------------------------------
def test_a_committed_batch_asks_the_agent_to_think(log):
    woken = threading.Event()
    ingress = ReplicationIngress(log, HighWaterMarks(log), on_arrival=woken.set)
    ingress.start()
    try:
        ingress.ingest(batch(1))
        assert woken.wait(5.0)
    finally:
        ingress.stop()


def test_a_duplicate_does_not_ask_the_agent_to_think(log):
    """A lost ack is the surrogate re-asking a question already answered. Waking the agent
    for it would turn a dropped response into a cognitive turn about nothing."""
    calls = []
    ingress = ReplicationIngress(log, HighWaterMarks(log), on_arrival=lambda: calls.append(1))
    ingress.start()
    try:
        ingress.ingest(batch(1))
        time.sleep(0.2)
        before = len(calls)

        ingress.ingest(batch(1))
        time.sleep(0.2)

        assert len(calls) == before
    finally:
        ingress.stop()


def test_a_rejected_batch_does_not_ask_the_agent_to_think(log):
    calls = []
    ingress = ReplicationIngress(log, HighWaterMarks(log), on_arrival=lambda: calls.append(1))
    ingress.start()
    try:
        ingress.ingest("{not json\n")
        time.sleep(0.2)

        assert calls == []
    finally:
        ingress.stop()


def test_an_ingress_without_a_callback_still_works(ingress):
    ingress.start()
    try:
        assert ingress.ingest(batch(1)).appended == 1
    finally:
        ingress.stop()


# --- Concurrency ----------------------------------------------------------------
def test_two_concurrent_deliveries_of_one_batch_commit_it_once(log, monkeypatch):
    """The failure this guards is not hypothetical and not the surrogate's fault: the spec
    allows one batch in flight, but an HTTP client that times out and retries puts two
    concurrent requests for one origin in front of the ingress while the first is still
    committing. If reading the mark, planning against it, writing and advancing are not one
    atomic sequence, both requests read the same mark, both plan a full commit, and the
    batch lands twice in the agent's permanent memory.

    Made deterministic by widening the window rather than hoping to hit it: the write is
    slowed, and both threads are released together by a barrier, so an unlocked ingress
    reliably interleaves. Verified against the unlocked implementation — see the plan's
    mutation gate."""
    marks = HighWaterMarks(log)
    ingress = ReplicationIngress(log, marks)

    real_append_many = log.append_many

    def slow_append_many(events):
        time.sleep(0.1)
        return real_append_many(events)

    monkeypatch.setattr(log, "append_many", slow_append_many)

    start = threading.Barrier(2)
    results = []

    def deliver():
        start.wait(timeout=5.0)
        results.append(ingress.ingest(batch(1, 2, 3)))

    threads = [threading.Thread(target=deliver) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert seqs_on(log) == [1, 2, 3], "the batch was committed more than once"
    assert sorted(r.appended for r in results) == [0, 3]
    assert all(r.status == 200 for r in results)


# --- Re-entrant ingest: a listener that forwards what it receives ---------------
def test_a_listener_that_ingests_raises_instead_of_deadlocking(log):
    """The relay bug, bounded. A listener that forwards what it receives calls `ingest`
    from inside the commit — on the thread that holds `_commit_lock`. Before the fix that
    blocked forever, so the outer ingest runs on a worker thread joined with a timeout:
    a wrong implementation fails by hitting the bound rather than hanging the suite."""
    ingress = ReplicationIngress(log, HighWaterMarks(log))
    errors: list[Exception] = []

    def relay(event):
        try:
            ingress.ingest(batch(4, 5))
        except Exception as error:
            errors.append(error)

    unsubscribe = log.subscribe(relay)
    # Daemon: a wrong implementation leaves this thread blocked forever, and a live
    # non-daemon thread would hang the suite at teardown — the timeout must be the
    # whole cost of a deadlock, not a build that never finishes.
    worker = threading.Thread(target=lambda: ingress.ingest(batch(1, 2, 3)), daemon=True)
    worker.start()
    worker.join(TIMEOUT)
    assert not worker.is_alive(), "the re-entrant ingest deadlocked instead of raising"
    unsubscribe()

    assert errors, "the re-entrant ingest never raised"
    assert all(isinstance(error, ReentrantIngest) for error in errors), (
        f"expected ReentrantIngest from the re-entry, got {errors!r}"
    )


def test_a_listener_ingesting_through_a_second_ingress_raises_instead_of_deadlocking(log):
    """The re-entry guard must live where the lock lives. With it on the ingress, two
    ingresses over one marks object each keep their own holder flag: a listener calling
    the *second* ingress's `ingest` sees None on its instance, passes the re-entry check,
    and blocks forever on the lock the first holds. Bounded as before — a wrong
    implementation fails by hitting the join timeout rather than hanging the suite."""
    marks = HighWaterMarks(log)
    first = ReplicationIngress(log, marks)
    second = ReplicationIngress(log, marks)
    errors: list[Exception] = []

    def relay(event):
        try:
            second.ingest(batch(4, 5))
        except Exception as error:
            errors.append(error)

    unsubscribe = log.subscribe(relay)
    worker = threading.Thread(target=lambda: first.ingest(batch(1, 2, 3)), daemon=True)
    worker.start()
    worker.join(TIMEOUT)
    assert not worker.is_alive(), "the re-entrant ingest deadlocked instead of raising"
    unsubscribe()

    assert errors, "the re-entrant ingest never raised"
    assert all(isinstance(error, ReentrantIngest) for error in errors), (
        f"expected ReentrantIngest from the re-entry, got {errors!r}"
    )


def test_the_reentry_error_names_the_cause(log):
    ingress = ReplicationIngress(log, HighWaterMarks(log))
    errors: list[Exception] = []

    def relay(event):
        try:
            ingress.ingest(batch(4, 5))
        except Exception as error:
            errors.append(error)

    unsubscribe = log.subscribe(relay)
    worker = threading.Thread(target=lambda: ingress.ingest(batch(1, 2, 3)), daemon=True)
    worker.start()
    worker.join(TIMEOUT)
    assert not worker.is_alive(), "the re-entrant ingest deadlocked instead of raising"
    unsubscribe()

    message = " ".join(str(error).lower() for error in errors)
    assert "listener" in message and "re-entrant" in message, (
        f"the re-entry error must name the cause, got: {errors!r}"
    )


def test_the_reentry_does_not_uncommit_the_batch_that_caused_it(log):
    """The relay firing is a bug in the listener, not a reason to un-commit. The batch
    that notified it is on disk and its mark advanced: the write happened before the
    notification, and the log swallows listener errors by design."""
    marks = HighWaterMarks(log)
    ingress = ReplicationIngress(log, marks)

    def relay(event):
        ingress.ingest(batch(4, 5))

    unsubscribe = log.subscribe(relay)
    worker = threading.Thread(target=lambda: ingress.ingest(batch(1, 2, 3)), daemon=True)
    worker.start()
    worker.join(TIMEOUT)
    assert not worker.is_alive(), "the re-entrant ingest deadlocked"
    unsubscribe()

    assert seqs_on(log) == [1, 2, 3], "the batch that notified the relay was un-committed"
    assert marks.high_water(SURROGATE) == 3, "the mark did not advance past the committed batch"


# --- The HTTP surface -----------------------------------------------------------
def test_the_endpoint_answers_2xx_for_a_committed_batch(ingress, log):
    client = TestClient(ingress.build_app())

    response = client.post("/replicate", content=batch(1, 2))

    assert response.status_code == 200
    assert response.json()["high_water"] == 2
    assert seqs_on(log) == [1, 2]


def test_the_endpoint_answers_4xx_with_the_reason(ingress):
    client = TestClient(ingress.build_app())

    response = client.post("/replicate", content="{not json\n")

    assert response.status_code == 400
    assert "not JSON" in response.json()["reason"]


def test_the_endpoint_can_be_mounted_on_an_existing_app(ingress, log):
    """A host already serving a chat UI should not need a second port for this."""
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    ingress.add_routes(app, path="/surrogate/replicate")
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.post("/surrogate/replicate", content=batch(1)).status_code == 200
    assert seqs_on(log) == [1]


def test_the_endpoint_reports_an_inferred_gap(ingress):
    client = TestClient(ingress.build_app())
    client.post("/replicate", content=batch(1))

    response = client.post("/replicate", content=batch(5))

    assert response.json()["inferred_gaps"] == [{"from_seq": 2, "to_seq": 4}]


def test_a_failed_write_does_not_move_the_mark(log, monkeypatch):
    """The ordering only matters when the write fails, which is the case worth having.

    A mark claiming events the log does not have silently discards the retry that would
    have delivered them — the surrogate resends, the host says "already got it", and the
    events are gone with nobody the wiser. If the process dies between the write and the
    advance instead, the mark is merely behind: the retry re-delivers a batch the host
    already has, dedupe answers 2xx, and nothing is written twice. Behind is recoverable;
    ahead is not."""
    marks = HighWaterMarks(log)
    ingress = ReplicationIngress(log, marks)
    ingress.ingest(batch(1, 2))

    def failing(events):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(log, "append_many", failing)
    with pytest.raises(OSError):
        ingress.ingest(batch(3, 4))
    monkeypatch.undo()

    assert marks.high_water(SURROGATE) == 2, "the mark moved for a write that never landed"
    assert ingress.ingest(batch(3, 4)).appended == 2, "the retry was discarded as a duplicate"


def test_the_endpoint_does_not_block_the_event_loop(log, monkeypatch):
    """`ingest` blocks on a lock and an fsync. Run on the event loop it would stall every
    other request in the process — the chat UI included, when they share an app — for the
    length of a disk write.

    Raced rather than inspected: a slow replicate and a prompt health check are issued
    together, and the health check must come back first. On the loop it cannot."""
    import asyncio

    import httpx

    ingress = ReplicationIngress(log, HighWaterMarks(log))
    real_ingest = ingress.ingest

    def slow_ingest(body):
        time.sleep(0.5)
        return real_ingest(body)

    monkeypatch.setattr(ingress, "ingest", slow_ingest)

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

    ingress.add_routes(app)

    async def race():
        order = []
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://ingress") as client:

            async def replicate():
                await client.post("/replicate", content=batch(1))
                order.append("replicate")

            async def check():
                await asyncio.sleep(0.05)  # let the slow request get in first
                await client.get("/health")
                order.append("health")

            await asyncio.gather(replicate(), check())
        return order

    assert asyncio.run(race()) == ["health", "replicate"]


def test_two_ingresses_over_one_log_do_not_double_append(log, monkeypatch):
    """`HighWaterMarks`' own docstring makes one instance per log the precondition, so
    sharing one marks object between two ingresses is the documented-correct arrangement —
    and it is exactly what a per-ingress lock breaks. Measured before the fix: the log came
    back holding seqs [1, 2, 3, 1, 2, 3]."""
    marks = HighWaterMarks(log)
    first = ReplicationIngress(log, marks)
    second = ReplicationIngress(log, marks)

    real_append_many = log.append_many

    def slow_append_many(events):
        time.sleep(0.1)
        return real_append_many(events)

    monkeypatch.setattr(log, "append_many", slow_append_many)

    start = threading.Barrier(2)
    appended = []

    def deliver(ingress):
        start.wait(timeout=5.0)
        appended.append(ingress.ingest(batch(1, 2, 3)).appended)

    threads = [threading.Thread(target=deliver, args=(i,)) for i in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert seqs_on(log) == [1, 2, 3], "the batch was committed by both ingresses"
    assert sorted(appended) == [0, 3]


def test_a_stop_that_times_out_does_not_leak_a_second_worker(recorder):
    """A cognitive cycle can outlast the stop timeout. If `stop` forgot the worker anyway,
    the next `start` would add a second beside the one still running — the failure
    `test_start_twice_does_not_run_two_workers` guards, reached by another route."""
    trigger = CoalescingTrigger(recorder, name="stubborn-worker")
    trigger.start()
    recorder.gate.clear()
    trigger.request()
    assert recorder.entered.wait(TIMEOUT)

    trigger.stop(timeout=0.05)  # the callback is still held open, so this times out
    trigger.start()

    live = [t for t in threading.enumerate() if t.name == "stubborn-worker"]
    recorder.gate.set()
    trigger.stop()
    assert len(live) == 1, f"{len(live)} workers running after a timed-out stop and a start"


def test_a_restart_does_not_fire_a_callback_nobody_asked_for(recorder):
    """`stop` sets the pending flag to wake the worker. Left set, the next `start` would run
    a cognitive turn with no arrival behind it."""
    trigger = CoalescingTrigger(recorder)
    trigger.start()
    trigger.stop()

    trigger.start()
    try:
        assert not recorder.entered.wait(0.2)
        assert recorder.calls == []
    finally:
        trigger.stop()


def test_an_oversized_body_is_refused_before_it_is_buffered(log):
    """The declared length is checked before `request.body()` reads anything. Measured
    without it: a 256 MB POST against a 1 KB limit answered a correct 413 after growing the
    process by half a gigabyte."""
    ingress = ReplicationIngress(log, HighWaterMarks(log), max_bytes=1024)
    client = TestClient(ingress.build_app())

    response = client.post("/replicate", content="x" * 200_000)

    assert response.status_code == 413
    assert "declares" in response.json()["reason"]
    assert log.read_all() == []


def _asgi_post(app, chunks: list[bytes], *, content_length: int | None):
    """Drive the ASGI app directly so a test can declare one length and send another, and
    count exactly how many body bytes the endpoint pulled before it answered. `TestClient`
    cannot lie about a declared length for us, and an in-process transport may pump a
    generator ahead of the app, so "how far did it get" is only measurable here."""
    import asyncio

    headers = [(b"content-type", b"application/json")]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/replicate",
        "raw_path": b"/replicate",
        "query_string": b"",
        "headers": headers,
        "server": ("testserver", 80),
        "client": ("testclient", 54321),
    }
    pending = list(chunks)
    pulled = 0

    async def receive():
        nonlocal pulled
        if pending:
            chunk = pending.pop(0)
            pulled += len(chunk)
            return {"type": "http.request", "body": chunk, "more_body": bool(pending)}
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))

    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, json.loads(body), pulled


def test_a_chunked_oversized_body_is_rejected(log):
    """No Content-Length: httpx sends a generator body chunked, which is the request shape
    the declared-length fast path cannot see. Over the limit it is refused and nothing
    lands."""
    ingress = ReplicationIngress(log, HighWaterMarks(log), max_bytes=1024)
    client = TestClient(ingress.build_app())

    def chunks():
        for _ in range(3):
            yield b"x" * 512

    response = client.post("/replicate", content=chunks())

    assert response.status_code == 413
    assert log.read_all() == []


def test_a_body_that_lies_about_its_length_is_rejected(log):
    """Declares a length under the limit and sends more than it declared. The fast path
    sees the lie as honest; the streaming bound sees the bytes."""
    ingress = ReplicationIngress(log, HighWaterMarks(log), max_bytes=1024)

    status, payload, _ = _asgi_post(
        ingress.build_app(),
        [b"x" * 600] * 3,
        content_length=600,
    )

    assert status == 413
    assert payload["reason"]
    assert log.read_all() == []


def test_the_stream_stops_early_on_an_oversized_body(log):
    """The bound is on the read, not the answer: the moment the running total passes the
    limit the endpoint stops pulling. An oversized stream is never fully consumed."""
    ingress = ReplicationIngress(log, HighWaterMarks(log), max_bytes=1024)

    status, payload, pulled = _asgi_post(
        ingress.build_app(),
        [b"x" * 512] * 64,
        content_length=None,
    )

    assert status == 413
    assert pulled <= 1024 + 512, f"endpoint kept reading: pulled {pulled} of 32768 bytes"
    assert log.read_all() == []
