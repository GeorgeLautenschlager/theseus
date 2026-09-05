from __future__ import annotations

import threading
import time

import pytest

from theseus.replication_ingress import CoalescingTrigger

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
