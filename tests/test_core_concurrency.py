from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from theseus.context_assembler import ContextAssembler
from theseus.ooda_core import OODACore
from theseus.stimulus_log import StimulusLog
from theseus.tools.tool import AssistantTurn


def make_core(tmp_path, provider):
    log = StimulusLog(path=tmp_path / "stimulus_log.jsonl")
    return OODACore(
        name="Tam",
        constitution="You are Tam.",
        persona="Direct.",
        context_assembler=ContextAssembler(stimulus_log=log),
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
