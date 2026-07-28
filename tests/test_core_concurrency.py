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
