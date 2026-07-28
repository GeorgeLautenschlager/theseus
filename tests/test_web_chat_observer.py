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
