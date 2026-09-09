"""Tests for the CommandChannel seam (issue #34, Task 2)."""

from __future__ import annotations

import threading
from datetime import datetime, timezone

import pytest

from theseus.commands import command_content, command_type
from theseus.stimulus_log import StimulusEvent
from theseus.surrogates.command_channel import CommandChannel, MemoryCommandChannel


def _command(verb: str, n: int) -> StimulusEvent:
    return StimulusEvent(
        id=f"cmd-{n}",
        ts=datetime.now(timezone.utc),
        actor="host",
        type=command_type(verb),
        content=command_content(target="tam", payload={"n": n}),
    )


def test_doorbell_shape_drains_and_returns() -> None:
    ch = MemoryCommandChannel()
    offered = [_command("say", n) for n in range(3)]
    for e in offered:
        ch.offer(e)

    got: list[StimulusEvent] = []
    done = threading.Event()

    def drain() -> None:
        got.extend(ch.stream())
        done.set()

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    # Bounded: a stream() that blocks instead of returning hangs this join.
    assert done.wait(timeout=5.0), "doorbell stream() did not return"
    assert got == offered


def test_live_shape_blocks_until_closed() -> None:
    ch = MemoryCommandChannel(block=True)
    ch.offer(_command("say", 1))

    got: list[StimulusEvent] = []
    first_seen = threading.Event()

    def consume() -> None:
        for e in ch.stream():
            got.append(e)
            first_seen.set()

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    assert first_seen.wait(timeout=5.0)
    # Offers have stopped; the stream must still be alive, waiting for more.
    assert t.is_alive()

    ch.offer(_command("say", 2))
    second_seen = threading.Event()

    def wait_second() -> None:
        deadline = datetime.now(timezone.utc).timestamp() + 5
        while len(got) < 2 and datetime.now(timezone.utc).timestamp() < deadline:
            pass
        second_seen.set()

    threading.Thread(target=wait_second, daemon=True).start()
    assert second_seen.wait(timeout=5.0), "blocking stream missed a later offer"

    ch.close()
    t.join(timeout=5.0)
    assert not t.is_alive(), "stream did not end after close()"
    assert [e.id for e in got] == ["cmd-1", "cmd-2"]


def test_close_from_another_thread_ends_stream_promptly() -> None:
    ch = MemoryCommandChannel(block=True)
    ended = threading.Event()

    def consume() -> None:
        list(ch.stream())
        ended.set()

    t = threading.Thread(target=consume, daemon=True)
    t.start()
    ch.close()
    # A poll-with-sleep implementation on a long interval fails this bound.
    assert ended.wait(timeout=1.0), "blocking stream did not end promptly on close()"


def test_nothing_yielded_twice() -> None:
    ch = MemoryCommandChannel()
    first = _command("say", 1)
    ch.offer(first)
    assert list(ch.stream()) == [first]
    assert list(ch.stream()) == []


def test_second_stream_resumes_with_new_offers() -> None:
    ch = MemoryCommandChannel()
    a, b = _command("say", 1), _command("say", 2)
    ch.offer(a)
    assert list(ch.stream()) == [a]
    ch.offer(b)
    assert list(ch.stream()) == [b]


def test_protocol_satisfied() -> None:
    assert isinstance(MemoryCommandChannel(), CommandChannel)

    class NotAChannel:
        pass

    assert not isinstance(NotAChannel(), CommandChannel)


def test_offer_after_close_raises() -> None:
    ch = MemoryCommandChannel()
    ch.close()
    with pytest.raises(RuntimeError):
        ch.offer(_command("say", 1))
