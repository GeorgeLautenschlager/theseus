"""The command channel round trip (issue #34, Task 5): host log → `CommandFeed` over a
real socket → surrogate's `CommandChannel` consumer, and the seam a composer needs.

The caller's loop is the thing under test: every test consumes through
`_caller_loop`, which takes a command, "executes" it, and only then advances the
cursor — the at-least-once decision. A test that advanced before executing would pass
while describing the wrong protocol, so the order is pinned here, written once, and
run unchanged against both `MemoryCommandChannel` and `SseCommandChannel`.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import httpx
import pytest

import theseus
from theseus.command_feed import CommandFeed
from theseus.commands import command_content, command_type
from theseus.high_water import HighWaterMarks
from theseus.replication_ingress import ReplicationIngress
from theseus.stimulus_log import StimulusEvent, StimulusLog
from theseus.surrogates.command_channel import MemoryCommandChannel
from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.http_transport import HttpTransport
from theseus.surrogates.replicator import Replicator
from theseus.surrogates.retry import RetryBudget
from theseus.surrogates.sse_command_channel import SseCommandChannel

HOST = "host"
SURROGATE = "kitchen"
TARGET = "tam"

# Fast heartbeats/polls so every idle path is bounded under a second; deadline asserts
# bound them independently of the intervals.
FAST = {"heartbeat_seconds": 0.05, "poll_seconds": 0.01}
FAST_BUDGET = RetryBudget(base_seconds=0.01, multiplier=2.0, jitter=0.0)


def _command(n: int) -> StimulusEvent:
    """A host-issued command carrying seq `n` — what the wire would carry."""
    return StimulusEvent(
        id=f"cmd-{n}",
        ts=datetime.now(timezone.utc),
        actor="george",
        type=command_type("say"),
        content=command_content(target=TARGET, payload={"n": n}),
        seq=n,
    )


def _issue(log: StimulusLog, n: int) -> StimulusEvent:
    """Append one command through the host's log, so it carries the log's seq."""
    return log.append(
        "george", command_type("say"), command_content(target=TARGET, payload={"n": n})
    )


def _cursor(tmp_path, name: str = "commands.json") -> AckedCursor:
    return AckedCursor(tmp_path / name, f"{SURROGATE}-surrogate")


def _caller_loop(channel, cursor, executed: list[StimulusEvent], *, want: int) -> None:
    """The consumer's loop, written once and run against both channels.

    Execute, then advance — never the other way round: the cursor means *executed*.
    A channel never advances it; that is the at-least-once decision this issue took,
    and this loop is the test that pins the caller's side of it.
    """
    for event in channel.stream():
        executed.append(event)
        cursor.advance(event.seq)
        if len(executed) >= want:
            break


def _run(channel, cursor, *, want: int) -> list[StimulusEvent]:
    """`_caller_loop` on a daemon thread, bounded: a stream that never produces fails
    the wait instead of hanging the suite. The channel is closed before joining, since
    an SSE stream left mid-iteration holds uvicorn's shutdown."""
    executed: list[StimulusEvent] = []
    got_all = threading.Event()

    def run() -> None:
        try:
            _caller_loop(channel, cursor, executed, want=want)
        finally:
            got_all.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert got_all.wait(10.0), f"caller loop did not receive {want} commands in time"
    channel.close()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "caller loop did not end after close()"
    return executed


def _sse_channel(base: str, cursor: AckedCursor, **kwargs) -> SseCommandChannel:
    """An SSE channel over the real socket, bounded by FAST reconnects."""
    return SseCommandChannel(
        f"{base}/commands/{TARGET}", cursor, client=httpx.Client(),
        max_reconnects=0, budget=FAST_BUDGET, **kwargs,
    )


# --- the round trip ----------------------------------------------------------------

def test_command_issued_during_downtime_is_delivered_on_reconnect(tmp_path, serve) -> None:
    """Hours of silence is the normal case: commands appended to the host's log while
    **no** channel is connected are queued in the log, not lost — a fresh cursor
    reconnecting afterwards gets every one, in order."""
    log = StimulusLog(path=tmp_path / "host.jsonl", origin=HOST)
    issued = [_issue(log, n) for n in range(3)]

    base = serve(CommandFeed(log, **FAST).build_app())
    cursor = _cursor(tmp_path)
    executed = _run(_sse_channel(base, cursor), cursor, want=3)

    assert [e.seq for e in executed] == [e.seq for e in issued]
    assert [e.id for e in executed] == [e.id for e in issued]


def test_no_replay_of_already_executed_commands(tmp_path, serve) -> None:
    """Consume several commands, advancing the cursor after each as the caller's loop
    does, then reconnect — the `Last-Event-ID` resume serves only what is left, and
    nothing executed arrives twice."""
    log = StimulusLog(path=tmp_path / "host.jsonl", origin=HOST)
    for n in range(1, 6):
        _issue(log, n)

    base = serve(CommandFeed(log, **FAST).build_app())
    cursor = _cursor(tmp_path)

    first = _run(_sse_channel(base, cursor), cursor, want=3)
    assert [e.seq for e in first] == [1, 2, 3]
    assert cursor.acked_seq == 3

    second = _run(_sse_channel(base, cursor), cursor, want=2)
    assert [e.seq for e in second] == [4, 5]  # 1–3 never replayed


def test_interleaved_issue_and_consume_stays_in_order(tmp_path, serve) -> None:
    """Commands appended while a stream is live arrive in seq order, none missed —
    the feed's subscribe-before-replay rule serving exactly once."""
    log = StimulusLog(path=tmp_path / "host.jsonl", origin=HOST)
    _issue(log, 0)

    base = serve(CommandFeed(log, **FAST).build_app())
    cursor = _cursor(tmp_path)
    executed: list[StimulusEvent] = []
    got_all = threading.Event()

    def run() -> None:
        try:
            _caller_loop(_sse_channel(base, cursor), cursor, executed, want=6)
        finally:
            got_all.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    writer = threading.Thread(
        target=lambda: [_issue(log, n) for n in range(1, 6)], daemon=True
    )
    writer.start()
    writer.join(timeout=5.0)
    assert got_all.wait(10.0), "live stream missed a command issued beside it"

    thread.join(timeout=5.0)
    assert not thread.is_alive()

    assert [e.seq for e in executed] == list(range(1, 7))


# --- one caller, two channels -------------------------------------------------------

@pytest.fixture(params=["memory", "sse"])
def channel_and_issuer(request, tmp_path, serve):
    """Both transports behind the seam: `memory` offers in-process, `sse` over a real
    socket from a real host log. The caller's loop is unchanged between them."""
    cursor = _cursor(tmp_path, f"commands-{request.param}.json")
    if request.param == "memory":
        channel = MemoryCommandChannel()

        def issuer(n):
            event = _command(n)
            channel.offer(event)
            return event

        return channel, issuer, cursor

    log = StimulusLog(path=tmp_path / "host.jsonl", origin=HOST)
    base = serve(CommandFeed(log, **FAST).build_app())
    cursor = _cursor(tmp_path)
    channel = _sse_channel(base, cursor)
    return channel, lambda n: _issue(log, n), cursor


def test_second_implementation_needs_no_caller_change(channel_and_issuer) -> None:
    """The fifth acceptance box: the consumer loop is literally one loop, parametrised
    over the two channels, asserting the same commands come out of both."""
    channel, issuer, cursor = channel_and_issuer

    issued = [issuer(n) for n in range(1, 4)]
    executed = _run(channel, cursor, want=3)

    assert [e.seq for e in executed] == [e.seq for e in issued]


# --- independence of the two directions ----------------------------------------------

def test_the_two_directions_are_independent(tmp_path, serve) -> None:
    """A `Replicator` drain (surrogate → host) and a command stream (host → surrogate)
    against the **same host**, in one test, do not interfere: each side's events reach
    the other's log, and neither cursor moves the other."""
    host_log = StimulusLog(path=tmp_path / "host.jsonl", origin=HOST)
    marks = HighWaterMarks(host_log)
    app = ReplicationIngress(host_log, marks).build_app()
    CommandFeed(host_log, **FAST).add_routes(app)
    base = serve(app)

    surrogate = StimulusLog(path=tmp_path / "surrogate.jsonl", origin=SURROGATE)
    ticks = [
        surrogate.append("sensor", "test.tick", {"n": n}, ts=datetime.now(timezone.utc))
        for n in range(3)
    ]
    up_cursor = AckedCursor(tmp_path / "up.json", SURROGATE)
    result = Replicator(
        surrogate, HttpTransport(f"{base}/replicate", client=httpx.Client()), up_cursor
    ).drain()

    # Upstream: the surrogate's events reached the host's log, and the host's own
    # commands are the only thing the command feed serves for our target.
    assert result.events_attempted == 3 and result.acked_seq == ticks[-1].seq
    replicated = [e for e in host_log.read_all() if e.origin == SURROGATE]
    # Ids are minted per ingest; what arrives is the same tape: type, content, seq.
    assert [(e.type, e.content, e.seq) for e in replicated] == [
        (e.type, e.content, e.seq) for e in ticks
    ]
    assert up_cursor.acked_seq == ticks[-1].seq

    # Downstream: a command the host issued reaches the surrogate over the same app.
    command = host_log.append(
        "george", command_type("say"), command_content(target=TARGET, payload={"n": 99})
    )
    down_cursor = _cursor(tmp_path, "down.json")
    executed = _run(_sse_channel(base, down_cursor), down_cursor, want=1)

    assert [e.seq for e in executed] == [command.seq]
    assert down_cursor.acked_seq == command.seq
    # Neither cursor moved the other: upstream stopped at the last tick, downstream
    # at the last executed command.
    assert up_cursor.acked_seq == ticks[-1].seq
    assert down_cursor.acked_seq == command.seq


def test_disconnected_stream_leaves_no_listener(tmp_path, serve) -> None:
    """After the round trip ends, the host log's listener list is empty — a listener
    left behind on every dropped connection would grow with reconnects, the normal
    case on a flaky link."""
    log = StimulusLog(path=tmp_path / "host.jsonl", origin=HOST)
    _issue(log, 1)

    base = serve(CommandFeed(log, **FAST).build_app())
    cursor = _cursor(tmp_path)
    executed = _run(_sse_channel(base, cursor), cursor, want=1)
    assert len(executed) == 1

    # The server notices the drop on its next poll (0.01s here); bounded, so a feed
    # that never unsubscribes fails this wait instead of hanging the suite.
    deadline = time.monotonic() + 5.0
    while log._listeners and time.monotonic() < deadline:
        time.sleep(0.01)
    assert log._listeners == [], "disconnected stream left a listener behind"


# --- the exports a composer needs ----------------------------------------------------

def test_exports_are_reachable() -> None:
    from theseus import (  # noqa: F401
        CommandChannel,
        CommandFeed,
        MemoryCommandChannel,
        SseCommandChannel,
        command_content,
        command_target,
        command_type,
        is_command,
    )

    for name in (
        "CommandFeed",
        "CommandChannel",
        "MemoryCommandChannel",
        "SseCommandChannel",
        "command_type",
        "command_content",
        "is_command",
        "command_target",
    ):
        assert name in theseus.__all__, f"{name} missing from theseus.__all__"
        assert hasattr(theseus, name)
