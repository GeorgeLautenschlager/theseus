"""Tests for CommandFeed (issue #34, Task 3) — the host side of the command channel.

The host serves its own commands to one surrogate as SSE, with a `Last-Event-ID` cursor
for reconnect. Starlette 1.3's TestClient blocks until the ASGI app returns, and an SSE
stream never returns on its own, so `client.stream` would hang: these tests drive
`CommandFeed._stream` — the exact generator the route serves — through a real starlette
`Request` whose `receive` the test controls.

Every wait is bounded. A stream that stops producing raises instead of hanging the
suite, which this repo has been bitten by before.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from starlette.requests import Request

from theseus.command_feed import CommandFeed
from theseus.commands import command_content, command_type
from theseus.stimulus_log import StimulusEvent, StimulusLog

HOST = "host"
TARGET_A = "kitchen"
TARGET_B = "study"
# A replicated event's origin is the producer's own name, not this log's.
FOREIGN_ORIGIN = "kitchen-surrogate"
ACTOR = "george"
TS = datetime.now(tz=timezone.utc).replace(microsecond=0)

# Fast enough that idle-path tests (heartbeats, drains) stay under a second, with the
# deadline asserts still bounding them independently.
FAST = {"heartbeat_seconds": 0.05, "poll_seconds": 0.01}


def _rig(tmp_path, *, feed_kwargs=None):
    log = StimulusLog(path=tmp_path / "host.jsonl", origin=HOST)
    feed = CommandFeed(log, **(feed_kwargs or {}))
    return SimpleNamespace(log=log, feed=feed)


def _command(log, target, *, payload=None, seq=None, origin=None):
    """Append one command. Local unless `seq` and `origin` are both supplied, per
    `StimulusLog.append`'s replicated contract."""
    return log.append(
        ACTOR,
        command_type("say"),
        command_content(target=target, payload=payload if payload is not None else {}),
        origin=origin,
        seq=seq,
    )


def _event_of(frame: str) -> StimulusEvent:
    """Rebuild the StimulusEvent an SSE frame carries: the `data:` lines, joined by
    newline, are the event's JSON."""
    lines = [line[len("data: ") :] for line in frame.splitlines() if line.startswith("data: ")]
    return StimulusEvent.from_json("\n".join(lines), default_origin=HOST)


class _Receive:
    """A receive the test controls. It answers immediately, because
    `Request.is_disconnected` polls it under an already-cancelled scope — a receive that
    checkpoints gets cancelled before answering — and reports disconnect once the test
    flips the flag."""

    def __init__(self) -> None:
        self.disconnected = False

    async def __call__(self, *args, **kwargs):
        if self.disconnected:
            return {"type": "http.disconnect"}
        return {"type": "http.request", "body": b""}


class _Stream:
    """`feed._stream` driven with a controlled disconnect."""

    def __init__(self, feed: CommandFeed, target: str, *, last_event_id=None) -> None:
        headers = (
            []
            if last_event_id is None
            else [(b"last-event-id", str(last_event_id).encode())]
        )
        self.receive = _Receive()
        request = Request({"type": "http", "headers": headers}, self.receive)
        self.gen = feed._stream(request, target)

    async def next(self, *, deadline: float = 5.0) -> str:
        """The next frame, bounded — a stream that stops producing fails, never hangs."""
        return await asyncio.wait_for(self.gen.__anext__(), timeout=deadline)

    async def events(self, count: int, *, deadline: float = 5.0) -> list[str]:
        """`count` SSE event frames, skipping the retry hint and heartbeat comments."""
        frames: list[str] = []
        end = time.monotonic() + deadline
        while len(frames) < count:
            remaining = end - time.monotonic()
            assert remaining > 0, (
                f"stream did not produce {count} events in time (got {len(frames)})"
            )
            frame = await self.next(deadline=remaining)
            if frame.startswith(":") or frame.startswith("retry:"):
                continue
            frames.append(frame)
        return frames

    async def drain(self, *, deadline: float = 0.3) -> list[str]:
        """Consume frames until `deadline`, skipping comments; returns any event frames
        seen. Proves absence — that a quiet stream carries nothing but heartbeats."""
        frames: list[str] = []
        end = time.monotonic() + deadline
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return frames
            try:
                frame = await self.next(deadline=remaining)
            except (asyncio.TimeoutError, TimeoutError):
                return frames
            if not (frame.startswith(":") or frame.startswith("retry:")):
                frames.append(frame)

    async def close(self) -> None:
        """Flip disconnect and expect the generator to end (unsubscribing in its
        finally), not to keep producing."""
        self.receive.disconnected = True
        try:
            await asyncio.wait_for(self.gen.__anext__(), timeout=5.0)
        except (asyncio.TimeoutError, TimeoutError):
            raise AssertionError("generator did not end after disconnect")
        except StopAsyncIteration:
            return
        raise AssertionError("generator kept producing after disconnect")


# --- pending: the replay half ----------------------------------------------------


def test_pending_filters_by_target_origin_and_cursor(tmp_path):
    rig = _rig(tmp_path)
    a1 = _command(rig.log, TARGET_A)
    _command(rig.log, TARGET_B)
    rig.log.append(ACTOR, "exchange", {"prompt": "hi", "response": "yo"})
    a2 = _command(rig.log, TARGET_A)

    page = rig.feed.pending(TARGET_A, after=None)
    assert [e.seq for e in page] == [a1.seq, a2.seq]  # seq order, not file order surprises
    assert [e.origin for e in page] == [HOST, HOST]

    assert [e.seq for e in rig.feed.pending(TARGET_A, after=a1.seq)] == [a2.seq]
    assert rig.feed.pending(TARGET_A, after=a2.seq) == []


def test_pending_ignores_foreign_origin_events(tmp_path):
    """A replicated event is command-typed and addressed to the target — but it entered
    the log from somewhere else, and its `seq` is the producer's numbering, not this
    log's cursor. The feed serves only what this host issued."""
    rig = _rig(tmp_path)
    local = _command(rig.log, TARGET_A)
    replicated = _command(rig.log, TARGET_A, seq=1, origin=FOREIGN_ORIGIN)

    assert replicated.type.startswith("command.")  # the trap is real: it looks servable
    assert [e.seq for e in rig.feed.pending(TARGET_A, after=None)] == [local.seq]
    assert [e.seq for e in rig.feed.pending(TARGET_A, after=1)] == []


# --- the endpoint: replay + live, cursor semantics -------------------------------


def test_endpoint_replays_from_last_event_id(tmp_path):
    rig = _rig(tmp_path)
    a1 = _command(rig.log, TARGET_A)
    a2 = _command(rig.log, TARGET_A)

    async def main():
        stream = _Stream(rig.feed, TARGET_A, last_event_id=a1.seq)
        frames = await stream.events(1)
        await stream.close()
        return frames

    frames = asyncio.run(main())
    assert [_event_of(f).seq for f in frames] == [a2.seq]  # above the cursor, not at it
    assert f"id: {a2.seq}" in frames[0]  # the cursor the surrogate holds comes back


def test_missing_last_event_id_replays_everything(tmp_path):
    """Not "nothing" and not "from now": a command issued while the surrogate was
    rebooting is queued, not lost."""
    rig = _rig(tmp_path)
    a1 = _command(rig.log, TARGET_A)
    a2 = _command(rig.log, TARGET_A)

    async def main():
        stream = _Stream(rig.feed, TARGET_A)
        frames = await stream.events(2)
        await stream.close()
        return frames

    frames = asyncio.run(main())
    assert [_event_of(f).seq for f in frames] == [a1.seq, a2.seq]


def test_malformed_last_event_id_treated_as_absent(tmp_path):
    """The cursor arrives off the network; refusing to serve is worse than replaying.
    `banana` is a fresh cursor, not a 4xx."""
    rig = _rig(tmp_path)
    a1 = _command(rig.log, TARGET_A)
    a2 = _command(rig.log, TARGET_A)

    async def main():
        stream = _Stream(rig.feed, TARGET_A, last_event_id="banana")
        frames = await stream.events(2)
        await stream.close()
        return frames

    frames = asyncio.run(main())
    assert [_event_of(f).seq for f in frames] == [a1.seq, a2.seq]


def test_command_appended_during_replay_is_not_dropped(tmp_path):
    """The ordering trap. The hook appends a command between the stream's subscribe and
    its replay read — the one window a read-first implementation leaves empty. With
    subscribe-first the command is on disk before the replay read and served once; with
    read-first it lands in neither the read nor the subscription and is served zero
    times. The hook appends *before* registering the stream's own listener precisely so
    the queue cannot rescue a swapped order.
    """
    rig = _rig(tmp_path, feed_kwargs=FAST)
    before = _command(rig.log, TARGET_A)

    real_subscribe = rig.log.subscribe

    def subscribe_hook(listener):
        rig.log.append(
            ACTOR, command_type("say"), command_content(target=TARGET_A, payload={})
        )
        return real_subscribe(listener)

    rig.log.subscribe = subscribe_hook

    async def main():
        stream = _Stream(rig.feed, TARGET_A)
        frames = await stream.events(2)
        assert await stream.drain() == []  # exactly once, not twice either
        await stream.close()
        return frames

    frames = asyncio.run(main())
    assert [_event_of(f).seq for f in frames] == [before.seq, before.seq + 1]


def test_command_at_boundary_delivered_exactly_once(tmp_path):
    """A command that lands in the subscription *and* the replay read — seq exactly at
    the replay/live boundary — is served once. Without the discard-at-boundary rule,
    subscribe-first order serves it twice."""
    rig = _rig(tmp_path, feed_kwargs=FAST)
    before = _command(rig.log, TARGET_A)

    real_subscribe = rig.log.subscribe

    def subscribe_hook(listener):
        unsub = real_subscribe(listener)  # the stream's listener first,
        _command(rig.log, TARGET_A)  # then the command — queue AND disk
        return unsub


    rig.log.subscribe = subscribe_hook

    async def main():
        stream = _Stream(rig.feed, TARGET_A)
        frames = await stream.events(2)
        assert await stream.drain() == []
        await stream.close()
        return frames

    frames = asyncio.run(main())
    assert [_event_of(f).seq for f in frames] == [before.seq, before.seq + 1]


def test_another_targets_command_never_appears(tmp_path):
    """Two targets, two streams, no crossover — in replay or live."""
    rig = _rig(tmp_path, feed_kwargs=FAST)
    a = _command(rig.log, TARGET_A)
    b = _command(rig.log, TARGET_B)

    async def main():
        stream_a = _Stream(rig.feed, TARGET_A)
        frames_a = await stream_a.events(1)
        # A command for B lands while A's stream is open.
        b_live = _command(rig.log, TARGET_B)
        crossover = await stream_a.drain()
        stream_b = _Stream(rig.feed, TARGET_B)
        frames_b = await stream_b.events(2)
        await stream_a.close()
        await stream_b.close()
        return frames_a, crossover, frames_b, b_live.seq

    frames_a, crossover, frames_b, b_live_seq = asyncio.run(main())
    assert [_event_of(f).seq for f in frames_a] == [a.seq]
    assert crossover == []
    assert [_event_of(f).seq for f in frames_b] == [b.seq, b_live_seq]


# --- idleness and teardown --------------------------------------------------------


def test_stream_heartbeats_when_idle(tmp_path):
    """An otherwise-silent connection carries comment lines and no events — and the
    heartbeat disturbs neither the cursor nor a later command's parse."""
    rig = _rig(tmp_path, feed_kwargs=FAST)

    async def main():
        stream = _Stream(rig.feed, TARGET_A)
        end = time.monotonic() + 2.0
        comments: list[str] = []
        while len(comments) < 2 and time.monotonic() < end:
            frame = await stream.next(deadline=end - time.monotonic())
            if frame.startswith(":"):
                comments.append(frame)
        cmd = _command(rig.log, TARGET_A)
        frames = await stream.events(1)
        await stream.close()
        return comments, frames, cmd.seq

    comments, frames, cmd_seq = asyncio.run(main())
    assert len(comments) == 2
    assert all("event:" not in c and not c.startswith("id:") for c in comments)
    assert [_event_of(f).seq for f in frames] == [cmd_seq]


def test_disconnect_ends_generator_and_unsubscribes(tmp_path):
    """After the client disconnects the log has no listener left — a listener left
    behind on every dropped connection grows with reconnects, the normal case on a
    flaky link."""
    rig = _rig(tmp_path, feed_kwargs=FAST)
    _command(rig.log, TARGET_A)

    async def main():
        stream = _Stream(rig.feed, TARGET_A)
        await stream.events(1)
        subscribed = len(rig.log._listeners)
        await stream.close()
        return subscribed

    subscribed = asyncio.run(main())
    assert subscribed == 1
    assert rig.log._listeners == []


def test_data_framing_survives_multi_line_payload(tmp_path):
    """A payload containing a newline round-trips through the framing back to an equal
    StimulusEvent — multiple `data:` lines joined by newline, never a raw newline
    inside one `data:` line."""
    rig = _rig(tmp_path)
    cmd = _command(rig.log, TARGET_A, payload={"text": "line1\nline2"})

    async def main():
        stream = _Stream(rig.feed, TARGET_A)
        frames = await stream.events(1)
        await stream.close()
        return frames

    frame = asyncio.run(main())[0]
    assert _event_of(frame) == cmd
    # The frame carries an id: line and an event: line around the data.
    assert f"id: {cmd.seq}" in frame
    assert f"event: {cmd.type}" in frame


def test_mounted_route_serves_the_stream(tmp_path):
    """`add_routes` / `build_app` mount the endpoint with the ingress's pattern — the
    route exists and answers with an event-stream."""
    rig = _rig(tmp_path)
    cmd = _command(rig.log, TARGET_A)
    app = rig.feed.build_app()

    routes = {route.path for route in app.routes if hasattr(route, "path")}
    assert f"/commands/{{target}}" in routes
