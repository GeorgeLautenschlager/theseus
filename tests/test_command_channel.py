"""Tests for the CommandChannel seam (issue #34, Tasks 2 and 4)."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import httpx
import pytest

from theseus.command_feed import CommandFeed, _format_sse
from theseus.commands import command_content, command_type
from theseus.stimulus_log import StimulusEvent, StimulusLog
from theseus.surrogates.command_channel import CommandChannel, MemoryCommandChannel
from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.retry import RetryBudget, backoff_delay
from theseus.surrogates.sse_command_channel import SseCommandChannel

# Fast enough that a reconnect-path test's real sleep stays under a millisecond; the
# backoff-shape tests below assert on recorded delays, never on timing.
FAST_BUDGET = RetryBudget(base_seconds=0.01, multiplier=2.0, jitter=0.0)


def test_doorbell_shape_drains_and_returns() -> None:
    ch = MemoryCommandChannel()
    offered = [_command("tam", n) for n in range(3)]
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
    ch.offer(_command("tam", 1))

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

    ch.offer(_command("tam", 2))
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
    first = _command("tam", 1)
    ch.offer(first)
    assert list(ch.stream()) == [first]
    assert list(ch.stream()) == []


def test_second_stream_resumes_with_new_offers() -> None:
    ch = MemoryCommandChannel()
    a, b = _command("tam", 1), _command("tam", 2)
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
        ch.offer(_command("tam", 1))


# --- the SSE transport (Task 4) ---------------------------------------------------

HOST = "host"
ACTOR = "george"

# Fast enough that idle-path tests stay under a second; deadline asserts bound them
# independently.
FAST = {"heartbeat_seconds": 0.05, "poll_seconds": 0.01}


def _command(target: str, n: int) -> StimulusEvent:
    """A host-issued command carrying seq `n` — what the wire would carry."""
    return StimulusEvent(
        id=f"cmd-{n}",
        ts=datetime.now(timezone.utc),
        actor=ACTOR,
        type=command_type("say"),
        content=command_content(target=target, payload={"n": n}),
        seq=n,
    )


def _log_command(log: StimulusLog, target: str, n: int) -> StimulusEvent:
    """Append one command through the host's log, so it carries the log's seq."""
    return log.append(
        ACTOR,
        command_type("say"),
        command_content(target=target, payload={"n": n}),
    )


class _RecordingApp:
    """Wraps an ASGI app and records the headers of every HTTP request it serves, so
    tests assert on what the host **received** rather than on what the client sent."""

    def __init__(self, app) -> None:
        self._app = app
        self.requests: list[dict[str, str]] = []
        self._lock = threading.Lock()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
            with self._lock:
                self.requests.append(headers)
        await self._app(scope, receive, send)


class _ScriptedHost:
    """A tiny ASGI app serving one scripted SSE body per connection, then closing.

    A script is a list whose items are frame strings sent as body chunks, or
    `threading.Event`s the connection blocks on until the test sets them (so the test
    controls exactly when the next bytes move). Request headers are recorded per
    connection — the reconnect test asserts on what the *second* connection received.
    """

    def __init__(self, scripts: list[list]) -> None:
        self.requests: list[dict[str, str]] = []
        self._scripts = list(scripts)
        self._lock = threading.Lock()

    async def __call__(self, scope, receive, send) -> None:
        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        with self._lock:
            self.requests.append(headers)
            script = self._scripts.pop(0) if self._scripts else []
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for item in script:
            if isinstance(item, threading.Event):
                item.wait(timeout=5.0)
            else:
                await send({"type": "http.response.body", "body": item.encode(), "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})


def _collect(channel, *, count: int, deadline: float = 10.0) -> list[StimulusEvent]:
    """Drain `stream()` in a daemon thread until `count` events or the stream ends.

    Bounded: a stream that never produces (or never ends) fails the wait instead of
    hanging the suite. `max_reconnects` bounds the stream itself; this bounds the wait.
    """
    got: list[StimulusEvent] = []
    done = threading.Event()

    def run() -> None:
        try:
            for event in channel.stream():
                got.append(event)
                if len(got) >= count:
                    break
        except Exception:  # noqa: BLE001 — teardown tearing the connection is not a test failure
            pass
        done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    # Close the channel (its client) before joining: the stream may still be connected
    # — on a failure, always is — and an open connection holds uvicorn's shutdown past
    # this fixture's join.
    produced = done.wait(deadline + 2.0)
    channel.close()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "stream thread did not end after close()"
    assert produced, f"stream did not yield {count} events in time"
    return got


def _cursor(tmp_path, origin: str = "tam-surrogate") -> AckedCursor:
    return AckedCursor(tmp_path / "command-cursor.json", origin)


def test_sse_resumes_from_the_cursor(tmp_path, serve) -> None:
    """With a cursor at seq N the request carries `Last-Event-ID: N` — asserted on the
    header the host actually received — and only later commands arrive."""
    log = StimulusLog(path=tmp_path / "host.jsonl", origin=HOST)
    first = _log_command(log, "tam", 1)
    cursor = _cursor(tmp_path)
    cursor.advance(first.seq)
    second = _log_command(log, "tam", 2)

    app = _RecordingApp(CommandFeed(log, **FAST).build_app())
    base = serve(app)
    channel = SseCommandChannel(
        f"{base}/commands/tam", cursor, client=httpx.Client(),
        max_reconnects=0, budget=FAST_BUDGET,
    )
    got = _collect(channel, count=1)

    assert [e.seq for e in got] == [second.seq]
    assert app.requests[0].get("last-event-id") == str(first.seq)


def test_sse_never_advanced_cursor_sends_no_header(tmp_path, serve) -> None:
    """The header is absent, not `"0"` and not `""`: sending 0 would claim a position
    the surrogate has never held."""
    log = StimulusLog(path=tmp_path / "host.jsonl", origin=HOST)
    _log_command(log, "tam", 1)

    app = _RecordingApp(CommandFeed(log, **FAST).build_app())
    base = serve(app)
    channel = SseCommandChannel(
        f"{base}/commands/tam", _cursor(tmp_path), client=httpx.Client(),
        max_reconnects=0, budget=FAST_BUDGET,
    )
    got = _collect(channel, count=1)

    assert len(got) == 1  # a fresh cursor replays from the beginning
    assert "last-event-id" not in app.requests[0]


def test_sse_channel_does_not_advance_the_cursor(tmp_path, serve) -> None:
    """Pins the at-least-once decision: the cursor means *executed*, not *received*,
    and only the caller moves it. A later change that "helpfully" advances here would
    convert delivery to at-most-once — a dropped command is invisible."""
    host = _ScriptedHost([[_format_sse(_command("tam", n)) for n in (3, 4)]])
    base = serve(host)
    cursor = _cursor(tmp_path)
    cursor.advance(2)
    channel = SseCommandChannel(
        f"{base}/commands/tam", cursor, client=httpx.Client(),
        max_reconnects=0, budget=FAST_BUDGET,
    )

    got = _collect(channel, count=2)

    assert [e.seq for e in got] == [3, 4]
    assert cursor.acked_seq == 2


def test_sse_heartbeats_yield_nothing_and_do_not_end_the_stream(tmp_path, serve) -> None:
    """A comment-line heartbeat yields no event and the stream survives it: the
    command sent *after* the heartbeat still arrives (and with max_reconnects=0, a
    stream the heartbeat had ended could never deliver it). What this does NOT pin is
    the `startswith(":")` skip branch in `_frames` — the following `data:` guard
    already ignores every comment line, so deleting that branch changes nothing here.
    The branch is defensive; this test proves the stream survives heartbeats, not that
    the branch exists."""
    host = _ScriptedHost([[": heartbeat\n\n", _format_sse(_command("tam", 7))]])
    base = serve(host)
    channel = SseCommandChannel(
        f"{base}/commands/tam", _cursor(tmp_path), client=httpx.Client(),
        max_reconnects=0, budget=FAST_BUDGET,
    )

    got = _collect(channel, count=1)

    assert [e.seq for e in got] == [7]


def test_sse_dropped_connection_reconnects_from_current_cursor(tmp_path, serve) -> None:
    """The second connection's `Last-Event-ID` reflects what the caller advanced to —
    not what the first connection last delivered. Here they differ: the first
    connection delivered seq 6, the caller executed only seq 5, so the reconnect must
    resume at 5, replaying 6."""
    advanced = threading.Event()
    fifth = _command("tam", 5)
    sixth = _command("tam", 6)
    host = _ScriptedHost(
        [
            [_format_sse(fifth), advanced, _format_sse(sixth)],
            [],  # second connection: headers recorded, then close
        ]
    )
    base = serve(host)
    cursor = _cursor(tmp_path)
    channel = SseCommandChannel(
        f"{base}/commands/tam", cursor, client=httpx.Client(),
        max_reconnects=1, budget=FAST_BUDGET,
    )
    got: list[StimulusEvent] = []
    done = threading.Event()

    def run() -> None:
        try:
            for event in channel.stream():
                got.append(event)
        except Exception:  # noqa: BLE001
            pass
        done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    # Execute the first command, then let the connection drop. The server is blocked
    # on `advanced` between the two frames, so the cursor is advanced before the
    # reconnect ever happens — the ordering the test exists to pin.
    deadline = time.monotonic() + 5.0
    while len(got) < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(got) == 1, "first command never arrived"
    cursor.advance(fifth.seq)
    advanced.set()
    assert done.wait(10.0), "stream did not end after reconnects were exhausted"
    channel.close()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "stream thread did not end after close()"

    assert [e.seq for e in got] == [fifth.seq, sixth.seq]  # executed 5, 6 replayed
    assert len(host.requests) == 2
    assert host.requests[1].get("last-event-id") == str(fifth.seq)


def test_sse_404_ends_the_stream_instead_of_looping(tmp_path, serve) -> None:
    """A permanent 4xx means the request will never work; retrying forever is 43k
    requests a day against a URL that cannot succeed. Bounded by the request count the
    host recorded."""

    async def not_found(scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 404})
        await send({"type": "http.response.body", "body": b""})

    base = serve(not_found)
    sleeps: list[float] = []
    channel = SseCommandChannel(
        f"{base}/commands/tam", _cursor(tmp_path),
        budget=FAST_BUDGET, sleep_fn=sleeps.append,
    )
    got = _collect(channel, count=1)

    assert got == []
    assert sleeps == [], "a permanent 4xx must not spend reconnects"


def test_sse_503_is_retried_and_429_is_retried(tmp_path) -> None:
    """A 5xx and a 429 are the host saying *later*, not *never* — they spend retry
    budget like a drop does, unlike a 404."""
    cursor = _cursor(tmp_path)
    statuses = iter([503, 429, 503])
    calls: list[str] = []
    sleeps: list[float] = []

    class _StatusHost:
        """Answers with the next scripted status, never a body."""

        def stream(self, method, url, headers=None):
            status = next(statuses)
            calls.append(str(status))
            request = httpx.Request(method, url)
            response = httpx.Response(status, request=request)

            class _Ctx:
                def __enter__(self) -> httpx.Response:
                    response.raise_for_status()
                    return response

                def __exit__(self, *args) -> None:
                    return None

            return _Ctx()

    channel = SseCommandChannel(
        "http://host/commands/tam", cursor,
        client=_StatusHost(),  # type: ignore[arg-type]
        budget=FAST_BUDGET, random_fn=lambda: 0.5, sleep_fn=sleeps.append,
        max_reconnects=2,
    )
    got = _collect(channel, count=1)

    assert got == []
    assert len(calls) == 3  # two reconnects, then max_reconnects stops the loop
    assert sleeps == [0.01, 0.02]  # first-attempt then second-attempt delay, unjittered


def test_sse_flapping_host_backoff_grows_and_stays_bounded(tmp_path, serve) -> None:
    """A host that answers 200 and closes immediately never satisfies the reset rule —
    resetting needs a connection that *stays* open, which the two tests below pin
    (delivered, and silent-but-lasting). What this proves is the other half: with no
    reset, the delays grow 0.01 → 0.02 → 0.04, so a flapping host is retried at a
    bounded rate instead of hammering it at the first-attempt delay forever."""
    host = _ScriptedHost([[], []])  # every connection closes immediately
    base = serve(host)
    sleeps: list[float] = []
    rng = iter([0.5, 0.5, 0.5])  # jitter factor 0 → delays are exactly the raw sequence
    channel = SseCommandChannel(
        f"{base}/commands/tam", _cursor(tmp_path), client=httpx.Client(),
        budget=FAST_BUDGET, random_fn=lambda: next(rng), sleep_fn=sleeps.append,
        max_reconnects=3,
    )
    got = _collect(channel, count=1)

    assert got == []
    # No connection lasted long enough to reset `failures`, so the delays are the raw
    # `backoff_delay` sequence, unjittered by the deterministic `random_fn`.
    assert sleeps == [0.01, 0.02, 0.04]


def test_backoff_resets_after_a_successful_connection(tmp_path) -> None:
    """Two drops, then a connection that delivers and stays open past the threshold
    before dropping: the last delay is the *first-attempt* delay again, not the grown
    one — a channel that has been up does not owe the ceiling."""
    request = httpx.Request("GET", "http://host/commands/tam")

    class _FlakyHost:
        def __init__(self) -> None:
            # Two drops, then a connection that delivers a command, then drops until
            # `max_reconnects` is spent.
            self.script: list[str] = ["drop", "drop", "ok", "drop", "drop", "drop"]

        def stream(self, method, url, headers=None):
            what = self.script.pop(0) if self.script else "drop"

            class _Ctx:
                def __enter__(self):
                    if what == "drop":
                        raise httpx.ConnectError("no route", request=request)
                    return self

                def __exit__(self, *args) -> None:
                    return None

                def raise_for_status(self) -> None:
                    pass

                def iter_lines(self):
                    # `httpx` yields lines, not whole frames.
                    yield from _format_sse(_command("tam", 1)).splitlines()

            return _Ctx()

    sleeps: list[float] = []
    channel = SseCommandChannel(
        "http://host/commands/tam", _cursor(tmp_path),
        client=_FlakyHost(),  # type: ignore[arg-type]
        budget=FAST_BUDGET, random_fn=lambda: 0.5, sleep_fn=sleeps.append,
        # Only the delivering connection is opened (drops fail at connect), so the
        # fake clock only needs an open time and a later close time: 4s of contact,
        # far past FAST_BUDGET's 0.01s threshold.
        now_fn=iter([1.0, 5.0]).__next__,
        max_reconnects=3,
    )
    got: list[StimulusEvent] = []
    done = threading.Event()

    def run() -> None:
        try:
            # Consume everything, not just the first command: the delays this test
            # exists to assert on come *after* the delivered one.
            got.extend(channel.stream())
        except Exception:  # noqa: BLE001 — teardown tearing the connection is not a test failure
            pass
        done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert done.wait(10.0), "stream did not end after reconnects were exhausted"
    channel.close()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "stream thread did not end after close()"

    assert [e.seq for e in got] == [1]
    # Two drops at attempts 1–2, the lasting connection resets the count, then the
    # post-drop retries start over at the first-attempt delay and grow to the stop.
    assert sleeps == [0.01, 0.02, 0.01, 0.02, 0.04]


def test_backoff_resets_on_a_healthy_silent_connection(tmp_path) -> None:
    """The finding: hours of silence is the *normal* case for a command channel, so a
    connection that stayed open but delivered nothing must still reset the backoff.
    Pins the lasted-not-spoke rule directly, so a delivery-based reset cannot return
    without this failing."""
    request = httpx.Request("GET", "http://host/commands/tam")

    class _SilentThenFlaky:
        def __init__(self) -> None:
            # Two drops, then a connection that opens, says nothing, and is dropped by
            # a proxy/NAT timeout, then drops until `max_reconnects` is spent.
            self.script: list[str] = ["drop", "drop", "silent", "drop", "drop", "drop"]

        def stream(self, method, url, headers=None):
            what = self.script.pop(0) if self.script else "drop"

            class _Ctx:
                def __enter__(self):
                    if what == "drop":
                        raise httpx.ConnectError("no route", request=request)
                    return self

                def __exit__(self, *args) -> None:
                    return None

                def raise_for_status(self) -> None:
                    pass

                def iter_lines(self):
                    return iter([])  # heartbeats consumed upstream; nothing to yield

            return _Ctx()

    sleeps: list[float] = []
    channel = SseCommandChannel(
        "http://host/commands/tam", _cursor(tmp_path),
        client=_SilentThenFlaky(),  # type: ignore[arg-type]
        budget=FAST_BUDGET, random_fn=lambda: 0.5, sleep_fn=sleeps.append,
        # Same fake clock as above: only the silent connection opens.
        now_fn=iter([1.0, 5.0]).__next__,
        max_reconnects=3,
    )
    got = list(channel.stream())

    assert got == []
    # Without the reset on the silent connection, the delays would grow to the ceiling:
    # [0.01, 0.02, 0.04]. With it, the post-drop retries restart at the first attempt.
    assert sleeps == [0.01, 0.02, 0.01, 0.02, 0.04]


def test_backoff_arithmetic_matches_retry_budget() -> None:
    # pin the deterministic shape the stream test relies on: base * multiplier**(n-1)
    assert backoff_delay(1, FAST_BUDGET, random_fn=lambda: 0.5) == 0.01
    assert backoff_delay(3, FAST_BUDGET, random_fn=lambda: 0.5) == 0.04


def test_sse_malformed_frame_is_skipped_not_fatal(tmp_path, serve, caplog) -> None:
    """One bad frame must not take the channel down — but a silently skipped command
    is the failure this protocol exists to prevent, so it is logged."""
    good = _command("tam", 9)
    host = _ScriptedHost([["data: {not json\n\n", _format_sse(good)]])
    base = serve(host)
    channel = SseCommandChannel(
        f"{base}/commands/tam", _cursor(tmp_path), client=httpx.Client(),
        max_reconnects=0, budget=FAST_BUDGET,
    )

    with caplog.at_level("WARNING"):
        got = _collect(channel, count=1)

    assert [e.seq for e in got] == [good.seq]
    assert any("malformed" in record.message.lower() for record in caplog.records)


class _HangingHost:
    """A fake host whose one connection blocks on `release` — the injected-client
    shape: nothing the channel owns can break the read, so only the stop signal can
    end the stream."""

    def __init__(self) -> None:
        self.connects = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def stream(self, *args, **kwargs):
        self.connects += 1
        return self

    def __enter__(self):
        self.entered.set()
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_lines(self):
        self.release.wait(timeout=5.0)
        return iter([])


def test_close_ends_injected_client_stream_by_returning(tmp_path) -> None:
    """With a client the channel does not own, close() cannot work by closing
    anything — the stop signal is the only thing that ends the stream, and it must
    end it by returning, never by raising out of the generator."""
    host = _HangingHost()
    channel = SseCommandChannel(
        "http://host/commands/tam", _cursor(tmp_path), client=host,  # type: ignore[arg-type]
        budget=FAST_BUDGET,
    )
    outcome: list[BaseException | None] = [None]
    done = threading.Event()

    def run() -> None:
        try:
            for _ in channel.stream():
                pass
        except BaseException as exc:  # noqa: BLE001 — the test asserts on exactly this
            outcome[0] = exc
        done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert host.entered.wait(5.0), "stream never connected"
    channel.close()
    host.release.set()
    assert done.wait(5.0), "stream did not end after close()"
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert outcome[0] is None, f"stream ended by raising: {outcome[0]!r}"
    assert host.connects == 1, "stream reconnected after close()"


def test_close_ends_owned_client_stream_by_returning(tmp_path, serve) -> None:
    """With an owned client the old behaviour made the stream *raise* — the closed
    client's read failed as an HTTPError, the loop treated it as transient and slept a
    full backoff before reconnecting onto a closed client. The stop signal must end
    it instead, promptly and by returning, on both sides of the backoff."""
    sleeps: list[float] = []

    async def dropping(scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b""})

    base = serve(dropping)
    channel = SseCommandChannel(
        f"{base}/commands/tam", _cursor(tmp_path),
        budget=RetryBudget(base_seconds=30.0, multiplier=1.0, jitter=0.0),
        # Record AND really sleep: the slices are the mechanism under test, and a
        # recording-only sleep_fn would let the loop burn the whole backoff before
        # close() could land between slices.
        sleep_fn=lambda s: (sleeps.append(s), time.sleep(s))[1],
    )
    outcome: list[BaseException | None] = [None]
    done = threading.Event()

    def run() -> None:
        try:
            for _ in channel.stream():
                pass
        except BaseException as exc:  # noqa: BLE001 — the test asserts on exactly this
            outcome[0] = exc
        done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    # Wait until the connection has dropped and the (30s) backoff has begun, so the
    # close() lands mid-backoff: the sleep must be interrupted, not slept through.
    deadline = time.monotonic() + 5.0
    while not sleeps and time.monotonic() < deadline:
        time.sleep(0.01)
    channel.close()
    assert done.wait(5.0), "stream did not end promptly after close() mid-backoff"
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert outcome[0] is None, f"stream ended by raising: {outcome[0]!r}"
    assert len(sleeps) == 1, "stream reconnected after close()"


def test_sse_channel_satisfies_the_protocol(tmp_path) -> None:
    """`close()` is on the protocol because every caller needs it to shut down; both
    implementations must satisfy the runtime check without callers noticing."""
    assert isinstance(
        SseCommandChannel(
            "http://host/commands/tam", _cursor(tmp_path),
            client=_HangingHost(),  # type: ignore[arg-type]
        ),
        CommandChannel,
    )
