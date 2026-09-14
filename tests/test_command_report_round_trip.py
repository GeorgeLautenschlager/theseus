"""Command execution reports, end to end (issue #35, Task 5): a report minted on the
surrogate's log reaches the host's log through the ordinary replication path.

The harness mirrors `test_command_round_trip.py` — a real socket in each direction:
host → surrogate over `CommandFeed` SSE, surrogate → host over `ReplicationIngress`
POST. The only new piece is the `CommandExecutor` on the surrogate side; the upstream
half is `Replicator.drain()` unchanged, which is what "no bypass" means here: nothing
report-specific is ever mounted, injected, or flushed. A report is a plain own-origin
event, and this file proves it rides home like any other one — correlated on the host
against the issued command by `(command_origin, command_seq)`, never by `id` (identity
is `(origin, seq)`; ids are re-minted when the ingress appends).
"""

from __future__ import annotations

import threading
import time

import httpx

from theseus.command_reports import BargedIn, Executed, Failed, Partial, is_report
from theseus.command_feed import CommandFeed
from theseus.commands import command_content, command_type
from theseus.high_water import HighWaterMarks
from theseus.replication_ingress import ReplicationIngress
from theseus.stimulus_log import StimulusEvent, StimulusLog
from theseus.surrogates.command_executor import CommandExecutor
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


def _issue(log: StimulusLog, n: int) -> StimulusEvent:
    """Append one command through the host's log, so it carries the log's seq and id."""
    return log.append(
        "george", command_type("say"), command_content(target=TARGET, payload={"n": n})
    )


def _reports(log: StimulusLog) -> list[StimulusEvent]:
    return [event for event in log.read_all() if is_report(event)]


def _run_executor(
    channel: SseCommandChannel,
    executor: CommandExecutor,
    surrogate_log: StimulusLog,
    *,
    want_reports: int,
) -> None:
    """`executor.run` on a daemon thread against the live channel.

    Polls for the expected reports on the surrogate log (deadline-bounded, never a bare
    sleep), then closes the channel — the channel's own client interrupts the in-flight
    stream read — and joins, so `run` is guaranteed to have returned before the caller
    drains upstream.
    """
    done = threading.Event()

    def run() -> None:
        try:
            executor.run(channel)
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while len(_reports(surrogate_log)) < want_reports:
        assert time.monotonic() < deadline, "expected reports never reached the surrogate log"
        time.sleep(0.01)
    channel.close()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "executor run did not end after close()"


def _drain_upstream(base: str, surrogate_log: StimulusLog, up_cursor: AckedCursor) -> None:
    """The ordinary upstream path: one `Replicator.drain()`, no report-specific route."""
    replicator = Replicator(
        surrogate_log, HttpTransport(f"{base}/replicate", client=httpx.Client()), up_cursor
    )
    result = replicator.drain()
    assert not result.unreachable and result.stopped_on is None, (
        f"upstream drain did not complete: {result}"
    )


# --- the round trip --------------------------------------------------------------


def test_muted_command_produces_failed_report_on_host_log(tmp_path, serve) -> None:
    """Acceptance #11: a command executed with muted output produces a `failed` stimulus
    on the host log. The renderer is injected and fakes the reflex layer (out of scope):
    it hands back `Failed("output muted")`, exactly as a real muted speaker would."""
    host_log = StimulusLog(path=tmp_path / "host.jsonl", origin=HOST)
    surrogate_log = StimulusLog(path=tmp_path / "surrogate.jsonl", origin=SURROGATE)
    issued = _issue(host_log, 1)

    ingress = ReplicationIngress(host_log, HighWaterMarks(host_log))
    app = ingress.build_app()
    CommandFeed(host_log, **FAST).add_routes(app)
    base = serve(app)

    cmd_cursor = AckedCursor(tmp_path / "commands.json", f"{SURROGATE}-commands")
    up_cursor = AckedCursor(tmp_path / "upstream.json", f"{SURROGATE}-upstream")
    channel = SseCommandChannel(
        f"{base}/commands/{TARGET}", cmd_cursor, max_reconnects=0, budget=FAST_BUDGET
    )
    executor = CommandExecutor(
        surrogate_log, lambda command: Failed("output muted"), cmd_cursor
    )

    _run_executor(channel, executor, surrogate_log, want_reports=1)
    _drain_upstream(base, surrogate_log, up_cursor)

    arrived = _reports(host_log)
    assert len(arrived) == 1
    report = arrived[0]
    # Arrived via the ordinary ingress path: same (origin, seq), re-minted id.
    minted = _reports(surrogate_log)[0]
    assert report.origin == SURROGATE
    assert report.seq == minted.seq
    assert report.id != minted.id
    # A failed report, correlated against the issued command by cross-node identity.
    assert report.type == "command_report.failed"
    assert report.content["reason"] == "output muted"
    assert report.content["command_seq"] == issued.seq
    assert report.content["command_origin"] == HOST
    assert report.content["command_id"] == issued.id


def test_partial_carries_progress_marker_end_to_end(tmp_path, serve) -> None:
    """A `Partial` outcome's progress marker survives the trip to the host intact."""
    host_log = StimulusLog(path=tmp_path / "host.jsonl", origin=HOST)
    surrogate_log = StimulusLog(path=tmp_path / "surrogate.jsonl", origin=SURROGATE)
    _issue(host_log, 1)

    ingress = ReplicationIngress(host_log, HighWaterMarks(host_log))
    app = ingress.build_app()
    CommandFeed(host_log, **FAST).add_routes(app)
    base = serve(app)

    cmd_cursor = AckedCursor(tmp_path / "commands.json", f"{SURROGATE}-commands")
    up_cursor = AckedCursor(tmp_path / "upstream.json", f"{SURROGATE}-upstream")
    channel = SseCommandChannel(
        f"{base}/commands/{TARGET}", cmd_cursor, max_reconnects=0, budget=FAST_BUDGET
    )
    executor = CommandExecutor(
        surrogate_log, lambda command: Partial("spoke 3 of 5 words"), cmd_cursor
    )

    _run_executor(channel, executor, surrogate_log, want_reports=1)
    _drain_upstream(base, surrogate_log, up_cursor)

    arrived = _reports(host_log)
    assert len(arrived) == 1
    report = arrived[0]
    assert report.origin == SURROGATE
    assert report.type == "command_report.partial"
    assert report.content["progress"] == "spoke 3 of 5 words"


def test_barged_in_carries_playback_position_end_to_end(tmp_path, serve) -> None:
    """A `BargedIn` outcome's playback position survives the trip to the host intact."""
    host_log = StimulusLog(path=tmp_path / "host.jsonl", origin=HOST)
    surrogate_log = StimulusLog(path=tmp_path / "surrogate.jsonl", origin=SURROGATE)
    _issue(host_log, 1)

    ingress = ReplicationIngress(host_log, HighWaterMarks(host_log))
    app = ingress.build_app()
    CommandFeed(host_log, **FAST).add_routes(app)
    base = serve(app)

    cmd_cursor = AckedCursor(tmp_path / "commands.json", f"{SURROGATE}-commands")
    up_cursor = AckedCursor(tmp_path / "upstream.json", f"{SURROGATE}-upstream")
    channel = SseCommandChannel(
        f"{base}/commands/{TARGET}", cmd_cursor, max_reconnects=0, budget=FAST_BUDGET
    )
    executor = CommandExecutor(surrogate_log, lambda command: BargedIn(1830), cmd_cursor)

    _run_executor(channel, executor, surrogate_log, want_reports=1)
    _drain_upstream(base, surrogate_log, up_cursor)

    arrived = _reports(host_log)
    assert len(arrived) == 1
    report = arrived[0]
    assert report.origin == SURROGATE
    assert report.type == "command_report.barged_in"
    assert report.content["playback_position"] == 1830


def test_one_report_per_delivered_command_failures_included(tmp_path, serve) -> None:
    """Three commands — one that succeeds, one whose renderer raises, one muted — yield
    exactly three reports on the host log: none missing, none doubled, and the command
    cursor advanced to the third command's seq (the cursor means *executed*)."""
    host_log = StimulusLog(path=tmp_path / "host.jsonl", origin=HOST)
    surrogate_log = StimulusLog(path=tmp_path / "surrogate.jsonl", origin=SURROGATE)
    issued = [_issue(host_log, n) for n in (1, 2, 3)]

    def render(command: StimulusEvent):
        n = command.content["payload"]["n"]
        if n == 1:
            return Executed()
        if n == 2:
            raise RuntimeError("speaker driver exploded")
        return Failed("output muted")

    ingress = ReplicationIngress(host_log, HighWaterMarks(host_log))
    app = ingress.build_app()
    CommandFeed(host_log, **FAST).add_routes(app)
    base = serve(app)

    cmd_cursor = AckedCursor(tmp_path / "commands.json", f"{SURROGATE}-commands")
    up_cursor = AckedCursor(tmp_path / "upstream.json", f"{SURROGATE}-upstream")
    channel = SseCommandChannel(
        f"{base}/commands/{TARGET}", cmd_cursor, max_reconnects=0, budget=FAST_BUDGET
    )
    executor = CommandExecutor(surrogate_log, render, cmd_cursor)

    _run_executor(channel, executor, surrogate_log, want_reports=3)
    _drain_upstream(base, surrogate_log, up_cursor)

    arrived = _reports(host_log)
    assert len(arrived) == 3
    by_seq = {event.content["command_seq"]: event for event in arrived}
    assert set(by_seq) == {1, 2, 3}
    assert by_seq[1].type == "command_report.executed"
    assert by_seq[2].type == "command_report.failed"
    assert by_seq[2].content["reason"] == "RuntimeError: speaker driver exploded"
    assert by_seq[3].type == "command_report.failed"
    assert by_seq[3].content["reason"] == "output muted"
    for event in arrived:
        assert event.origin == SURROGATE
        assert event.content["command_origin"] == HOST
        assert event.content["command_id"] == {
            i.seq: i.id for i in issued
        }[event.content["command_seq"]]
    # Report-then-advance: the command cursor means *executed*, and all three were.
    assert cmd_cursor.acked_seq == 3
