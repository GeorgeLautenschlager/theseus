"""Tests for CommandExecutor (issue #35, Task 2): the reporting primitive.

Every delivered command appends exactly one report to the local log — failures
included. A renderer is an injected fake; no speaking, VAD, or playback exists
here."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from theseus.command_reports import BargedIn, Executed, Failed, Partial
from theseus.commands import command_content, command_type
from theseus.stimulus_log import StimulusEvent, StimulusLog
from theseus.surrogates.command_channel import MemoryCommandChannel
from theseus.surrogates.command_executor import CommandExecutor
from theseus.surrogates.cursor import AckedCursor

HOST = "host"
SURROGATE = "surrogate"


def _parts(tmp_path: Path):
    """A surrogate log, a real cursor, and a host log to mint commands from."""
    log = StimulusLog(tmp_path / "surrogate.jsonl", origin=SURROGATE)
    cursor = AckedCursor(tmp_path / "acked.json", origin=HOST)
    host = StimulusLog(tmp_path / "host.jsonl", origin=HOST)
    return log, cursor, host


def _command(host: StimulusLog) -> StimulusEvent:
    """A host-issued command off the host's own log: real seq, origin, id."""
    return host.append(
        "george",
        command_type("say"),
        command_content(target=SURROGATE, payload={"text": "hi"}),
    )


class _ScriptedRenderer:
    """Returns one scripted Outcome per call; raises if asked for more."""

    def __init__(self, outcomes) -> None:
        self._outcomes = list(outcomes)

    def __call__(self, command: StimulusEvent):
        return self._outcomes.pop(0)


def _executor(tmp_path: Path, render):
    log, cursor, host = _parts(tmp_path)
    return log, cursor, host, CommandExecutor(log, render, cursor)


# --- execute_one: one report per command ---------------------------------------


def test_executed_yields_one_executed_report(tmp_path):
    log, cursor, host, ex = _executor(tmp_path, lambda cmd: Executed())
    cmd = _command(host)
    report = ex.execute_one(cmd)

    events = log.read_all()
    assert len(events) == 1
    e = events[0]
    assert e.type == "command_report.executed"
    assert (e.id, e.seq) == (report.id, report.seq)
    assert e.content["command_seq"] == cmd.seq
    assert e.content["command_origin"] == cmd.origin
    assert e.content["command_id"] == cmd.id


def test_partial_carries_progress(tmp_path):
    log, cursor, host, ex = _executor(tmp_path, lambda cmd: Partial("halfway"))
    ex.execute_one(_command(host))

    (e,) = log.read_all()
    assert e.type == "command_report.partial"
    assert e.content["progress"] == "halfway"


def test_barged_in_carries_playback_position(tmp_path):
    log, cursor, host, ex = _executor(
        tmp_path, _ScriptedRenderer([BargedIn(1234), BargedIn("2:34")])
    )
    ex.execute_one(_command(host))
    ex.execute_one(_command(host))

    e1, e2 = log.read_all()
    assert e1.type == "command_report.barged_in"
    assert e1.content["playback_position"] == 1234
    # A string position round-trips too.
    assert e2.content["playback_position"] == "2:34"


def test_failed_carries_reason(tmp_path):
    log, cursor, host, ex = _executor(tmp_path, lambda cmd: Failed("output muted"))
    ex.execute_one(_command(host))

    (e,) = log.read_all()
    assert e.type == "command_report.failed"
    assert e.content["reason"] == "output muted"


def test_renderer_that_raises_yields_one_failed_report(tmp_path):
    def render(cmd):
        raise RuntimeError("boom")

    log, cursor, host, ex = _executor(tmp_path, render)
    report = ex.execute_one(_command(host))  # must not re-raise

    events = log.read_all()
    assert len(events) == 1
    assert report.type == "command_report.failed"
    assert "boom" in report.content["reason"]


def test_misbehaving_renderer_is_reported_not_crashed(tmp_path):
    # Partial("") makes the partial constructor raise: the renderer is misbehaving,
    # and the answer is one failed report, not an escape.
    log, cursor, host, ex = _executor(tmp_path, lambda cmd: Partial(""))
    ex.execute_one(_command(host))

    (e,) = log.read_all()
    assert e.type == "command_report.failed"


def test_mixed_commands_yield_one_report_each_in_order(tmp_path):
    outcomes = [
        Executed(),
        Partial("halfway"),
        Failed("output muted"),
        BargedIn(1234),
        Executed(),
        Partial("more"),
    ]
    log, cursor, host, ex = _executor(tmp_path, _ScriptedRenderer(outcomes))
    commands = [_command(host) for _ in outcomes]
    for cmd in commands:
        ex.execute_one(cmd)

    events = log.read_all()
    assert len(events) == len(commands)
    assert [e.type for e in events] == [
        "command_report.executed",
        "command_report.partial",
        "command_report.failed",
        "command_report.barged_in",
        "command_report.executed",
        "command_report.partial",
    ]
    for event, cmd in zip(events, commands):
        assert event.content["command_seq"] == cmd.seq
        assert event.content["command_origin"] == cmd.origin
        assert event.content["command_id"] == cmd.id


def test_out_of_contract_command_raises_before_any_append(tmp_path):
    log, cursor, host, ex = _executor(tmp_path, lambda cmd: Executed())
    bad = StimulusEvent(
        id="cmd-bad",
        ts=datetime.now(timezone.utc),
        actor="george",
        type=command_type("say"),
        content=command_content(target=SURROGATE, payload={}),
        seq=None,
    )

    with pytest.raises(ValueError):
        ex.execute_one(bad)
    assert log.read_all() == []


# --- run: report, then advance (issue #35, Task 3) ------------------------------


def test_run_reports_each_command_then_advances_cursor(tmp_path):
    log, cursor, host, ex = _executor(tmp_path, lambda cmd: Executed())
    channel = MemoryCommandChannel(block=False)
    commands = [_command(host) for _ in range(3)]
    for cmd in commands:
        channel.offer(cmd)

    ex.run(channel)

    events = log.read_all()
    assert len(events) == 3
    assert [e.content["command_seq"] for e in events] == [c.seq for c in commands]
    assert cursor.acked_seq == commands[-1].seq


def test_run_reports_before_advancing_cursor(tmp_path):
    # A final-counts-only test would pass on the wrong order; this pins it: a log
    # listener (fired inside append, before execute_one returns) records the
    # cursor at report time — and it must still sit at the *previous* command's
    # position, i.e. the advance for command n happens strictly after command
    # n's report.
    log, cursor, host, ex = _executor(tmp_path, lambda cmd: Executed())
    observed: list[tuple[int, int | None]] = []
    log.subscribe(
        lambda e: observed.append((e.content["command_seq"], cursor.acked_seq))
    )
    channel = MemoryCommandChannel(block=False)
    commands = [_command(host) for _ in range(3)]
    for cmd in commands:
        channel.offer(cmd)

    ex.run(channel)

    seqs = [c.seq for c in commands]
    assert observed == [(seqs[0], None), (seqs[1], seqs[0]), (seqs[2], seqs[1])]
    assert cursor.acked_seq == seqs[-1]


def test_run_over_empty_doorbell_returns_immediately(tmp_path):
    log, cursor, host, ex = _executor(tmp_path, lambda cmd: Executed())
    channel = MemoryCommandChannel(block=False)

    ex.run(channel)

    assert log.read_all() == []
    assert cursor.acked_seq is None
