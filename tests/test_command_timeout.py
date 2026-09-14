from datetime import datetime, timedelta, timezone

from theseus.command_reports import Executed
from theseus.surrogates.command_executor import CommandExecutor
from theseus.surrogates.command_channel import MemoryCommandChannel
from test_command_executor import _Clock, _command, _parts


def test_timeout_expiry_skips_renderer_and_run_advances(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    log, cursor, host = _parts(tmp_path)
    called = []
    ex = CommandExecutor(log, lambda c: called.append(c) or Executed(), cursor, clock=_Clock(now))
    commands = [_command(host, ts=now), _command(host, ts=now-timedelta(hours=7)),
                _command(host, ts=now+timedelta(hours=1))]
    channel = MemoryCommandChannel(block=False)
    for c in commands: channel.offer(c)
    ex.run(channel)
    events = log.read_all()
    assert [e.type for e in events] == ["command_report.executed", "command_report.expired", "command_report.expired"]
    assert events[1].content["reason"] == "ttl_exceeded"
    assert events[2].content["reason"] == "clock_unreliable"
    assert len(called) == 1
    assert cursor.acked_seq == commands[-1].seq


def test_per_command_ttl_expiry(tmp_path):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    log, cursor, host = _parts(tmp_path)
    called = []
    ex = CommandExecutor(log, lambda c: called.append(c) or Executed(), cursor, clock=_Clock(now))
    ex.execute_one(_command(host, ts=now-timedelta(seconds=120), ttl_seconds=60))
    ex.execute_one(_command(host, ts=now-timedelta(seconds=30), ttl_seconds=60))
    assert [e.type for e in log.read_all()] == ["command_report.expired", "command_report.executed"]
    assert len(called) == 1
