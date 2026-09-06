"""Tests for the Replicator drain loop (Task 3).

Driven entirely by fake transports — the offline suite never touches a live server, and two
deliberately differently-built fakes prove the seam holds. Every thread join is bounded so a
broken implementation fails instead of hanging the suite.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import pytest

from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.replicator import DrainResult, Replicator
from theseus.surrogates.transport import TransportResult
from theseus.stimulus_log import StimulusLog

ORIGIN = "alpha"


def make_log(tmp_path: Path) -> StimulusLog:
    return StimulusLog(tmp_path / "log.jsonl", origin=ORIGIN)


def append_n(log: StimulusLog, n: int, start: int = 0) -> None:
    for i in range(start, start + n):
        log.append("tester", "test.event", {"n": i})


def seqs_in(bodies: list[str]) -> list[int]:
    """The seq run carried by a set of batch bodies, in wire order."""
    out: list[int] = []
    for body in bodies:
        for line in body.splitlines():
            if line:
                out.append(json.loads(line)["seq"])
    return out


class ScriptedTransport:
    """First fake: plays a script of statuses or exceptions, records every body it was given."""

    def __init__(self, script: list[int | BaseException] | None = None) -> None:
        self._script = list(script or [])
        self.bodies: list[str] = []
        self.calls = 0

    def send(self, body: str) -> TransportResult:
        self.calls += 1
        self.bodies.append(body)
        outcome = self._script.pop(0) if self._script else 200
        if isinstance(outcome, BaseException):
            raise outcome
        return TransportResult(status=outcome)


class ConcurrencyProbeTransport:
    """Counts overlapping sends: sleeps inside `send` so in-flight overlap is observable."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.max_in_flight = 0
        self.bodies: list[str] = []
        self._lock = threading.Lock()
        self._in_flight = 0

    def send(self, body: str) -> TransportResult:
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
        time.sleep(self.delay)
        with self._lock:
            self._in_flight -= 1
            self.bodies.append(body)
        return TransportResult(status=200)


class QueueTransport:
    """Second fake, built differently (queue-driven): proves any object satisfying the
    protocol drives the same Replicator unchanged."""

    def __init__(self, statuses: list[int]) -> None:
        self._statuses: queue.Queue[int] = queue.Queue()
        for status in statuses:
            self._statuses.put(status)
        self.sent: list[str] = []

    def send(self, body: str) -> TransportResult:
        self.sent.append(body)
        return TransportResult(status=self._statuses.get())


def test_backlog_drains_in_seq_order_across_batches(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 25)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport()
    rep = Replicator(log, transport, cursor, max_events=7, max_bytes=1_000_000)

    result = rep.drain()

    # 25 events at 7 per batch: 7+7+7+4, nothing missing or repeated across the bodies.
    assert len(transport.bodies) == 4
    assert seqs_in(transport.bodies) == list(range(1, 26))
    assert result == DrainResult(batches_attempted=4, events_attempted=25, acked_seq=25, stopped_on=None)


def test_cursor_advances_only_on_2xx(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 12)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([200, 500])
    rep = Replicator(log, transport, cursor, max_events=5, max_bytes=1_000_000)

    result = rep.drain()

    # Batches are 5+5+2; the 500 on the second must not move the cursor past the first.
    assert transport.calls == 2
    assert cursor.acked_seq == 5
    assert result.acked_seq == 5
    assert result.stopped_on == 500


def test_non_2xx_stops_the_drain_immediately(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 12)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([200, 500])
    rep = Replicator(log, transport, cursor, max_events=4, max_bytes=1_000_000)

    result = rep.drain()

    # Three batches pending (4+4+4); the failed second is never followed by a third send.
    assert transport.calls == 2
    assert result.stopped_on == 500


def test_never_more_than_one_request_in_flight(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 20)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ConcurrencyProbeTransport(delay=0.05)
    rep = Replicator(log, transport, cursor, max_events=4, max_bytes=1_000_000)

    gate = threading.Barrier(4)

    def go() -> None:
        gate.wait(timeout=5)
        rep.drain()

    threads = [threading.Thread(target=go) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads), "drain deadlocked; fail, do not hang"

    assert transport.max_in_flight == 1


def test_only_the_surrogates_own_origin_is_shipped(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 5)
    for seq in (101, 102):
        # Host-origin events as they will arrive once the command channel (#34) exists.
        log.append("host", "cmd.issued", {"seq": seq}, origin="host", seq=seq)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport()
    rep = Replicator(log, transport, cursor, max_events=50, max_bytes=1_000_000)

    result = rep.drain()

    assert seqs_in(transport.bodies) == [1, 2, 3, 4, 5]
    assert result.events_attempted == 5


def test_already_acked_events_are_not_resent(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 6)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport()
    rep = Replicator(log, transport, cursor, max_events=3, max_bytes=1_000_000)

    first = rep.drain()
    assert seqs_in(transport.bodies) == list(range(1, 7))

    append_n(log, 4, start=6)
    second = rep.drain()

    new_bodies = transport.bodies[first.batches_attempted:]
    assert seqs_in(new_bodies) == [7, 8, 9, 10]


def test_empty_backlog_never_touches_the_transport(tmp_path):
    log = make_log(tmp_path)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport()
    rep = Replicator(log, transport, cursor)  # constructor defaults: the imported limits

    result = rep.drain()

    assert result == DrainResult(batches_attempted=0, events_attempted=0, acked_seq=None, stopped_on=None)
    assert transport.calls == 0


def test_drain_resumes_from_the_persisted_cursor_after_a_restart(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 5)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport()
    Replicator(log, transport, cursor, max_events=3, max_bytes=1_000_000).drain()

    # Restart: fresh objects over the same paths.
    log2 = StimulusLog(tmp_path / "log.jsonl", origin=ORIGIN)
    cursor2 = AckedCursor(tmp_path / "cursor.json", origin=ORIGIN)
    assert cursor2.acked_seq == 5
    append_n(log2, 3, start=5)
    transport2 = ScriptedTransport()
    Replicator(log2, transport2, cursor2, max_events=3, max_bytes=1_000_000).drain()

    assert seqs_in(transport2.bodies) == [6, 7, 8]


def test_interrupted_drain_keeps_the_batches_it_did_ack(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 10)
    cursor_path = tmp_path / "cursor.json"
    cursor = AckedCursor(cursor_path, origin=log.origin)
    transport = ScriptedTransport([200, ConnectionError("network down")])
    rep = Replicator(log, transport, cursor, max_events=4, max_bytes=1_000_000)

    with pytest.raises(ConnectionError):
        rep.drain()

    # Batches are 4+4+2: the acked first batch survives, the rest is still owed.
    assert cursor.acked_seq == 4
    assert AckedCursor(cursor_path, origin=ORIGIN).acked_seq == 4


def test_a_batch_the_host_will_always_reject_stalls_the_drain(tmp_path):
    # Known stall #32 must close: a single event whose line exceeds max_bytes becomes a lone
    # batch the host will always 413. With no abandon rule yet, stopping is the honest end.
    log = make_log(tmp_path)
    log.append("tester", "test.event", {"blob": "x" * 500})
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([413])
    rep = Replicator(log, transport, cursor, max_events=100, max_bytes=100)

    result = rep.drain()

    assert transport.calls == 1
    assert result.stopped_on == 413
    assert cursor.acked_seq is None


def test_a_second_fake_transport_drives_the_same_replicator(tmp_path):
    # The issue's design-for-deletion acceptance box: a differently-implemented fake, same
    # Replicator unchanged.
    log = make_log(tmp_path)
    append_n(log, 9)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = QueueTransport([200, 200, 200])
    rep = Replicator(log, transport, cursor, max_events=4, max_bytes=1_000_000)

    result = rep.drain()

    assert seqs_in(transport.sent) == list(range(1, 10))
    assert result == DrainResult(batches_attempted=3, events_attempted=9, acked_seq=9, stopped_on=None)


def test_the_drain_orders_by_seq_not_by_file_order(tmp_path):
    """The log is arrival-ordered and a surrogate is the sole writer of its own origin, so
    seq order and file order normally coincide. The host's dedupe depends on ascending seq,
    not on that coincidence — a foreign writer or a hand-edited log breaks it, and the
    straddle rule would then silently drop the events that arrived out of order.

    Written as a raw file so file order genuinely disagrees with seq order.
    """
    path = tmp_path / "log.jsonl"
    out_of_order = [3, 1, 4, 2]
    path.write_text(
        "".join(
            json.dumps({
                "id": f"01OUTOFORDER{seq:017d}",
                "ts": f"2026-09-04T16:00:{seq:02d}+00:00",
                "actor": "sensor",
                "type": "observation",
                "content": {"n": seq},
                "origin": ORIGIN,
                "seq": seq,
                "appended_ts": f"2026-09-04T16:00:{seq:02d}+00:00",
            }) + "\n"
            for seq in out_of_order
        ),
        encoding="utf-8",
    )
    log = StimulusLog(path, origin=ORIGIN)
    transport = ScriptedTransport([200])
    cursor = AckedCursor(tmp_path / "cursor.json", ORIGIN)

    Replicator(log, transport, cursor).drain()

    assert seqs_in(transport.bodies) == [1, 2, 3, 4]
    assert cursor.acked_seq == 4


def _raw_line(n: int, *, origin: str | None = None, seq: int | None = None) -> str:
    """One pre-envelope log line: no `origin`, no `seq` — exactly what a log that
    outlived the envelope upgrade contains. `log.append()` cannot produce these."""
    d: dict = {
        "id": f"raw-{n}",
        "ts": f"2026-01-01T00:00:{n % 60:02d}+00:00",
        "actor": "legacy",
        "type": "legacy.event",
        "content": {"n": n},
    }
    if origin is not None:
        d["origin"] = origin
    if seq is not None:
        d["seq"] = seq
    return json.dumps(d) + "\n"


def test_pre_envelope_events_are_skipped_counted_and_reported(tmp_path):
    """Lines that predate the seq envelope carry no dedupe identity, so they are not
    part of the stream. Skipping them is correct; shipping them earns a 400. The defect
    this pins is silence: they must be counted and reported, not vanish."""
    path = tmp_path / "log.jsonl"
    path.write_text(_raw_line(1) + _raw_line(2), encoding="utf-8")
    log = StimulusLog(path, origin=ORIGIN)
    append_n(log, 2)
    transport = ScriptedTransport()
    cursor = AckedCursor(tmp_path / "cursor.json", ORIGIN)

    result = Replicator(log, transport, cursor).drain()

    assert seqs_in(transport.bodies) == [1, 2]   # only the real events shipped
    assert result.events_attempted == 2
    assert result.stopped_on is None             # and the drain still completes
    assert result.skipped_unsequenced == 2
    assert result.skipped_duplicate == 0


def test_pre_envelope_skip_is_logged_once_not_per_event(tmp_path, caplog):
    """A 468-line legacy log must not produce 468 log lines: one warning per drain."""
    path = tmp_path / "log.jsonl"
    path.write_text(_raw_line(1) + _raw_line(2) + _raw_line(3), encoding="utf-8")
    log = StimulusLog(path, origin=ORIGIN)
    append_n(log, 1)
    transport = ScriptedTransport()
    cursor = AckedCursor(tmp_path / "cursor.json", ORIGIN)

    with caplog.at_level(logging.WARNING, logger="theseus.surrogates.replicator"):
        Replicator(log, transport, cursor).drain()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "predate" in warnings[0].getMessage().lower()


def test_duplicate_seq_does_not_wedge_the_channel(tmp_path):
    """A duplicate seq means two processes wrote one log (forbidden but unenforceable)
or a restored backup. The host 400s any batch whose seqs do not strictly ascend, so
    without dedupe this log ships nothing — not even seq 1 — on every drain, forever.
    Dropping the later occurrence is safe: the host would dedupe it anyway."""
    path = tmp_path / "log.jsonl"
    lines = []
    for n, seq in enumerate([1, 2, 2, 3]):
        d = json.loads(_raw_line(n))
        d["id"] = f"dup-{seq}-{n}"   # the two seq-2 events are distinguishable
        d["origin"] = ORIGIN
        d["seq"] = seq
        lines.append(json.dumps(d) + "\n")
    path.write_text("".join(lines), encoding="utf-8")
    log = StimulusLog(path, origin=ORIGIN)
    transport = ScriptedTransport()
    cursor = AckedCursor(tmp_path / "cursor.json", ORIGIN)

    result = Replicator(log, transport, cursor).drain()

    assert seqs_in(transport.bodies) == [1, 2, 3]   # the duplicate never reaches the wire
    shipped = {json.loads(l)["id"]: json.loads(l)["seq"] for b in transport.bodies for l in b.splitlines() if l}
    assert shipped["dup-2-1"] == 2                  # first occurrence kept, later one dropped
    assert "dup-2-2" not in shipped
    assert result.skipped_duplicate == 1
    assert result.stopped_on is None                # the channel is not wedged
    assert cursor.acked_seq == 3


def test_clean_drain_reports_zero_for_both_counters(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 5)
    transport = ScriptedTransport()
    cursor = AckedCursor(tmp_path / "cursor.json", ORIGIN)

    result = Replicator(log, transport, cursor).drain()

    assert result.skipped_unsequenced == 0
    assert result.skipped_duplicate == 0
    assert result.stopped_on is None


def test_3xx_stops_the_drain_and_does_not_advance_cursor(tmp_path):
    """A 301/302 from a misconfigured host or proxy is not an ack. The success check is
    pinned here: mutating it to `status >= 400` must fail this test."""
    log = make_log(tmp_path)
    append_n(log, 1)
    transport = ScriptedTransport([302])
    cursor = AckedCursor(tmp_path / "cursor.json", ORIGIN)

    result = Replicator(log, transport, cursor).drain()

    assert result.stopped_on == 302
    assert result.acked_seq is None   # the cursor did not advance


def test_the_batch_limits_are_the_hosts_own(tmp_path):
    """Two constants for one protocol limit is how a surrogate and a host come to disagree
    about what fits. A surrogate whose default `max_bytes` is larger than the host's builds
    batches the host 413s, and the drain stalls on a batch it believes is legal.

    Asserted on the defaults directly: a round-trip test cannot see this, because the host
    rejects the batch either way — it just costs a wasted request and a stall.
    """
    from theseus.replication_batch import DEFAULT_MAX_BATCH_BYTES, DEFAULT_MAX_BATCH_EVENTS

    replicator = Replicator(
        make_log(tmp_path), ScriptedTransport([]), AckedCursor(tmp_path / "c.json", ORIGIN)
    )

    assert replicator._max_events == DEFAULT_MAX_BATCH_EVENTS
    assert replicator._max_bytes == DEFAULT_MAX_BATCH_BYTES
