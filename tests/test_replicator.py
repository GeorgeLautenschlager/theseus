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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.responses import JSONResponse

from theseus.replication_events import BATCH_REJECTED, GAP
from theseus.surrogates.clock import SystemClock
from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.http_transport import HttpTransport
from theseus.surrogates.replicator import DrainResult, Replicator
from theseus.surrogates.retry import RetryBudget
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
        self.script = list(script or [])
        self.bodies: list[str] = []
        self.calls = 0

    def send(self, body: str) -> TransportResult:
        self.calls += 1
        self.bodies.append(body)
        outcome = self.script.pop(0) if self.script else 200
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, TransportResult):
            return outcome
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
    transport = ScriptedTransport([200, 302])
    rep = Replicator(log, transport, cursor, max_events=5, max_bytes=1_000_000)

    result = rep.drain()

    # Batches are 5+5+2; the 302 on the second must not move the cursor past the first.
    # A 5xx would retry now (#32), so the non-ack here is a 3xx.
    assert transport.calls == 2
    assert cursor.acked_seq == 5
    assert result.acked_seq == 5
    assert result.stopped_on == 302


def test_non_2xx_stops_the_drain_immediately(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 12)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([200, 302])
    rep = Replicator(log, transport, cursor, max_events=4, max_bytes=1_000_000)

    result = rep.drain()

    # Three batches pending (4+4+4); the non-2xx second is never followed by a third send.
    assert transport.calls == 2
    assert result.stopped_on == 302


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

    result = rep.drain()

    # Batches are 4+4+2: the acked first batch survives, the rest is still owed. The raise
    # no longer propagates (#32): an unreachable host stops the drain cleanly.
    assert result.unreachable is True
    assert result.stopped_on is None
    assert cursor.acked_seq == 4
    assert AckedCursor(cursor_path, origin=ORIGIN).acked_seq == 4


def test_a_batch_the_host_will_always_reject_is_stepped_over(tmp_path):
    # The stall #32 closed: a single event whose line exceeds max_bytes becomes a lone batch
    # the host will always 413 — a 4xx is permanent, so it is rejected and stepped over.
    log = make_log(tmp_path)
    log.append("tester", "test.event", {"blob": "x" * 500})
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([413])
    rep = Replicator(log, transport, cursor, max_events=100, max_bytes=100)

    result = rep.drain()

    assert transport.calls == 1
    assert result.stopped_on is None
    assert result.rejected_batches == 1
    assert cursor.acked_seq == 1


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
                "ts": datetime.now(timezone.utc).isoformat(),
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
    outlived the envelope upgrade contains. `log.append()` cannot produce these.
    The `ts` is fresh: a stale one would now trip the age abandonment (Task 4)
    before the behaviour under test ever runs."""
    d: dict = {
        "id": f"raw-{n}",
        "ts": datetime.now(timezone.utc).isoformat(),
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


# --- Task 1: the clock seam and the host's reason ---------------------------


def test_system_clock_is_timezone_aware_utc():
    """A naive `now()` would compare wrongly against event timestamps, which are all
    timezone-aware; the one clock the system ships must therefore be aware and UTC."""
    now = SystemClock().now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_transport_result_carries_the_hosts_reason():
    """The host's own words travel with the status; the default stays `""` so every
    existing construction keeps type-checking."""
    assert TransportResult(status=400, reason="batch too large").reason == "batch too large"
    assert TransportResult(status=200).reason == ""


def test_transport_reads_reason_from_a_rejection_body():
    """The ingress answers a rejection with a JSON body carrying `reason`; the transport
    surfaces it so the log can record why, not just that."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.post("/replicate")
    def replicate():
        return JSONResponse(
            status_code=413, content={"reason": "batch is 9000 bytes, over the 8192 limit"}
        )

    transport = HttpTransport("http://testserver/replicate", client=TestClient(app))
    result = transport.send("{}")
    assert result.status == 413
    assert result.reason == "batch is 9000 bytes, over the 8192 limit"


def test_transport_survives_a_non_json_error_body():
    """A transport that raises while parsing an error response turns a clean `4xx` into
    what looks like an unreachable host — the distinction the whole seam preserves."""
    from fastapi import FastAPI, Response
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.post("/replicate")
    def replicate():
        return Response(content="nope", status_code=400, media_type="text/plain")

    transport = HttpTransport("http://testserver/replicate", client=TestClient(app))
    result = transport.send("{}")
    assert result.status == 400
    assert result.reason == ""


def test_transport_bounds_an_over_long_reason():
    """A remote party's words are bounded in the transport too, not only at the tape."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from theseus.replication_events import MAX_REASON_CHARS

    app = FastAPI()

    @app.post("/replicate")
    def replicate():
        return JSONResponse(status_code=400, content={"reason": "x" * 10_000})

    transport = HttpTransport("http://testserver/replicate", client=TestClient(app))
    result = transport.send("{}")
    assert result.status == 400
    # One over the limit, deliberately: `_clean_reason` marks a reason it had to cut,
    # and slicing to exactly the limit here would hide from it that anything was cut.
    assert len(result.reason) == MAX_REASON_CHARS + 1


class FakeClock:
    """Records sleeps and advances `now` by them: backoff without real time passing."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []
        self._now = datetime.now(tz=timezone.utc)

    def now(self) -> datetime:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self._now += timedelta(seconds=seconds)


def test_a_5xx_then_a_2xx_retries_the_same_range(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 3)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([500, 200])
    rep = Replicator(log, transport, cursor)

    result = rep.drain()

    assert transport.calls == 2
    assert transport.bodies[0] == transport.bodies[1]
    # Each delivery carries the full range; no event is duplicated within one delivery.
    assert seqs_in(transport.bodies) == [1, 2, 3, 1, 2, 3]
    assert cursor.acked_seq == 3
    assert result.stopped_on is None


def test_retry_sleeps_on_the_injected_clock(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 2)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([500] * 4 + [200])
    clock = FakeClock()
    # random 0.5 zeroes the jitter, so delays are the bare 2/6/18/54 ladder.
    rep = Replicator(log, transport, cursor, clock=clock, random_fn=lambda: 0.5)

    result = rep.drain()

    # The four waits between five attempts, per #39: 6 + 18 + 54 + 120 = 198s, and the
    # 120s ceiling actually fires on the last one.
    assert clock.sleeps == [6.0, 18.0, 54.0, 120.0]
    assert sum(clock.sleeps) == 198.0
    assert transport.calls == 5
    assert result.stopped_on is None


def test_a_4xx_is_never_retried(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 2)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([422])
    rep = Replicator(log, transport, cursor)

    rep.drain()

    assert transport.calls == 1


def test_a_4xx_records_a_batch_rejected_on_the_surrogates_own_log(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 2)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([TransportResult(status=413, reason="too big")])
    rep = Replicator(log, transport, cursor)

    rep.drain()

    markers = [e for e in log.read_all() if e.type == BATCH_REJECTED]
    assert len(markers) == 1
    event = markers[0]
    assert event.origin == log.origin
    assert event.seq is not None
    assert event.content == {
        "origin": ORIGIN,
        "from_seq": 1,
        "to_seq": 2,
        "status": 413,
        "reason": "too big",
    }


def test_a_4xx_steps_over_and_keeps_draining(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 6)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([200, 409, 200])
    rep = Replicator(log, transport, cursor, max_events=2, max_bytes=1_000_000)

    result = rep.drain()

    assert transport.calls == 3
    assert seqs_in(transport.bodies) == [1, 2, 3, 4, 5, 6]
    assert cursor.acked_seq == 6
    assert result.rejected_batches == 1
    assert result.stopped_on is None


def test_an_unreachable_host_stops_the_drain_and_abandons_nothing(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 2)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([ConnectionError("link down")])
    rep = Replicator(log, transport, cursor)

    result = rep.drain()

    assert result.unreachable is True
    assert cursor.acked_seq is None
    assert not [e for e in log.read_all() if e.type == BATCH_REJECTED]

    # The next drain re-sends the same range: nothing was abandoned or acked.
    transport.script = [200]
    rep.drain()
    assert seqs_in(transport.bodies) == [1, 2, 1, 2]


def test_an_unreachable_host_spends_no_attempts(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 2)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([ConnectionError("down")] * 50)
    rep = Replicator(log, transport, cursor)

    rep.drain()

    # One call, not max_attempts: only a host that answered can cost budget.
    assert transport.calls == 1


def test_a_3xx_stops_without_retrying_or_abandoning(tmp_path):
    log = make_log(tmp_path)
    append_n(log, 2)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([302])
    rep = Replicator(log, transport, cursor)

    result = rep.drain()

    assert transport.calls == 1
    assert result.stopped_on == 302
    assert cursor.acked_seq is None
    assert not [e for e in log.read_all() if e.type == BATCH_REJECTED]


def gaps_in(log: StimulusLog) -> list[dict]:
    """The declared-gap markers a drain left on the surrogate's own log."""
    return [e.content for e in log.read_all() if e.type == GAP]


# --- Task 4: the abandon rule -----------------------------------------------


def test_attempt_exhaustion_abandons_and_moves_on(tmp_path):
    """Five 500s spend the whole budget on batch 1; the surrogate declares the gap on its
    own log, steps past it, and batch 2 still ships. The channel does not wedge."""
    log = make_log(tmp_path)
    append_n(log, 4)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([500] * 5 + [200])
    rep = Replicator(
        log, transport, cursor, max_events=2, clock=FakeClock(), random_fn=lambda: 0.5
    )

    result = rep.drain()

    # Exactly max_attempts calls for batch 1, then batch 2 once.
    assert transport.calls == 6
    assert seqs_in(transport.bodies) == [1, 2] * 5 + [3, 4]
    gaps = gaps_in(log)
    assert len(gaps) == 1
    assert gaps[0]["reason"] == "retry_exhausted"
    assert gaps[0]["declared"] is True
    assert gaps[0]["from_seq"] == 1 and gaps[0]["to_seq"] == 2
    assert cursor.acked_seq == 4
    assert result.stopped_on is None
    assert result.abandoned_batches == 1


def test_age_exhaustion_abandons_without_sending(tmp_path):
    """A batch whose oldest event is past max_age is abandoned before any attempt: the
    transport is never called for it, and the next batch drains."""
    log = make_log(tmp_path)
    clock = FakeClock()
    stale = clock.now() - timedelta(hours=7)
    log.append("tester", "test.event", {"n": 0}, ts=stale)
    log.append("tester", "test.event", {"n": 1}, ts=stale)
    append_n(log, 2, start=2)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([200])
    rep = Replicator(log, transport, cursor, max_events=2, clock=clock)

    result = rep.drain()

    assert transport.calls == 1
    assert seqs_in(transport.bodies) == [3, 4]
    gaps = gaps_in(log)
    assert len(gaps) == 1
    assert gaps[0]["reason"] == "retry_exhausted"
    assert gaps[0]["from_seq"] == 1 and gaps[0]["to_seq"] == 2
    assert cursor.acked_seq == 4
    assert result.abandoned_batches == 1


def test_age_exhaustion_with_attempts_remaining(tmp_path):
    """Age and attempts are independent bounds: the aged batch is abandoned although it
    never spent a single attempt."""
    log = make_log(tmp_path)
    clock = FakeClock()
    stale = clock.now() - timedelta(hours=7)
    log.append("tester", "test.event", {"n": 0}, ts=stale)
    log.append("tester", "test.event", {"n": 1}, ts=stale)
    append_n(log, 2, start=2)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([200])
    rep = Replicator(
        log,
        transport,
        cursor,
        max_events=2,
        clock=clock,
        budget=RetryBudget(max_attempts=9),
    )

    rep.drain()

    assert transport.calls == 1  # batch 2 only: batch 1 cost no attempt at all
    assert len(gaps_in(log)) == 1


def test_a_batch_that_ages_out_mid_retry_is_abandoned(tmp_path):
    """5xx, then the fake clock's backoff sleep pushes the batch past max_age: the next
    pre-attempt check abandons it rather than retrying into staleness."""
    log = make_log(tmp_path)
    clock = FakeClock()
    borderline = clock.now() - timedelta(seconds=5)
    log.append("tester", "test.event", {"n": 0}, ts=borderline)
    log.append("tester", "test.event", {"n": 1}, ts=borderline)
    append_n(log, 2, start=2)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    # One 500 for the batch that ages out, then a clean 200 for the next one — so the
    # only retry in this test is the one under examination.
    transport = ScriptedTransport([500, 200])
    rep = Replicator(
        log,
        transport,
        cursor,
        max_events=2,
        clock=clock,
        random_fn=lambda: 0.5,
        budget=RetryBudget(max_age=timedelta(seconds=10)),
    )

    result = rep.drain()

    # One attempt at age 5s, then the 6s backoff puts it at 11s: too old to retry.
    assert clock.sleeps == [6.0]
    assert transport.calls == 2
    assert seqs_in(transport.bodies) == [1, 2, 3, 4]
    assert len(gaps_in(log)) == 1
    assert cursor.acked_seq == 4
    assert result.abandoned_batches == 1
    assert result.stopped_on is None


def test_the_gap_span_is_the_batchs_own_event_ts(tmp_path):
    """span_start/span_end are the batch's oldest and newest event_ts — the surrogate
    knows exactly what it dropped, which is why a declared gap beats an inferred one."""
    log = make_log(tmp_path)
    clock = FakeClock()
    base = clock.now()
    log.append("tester", "test.event", {"n": 0}, ts=base - timedelta(hours=3))
    log.append("tester", "test.event", {"n": 1}, ts=base - timedelta(hours=1))
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    rep = Replicator(
        log,
        ScriptedTransport(),
        cursor,
        max_events=2,
        clock=clock,
        budget=RetryBudget(max_age=timedelta(hours=2)),
    )

    rep.drain()

    (gap,) = gaps_in(log)
    from datetime import datetime as _dt

    assert _dt.fromisoformat(gap["span_start"]) == base - timedelta(hours=3)
    assert _dt.fromisoformat(gap["span_end"]) == base - timedelta(hours=1)


def test_abandonment_does_not_lose_the_events_behind_it(tmp_path):
    """Batch 1 abandoned, batches 2-3 fine: every seq behind the gap still reaches the
    transport, and the cursor lands past all of them."""
    log = make_log(tmp_path)
    append_n(log, 6)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([500] * 5 + [200, 200])
    rep = Replicator(
        log, transport, cursor, max_events=2, clock=FakeClock(), random_fn=lambda: 0.5
    )

    result = rep.drain()

    assert seqs_in(transport.bodies) == [1, 2] * 5 + [3, 4, 5, 6]
    assert cursor.acked_seq == 6
    assert result.abandoned_batches == 1
    assert result.stopped_on is None


def test_age_uses_the_batchs_oldest_ts_not_its_first(tmp_path):
    """`min(e.ts ...)`, not `batch[0].ts`. Events are ordered by `seq`, and `ts` need not
    ascend with `seq` — a producer whose clock stepped backwards, or two sensors with
    skewed clocks, puts the oldest event anywhere in the batch.

    Taking the first event's ts flatters the age of exactly that batch, so a batch holding
    something genuinely stale is retried into staleness instead of being abandoned.
    """
    log = make_log(tmp_path)
    clock = FakeClock()
    # seq 1 is recent, seq 2 is ancient: the batch's oldest ts is NOT its first event's.
    log.append("tester", "test.event", {"n": 0}, ts=clock.now() - timedelta(minutes=1))
    log.append("tester", "test.event", {"n": 1}, ts=clock.now() - timedelta(hours=9))
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([200])

    result = Replicator(log, transport, cursor, max_events=2, clock=clock).drain()

    assert transport.calls == 0, "a batch holding a 9h-old event must not be sent"
    assert result.abandoned_batches == 1
    gaps = [e.content for e in log.read_all() if e.type == GAP]
    assert gaps and gaps[0]["reason"] == "retry_exhausted"


def test_abandoning_the_last_batch_still_advances_the_cursor(tmp_path):
    """The advance inside `abandon()` is the whole point of abandoning.

    Every other abandon test has a later batch that acks past the abandoned range, and
    `AckedCursor` never moves backwards — so the cursor lands on the right number whether
    or not `abandon()` advanced it. Deleting that advance passed the entire suite. What it
    restores is the pre-#32 stall: the batch is re-sent and re-abandoned every drain, and
    the tape grows a marker each time.
    """
    log = make_log(tmp_path)
    append_n(log, 2)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([500] * 50)
    rep = Replicator(
        log, transport, cursor, max_events=2, clock=FakeClock(), random_fn=lambda: 0.5
    )

    first = rep.drain()

    assert first.abandoned_batches == 1
    assert cursor.acked_seq == 2, "abandoning must step the cursor past the batch"
    sent_first = len(transport.bodies)
    assert len(gaps_in(log)) == 1

    # The abandoned range is behind the cursor now, so a second drain never re-sends seq
    # 1 or 2. It does send the gap marker itself (seq 3) — that marker is an ordinary
    # event on the surrogate's own tape and replicates like any other.
    rep.drain()

    resent = seqs_in(transport.bodies[sent_first:])
    assert 1 not in resent and 2 not in resent, "the abandoned batch was re-sent"
    assert len([g for g in gaps_in(log) if (g["from_seq"], g["to_seq"]) == (1, 2)]) == 1


def test_a_blank_reason_from_the_host_does_not_crash_the_drain(tmp_path):
    """`" "` is truthy, so it sails past an `or`-fallback and reaches a constructor that
    rejects a blank reason. The marker build sits outside the transport's try, so the
    ValueError escaped `drain()` entirely and the cursor never moved — every later drain
    then crashed on the same batch. That is the head-of-line wedge this issue closes,
    reachable from a remote party's response body.
    """
    log = make_log(tmp_path)
    append_n(log, 1)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([TransportResult(status=400, reason="   ")])

    result = Replicator(log, transport, cursor, clock=FakeClock()).drain()

    assert result.rejected_batches == 1
    assert cursor.acked_seq == 1
    marker = [e.content for e in log.read_all() if e.type == BATCH_REJECTED][0]
    assert marker["reason"].strip(), "a blank reason must fall back to a truthful one"


def test_an_age_abandoned_batch_is_not_counted_as_attempted(tmp_path):
    """`batches_attempted` says "sent to the transport, acked or not". A batch abandoned by
    age is never handed to the transport at all, so counting it makes the field disagree
    with its own comment and with `transport.calls` — and a caller reconciling the two has
    no way to tell a silent send from a skipped one."""
    log = make_log(tmp_path)
    clock = FakeClock()
    stale = clock.now() - timedelta(hours=9)
    log.append("tester", "test.event", {"n": 0}, ts=stale)
    log.append("tester", "test.event", {"n": 1}, ts=stale)
    append_n(log, 2, start=2)
    cursor = AckedCursor(tmp_path / "cursor.json", origin=log.origin)
    transport = ScriptedTransport([200])

    result = Replicator(log, transport, cursor, max_events=2, clock=clock).drain()

    assert result.abandoned_batches == 1
    assert transport.calls == 1, "only the fresh batch was sent"
    assert result.batches_attempted == 1
    assert result.events_attempted == 2
