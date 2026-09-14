"""Spec acceptance suite (§ Acceptance scenarios), entirely offline.

Scenarios 1–9 and 12 use an upstream in-process ``TestClient`` rig; the SSE
scenarios 10–11 use the real ``serve`` socket.  No test uses a live network.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from theseus.high_water import HighWaterMarks
from theseus.command_feed import CommandFeed
from theseus.command_reports import Failed, is_report
from theseus.commands import command_content, command_type
from theseus.replication_events import BATCH_REJECTED, GAP, declared_gap
from theseus.replication_ingress import ReplicationIngress
from theseus.stimulus_log import StimulusLog
from theseus.surrogates.buffer import BufferPolicy, BufferedStimulusLog
from theseus.surrogates.command_executor import CommandExecutor
from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.http_transport import HttpTransport
from theseus.surrogates.replicator import Replicator
from theseus.surrogates.retry import RetryBudget
from theseus.surrogates.sse_command_channel import SseCommandChannel

HOST = "local"
SURROGATE = "kitchen"
TARGET = "tam"
URL = "http://testserver/replicate"
FAST = {"heartbeat_seconds": 0.05, "poll_seconds": 0.01}
FAST_BUDGET = RetryBudget(base_seconds=0.01, multiplier=2.0, jitter=0.0)
BASE = datetime.now(tz=timezone.utc).replace(microsecond=0)


class FakeClock:
    """A clock whose sleeps advance simulated time without waiting."""

    def __init__(self, start: datetime) -> None:
        self._now = start
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self._now += timedelta(seconds=seconds)


def _rig(tmp_path):
    host_log = StimulusLog(tmp_path / "host.jsonl", origin=HOST)
    marks = HighWaterMarks(host_log)
    app = ReplicationIngress(host_log, marks).build_app()
    return SimpleNamespace(
        host_log=host_log,
        marks=marks,
        client=TestClient(app),
        surrogate=StimulusLog(tmp_path / "surrogate.jsonl", origin=SURROGATE),
    )


def _rig_with_host_limit(tmp_path, limit):
    """Like ``_rig`` but the host ingress caps a batch at ``limit`` bytes; the
    surrogate has no such cap, so it can form a batch the host must answer 4xx."""
    host_log = StimulusLog(tmp_path / "host.jsonl", origin=HOST)
    marks = HighWaterMarks(host_log)
    app = ReplicationIngress(host_log, marks, max_bytes=limit).build_app()
    return SimpleNamespace(host_log=host_log, marks=marks, client=TestClient(app), surrogate=StimulusLog(tmp_path / "surrogate.jsonl", origin=SURROGATE))


def _drain(rig, tmp_path, cursor_name="cursor.json", **kwargs):
    cursor = AckedCursor(tmp_path / cursor_name, SURROGATE)
    transport = HttpTransport(URL, client=rig.client)
    result = Replicator(rig.surrogate, transport, cursor, **kwargs).drain()
    return cursor, result


def _host_events(rig):
    return [event for event in rig.host_log.read_all() if event.origin == SURROGATE]


def _pressure_rig(tmp_path):
    """A buffer that evicts exactly once: six ~315 B lines total 1890 B over the 1600 B
    cap; one eviction cuts back to the 800 B low water, which the survivors (630 B) plus
    the ~384 B marker stay under, so no second rewrite muddies the expected range."""
    host_log = StimulusLog(tmp_path / "host.jsonl", origin=HOST)
    marks = HighWaterMarks(host_log)
    app = ReplicationIngress(host_log, marks).build_app()
    surrogate = BufferedStimulusLog(
        tmp_path / "surrogate.jsonl", origin=SURROGATE,
        policy=BufferPolicy(max_bytes=1600, low_water=0.5),
    )
    rig = SimpleNamespace(host_log=host_log, marks=marks, client=TestClient(app), surrogate=surrogate)
    for i in range(1, 7):
        surrogate.append("sensor", "test.tick", {"n": i, "pad": "x" * 100}, ts=BASE + timedelta(seconds=i))
    return rig


def _flaky(rig, statuses):
    """Rebuild the rig's client over an ASGI wrapper that answers the scripted statuses
    on /replicate (in order, one per call) before delegating to the real ingress app.
    The transport and host stay real; only the first answers are bent."""
    app = rig.client.app
    calls = 0

    async def wrapper(scope, receive, send):
        nonlocal calls
        if scope["type"] == "http" and scope["path"] == "/replicate" and calls < len(statuses):
            status = statuses[calls]
            calls += 1
            if status >= 300:
                while (await receive()).get("more_body"):
                    pass
                await send({"type": "http.response.start", "status": status, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
        await app(scope, receive, send)

    rig.client = TestClient(wrapper, follow_redirects=False)


def _body(events):
    return "\n".join(event.to_json() for event in events) + "\n"


def test_scenario_01_normal_batch_appends_in_arrival_order(tmp_path):
    """Spec acceptance scenario 01: Normal batch appends in arrival order; Assembler window sorts by `event_ts`."""
    rig = _rig(tmp_path)
    timestamps = [BASE + timedelta(seconds=3), BASE + timedelta(seconds=1), BASE + timedelta(seconds=2)]
    for i, ts in enumerate(timestamps, 1):
        rig.surrogate.append("sensor", "test.tick", {"n": i}, ts=ts)

    _drain(rig, tmp_path)
    events = _host_events(rig)

    assert [event.seq for event in events] == [1, 2, 3]
    assert [event.ts for event in events] == timestamps
    assert [event.seq for event in sorted(events, key=lambda event: event.ts)] == [2, 3, 1]


def test_scenario_02_duplicate_batch_appends_nothing(tmp_path):
    """Spec acceptance scenario 02: Duplicate batch returns `2xx` and appends nothing."""
    rig = _rig(tmp_path)
    for i in range(1, 4):
        rig.surrogate.append("sensor", "test.tick", {"n": i}, ts=BASE + timedelta(seconds=i))
    _drain(rig, tmp_path)
    before = len(_host_events(rig))
    response = rig.client.post(URL, content=_body(rig.surrogate.read_all()))

    assert 200 <= response.status_code < 300
    assert len(_host_events(rig)) == before


def test_scenario_03_lost_ack_retry_no_duplicate(tmp_path):
    """Spec acceptance scenario 03: Lost ack → retry → `2xx`, no duplicate events on the host log."""
    rig = _rig(tmp_path)
    for i in range(1, 6):
        rig.surrogate.append("sensor", "test.tick", {"n": i}, ts=BASE + timedelta(seconds=i))

    # A first drain fully acks the host; the second drives the lost-ack case with a
    # fresh cursor that believes nothing landed, so it re-sends and the host dedupes.
    _drain(rig, tmp_path, cursor_name="first.json")
    cursor, result = _drain(rig, tmp_path, cursor_name="retry.json")
    seqs = [event.seq for event in _host_events(rig)]

    assert result.stopped_on is None
    assert seqs == [1, 2, 3, 4, 5]
    assert len(seqs) == len(set(seqs))
    assert cursor.acked_seq == 5


def test_scenario_04_declared_gap_advances_high_water_past_the_hole(tmp_path):
    """Spec acceptance scenario 04: Declared gap: surrogate abandons a range, emits `stimulus.gap`, host appends both marker and subsequent events without error; high-water mark advances past the hole."""
    rig = _rig(tmp_path)
    first = rig.surrogate.append("sensor", "test.tick", {"n": 1}, ts=BASE)
    marker = rig.surrogate.append(
        "sensor",
        GAP,
        declared_gap(
            origin=SURROGATE,
            from_seq=2,
            to_seq=4,
            reason="link_down",
            span_start=BASE + timedelta(seconds=1),
            span_end=BASE + timedelta(seconds=4),
        ),
        ts=BASE + timedelta(seconds=5),
    )
    later = rig.surrogate.append("sensor", "test.tick", {"n": 6}, ts=BASE + timedelta(seconds=6))

    # Rewrite the three seqs to 1→5→6 so the batch carries the 2–4 hole the marker
    # declares; the Replicator numbers its own appends densely and can't produce it.
    lines = []
    for event, seq in ((first, 1), (marker, 5), (later, 6)):
        payload = json.loads(event.to_json())
        payload["seq"] = seq
        lines.append(json.dumps(payload))
    response = rig.client.post(URL, content="\n".join(lines) + "\n")

    events = _host_events(rig)
    gap = next(event for event in events if event.type == GAP)
    assert 200 <= response.status_code < 300
    assert (gap.content["from_seq"], gap.content["to_seq"]) == (2, 4)
    assert gap.content["declared"] is True
    assert gap.content["reason"] == "link_down"
    assert [event.content["n"] for event in events if event.type == "test.tick"] == [1, 6]
    assert rig.marks.high_water(SURROGATE) == 6
    assert "inferred_gaps" not in response.json()


def test_scenario_05_inferred_gap_is_recorded_by_the_host(tmp_path):
    """Spec acceptance scenario 05: Inferred gap: `seq` jump with no marker is appended, and the host records an inferred-gap event."""
    rig = _rig(tmp_path)
    later = rig.surrogate.append("sensor", "test.tick", {"n": 4}, ts=BASE + timedelta(seconds=3))
    payload = json.loads(later.to_json())
    payload["seq"] = 4
    response = rig.client.post(URL, content=json.dumps(payload) + "\n")

    assert 200 <= response.status_code < 300
    assert [event.seq for event in _host_events(rig) if event.type == "test.tick"] == [4]
    inferred = response.json()["inferred_gaps"]
    assert inferred == [{"from_seq": 1, "to_seq": 3}]


def test_scenario_06_oversized_or_malformed_batch_is_rejected_and_marked(tmp_path):
    """Spec acceptance scenario 06: Oversized/malformed batch returns `4xx`; surrogate advances and emits `batch_rejected`."""
    # Keep the host limit below the payload while the surrogate can form the batch.
    rig = _rig_with_host_limit(tmp_path, 500)
    poison = rig.surrogate.append("sensor", "test.blob", {"blob": "x" * 500}, ts=BASE)
    response = rig.client.post(URL, content=_body([poison]))
    assert 400 <= response.status_code < 500
    cursor, result = _drain(rig, tmp_path, max_bytes=100)
    assert result.rejected_batches == 1
    assert cursor.acked_seq == 1
    assert _host_events(rig) == []
    cursor, result = _drain(rig, tmp_path)
    markers = [e for e in _host_events(rig) if e.type == BATCH_REJECTED]
    assert result.stopped_on is None and len(markers) == 1
    assert markers[0].content["from_seq"] == markers[0].content["to_seq"] == 1
    assert 400 <= markers[0].content["status"] < 500


def test_scenario_07_retry_exhaustion_abandons_and_drains_the_rest(tmp_path):
    """Spec acceptance scenario 07: Retry exhaustion: undeliverable batch is abandoned after max attempts/age; the channel drains subsequent batches rather than blocking on it."""
    rig = _rig(tmp_path)
    _flaky(rig, [500, 500])
    for i in range(1, 5):
        rig.surrogate.append("sensor", "test.tick", {"n": i}, ts=BASE + timedelta(seconds=i))
    clock = FakeClock(BASE)
    cursor, result = _drain(rig, tmp_path, max_events=2, budget=RetryBudget(max_attempts=2), clock=clock)
    assert result.abandoned_batches >= 1 and result.stopped_on is None
    assert [e.content["n"] for e in _host_events(rig) if e.type == "test.tick"] == [3, 4]
    _drain(rig, tmp_path, cursor_name="marker.json", max_events=2,
           budget=RetryBudget(max_attempts=2), clock=clock)
    assert any(e.type == GAP and e.content["reason"] == "retry_exhausted" for e in _host_events(rig))
    assert cursor.acked_seq == 4
    assert clock.sleeps


def test_scenario_08_storage_pressure_evicts_declares_and_keeps_observing(tmp_path):
    """Spec acceptance scenario 08: Storage pressure: surrogate evicts oldest buffered events, emits `storage_pressure` gap, and keeps observing."""
    rig = _pressure_rig(tmp_path)
    buffered = rig.surrogate.read_all()
    survivors = [e.seq for e in buffered if e.type != GAP]
    gaps = [e for e in buffered if e.type == GAP and e.content["reason"] == "storage_pressure"]
    assert survivors and gaps and min(survivors) > 1
    assert gaps[0].content["from_seq"] == 1
    cursor, result = _drain(rig, tmp_path)
    assert result.stopped_on is None
    host = _host_events(rig)
    assert any(e.type == GAP and e.content["reason"] == "storage_pressure" for e in host)
    assert [e.seq for e in host if e.type != GAP] == survivors


def test_scenario_09_hour_offline_then_drains_backlog_in_chunks(tmp_path):
    """Spec acceptance scenario 09: Surrogate offline 1h, then drains backlog in chunks with correct final high-water mark."""
    rig = _rig(tmp_path)
    n = 240
    max_events = 20
    # An hour's backlog at 15s spacing, drained from a clock an hour past BASE: the
    # hour-span is context for the reader; only the chunking and final mark are asserted.
    clock = FakeClock(BASE + timedelta(hours=1))
    for i in range(1, n + 1):
        rig.surrogate.append(
            "sensor", "test.tick", {"n": i},
            ts=BASE + timedelta(seconds=15 * (i - 1)),
        )

    class CountingClient:
        def __init__(self, client):
            self.client = client
            self.posts = 0

        def post(self, *args, **kwargs):
            self.posts += 1
            return self.client.post(*args, **kwargs)

    client = CountingClient(rig.client)
    rig.client = client
    cursor, result = _drain(rig, tmp_path, max_events=max_events, clock=clock)
    events = _host_events(rig)
    seqs = [event.seq for event in events]

    assert result.stopped_on is None
    assert seqs == list(range(1, n + 1))
    assert len(seqs) == n
    assert client.posts == (n + max_events - 1) // max_events
    assert rig.marks.high_water(SURROGATE) == n
    assert cursor.acked_seq == n
    # No simulated (or real) time spent: every batch acked first try, no backoff.
    assert clock.sleeps == []


def test_scenario_10_command_issued_during_downtime_is_delivered_on_reconnect(tmp_path, serve):
    """Spec acceptance scenario 10: Command issued during surrogate downtime is delivered on reconnect via cursor."""
    host_log = StimulusLog(tmp_path / "host.jsonl", origin=HOST)
    issued = [host_log.append("george", command_type("say"),
                             command_content(target=TARGET, payload={"n": n}))
              for n in range(3)]
    base = serve(CommandFeed(host_log, **FAST).build_app())
    cursor = AckedCursor(tmp_path / "commands.json", f"{SURROGATE}-surrogate")
    channel = SseCommandChannel(
        f"{base}/commands/{TARGET}", cursor, client=httpx.Client(),
        max_reconnects=0, budget=FAST_BUDGET)
    executed = []
    done = threading.Event()

    def consume():
        try:
            for event in channel.stream():
                executed.append(event)
                cursor.advance(event.seq)
                if len(executed) == len(issued):
                    break
        finally:
            done.set()

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    assert done.wait(10), "commands were not delivered"
    channel.close()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert [event.seq for event in executed] == [event.seq for event in issued]
    assert cursor.acked_seq == issued[-1].seq


def test_scenario_11_muted_command_produces_failed_report_on_host_log(tmp_path, serve):
    """Spec acceptance scenario 11: Command executed with muted output produces a `failed` stimulus on the host log."""
    host_log = StimulusLog(tmp_path / "host.jsonl", origin=HOST)
    surrogate_log = StimulusLog(tmp_path / "surrogate.jsonl", origin=SURROGATE)
    issued = host_log.append("george", command_type("say"),
                             command_content(target=TARGET, payload={"n": 1}))
    app = ReplicationIngress(host_log, HighWaterMarks(host_log)).build_app()
    CommandFeed(host_log, **FAST).add_routes(app)
    base = serve(app)
    cmd_cursor = AckedCursor(tmp_path / "commands.json", f"{SURROGATE}-commands")
    up_cursor = AckedCursor(tmp_path / "upstream.json", f"{SURROGATE}-upstream")
    channel = SseCommandChannel(f"{base}/commands/{TARGET}", cmd_cursor,
                                max_reconnects=0, budget=FAST_BUDGET)
    executor = CommandExecutor(surrogate_log, lambda command: Failed("output muted"), cmd_cursor)
    thread = threading.Thread(target=executor.run, args=(channel,), daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not [event for event in surrogate_log.read_all() if is_report(event)]:
        assert time.monotonic() < deadline, "report was not produced"
        time.sleep(0.01)
    channel.close()
    thread.join(timeout=5)
    assert not thread.is_alive()
    Replicator(surrogate_log, HttpTransport(f"{base}/replicate", client=httpx.Client()), up_cursor).drain()
    reports = [event for event in host_log.read_all() if is_report(event)]
    assert len(reports) == 1
    report = reports[0]
    assert report.origin == SURROGATE
    assert report.type == "command_report.failed"
    assert report.content["reason"] == "output muted"
    assert (report.content["command_origin"], report.content["command_seq"]) == (HOST, issued.seq)


def test_scenario_12_clock_skew_is_derivable_from_both_timestamps(tmp_path):
    """Spec acceptance scenario 12: Surrogate clock skewed 900ms: both timestamps present, skew derivable."""
    rig = _rig(tmp_path)
    surrogate_ts = datetime.now(tz=timezone.utc) - timedelta(milliseconds=900)
    rig.surrogate.append("sensor", "test.tick", {"n": 1}, ts=surrogate_ts)

    _drain(rig, tmp_path)
    events = _host_events(rig)

    assert events
    for event in events:
        assert event.ts is not None
        assert event.appended_ts is not None
        skew = event.appended_ts - event.ts
        assert timedelta(milliseconds=900) <= skew < timedelta(milliseconds=900, seconds=5)
