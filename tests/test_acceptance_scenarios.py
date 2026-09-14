"""Spec acceptance suite (§ Acceptance scenarios), entirely offline.

Scenarios 1–9 and 12 use an upstream in-process ``TestClient`` rig; the SSE
scenarios 10–11 use the real ``serve`` socket.  No test uses a live network.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from fastapi.testclient import TestClient

from theseus.high_water import HighWaterMarks
from theseus.replication_events import BATCH_REJECTED, GAP, declared_gap
from theseus.replication_ingress import ReplicationIngress
from theseus.stimulus_log import StimulusLog
from theseus.surrogates.buffer import BufferPolicy, BufferedStimulusLog
from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.http_transport import HttpTransport
from theseus.surrogates.replicator import Replicator
from theseus.surrogates.retry import RetryBudget

HOST = "local"
SURROGATE = "kitchen"
URL = "http://testserver/replicate"
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


def _drain(rig, tmp_path, cursor_name="cursor.json", **kwargs):
    cursor = AckedCursor(tmp_path / cursor_name, SURROGATE)
    transport = HttpTransport(URL, client=rig.client)
    result = Replicator(rig.surrogate, transport, cursor, **kwargs).drain()
    return cursor, result


def _host_events(rig):
    return [event for event in rig.host_log.read_all() if event.origin == SURROGATE]


def _pressure_rig(tmp_path):
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
    """Spec acceptance scenario 01: normal batch appends in arrival order."""
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
    """Spec acceptance scenario 02: duplicate batch returns 2xx and appends nothing."""
    rig = _rig(tmp_path)
    for i in range(1, 4):
        rig.surrogate.append("sensor", "test.tick", {"n": i}, ts=BASE + timedelta(seconds=i))
    _drain(rig, tmp_path)
    before = len(_host_events(rig))
    response = rig.client.post(URL, content=_body(rig.surrogate.read_all()))

    assert 200 <= response.status_code < 300
    assert len(_host_events(rig)) == before


def test_scenario_04_declared_gap_advances_high_water_past_the_hole(tmp_path):
    """Spec acceptance scenario 04: a declared marker lets high-water pass a hole."""
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
    """Spec acceptance scenario 05: a sequence jump is accepted and diagnosed."""
    rig = _rig(tmp_path)
    later = rig.surrogate.append("sensor", "test.tick", {"n": 4}, ts=BASE + timedelta(seconds=3))
    payload = json.loads(later.to_json())
    payload["seq"] = 4
    response = rig.client.post(URL, content=json.dumps(payload) + "\n")

    assert 200 <= response.status_code < 300
    assert [event.seq for event in _host_events(rig) if event.type == "test.tick"] == [4]
    inferred = response.json()["inferred_gaps"]
    assert inferred == [{"from_seq": 1, "to_seq": 3}]


def test_scenario_03_lost_ack_retry_no_duplicate(tmp_path):
    """Spec acceptance scenario 03: lost ack retry creates no duplicate events."""
    rig = _rig(tmp_path)
    for i in range(1, 6):
        rig.surrogate.append("sensor", "test.tick", {"n": i}, ts=BASE + timedelta(seconds=i))

    _drain(rig, tmp_path, cursor_name="first.json")
    cursor, result = _drain(rig, tmp_path, cursor_name="retry.json", clock=FakeClock(BASE))
    seqs = [event.seq for event in _host_events(rig)]

    assert result.stopped_on is None
    assert seqs == [1, 2, 3, 4, 5]
    assert len(seqs) == len(set(seqs))
    assert cursor.acked_seq == 5


def test_scenario_06_oversized_or_malformed_batch_is_rejected_and_marked(tmp_path):
    """Spec acceptance scenario 06: a host 4xx advances the surrogate and is marked."""
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


def _rig_with_host_limit(tmp_path, limit):
    host_log = StimulusLog(tmp_path / "host.jsonl", origin=HOST)
    marks = HighWaterMarks(host_log)
    app = ReplicationIngress(host_log, marks, max_bytes=limit).build_app()
    return SimpleNamespace(host_log=host_log, marks=marks, client=TestClient(app), surrogate=StimulusLog(tmp_path / "surrogate.jsonl", origin=SURROGATE))


def test_scenario_07_retry_exhaustion_abandons_and_drains_the_rest(tmp_path):
    """Spec acceptance scenario 07: exhausted batches do not block later batches."""
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


def test_scenario_09_hour_offline_then_drains_backlog_in_chunks(tmp_path):
    """Spec acceptance scenario 09: an hour of backlog drains in bounded batches."""
    rig = _rig(tmp_path)
    n = 240
    max_events = 20
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


def test_scenario_08_storage_pressure_evicts_declares_and_keeps_observing(tmp_path):
    """Spec acceptance scenario 08: eviction records a hole without pausing observation."""
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
