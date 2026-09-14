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
from theseus.replication_events import GAP, declared_gap
from theseus.replication_ingress import ReplicationIngress
from theseus.stimulus_log import StimulusLog
from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.http_transport import HttpTransport
from theseus.surrogates.replicator import Replicator

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
    """Spec acceptance scenario 04: a declared hole and later events drain normally."""
    rig = _rig(tmp_path)
    rig.surrogate.append("sensor", "test.tick", {"n": 1}, ts=BASE)
    rig.surrogate.append(
        "sensor",
        GAP,
        declared_gap(
            origin=SURROGATE,
            from_seq=2,
            to_seq=2,
            reason="link_down",
            span_start=BASE + timedelta(seconds=1),
            span_end=BASE + timedelta(seconds=2),
        ),
        ts=BASE + timedelta(seconds=2),
    )
    rig.surrogate.append("sensor", "test.tick", {"n": 3}, ts=BASE + timedelta(seconds=3))

    cursor, result = _drain(rig, tmp_path)
    events = _host_events(rig)
    gap = next(event for event in events if event.type == GAP)

    assert gap.content["declared"] is True
    assert gap.content["from_seq"] == 2 and gap.content["to_seq"] == 2
    assert gap.content["reason"] == "link_down"
    assert [event.content["n"] for event in events if event.type == "test.tick"] == [1, 3]
    assert result.stopped_on is None
    assert cursor.acked_seq == rig.marks.high_water(SURROGATE) == 3


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
