"""A real `Replicator` over a real `HttpTransport` against a real `ReplicationIngress`.

Nothing is faked but the network — and the network is an in-process ASGI hop
(`TestClient`), so there is no socket and no server process either. The two logs
carry DIFFERENT origins: the host rejects a batch claiming its own origin, so they must
differ or every test fails at the door.
"""

from __future__ import annotations

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
BASE = datetime(2026, 9, 6, 12, 0, 0, tzinfo=timezone.utc)


def _rig(tmp_path, *, host_max_bytes: int | None = None):
    """A host (log + marks + ingress mounted in an ASGI app) and a surrogate log."""
    host_log = StimulusLog(path=tmp_path / "host.jsonl", origin=HOST)
    marks = HighWaterMarks(host_log)
    kwargs = {} if host_max_bytes is None else {"max_bytes": host_max_bytes}
    app = ReplicationIngress(host_log, marks, **kwargs).build_app()
    # TestClient is an httpx.Client subclass that drives the ASGI app in-process — no
    # socket, no server process — which is exactly the seam `HttpTransport.client` takes.
    client = TestClient(app)
    surrogate = StimulusLog(path=tmp_path / "surrogate.jsonl", origin=SURROGATE)
    return SimpleNamespace(host_log=host_log, marks=marks, client=client, surrogate=surrogate)


def _drain(rig, tmp_path, cursor_name: str = "cursor.json", **replicator_kwargs):
    cursor = AckedCursor(tmp_path / cursor_name, SURROGATE)
    transport = HttpTransport(URL, client=rig.client)
    result = Replicator(rig.surrogate, transport, cursor, **replicator_kwargs).drain()
    return cursor, result


def test_a_backlog_drains_in_seq_order_across_batches(tmp_path):
    """N above the batch limit takes several batches; the host holds all of them in order."""
    rig = _rig(tmp_path)
    n = 750  # default max_events is 500, so this is at least two batches
    for i in range(1, n + 1):
        rig.surrogate.append("sensor", "test.tick", {"n": i}, ts=BASE + timedelta(seconds=i))

    cursor, result = _drain(rig, tmp_path)

    assert result.stopped_on is None
    assert result.events_attempted == n
    host_events = [e for e in rig.host_log.read_all() if e.origin == SURROGATE]
    assert [e.seq for e in host_events] == list(range(1, n + 1))
    # The surrogate's own clock survives the trip: ts is meaning, not arrival order.
    assert [e.ts for e in host_events] == [BASE + timedelta(seconds=i) for i in range(1, n + 1)]
    assert cursor.acked_seq == n


def test_the_surrogates_cursor_ends_where_the_hosts_mark_ends(tmp_path):
    """The two halves agree on what was delivered."""
    rig = _rig(tmp_path)
    for i in range(1, 11):
        rig.surrogate.append("sensor", "test.tick", {"n": i}, ts=BASE + timedelta(seconds=i))

    cursor, result = _drain(rig, tmp_path)

    assert result.stopped_on is None
    assert cursor.acked_seq == rig.marks.high_water(SURROGATE)


def test_a_re_drain_after_a_lost_ack_is_a_no_op_on_the_host(tmp_path):
    """The host committed, the ack died in flight. The re-send must append nothing."""
    rig = _rig(tmp_path)
    for i in range(1, 26):
        rig.surrogate.append("sensor", "test.tick", {"n": i}, ts=BASE + timedelta(seconds=i))

    _drain(rig, tmp_path, cursor_name="first.json")
    before = (tmp_path / "host.jsonl").read_bytes()

    # A fresh cursor believes nothing was acked — the lost-ack case, end to end.
    cursor, result = _drain(rig, tmp_path, cursor_name="second.json")

    assert result.stopped_on is None  # the host answered 2xx on every re-sent batch
    after = (tmp_path / "host.jsonl").read_bytes()
    assert after == before
    assert cursor.acked_seq == rig.marks.high_water(SURROGATE)


def test_a_declared_gap_survives_the_round_trip(tmp_path):
    """The host re-validates markers at ingress; this proves the surrogate builds ones
    that survive that check."""
    rig = _rig(tmp_path)
    for i in range(1, 4):
        rig.surrogate.append("sensor", "test.tick", {"n": i}, ts=BASE + timedelta(seconds=i))
    content = declared_gap(
        origin=SURROGATE,
        from_seq=2,
        to_seq=2,
        reason="link_down",
        span_start=BASE,
        span_end=BASE + timedelta(minutes=1),
    )
    rig.surrogate.append("sensor", GAP, content, ts=BASE + timedelta(seconds=4))

    cursor, result = _drain(rig, tmp_path)

    assert result.stopped_on is None
    gaps = [e for e in rig.host_log.read_all() if e.type == GAP]
    assert len(gaps) == 1
    got = gaps[0].content
    assert got["declared"] is True
    assert got["origin"] == SURROGATE
    assert got["from_seq"] == 2 and got["to_seq"] == 2
    assert got["reason"] == "link_down"
    assert (got["span_start"], got["span_end"]) == (content["span_start"], content["span_end"])


def test_an_oversized_batch_is_refused_end_to_end(tmp_path):
    """The surrogate's limits are its own view, not the host's. A batch that fits the
    surrogate but not the host is refused, and the cursor does not advance on a 4xx."""
    rig = _rig(tmp_path, host_max_bytes=300)
    rig.surrogate.append("sensor", "test.blob", {"blob": "x" * 500}, ts=BASE + timedelta(seconds=1))

    cursor, result = _drain(rig, tmp_path, max_bytes=4096)

    assert result.stopped_on == 413
    assert cursor.acked_seq is None
    assert [e for e in rig.host_log.read_all() if e.origin == SURROGATE] == []
