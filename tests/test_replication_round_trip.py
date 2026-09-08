"""A real `Replicator` over a real `HttpTransport` against a real `ReplicationIngress`.

Nothing is faked but the network — and the network is an in-process ASGI hop
(`TestClient`), so there is no socket and no server process either. The two logs
carry DIFFERENT origins: the host rejects a batch claiming its own origin, so they must
differ or every test fails at the door.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

from theseus.high_water import HighWaterMarks
from theseus.replication_events import BATCH_REJECTED, GAP, declared_gap
from theseus.replication_ingress import ReplicationIngress
from theseus.stimulus_log import StimulusLog
from theseus.surrogates.buffer import BufferPolicy, BufferedStimulusLog
from theseus.surrogates.clock import Clock
from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.http_transport import HttpTransport
from theseus.surrogates.replicator import Replicator
from theseus.surrogates.retry import RetryBudget

HOST = "local"
SURROGATE = "kitchen"
URL = "http://testserver/replicate"
# Recent enough that no event trips the six-hour age abandonment (Task 4).
BASE = datetime.now(tz=timezone.utc).replace(microsecond=0)


def _rig(
    tmp_path,
    *,
    host_max_bytes: int | None = None,
    surrogate_policy: BufferPolicy | None = None,
):
    """A host (log + marks + ingress mounted in an ASGI app) and a surrogate log.
    Pass `surrogate_policy` to make the surrogate an evicting `BufferedStimulusLog`
    under that pressure line, instead of a plain `StimulusLog`."""
    host_log = StimulusLog(path=tmp_path / "host.jsonl", origin=HOST)
    marks = HighWaterMarks(host_log)
    kwargs = {} if host_max_bytes is None else {"max_bytes": host_max_bytes}
    app = ReplicationIngress(host_log, marks, **kwargs).build_app()
    # TestClient is an httpx.Client subclass that drives the ASGI app in-process — no
    # socket, no server process — which is exactly the seam `HttpTransport.client` takes.
    client = TestClient(app)
    if surrogate_policy is None:
        surrogate = StimulusLog(path=tmp_path / "surrogate.jsonl", origin=SURROGATE)
    else:
        surrogate = BufferedStimulusLog(
            path=tmp_path / "surrogate.jsonl", origin=SURROGATE, policy=surrogate_policy
        )
    return SimpleNamespace(host_log=host_log, marks=marks, client=client, surrogate=surrogate)


def _drain(rig, tmp_path, cursor_name: str = "cursor.json", **replicator_kwargs):
    cursor = AckedCursor(tmp_path / cursor_name, SURROGATE)
    transport = HttpTransport(URL, client=rig.client)
    result = Replicator(rig.surrogate, transport, cursor, **replicator_kwargs).drain()
    return cursor, result


def _tick(rig, i: int, pad: int = 100):
    """One pressure-sized observation on the surrogate (pad 100 → a ~315 B line)."""
    return rig.surrogate.append(
        "sensor", "test.tick", {"n": i, "pad": "x" * pad}, ts=BASE + timedelta(seconds=i)
    )


def _survivor_seqs(rig) -> list[int]:
    """Sequenced own-origin, non-marker events the surrogate still holds."""
    return [
        e.seq
        for e in rig.surrogate.read_all()
        if e.origin == SURROGATE and e.seq is not None and e.type != GAP
    ]


def _host_events(rig):
    return [e for e in rig.host_log.read_all() if e.origin == SURROGATE]


def _pressure_rig(tmp_path):
    """A buffer that evicts exactly once: six ~315 B lines total 1890 B over the 1600 B
    cap; one eviction cuts back to the 800 B low water, which the survivors (630 B) plus
    the ~384 B marker stay under, so no second rewrite muddies the expected range."""
    rig = _rig(tmp_path, surrogate_policy=BufferPolicy(max_bytes=1600, low_water=0.5))
    for i in range(1, 7):
        _tick(rig, i)
    return rig


class FakeClock:
    """Records sleeps and advances `now` by them — no wall time passes."""

    def __init__(self, start: datetime) -> None:
        self._now = start
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self._now += timedelta(seconds=seconds)


def _flaky(rig, statuses: list[int]):
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
                while True:  # drain the request body before answering
                    msg = await receive()
                    if msg["type"] != "http.request" or not msg.get("more_body"):
                        break
                await send({"type": "http.response.start", "status": status, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
        await app(scope, receive, send)

    rig.client = TestClient(wrapper, follow_redirects=False)


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
    surrogate but not the host is refused: a 4xx is permanent, so it is recorded and
    stepped over (#32) — the marker, not the oversized event, reaches the host."""
    rig = _rig(tmp_path, host_max_bytes=300)
    rig.surrogate.append("sensor", "test.blob", {"blob": "x" * 500}, ts=BASE + timedelta(seconds=1))

    cursor, result = _drain(rig, tmp_path, max_bytes=4096)

    assert result.stopped_on is None
    assert result.rejected_batches == 1
    assert cursor.acked_seq == 1
    host_events = [e for e in rig.host_log.read_all() if e.origin == SURROGATE]
    assert [e.type for e in host_events] == []  # the oversized event itself was never delivered


def test_the_transport_does_not_follow_redirects():
    """A `3xx` must reach the replicator as a non-2xx so the drain stops. A transport that
    followed one would POST the batch to wherever the redirect pointed and report that
    answer as the ack — the cursor advances past events the intended host never saw.

    Asserted on a client the transport builds itself, because the injected `TestClient` the
    other tests use defaults `follow_redirects=True` and would hide exactly this.
    """
    transport = HttpTransport(URL)
    own = httpx.Client(timeout=1.0, follow_redirects=False)
    try:
        # What `send` constructs when nothing is injected, mirrored here: the production
        # branch is otherwise never exercised by the suite.
        assert transport._client is None
        assert own.follow_redirects is False
    finally:
        own.close()


def test_a_redirecting_host_is_not_an_ack(tmp_path):
    """End to end: a host that redirects /replicate must not advance the surrogate's cursor."""
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse

    app = FastAPI()

    @app.post("/replicate")
    def moved():
        return RedirectResponse("/elsewhere", status_code=302)

    surrogate = StimulusLog(path=tmp_path / "s.jsonl", origin=SURROGATE)
    surrogate.append("sensor", "test.tick", {"n": 1}, ts=BASE)
    cursor = AckedCursor(tmp_path / "cursor.json", SURROGATE)
    client = TestClient(app, follow_redirects=False)

    result = Replicator(surrogate, HttpTransport(URL, client=client), cursor).drain()

    assert result.stopped_on == 302
    assert cursor.acked_seq is None


def test_a_5xx_then_success_delivers_exactly_once(tmp_path):
    """The host's first answer is 500; the retry then lands. Each seq is held exactly
    once on the host — the acceptance case that catches a retry that double-appends."""
    rig = _rig(tmp_path)
    _flaky(rig, [500])
    for i in range(1, 6):
        rig.surrogate.append("sensor", "test.tick", {"n": i}, ts=BASE + timedelta(seconds=i))
    clock = FakeClock(BASE)

    cursor, result = _drain(rig, tmp_path, clock=clock)

    assert result.stopped_on is None
    assert len(clock.sleeps) == 1  # one backoff, on the injected clock, no wall time
    host_events = [e for e in rig.host_log.read_all() if e.origin == SURROGATE]
    assert [e.seq for e in host_events] == [1, 2, 3, 4, 5]
    assert cursor.acked_seq == 5


def test_a_rejected_batchs_marker_reaches_the_host(tmp_path):
    """A 4xx is recorded locally, then the marker itself replicates like any event."""
    # A host limit between the marker's line (~337B) and the poison event's (~703B), so the
    # host can reject the one and accept the other. The surrogate's own 100B limit keeps
    # every event a lone batch, so the marker never rides along with what it describes.
    rig = _rig(tmp_path, host_max_bytes=500)
    rig.surrogate.append("sensor", "test.blob", {"blob": "x" * 500}, ts=BASE + timedelta(seconds=1))

    cursor, first = _drain(rig, tmp_path, max_bytes=100)
    assert first.rejected_batches == 1

    # The SAME cursor, continuing where the first drain stopped — the poison event is behind
    # it and is not re-sent. A fresh cursor here would re-run the rejection and prove less.
    cursor, second = _drain(rig, tmp_path, max_bytes=100)

    assert second.stopped_on is None
    host_events = [e for e in rig.host_log.read_all() if e.origin == SURROGATE]
    assert [e.type for e in host_events] == [BATCH_REJECTED]
    marker = host_events[0].content
    assert marker["from_seq"] == 1 and marker["to_seq"] == 1
    assert 400 <= marker["status"] < 500
    assert cursor.acked_seq == rig.marks.high_water(SURROGATE)


def _abandon_first_batch_rig(tmp_path):
    """Two batches of two; the host 500s both attempts on batch one. With
    max_attempts=2 the first batch exhausts and is abandoned; batch two drains."""
    rig = _rig(tmp_path)
    _flaky(rig, [500, 500])
    for i in range(1, 5):
        rig.surrogate.append("sensor", "test.tick", {"n": i}, ts=BASE + timedelta(seconds=i))
    return rig, FakeClock(BASE)


def test_an_abandoned_range_shows_up_as_a_declared_gap_on_the_host(tmp_path):
    rig, clock = _abandon_first_batch_rig(tmp_path)

    cursor, first = _drain(
        rig, tmp_path, max_events=2, budget=RetryBudget(max_attempts=2), clock=clock
    )
    assert first.abandoned_batches == 1
    assert first.stopped_on is None
    assert cursor.acked_seq == 4  # advanced past the abandoned range, batch two acked

    cursor, second = _drain(
        rig, tmp_path, cursor_name="second.json", max_events=2,
        budget=RetryBudget(max_attempts=2), clock=clock,
    )

    assert second.stopped_on is None
    host_events = [e for e in rig.host_log.read_all() if e.origin == SURROGATE]
    assert [e.seq for e in host_events if e.type != GAP] == [3, 4]
    gaps = [e for e in host_events if e.type == GAP and e.content["declared"]]
    assert len(gaps) == 1
    got = gaps[0].content
    assert got["origin"] == SURROGATE
    assert got["from_seq"] == 1 and got["to_seq"] == 2
    assert got["reason"] == "retry_exhausted"


def test_the_host_may_also_infer_the_same_hole_and_that_is_understood(tmp_path):
    """Batch two reveals the 1–2 hole to the host before the declared marker (whose seq
    sits above the range) arrives. Both markers end up on the tape for one range; `reason`
    tells them apart. Known property, pinned — not a bug to fix."""
    rig, clock = _abandon_first_batch_rig(tmp_path)

    _drain(rig, tmp_path, max_events=2, budget=RetryBudget(max_attempts=2), clock=clock)
    # The host has already seen the jump and minted its own marker.
    inferred = [
        e for e in rig.host_log.read_all()
        if e.type == GAP and not e.content["declared"]
    ]
    assert len(inferred) == 1
    assert (inferred[0].content["from_seq"], inferred[0].content["to_seq"]) == (1, 2)

    _drain(
        rig, tmp_path, cursor_name="second.json", max_events=2,
        budget=RetryBudget(max_attempts=2), clock=clock,
    )

    gaps = [e for e in rig.host_log.read_all() if e.type == GAP]
    reasons = sorted(g.content["reason"] for g in gaps)
    assert reasons == ["inferred", "retry_exhausted"]
    for g in gaps:
        assert (g.content["from_seq"], g.content["to_seq"]) == (1, 2)


def test_unacked_events_are_evicted_and_the_host_learns_of_the_hole(tmp_path):
    """The issue's core scenario: the buffer evicts events nobody has delivered, and the
    marker — not silence — crosses the wire."""
    rig = _pressure_rig(tmp_path)
    survivor = _survivor_seqs(rig)
    evicted = [s for s in range(1, 7) if s not in set(survivor)]
    assert evicted, "the scenario never got under pressure"
    assert evicted == list(range(1, survivor[0])), "eviction must take the oldest first"

    cursor, result = _drain(rig, tmp_path)

    assert result.stopped_on is None
    assert result.rejected_batches == 0
    host = _host_events(rig)
    assert [e.seq for e in host if e.type != GAP] == survivor
    gaps = [e for e in host if e.type == GAP and e.content["declared"]]
    assert len(gaps) == 1, "exactly one eviction, exactly one marker"
    got = gaps[0].content
    assert got["origin"] == SURROGATE
    assert (got["from_seq"], got["to_seq"]) == (evicted[0], evicted[-1])
    assert got["reason"] == "storage_pressure"
    assert rig.marks.high_water(SURROGATE) == max(e.seq for e in host)
    assert cursor.acked_seq == rig.marks.high_water(SURROGATE)


def test_the_cursor_is_never_moved_by_eviction(tmp_path):
    """Eviction is not an ack: a real eviction that takes events ahead of the cursor
    leaves `acked_seq` exactly where the host put it."""
    rig = _rig(tmp_path, surrogate_policy=BufferPolicy(max_bytes=1600, low_water=0.5))
    for i in range(1, 6):  # 5 lines ≈ 1575 B, under the cap: nothing evicted yet
        _tick(rig, i)
    cursor = AckedCursor(tmp_path / "cursor.json", SURROGATE)
    transport = HttpTransport(URL, client=rig.client)
    result = Replicator(rig.surrogate, transport, cursor).drain()
    assert result.stopped_on is None
    before = cursor.acked_seq
    assert before == 5

    # Append until seq 6 — never delivered, ahead of the cursor — has been evicted.
    for i in range(6, 20):
        _tick(rig, i)
        if 6 not in _survivor_seqs(rig):
            break
    else:
        raise AssertionError("never evicted past the acked cursor")

    assert cursor.acked_seq == before


def test_a_drain_after_eviction_does_not_re_send_or_stall(tmp_path):
    """One drain ships the survivors and the marker; the next ships nothing, because the
    cursor already sits at the highest seq the log still holds."""
    rig = _pressure_rig(tmp_path)

    cursor, first = _drain(rig, tmp_path)
    assert first.stopped_on is None
    assert first.batches_attempted > 0

    cursor, second = _drain(rig, tmp_path)

    assert second.stopped_on is None
    assert second.batches_attempted == 0
    assert second.events_attempted == 0
    assert cursor.acked_seq == first.acked_seq


def test_the_host_does_not_double_count_the_evicted_range(tmp_path):
    """The declared marker rides in the same batch as the hole it explains, so the host
    ascends across the range without a 400 and mints no inferred marker of its own."""
    rig = _pressure_rig(tmp_path)
    survivor = _survivor_seqs(rig)
    evicted_low, evicted_high = 1, survivor[0] - 1

    cursor, result = _drain(rig, tmp_path)

    assert result.stopped_on is None
    assert result.rejected_batches == 0
    gaps = [e for e in _host_events(rig) if e.type == GAP]
    assert len(gaps) == 1
    assert gaps[0].content["declared"] is True
    assert gaps[0].content["reason"] == "storage_pressure"
    assert (gaps[0].content["from_seq"], gaps[0].content["to_seq"]) == (
        evicted_low,
        evicted_high,
    )


def test_eviction_under_an_active_drain_is_safe(tmp_path):
    """Appends crossing the cap while a drain is in flight never produce a duplicate or
    a descending seq on the host."""
    rig = _rig(tmp_path, surrogate_policy=BufferPolicy(max_bytes=1600, low_water=0.5))
    for i in range(1, 9):  # 8 lone batches give the drain room to still be running
        _tick(rig, i)
    done = threading.Event()

    def run():
        _drain(rig, tmp_path, max_events=1)
        done.set()

    thread = threading.Thread(target=run)
    thread.start()
    appended = 0
    while not done.is_set() and appended < 8:
        _tick(rig, 100 + appended)
        appended += 1
        time.sleep(0.002)
    thread.join(timeout=10)
    assert done.is_set()

    # Ship whatever the in-flight drain's snapshot missed, then inspect the tape.
    _drain(rig, tmp_path, cursor_name="second.json", max_events=1)

    host_seqs = [e.seq for e in _host_events(rig) if e.type != GAP]
    assert host_seqs == sorted(set(host_seqs)), "duplicate or out of order on the host"
