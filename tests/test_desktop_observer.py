from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import threading
from unittest.mock import Mock

import pytest

from theseus.auto_core import Autocore
from theseus.high_water import HighWaterMarks
from theseus.replication_ingress import ReplicationIngress
from theseus.stimulus_log import StimulusLog
from theseus.surrogates.desktop_observer import DesktopObserver, DesktopState
from theseus.surrogates.cursor import AckedCursor
from theseus.surrogates.runtime import SurrogateRuntime
from theseus.surrogates.transport import TransportResult
from theseus.tools.desktop_state import DesktopStateTool
from theseus.wake_policy import DESKTOP_OBSERVATION


def test_transitions_heartbeat_and_unknown_recovery(tmp_path):
    log = StimulusLog(tmp_path/'edge', origin='windows-desktop')
    state = DesktopState(1, 42, 'editor.exe', 'Document', 'active', False)
    source = Mock()
    source.sample.return_value = state
    clock = [0.0]
    observer = DesktopObserver(log, source, monotonic=lambda: clock[0])
    observer.sample_once()
    clock[0] = 59
    observer.sample_once()
    assert len(log.read_all()) == 1
    clock[0] = 60
    observer.sample_once()
    for updated in (replace(state, title='Other'), replace(state, activity='idle'),
                    DesktopState(activity='idle', locked=True)):
        source.sample.return_value = updated
        observer.sample_once()
    source.sample.side_effect = OSError('unavailable')
    observer.sample_once()
    source.sample.side_effect = None
    source.sample.return_value = state
    observer.sample_once()
    events = log.read_all()
    assert [e.content['reason'] for e in events] == ['initial', 'heartbeat'] + ['changed']*5
    assert events[-2].content['state']['locked'] is None
    assert events[-1].content['state']['process_name'] == 'editor.exe'
    assert all(e.type == DESKTOP_OBSERVATION for e in events)


def test_failed_append_is_retried(tmp_path):
    log = Mock()
    log.append.side_effect = [OSError('disk'), None]
    observer = DesktopObserver(log, Mock(sample=lambda: DesktopState()))
    with pytest.raises(OSError):
        observer.sample_once()
    observer.sample_once()
    assert log.append.call_count == 2
    assert log.append.call_args.kwargs['content']['reason'] == 'initial'


def test_runtime_replication_and_shutdown(tmp_path):
    edge = StimulusLog(tmp_path/'edge', origin='windows-desktop')
    host = StimulusLog(tmp_path/'host', origin='host')
    ingress = ReplicationIngress(host, HighWaterMarks(host), on_arrival=lambda: None)
    ingress._trigger.request = Mock()
    delivered = threading.Event()

    class Transport:
        def send(self, body):
            result = ingress.ingest(body)
            delivered.set()
            return TransportResult(status=result.status)

    source = Mock(sample=lambda: DesktopState(activity='active', locked=False))
    observer = DesktopObserver(edge, source, poll_seconds=60)
    runtime = SurrogateRuntime(edge, Transport(), AckedCursor(tmp_path/'cursor', edge.origin),
                               desktop_observer=observer)
    try:
        runtime.start()
        runtime.start()
        assert delivered.wait(3)
        assert DesktopStateTool(edge.origin, host).execute().details['available']
        ingress._trigger.request.assert_not_called()
    finally:
        runtime.stop()
    assert observer._thread is None
    assert runtime._flush_thread is None
    runtime.stop()
    delivered.clear()
    try:
        runtime.start()
        assert delivered.wait(3)
    finally:
        runtime.stop()
    assert len(host.read_all()) == 2


def test_both_wake_paths_and_mixed_batch(tmp_path):
    core = Autocore(name='test', home_directory=tmp_path/'home', tools={})
    core._consume_wake()
    host = core.stimulus_log
    ingress = ReplicationIngress(host, HighWaterMarks(host), on_arrival=core.wake)
    ingress._trigger.request = Mock()
    edge = StimulusLog(tmp_path/'edge', origin='windows-desktop')
    observer = DesktopObserver(edge, Mock(sample=lambda: DesktopState()))
    observer.sample_once()
    ambient = edge.read_all()[0]
    assert ingress.ingest(ambient.to_json()+'\n').status == 200
    ingress._trigger.request.assert_not_called()
    assert not core._wake.is_set()
    # Mixed batch contains a duplicate ambient event and a new chat event.
    chat = edge.append('user', 'chat_message', {'message': 'hello'})
    batch = ambient.to_json()+'\n'+chat.to_json()+'\n'
    assert ingress.ingest(batch).appended == 1
    ingress._trigger.request.assert_called_once()
    assert core._wake.is_set()
    core._consume_wake()
    ingress._trigger.request.reset_mock()
    assert ingress.ingest(batch).appended == 0
    assert not core._wake.is_set()
    ingress._trigger.request.assert_not_called()


def test_custom_wake_policy_can_include_ambient(tmp_path):
    core = Autocore(name='test', home_directory=tmp_path/'home', tools={}, wake_on=lambda e: True)
    core._consume_wake()
    core.stimulus_log.append('observer', DESKTOP_OBSERVATION, {})
    assert core._wake.is_set()


@pytest.mark.parametrize('offset,stale', [(0, False), (-180, True), (60, True)])
def test_latest_state_recovered_after_restart_and_age(tmp_path, offset, stale):
    path = tmp_path/'host'
    log = StimulusLog(path, origin='host')
    edge = StimulusLog(tmp_path/'edge', origin='windows-desktop')
    captured = datetime.now(timezone.utc) + timedelta(seconds=offset)
    first = edge.append('desktop_observer', DESKTOP_OBSERVATION,
                        {'captured_at': captured.isoformat(), 'state': {'title': 'old'}})
    second = edge.append('desktop_observer', DESKTOP_OBSERVATION,
                         {'captured_at': captured.isoformat(), 'state': {'title': 'new'}})
    # File arrival order deliberately differs from source order.
    log.append_many([second, first])
    result = DesktopStateTool(edge.origin, StimulusLog(path, origin='host')).execute().details
    assert result['observation']['state']['title'] == 'new'
    assert result['stale'] is stale
    assert result['clock_skew'] is (offset > 0)
    assert not DesktopStateTool('other', log).execute().details['available']


@pytest.mark.parametrize('timestamp', ['bad', '2026-09-28T12:00:00', None])
def test_bad_timestamp_is_not_fresh(tmp_path, timestamp):
    log = StimulusLog(tmp_path/'log', origin='windows-desktop')
    log.append('observer', DESKTOP_OBSERVATION, {'captured_at': timestamp})
    result = DesktopStateTool(log.origin, log).execute().details
    assert result['stale'] and not result['available']


def test_host_tool_registration_and_edge_opt_out(tmp_path):
    from theseus.agents.surrogate_host import build_surrogate_host
    from theseus.surrogate_service import SurrogateService
    from theseus.surrogates.windows_surrogate import build_windows_surrogate

    log = StimulusLog(tmp_path/'host', origin='host')
    host = build_surrogate_host(log, HighWaterMarks(log), lambda: None, target='my-pc')
    assert host.desktop_state_tool.target == 'my-pc'
    core = Mock(stimulus_log=log, tools={})
    service = SurrogateService(core, Mock(target='my-pc', host='127.0.0.1', port=8800))
    assert core.tools['desktop_state'].target == 'my-pc'
    service.stop()
    app = build_windows_surrogate('http://host', data_dir=tmp_path/'edge',
                                  headless=True, observe_desktop=False)
    assert app.runtime._desktop_observer is None
    app.runtime.stop()


def test_ingress_wake_override(tmp_path):
    host = StimulusLog(tmp_path/'host', origin='host')
    edge = StimulusLog(tmp_path/'edge', origin='edge')
    ingress = ReplicationIngress(host, HighWaterMarks(host), on_arrival=lambda: None,
                                 wake_on=lambda e: True)
    ingress._trigger.request = Mock()
    event = edge.append('observer', DESKTOP_OBSERVATION, {})
    ingress.ingest(event.to_json()+'\n')
    ingress._trigger.request.assert_called_once()
