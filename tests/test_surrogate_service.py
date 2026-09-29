from dataclasses import replace
from datetime import datetime, timezone
import runpy
import threading

from fastapi.testclient import TestClient
import pytest

from theseus.assembly import AgentSpec, InterfaceSpec, ModelSpec, SurrogateSpec, build_agent, assemble
from theseus.deployment import DeploymentSpec
from theseus.deployment_bundle import assemble_compose
from theseus.deployment_control import ActivationStore, LifecycleStatusStore
from theseus.stimulus_log import StimulusEvent


def spec():
    return AgentSpec(name='test', constitution='test', core='auto',
                     models=(ModelSpec('ollama', 'offline'),),
                     interface=InterfaceSpec('none'), surrogate=SurrogateSpec())


def test_roundtrip_and_replication_share_existing_core(tmp_path):
    s = spec()
    launcher = assemble(s, tmp_path / 'launcher')
    assert runpy.run_path(str(launcher))['SPEC'] == s
    agent = build_agent(s, tmp_path / 'state')
    service = agent.surrogate
    fired = threading.Event()
    # Coalescing ingress callback is captured at construction.
    service.ingress._trigger._callback = fired.set
    event = StimulusEvent(id='test-1', ts=datetime.now(timezone.utc), actor='user',
                          type='chat_message', content={'message': 'test'},
                          origin='windows-desktop', seq=1)
    with TestClient(service.app) as client:
        assert client.get('/health').json()['target'] == 'windows-desktop'
        for _ in range(2):
            assert client.post('/replicate', content=event.to_json()+'\n').status_code == 200
        assert fired.wait(2)
        assert len([e for e in agent.core.stimulus_log.read_all() if e.origin == event.origin and e.seq == event.seq]) == 1
        agent.core.tools['say_to_surrogate'].execute('hello')
        assert len(service.feed.pending('windows-desktop', after=None)) == 1
    assert service.ingress._trigger._thread is None


def test_compose_exposes_only_host_loopback(tmp_path):
    s = replace(spec(), surrogate=SurrogateSpec(host='0.0.0.0'))
    out = assemble_compose(DeploymentSpec(id='test', agents={'test': s}),
                           tmp_path/'bundle', definition_root=tmp_path)
    assert '"127.0.0.1:8800:8800"' in (out/'compose.yaml').read_text()
    with pytest.raises(ValueError, match='ports must be unique'):
        assemble_compose(DeploymentSpec(id='test', agents={'one': s, 'two': s}),
                         tmp_path/'bad', definition_root=tmp_path)


def test_managed_shutdown_owns_surrogate_worker(tmp_path):
    agent = build_agent(spec(), tmp_path/'state')
    entered = threading.Event()
    # Keep the server deterministic here; its protocol is tested above.
    def serve():
        entered.set()
        while not agent.surrogate.server.should_exit:
            agent._shutdown_requested.wait(.01)
    agent.surrogate.run = serve
    agent.core.loop = lambda: agent.core._shutdown_requested.wait(3)
    control = ActivationStore(tmp_path/'control', 'test')
    control.set('active')
    status = tmp_path/'status.json'
    results = []
    thread = threading.Thread(target=lambda: results.append(agent.run_managed(
        deployment_id='test', agent_id='test', control_path=control.path,
        status_path=status, shutdown_timeout_seconds=2)))
    thread.start()
    assert entered.wait(2)
    agent.request_shutdown()
    thread.join(3)
    assert results == [True]
    assert LifecycleStatusStore(status, 'test').read().clean


def test_telegram_and_surrogate_tools_coexist(tmp_path, monkeypatch):
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'offline-test-token')
    s = replace(spec(), interface=InterfaceSpec('telegram', allowed_user_ids=(123,)))
    agent = build_agent(s, tmp_path/'state')
    assert agent.observer.stimulus_log is agent.core.stimulus_log
    assert 'say_to_surrogate' in agent.core.tools
    assert any(type(tool).__name__ == 'TelegramTool' for tool in agent.core.tools.values())
    agent._close_resources()


def test_live_server_drains_connected_command_stream(tmp_path):
    import socket
    import time
    import httpx

    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    agent = build_agent(replace(spec(), surrogate=SurrogateSpec(port=port)), tmp_path/'state')
    agent.core.loop = lambda: agent.core._shutdown_requested.wait(15)
    control = ActivationStore(tmp_path/'control', 'test')
    control.set('active')
    status = tmp_path/'status.json'
    results = []
    thread = threading.Thread(target=lambda: results.append(agent.run_managed(
        deployment_id='test', agent_id='test', control_path=control.path,
        status_path=status, shutdown_timeout_seconds=8)))
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not agent.surrogate.server.started and time.monotonic() < deadline:
            time.sleep(.01)
        assert agent.surrogate.server.started
        with httpx.stream('GET', f'http://127.0.0.1:{port}/commands/windows-desktop', timeout=5) as stream:
            assert stream.status_code == 200
            assert next(stream.iter_lines()) == 'retry: 2000'
            agent.request_shutdown()
            thread.join(10)
        assert results == [True]
        assert LifecycleStatusStore(status, 'test').read().clean
        assert agent.surrogate.ingress._trigger._thread is None
    finally:
        agent.request_shutdown()
        thread.join(10)
