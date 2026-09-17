from __future__ import annotations

import subprocess
import threading
import time

import pytest

from theseus.assembly import AgentSpec, InterfaceSpec, MemorySpec, ModelSpec, build_agent
from theseus.auto_core import Autocore
from theseus.deployment_control import (
    ActivationStore,
    DeploymentController,
    LifecycleStatusStore,
    OperationJournal,
)


def managed_spec() -> AgentSpec:
    return AgentSpec(
        name="managed-test",
        constitution="test",
        core="auto",
        models=(ModelSpec("ollama", "offline"),),
        interface=InterfaceSpec("none"),
        memory=MemorySpec(),
    )


def test_shutdown_during_cadence_sleep_starts_no_next_step(tmp_path):
    core = Autocore("test", tmp_path / "state", {})
    sleeping = threading.Event()
    calls = []
    core.step = lambda: calls.append("step")

    def sleep():
        sleeping.set()
        core._shutdown_requested.wait(5)
        return True

    core._sleep = sleep
    thread = threading.Thread(target=core.loop)
    thread.start()
    assert sleeping.wait(1)
    core.request_shutdown()
    thread.join(1)
    assert not thread.is_alive()
    assert calls == ["step"]


def test_shutdown_during_step_allows_full_step_then_stops(tmp_path):
    core = Autocore("test", tmp_path / "state", {})
    entered = threading.Event()
    release = threading.Event()
    completed = []

    def step():
        entered.set()
        assert release.wait(2)
        completed.append(True)  # represents action, reminders, and consolidation

    core.step = step
    thread = threading.Thread(target=core.loop)
    thread.start()
    assert entered.wait(1)
    core.request_shutdown()
    release.set()
    thread.join(1)
    assert completed == [True]
    assert not thread.is_alive()


def test_shutdown_during_consolidation_finishes_consolidation_only(tmp_path):
    core = Autocore("test", tmp_path / "state", {})
    entered = threading.Event()
    release = threading.Event()
    consolidated = []

    class Consolidator:
        def tick(self):
            entered.set()
            assert release.wait(2)
            consolidated.append(True)

    core.memory_consolidator = Consolidator()

    def step():
        core.memory_consolidator.tick()

    core.step = step
    thread = threading.Thread(target=core.loop)
    thread.start()
    assert entered.wait(1)
    core.request_shutdown()
    release.set()
    thread.join(1)
    assert consolidated == [True]
    assert not thread.is_alive()


def _managed_agent(tmp_path):
    agent = build_agent(managed_spec(), tmp_path / "state")
    control = ActivationStore(tmp_path / "control", "deployment")
    control.set("active")
    status = tmp_path / "logs" / "lifecycle-status.json"
    return agent, control.path, status


def test_managed_drain_joins_workers_and_writes_clean_ack(tmp_path):
    agent, control, status = _managed_agent(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def loop():
        entered.set()
        release.wait(2)

    agent.core.loop = loop
    result = []
    supervisor = threading.Thread(target=lambda: result.append(agent.run_managed(
        deployment_id="deployment", agent_id="agent", control_path=control,
        status_path=status, shutdown_timeout_seconds=1,
    )))
    supervisor.start()
    assert entered.wait(1)
    agent.request_shutdown()
    release.set()
    supervisor.join(2)
    record = LifecycleStatusStore(status, "agent").read()
    assert result == [True]
    assert record is not None and record.state == "stopped" and record.clean is True


def test_deadline_and_child_writer_never_produce_clean_ack(tmp_path):
    agent, control, status = _managed_agent(tmp_path)
    entered = threading.Event()
    release = threading.Event()

    def loop():
        entered.set()
        release.wait(2)

    agent.core.loop = loop
    result = []
    supervisor = threading.Thread(target=lambda: result.append(agent.run_managed(
        deployment_id="deployment", agent_id="agent", control_path=control,
        status_path=status, shutdown_timeout_seconds=0.05,
    )))
    supervisor.start()
    assert entered.wait(1)
    agent.request_shutdown()
    supervisor.join(1)
    release.set()
    record = LifecycleStatusStore(status, "agent").read()
    assert result == [False]
    assert record is not None and record.clean is False
    assert "threads still alive" in (record.detail or "")


def test_live_tool_child_is_forced_down_and_marks_shutdown_unclean(tmp_path):
    agent, control, status = _managed_agent(tmp_path)

    class ChildWriter:
        terminated = False

        def has_live_children(self):
            return not self.terminated

        def terminate_children(self):
            self.terminated = True

    child = ChildWriter()
    agent.core.tools["child"] = child
    agent.core.loop = lambda: agent.core._shutdown_requested.wait(1)
    result = []
    supervisor = threading.Thread(target=lambda: result.append(agent.run_managed(
        deployment_id="deployment", agent_id="agent", control_path=control,
        status_path=status, shutdown_timeout_seconds=1,
    )))
    supervisor.start()
    while LifecycleStatusStore(status, "agent").read() is None:
        time.sleep(0.005)
    agent.request_shutdown()
    supervisor.join(2)
    record = LifecycleStatusStore(status, "agent").read()
    assert result == [False] and child.terminated is True
    assert record is not None and record.clean is False
    assert "child processes" in (record.detail or "")


@pytest.mark.parametrize("state", [None, "suspended", "retired"])
def test_absent_suspended_or_retired_activation_blocks_boot(tmp_path, state):
    agent = build_agent(managed_spec(), tmp_path / "state")
    activation = ActivationStore(tmp_path / "control", "deployment")
    if state is not None:
        activation.set(state)
    with pytest.raises(PermissionError):
        agent.run_managed(
            deployment_id="deployment", agent_id="agent", control_path=activation.path,
            status_path=tmp_path / "status.json", shutdown_timeout_seconds=1,
        )


def test_stale_lifecycle_ack_is_rejected(tmp_path):
    store = LifecycleStatusStore(tmp_path / "status.json", "agent")
    stale = store.running()
    current = store.running()
    with pytest.raises(RuntimeError, match="stale"):
        store.stopped(stale.run_id, clean=True)
    assert store.stopped(current.run_id, clean=True).clean is True


def test_interrupted_operation_blocks_start_and_preserves_recovery_context(tmp_path):
    journal = OperationJournal(tmp_path / "root" / "control", "deployment")
    interrupted = journal.begin("migration", ("agent",))
    journal.transition(interrupted.operation_id, "copying")
    calls = []

    def run(*args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    controller = DeploymentController(
        root=tmp_path / "root", deployment_id="deployment", agent_ids=("agent",),
        compose_file=tmp_path / "compose.yaml", run=run,
    )
    with pytest.raises(RuntimeError, match="inspect interrupted"):
        controller.start()
    recovery = controller.interrupted_operation()
    assert recovery is not None
    assert recovery.phase == "copying" and recovery.prior_running_services == ("agent",)
    assert calls == []


@pytest.mark.parametrize("kind", ["backup", "restore", "migration"])
def test_host_data_operations_share_lock_and_durable_journal(tmp_path, kind):
    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="agent\n", stderr="")

    controller = DeploymentController(
        root=tmp_path / "root", deployment_id="deployment", agent_ids=("agent",),
        compose_file=tmp_path / "compose.yaml", run=run,
    )
    with controller.operation(kind) as operation:
        assert operation.prior_running_services == ("agent",)
        controller.journal.transition(operation.operation_id, "copying")
    finished = controller.journal.read()
    assert finished is not None and finished.kind == kind and finished.phase == "completed"


def test_controller_stop_requires_fresh_running_ack(tmp_path):
    root = tmp_path / "root"
    status = root / "data" / "agents" / "agent" / "logs" / "lifecycle-status.json"
    lifecycle = LifecycleStatusStore(status, "agent")
    lifecycle.stopped(lifecycle.running().run_id, clean=True)  # stale clean record

    def run(command, **kwargs):
        stdout = "agent\n" if "ps" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    controller = DeploymentController(
        root=root, deployment_id="deployment", agent_ids=("agent",),
        compose_file=tmp_path / "compose.yaml", run=run,
    )
    with pytest.raises(RuntimeError, match="stale or missing"):
        controller.stop(timeout_seconds=1)
    assert controller.journal.read().phase == "failed"


def test_preflight_reads_local_state_without_selecting_model(tmp_path):
    agent = build_agent(managed_spec(), tmp_path / "state")
    agent.core._select_model_provider = lambda: pytest.fail("preflight selected a model")
    assert agent.preflight()["ready"] is True
