from __future__ import annotations

import json
import runpy

import pytest

from theseus.assembly import AgentSpec, InterfaceSpec, ModelSpec, PairingSpec, assemble, build_agent
from theseus.context_assembler import ContextAssembler
from theseus.durable_delivery import DeliveryJournal, DeliveryOutcome, DurableOutbox
from theseus.stimulus_log import PairedStimulusLog, StimulusLog
from theseus.telegram_observer import TELEGRAM_TRANSPORT, TelegramObserver
from theseus.durable_delivery import DurableInbox
from theseus.model_providers import PROVIDER_REGISTRY
from theseus.tools.tool import AssistantTurn


def test_reciprocal_logs_keep_writes_local_and_read_peer_after_startup(tmp_path):
    a_path, b_path = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a = PairedStimulusLog(a_path, b_path, peer_name="Beta")
    assert a.read_peer_all() == []
    missing_context = ContextAssembler(a).assemble_context()
    assert not missing_context.peer_available
    notifications = []
    a.subscribe(notifications.append)
    b = PairedStimulusLog(b_path, a_path, peer_name="Alpha")
    peer_event = b.append("Beta", "decision", {"text": "working"})
    assert a.read_all() == []
    assert notifications == []
    assert a.read_peer_all() == [peer_event]
    assert list(a.peer_log) == [peer_event]
    assert a.peer_log.read_range(peer_event.id, peer_event.id) == [peer_event]
    assert not hasattr(a.peer_log, "append")
    assert ContextAssembler(a).assemble_context().peer_available
    own_event = a.append("Alpha", "decision", {"text": "observed"})
    assert b.read_peer_all() == [own_event]
    assert [event.seq for event in a.read_all()] == [1]
    assert [event.seq for event in b.read_all()] == [1]


def test_peer_alias_and_corruption_handling(tmp_path):
    own = tmp_path / "own.jsonl"
    StimulusLog(own)
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(own)
    with pytest.raises(ValueError, match="distinct"):
        PairedStimulusLog(own, alias, peer_name="Other")
    peer = tmp_path / "peer.jsonl"
    log = PairedStimulusLog(own, peer, peer_name="Other")
    peer.write_text("broken", encoding="utf-8")
    assert log.read_peer_all() == []  # crash-torn final record
    peer.write_text("broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt interior"):
        log.read_peer_all()


def test_peer_context_is_attributed_and_capped(tmp_path):
    a = PairedStimulusLog(tmp_path / "a", tmp_path / "b", peer_name="Beta")
    b = StimulusLog(tmp_path / "b")
    for n in range(12):
        a.append("Alpha", "observation", {"message": f"local {n}"})
        b.append("Beta", "decision", {"message": f"peer {n}"})
    context = ContextAssembler(a, token_budget=1500).assemble_context()
    assert context.peer_name == "Beta"
    assert context.peer_events
    assert "peer 11" in context.peer_events
    assert "local 11" in context.recent_events
    assert all(json.loads(line)["actor"] == "Beta" for line in context.peer_events.splitlines())
    assert len(context.peer_events) / 2.78 <= (context.budget_tokens - 70) * 0.25
    b.append("Beta", "tool_result", {"output": "x" * 100_000})
    huge = ContextAssembler(a, token_budget=300).assemble_context()
    assert len(huge.peer_events) / 2.78 <= 300 * 0.25
    unbounded = ContextAssembler(a, window_size=8, token_budget=None).assemble_context()
    assert len(unbounded.peer_events.splitlines()) == 2
    assert len(unbounded.recent_events.splitlines()) == 8


def test_assembly_round_trip_with_missing_peer_and_auto_core(tmp_path):
    definition = AgentSpec(
        name="Alpha", constitution="Observe Beta.", core="auto",
        models=(ModelSpec("ollama", "example", context=32768),),
        interface=InterfaceSpec("none"),
        pairing=PairingSpec("../../beta/state/stimulus_log.jsonl", "Beta"),
    )
    launcher = assemble(definition, tmp_path / "alpha")
    loaded = runpy.run_path(str(launcher))["SPEC"]
    assert loaded == definition
    agent = build_agent(loaded, tmp_path / "alpha" / "state")
    assert isinstance(agent.core.stimulus_log, PairedStimulusLog)
    assert agent.core.stimulus_log.read_peer_all() == []


def test_ooda_prompt_contains_peer_history_without_copying_it(tmp_path, monkeypatch):
    prompts = []

    class Provider:
        def __init__(self, model):
            pass

        def is_available(self):
            return True

        def complete_with_tools(self, messages, tools):
            prompts.append(messages[-1]["content"])
            return AssistantTurn(text="", tool_calls=[])

    monkeypatch.setitem(PROVIDER_REGISTRY, "paired-fake", Provider)
    home = tmp_path / "alpha" / "state"
    peer_path = tmp_path / "beta" / "state" / "stimulus_log.jsonl"
    peer = StimulusLog(peer_path)
    peer.append("Beta", "observation", {"message": "I found a clue"})
    definition = AgentSpec(
        name="Alpha", constitution="Observe Beta.", core="ooda",
        models=(ModelSpec("paired-fake", "fake", context=32768),),
        pairing=PairingSpec("../../beta/state/stimulus_log.jsonl", "Beta"),
    )
    agent = build_agent(definition, home)
    agent.core.orient()
    assert "<peer_stimulus_log name='Beta'>" in prompts[0]
    assert "I found a clue" in prompts[0]
    assert all("I found a clue" not in event.to_json() for event in agent.core.stimulus_log)


class _Sender:
    def __init__(self):
        self.sent = []

    def send(self, item):
        self.sent.append(item)
        return DeliveryOutcome.delivered(str(len(self.sent)))


def test_durable_telegram_pacing_survives_restart_and_keeps_parts_together(tmp_path):
    clock = [100.0]
    journal = DeliveryJournal(tmp_path / "delivery.sqlite3")
    sender = _Sender()
    outbox = DurableOutbox(
        journal, TELEGRAM_TRANSPORT, sender, now=lambda: clock[0],
        min_group_interval_seconds=5,
    )
    first = outbox.enqueue("group", [{"text": "one"}, {"text": "two"}])
    second = outbox.enqueue("group", [{"text": "three"}])
    assert outbox.drain() == 2
    assert [item.group_id for item in sender.sent] == [first[0].group_id] * 2
    restarted = DurableOutbox(
        DeliveryJournal(journal.path), TELEGRAM_TRANSPORT, sender,
        now=lambda: clock[0], min_group_interval_seconds=5,
    )
    restarted.recover()
    assert restarted.drain() == 0
    clock[0] = 105.0
    assert restarted.drain() == 1
    assert sender.sent[-1].group_id == second[0].group_id


def test_peer_telegram_message_wakes_once_without_peer_file_wake(tmp_path):
    log = PairedStimulusLog(tmp_path / "a", tmp_path / "b", peer_name="Beta")
    StimulusLog(tmp_path / "b").append("Beta", "decision", {})
    wakes = []
    inbox = DurableInbox(DeliveryJournal(tmp_path / "delivery.sqlite3"), TELEGRAM_TRANSPORT)

    class API:
        def get_updates(self, *, offset, timeout):
            return [{"update_id": 7, "message": {
                "message_id": 9, "from": {"id": 22, "is_bot": True},
                "chat": {"id": -100, "type": "supergroup"}, "text": "Ready?",
            }}]

    observer = TelegramObserver(log, lambda: wakes.append("wake"), API(), inbox,
                                allowed_user_ids=(22,), allowed_chat_ids=(-100,))
    assert wakes == []
    observer.poll_once()
    assert wakes == ["wake"]
    assert len(log.read_all()) == 1
    observer.recover_pending()
    assert wakes == ["wake"]
