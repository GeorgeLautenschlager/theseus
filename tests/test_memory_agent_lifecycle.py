"""Exercise formation and recall through assembled agents, with scripted models."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from theseus.assembly import AgentSpec, InterfaceSpec, MemorySpec, ModelSpec, PairingSpec, build_agent
from theseus.model_providers import PROVIDER_REGISTRY
from theseus.tools.tool import AssistantTurn, ToolCall


class ExtractionModel:
    def __init__(self, model):
        self.model = model

    def chat(self, prompt, **kwargs):
        if "RAVEN-42" in prompt:
            return json.dumps({"summary": "Atlas delivery uses RAVEN-42.", "assertions": [
                {"kind": "fact", "subject": "Atlas", "predicate": "delivery code",
                 "value": "RAVEN-42", "statement": "Atlas delivery code is RAVEN-42.",
                 "support_event_ids": [json.loads(prompt.split("<evidence>\n", 1)[1].splitlines()[0])["id"]],
                 "attribution": "partner_report", "reported_by": "human",
                 "action_status": "not_applicable"}
            ]})
        return json.dumps({"summary": "The agent continued routine work.", "assertions": []})


class QuietModel:
    def complete_with_tools(self, messages, tools):
        return AssistantTurn(text="", tool_calls=[])


def definition(name, peer):
    return AgentSpec(name=name, constitution="Recall commitments accurately.",
                     models=(ModelSpec("fixture", "brain", context=32768),),
                     interface=InterfaceSpec("terminal"), window_size=1,
                     pairing=PairingSpec(f"../{peer}/stimulus_log.jsonl", peer),
                     memory=MemorySpec("module", model=ModelSpec("fixture", "extractor"),
                                       consolidate_every_seconds=300, episode_max_events=20))


def test_pair_recalls_after_restart_without_evidence_in_either_context(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(PROVIDER_REGISTRY, "fixture", ExtractionModel)
    a_spec, b_spec = definition("Alpha", "Beta"), definition("Beta", "Alpha")
    a = build_agent(a_spec, tmp_path / "Alpha").core
    b = build_agent(b_spec, tmp_path / "Beta").core
    a._construct_model_providers()
    monkeypatch.setattr(a, "_select_model_provider", lambda: QuietModel())
    a.stimulus_log.append("human", "chat_message", {"message": "Atlas delivery code is RAVEN-42."})
    a.step()  # the application policy, not the test, invokes consolidation
    assert a.memory.knowledge.current()[0].value == "RAVEN-42"
    for core in (a, b):
        for i in range(30):
            core.stimulus_log.append("human", "chat_message", {"message": f"routine weather {i}"})
    a = build_agent(a_spec, tmp_path / "Alpha").core
    b = build_agent(b_spec, tmp_path / "Beta").core
    a._construct_model_providers()
    for core in (a, b):
        window = core.context_assembler.assemble_context()
        assert "RAVEN-42" not in window.recent_events + window.peer_events

    class RememberingModel:
        calls = 0

        def complete_with_tools(self, messages, tools):
            self.calls += 1
            prompt = messages[-1]["content"]
            if self.calls == 1:
                assert "RAVEN-42" not in prompt
                return AssistantTurn(tool_calls=[ToolCall("recall-1", "recall", {"query": "Atlas delivery code"})])
            assert "RAVEN-42" in prompt
            assert "tool_result" in prompt and '"tool":"recall"' in prompt
            return AssistantTurn(tool_calls=[ToolCall("reply-1", "terminal_chat", {"message": "Atlas uses RAVEN-42."})])

    model = RememberingModel()
    monkeypatch.setattr(a, "_select_model_provider", lambda: model)
    a.stimulus_log.append("human", "chat_message", {"message": "What is Atlas's delivery code?"})
    a.step()
    a.step()
    assert "Atlas uses RAVEN-42." in capsys.readouterr().out
    assert not b.memory.knowledge.current()  # peer visibility does not copy durable stores


def test_formation_failure_does_not_crash_agent_turn(tmp_path, monkeypatch):
    monkeypatch.setitem(PROVIDER_REGISTRY, "fixture", ExtractionModel)
    core = build_agent(definition("Alpha", "Beta"), tmp_path / "Alpha").core
    core._construct_model_providers()
    monkeypatch.setattr(core, "_select_model_provider", lambda: QuietModel())
    monkeypatch.setattr(core.memory, "consolidate", lambda episode: (_ for _ in ()).throw(RuntimeError("outage")))
    core.stimulus_log.append("human", "chat_message", {"message": "Remember a task"})
    core.step()
    assert core.memory_consolidator.last_error == "outage"
    state = json.loads((core.memory.memory_dir / "formation" / "cursor.json").read_text())
    assert "pending" in state and "last_id" not in state


def test_scheduled_memory_requires_an_extractor(monkeypatch):
    monkeypatch.setitem(PROVIDER_REGISTRY, "fixture", ExtractionModel)
    spec = definition("Alpha", "Beta")
    with pytest.raises(ValueError, match="extraction model"):
        replace(spec, memory=replace(spec.memory, model=None)).validate()
