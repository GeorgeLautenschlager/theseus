from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from theseus.assembly import (
    AgentSpec, InterfaceSpec, MemorySpec, ModelSpec, assemble, build_agent,
)
from theseus.model_providers import PROVIDER_REGISTRY
from theseus.tools.tool import AssistantTurn, ToolCall
from theseus.tools.web_chat import WebChat

ROOT = Path(__file__).resolve().parents[1]


def spec(**changes):
    return replace(AgentSpec(
        name="Test Agent", constitution="Identity: café, \"quotes\", \\paths\nand newlines.",
        core="ooda", models=(ModelSpec("ollama", "test-model"),),
    ), **changes)


def test_cli_snapshot_is_deterministic_and_independent_of_definition(tmp_path):
    definition = tmp_path / "definition.py"
    definition.write_text(
        "from theseus.assembly import AgentSpec, ModelSpec, MemorySpec, InterfaceSpec\n"
        f"SPEC = {spec()!r}\n"
    )
    output = tmp_path / "output with spaces"
    command = [sys.executable, "-m", "theseus.assemble", str(definition), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True)
    before = (output / "agent.py").read_bytes()
    subprocess.run(command, check=True, capture_output=True)
    assert (output / "agent.py").read_bytes() == before
    definition.unlink()
    subprocess.run([sys.executable, str(output / "agent.py"), "--check"], check=True, capture_output=True)
    assert not (output / "state").exists()
    assert runpy.run_path(str(output / "agent.py"))["SPEC"] == spec()


def test_reassembly_applies_changes_on_next_boot_and_preserves_state(tmp_path):
    output = tmp_path / "agent"
    launcher = assemble(spec(), output)
    home = output / "state"
    original = build_agent(runpy.run_path(str(launcher))["SPEC"], home)
    original.core.stimulus_log.append(actor="user", type="chat_message", content={"message": "Remember me"})
    runtime = {
        "a_mem.jsonl": '{"memory": "keep"}\n', "memory/knowledge.jsonl": "knowledge\n",
        "GOALS.md": "A current goal", "TASKS.md": "An unfinished task",
        "CURRENT_TASK.md": "Current task", "SCHEDULE.md": "An edited schedule", ".env": "EXAMPLE=keep",
    }
    for name, text in runtime.items():
        path = home / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    before = {path: path.read_bytes() for path in home.rglob("*") if path.is_file()}
    updated = spec(name="Changed", constitution="New identity", window_size=12,
                   models=(ModelSpec("ollama", "changed-model"),), tools=("read",))
    assemble(updated, output)
    assert all(path.read_bytes() == content for path, content in before.items())
    rebuilt = build_agent(runpy.run_path(str(launcher))["SPEC"], home)
    assert rebuilt.core.name == "Changed"
    assert rebuilt.core.constitution == "New identity"
    assert rebuilt.core.context_assembler.window_size == 12
    assert rebuilt.core.model_providers[0].model == "changed-model"
    assert "read" in rebuilt.core.tools
    assert len(rebuilt.core.stimulus_log.read_all()) == 1
    assert all((home / name).read_text() == text for name, text in runtime.items())


def test_isolated_agent_completes_real_cognitive_turn(tmp_path, monkeypatch, capsys):
    class FakeModel:
        def __init__(self, model):
            pass

        def is_available(self):
            return True

        def complete_with_tools(self, messages, tools):
            return AssistantTurn(text="", tool_calls=[ToolCall("call-1", "terminal_chat", {"message": "Hello test"})])

    monkeypatch.setitem(PROVIDER_REGISTRY, "offline", FakeModel)
    definition = spec(models=(ModelSpec("offline", "test"),))
    launcher = assemble(definition, tmp_path / "build")
    loaded = runpy.run_path(str(launcher))["SPEC"]
    agent = build_agent(loaded, tmp_path / "isolated")
    monkeypatch.setattr("builtins.input", lambda _: "Hello agent")
    agent.observer.observe_chat_message()
    assert "Hello test" in capsys.readouterr().out
    events = agent.core.stimulus_log.read_all()
    assert [e.type for e in events] == ["chat_message", "decision", "tool_result"]
    assert not events[-1].content["is_error"]
    second = build_agent(loaded, tmp_path / "other")
    assert second.core.stimulus_log.read_all() == []


@pytest.mark.parametrize("changes, message", [
    ({"models": ()}, "nonempty tuple"),
    ({"models": (ModelSpec("unknown", "test"),)}, "Unknown provider"),
    ({"tools": ("not_a_tool",)}, "Unknown tools"),
    ({"constitution": ""}, "constitution"),
    ({"core": "missing"}, "core must"),
    ({"window_size": 0}, "window_size"),
    ({"interface": InterfaceSpec("discord")}, "interface must"),
    ({"interface": InterfaceSpec("web", port=0)}, "port"),
    ({"core": "auto", "models": (ModelSpec("ollama", "x", during="99:00-00:00"), ModelSpec("ollama", "y"))}, "Invalid cadence"),
    ({"core": "auto", "models": (ModelSpec("ollama", "x"), ModelSpec("ollama", "y"))}, "exactly one default"),
    ({"memory": MemorySpec("amem")}, "ModelSpec"),
    ({"core": "auto", "memory": MemorySpec("amem")}, "requires OODA"),
    ({"core": "auto", "core_factory": "theseus.auto_core:Missing"}, "Missing"),
])
def test_invalid_reassembly_keeps_previous_output(tmp_path, changes, message):
    launcher = assemble(spec(), tmp_path)
    before = launcher.read_bytes()
    with pytest.raises((ValueError, AttributeError), match=message):
        assemble(spec(**changes), tmp_path)
    assert launcher.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["agent.py"]


def test_output_refuses_unowned_file_and_symlink(tmp_path):
    output = tmp_path / "agent.py"
    output.write_text("my existing agent")
    with pytest.raises(ValueError, match="non-assembler"):
        assemble(spec(), tmp_path)
    assert output.read_text() == "my existing agent"
    output.unlink()
    target = tmp_path / "target"
    target.write_text("precious")
    output.symlink_to(target)
    with pytest.raises(ValueError, match="non-assembler"):
        assemble(spec(), tmp_path)
    assert target.read_text() == "precious"


def test_failed_replace_keeps_previous_output_and_cleans_temporary(tmp_path, monkeypatch):
    launcher = assemble(spec(), tmp_path)
    before = launcher.read_bytes()

    def fail(*args):
        raise OSError("Disk error")

    monkeypatch.setattr("theseus.assembly.os.replace", fail)
    with pytest.raises(OSError, match="Disk error"):
        assemble(spec(name="new"), tmp_path)
    assert launcher.read_bytes() == before
    assert list(tmp_path.iterdir()) == [launcher]


def test_auto_web_wiring_and_cadence(tmp_path):
    definition = spec(core="auto", interface=InterfaceSpec("web"), tools=("read",))
    agent = build_agent(definition, tmp_path)
    assert agent.observer.stimulus_log is agent.core.stimulus_log
    assert agent.observer.orient_chat_message_callback == agent.core.wake
    assert agent.core.tools[WebChat.name].web_observer is agent.observer
    assert (tmp_path / "CADENCE.md").read_text() == definition.cadence_text()
    assert (tmp_path / "SCHEDULE.md").exists()


def test_amem_wires_recall_to_same_persistent_store(tmp_path):
    memory = MemorySpec("amem", model=ModelSpec("ollama", "test"),
                        embedding=ModelSpec("ollama", "nomic-embed-text"))
    agent = build_agent(spec(memory=memory), tmp_path)
    assert agent.core.tools["recall"].memory is agent.core.memory
    assert agent.core.memory.store.path == tmp_path / "a_mem.jsonl"


def test_example_variant_reuses_defaults():
    base = runpy.run_path(str(ROOT / "agents/minimal.py"))["SPEC"]
    variant = runpy.run_path(str(ROOT / "agents/experiment.py"))["SPEC"]
    variant.validate()
    assert base.models == variant.models
    assert variant.name != base.name


def test_custom_tam_style_parts_receive_isolated_home_without_starting(tmp_path, monkeypatch):
    import types
    from theseus.auto_core import Autocore

    module = types.ModuleType("assembly_test_parts")

    class LocalCore(Autocore):
        pass

    class LocalMemory:
        def __init__(self, stimulus_log, memory_dir):
            self.log = stimulus_log
            self.directory = memory_dir
            self.started = False

        def start(self):
            self.started = True

        def retrieve(self, query):
            return "A remembered fact"

    module.LocalCore = LocalCore
    module.LocalMemory = LocalMemory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    definition = spec(
        core="auto", core_factory="assembly_test_parts:LocalCore",
        memory=MemorySpec("custom", factory="assembly_test_parts:LocalMemory"),
    )
    launcher = assemble(definition, tmp_path / "build")
    agent = build_agent(runpy.run_path(str(launcher))["SPEC"], tmp_path / "state")
    assert isinstance(agent.core, LocalCore)
    assert agent.core.memory.log is agent.core.stimulus_log
    assert agent.core.memory.directory == tmp_path / "state/memory"
    assert not agent.core.memory.started
    assert agent.core.tools["recall"].execute("a fact").content == "A remembered fact"


def test_generated_agent_boots_as_separate_process(tmp_path):
    launcher = assemble(spec(), tmp_path / "build")
    home = tmp_path / "elsewhere"
    result = subprocess.run(
        [sys.executable, str(launcher), "--home", str(home)],
        input="", text=True, capture_output=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert (home / "CONSTITUTION.md").read_text() == spec().constitution
    assert not (tmp_path / "build/state").exists()
