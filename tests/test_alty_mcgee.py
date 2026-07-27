from __future__ import annotations

from theseus.agents.alty_mcgee import AltyMcGee
from theseus.memory_store import MemoryStore
from theseus.stimulus_log import StimulusLog


def test_injected_stores_are_used(tmp_path):
    """An injected *empty* MemoryStore must still win.

    MemoryStore defines __len__, so an empty one is falsy — `store or default()` quietly
    discards it and falls back to the working-directory file. That is how a test run ends
    up reading and writing the agent's real memory, which both destroys isolation and
    makes retention tests pass on notes they never formed.
    """
    log = StimulusLog(path=tmp_path / "log.jsonl")
    store = MemoryStore(tmp_path / "a_mem.jsonl")
    assert not store, "precondition: an empty store is falsy, which is the whole trap"

    agent = AltyMcGee(stimulus_log=log, memory_store=store)

    assert agent.memory.store.path == store.path
    assert agent.stimulus_log.path == log.path
    # The recall tool must reach the same store the agent forms notes into.
    assert agent.core.tools["recall"].memory.store.path == store.path


def test_defaults_to_working_directory_files_when_nothing_is_injected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    agent = AltyMcGee()

    assert agent.stimulus_log.path == StimulusLog("stimulus_log.jsonl").path
    assert agent.memory.store.path == MemoryStore("a_mem.jsonl").path
