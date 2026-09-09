# Agent Assembler

Keep an agent's definition in Python, assemble it into a runnable directory, and
repeat as its settings change. Requires Theseus installed in the Python environment.
From this repository:

```sh
poetry run python -m theseus.assemble agents/minimal.py --output build/test-agent
poetry run python build/test-agent/agent.py --check
poetry run python build/test-agent/agent.py
```

The last command starts a terminal agent and needs the configured model backend.
Assembly and `--check` validate settings without contacting a model, constructing
providers, starting threads, or creating runtime state. Definitions are trusted Python: their top-level code executes when loaded.

## Definitions and experiments

A definition exports `SPEC`:

```python
from theseus.assembly import AgentSpec, InterfaceSpec, ModelSpec

SPEC = AgentSpec(
    name="Researcher",
    constitution="Help me investigate questions and keep track of evidence.",
    models=(ModelSpec("ollama", "gemma4:e4b", context=131072, tick=300),),
    core="auto",
    tools=("read", "ls", "find", "grep"),
    interface=InterfaceSpec("web", port=8001),
    window_size=60,
)
```

Use ordinary functions, imports, and `dataclasses.replace` for shared defaults.
`agents/experiment.py` shows a variant of `agents/minimal.py`. To read identity
from a file, anchor its path to the definition:

```python
from pathlib import Path
constitution = Path(__file__).with_name("CONSTITUTION.md").read_text(encoding="utf-8")
```

- `core="auto"` uses Autocore. Models may have `during="08:00-22:00"`; exactly one
  must omit `during` to supply the default. `tick` is seconds; set `context` to
  the backend’s configured token window (the default is a conservative 4096). Existing cadence
  parsing checks time windows. Provider endpoint/key configuration stays in the
  providers' existing environment variables.
- `core="ooda"` is driven by incoming chat. Models are ordered fallbacks, must
  share the same context budget, and cannot specify `during`. `tick` is unused.
- `tools` names the existing `read`, `write`, `edit`, `ls`, `find`, `grep`, and `bash`
  tools. Their working directory is the runtime home. Omitted tools are unavailable;
  included tools execute with the substrate's existing behavior. This is not a
  sandbox or an `allow`/`ask`/`deny` policy system.
- `interface` is one terminal or web interface. Its reply tool is wired automatically.
- Memory defaults to none. Autocore accepts either `MemorySpec("amem", ...)`
  (AgenticMemory) or `MemorySpec("module", ...)` (MemoryModule). Both wire recall
  to the selected module and the core's stimulus log. For example:

```python
memory=MemorySpec(
    "module",  # or "amem"
    model=ModelSpec("ollama", "gemma4:e4b"),
    embedding=ModelSpec("ollama", "nomic-embed-text"),
    recall_budget_tokens=2000,
)
```

A-MEM requires both providers. MemoryModule accepts optional providers: without
an embedder it can still recall keyword facts and recent events. Its recall output
respects `recall_budget_tokens`; A-MEM retains its existing retrieval settings.
A-MEM stores notes in `a_mem.jsonl`; MemoryModule stores layers in `memory/`.
Switching modules preserves both stores but does not migrate between them.

Autocore holds the module without automatically forming or consolidating memories.
The agent's episode/scheduling policy still calls `memory.form()` for A-MEM or
`memory.consolidate(episode)` for MemoryModule. This assembler does not invent that
policy. OODA supports A-MEM and retains its existing end-of-turn `form()` call;
MemoryModule does not implement that lifecycle and is rejected with OODA.

## Reassembly and state

Edit the definition and run the same assembly command. Restart the agent to use
its new configuration. The generated `agent.py` snapshots all values, including
identity text; the original definition and source identity files are not needed
at runtime. Theseus still needs to be installed.

Assembly replaces only an assembler-marked `agent.py`, using an atomic rename.
Validation or a failed write leaves the previous launcher intact. An unrelated
file or symlink named `agent.py` is rejected. Generated files are disposable; edit
the source definition instead.

The default runtime home is `state/` next to the launcher. Override it explicitly:

```sh
poetry run python build/test-agent/agent.py --home /path/to/agent-home
```

**On boot**, the definition reapplies `CONSTITUTION.md`, `PERSONA.md`, and
`CADENCE.md` in that home. Runtime edits to those three files last until the next
boot; copy intentional changes back into the definition or its source files.
Assembly itself never touches the runtime home. Logs, memory, goals, tasks,
current task, schedule, and credentials are preserved. Do not run two agents
against the same home. For experiments and tests, give each agent a fresh home.

## Tam

`agents/tam.py` reads Tam's identity and cadence and assembles plain Autocore.
Set `TAM_HOME` to those source files. `TAM_MEMORY` selects `module` (the default)
or `amem`; `TAM_MEMORY_MODEL` selects the Ollama memory model (default
`gemma4:e4b`). Edit the definition to choose other providers or embedding models.
No agent-specific Python modules are required.

```sh
TAM_HOME=/path/to/tam TAM_MEMORY=module poetry run python -m theseus.assemble \
  agents/tam.py --output build/tam
poetry run python build/tam/agent.py --check
poetry run python build/tam/agent.py --home /path/to/test-home
```

The first two commands do not start Tam or touch his live state. The third starts
an isolated instance using the source configuration and web port 1337; use another
port for parallel experiments. To deploy, stop the existing agent and run the
launcher against its existing home, with the configured providers available in
the service environment. The tool does not edit or restart services. Memory
consolidation is configured separately as described above.

## Tests without a service

```python
from theseus.assembly import build_agent

agent = build_agent(SPEC, tmp_path / "agent")
# Construction does not start loops or servers.
# Substitute a fake provider before driving a cognitive turn.
agent.core.model_providers = [fake_provider]  # OODA
agent.core.orient_and_wait()
```

`tests/test_assembly.py` exercises the full observer → core → reply path with an
offline fake provider, as well as generated-process boot and state preservation.
