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
providers, starting threads, or creating runtime state. Definitions and custom
modules are trusted Python: their top-level code executes when loaded.

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
- Memory defaults to none. For OODA, `MemorySpec("amem", model=ModelSpec(...),
  embedding=ModelSpec(...))` adds A-MEM and recall. OODA owns its existing `form()`
  lifecycle. For auto, a custom memory factory owns its own consolidation lifecycle.

## Reassembly and state

Edit the definition and run the same assembly command. Restart the agent to use
its new configuration. The generated `agent.py` snapshots all values, including
identity text; the original definition and source identity files are not needed
at runtime. Theseus and any custom Python modules still need to be installed.

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

`agents/tam.py` reads Tam's identity and cadence and references his existing
`tam:TamCore` and `tam_memory:TamMemory`. Set `TAM_HOME` to those source files and
make the modules importable (an installed package, or `PYTHONPATH` for his current
flat repository):

```sh
TAM_HOME=/path/to/tam PYTHONPATH=/path/to/tam poetry run python -m theseus.assemble \
  agents/tam.py --output build/tam
PYTHONPATH=/path/to/tam poetry run python build/tam/agent.py --check
PYTHONPATH=/path/to/tam poetry run python build/tam/agent.py --home /path/to/test-home
```

The first two commands do not start Tam or touch his live state. The third starts
an isolated instance using the source configuration and web port 1337; stop any
other listener on that port or change the definition for parallel experiments.
To deploy, stop the existing agent and run the generated launcher against its
existing home, with the same environment and dependencies as the existing service.
Keep the service's PATH configured to find Tam's intended Claude executable.
The tool does not edit or restart services.

Custom auto core factories accept `name`, `home_directory`, and `tools` keyword
arguments and behave like Autocore. Custom memory factories accept `stimulus_log`
and `memory_dir`, provide `retrieve(query)` for recall, and may provide `start()`;
`run()` calls it before starting the cognitive loop. Import references keep the
snapshot portable without trying to serialize Python closures. Further custom
configuration belongs in a small factory function in the agent's own package.

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
