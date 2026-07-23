# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Theseus is a **modular construction kit for cognitive language agents**, packaged as a conventional Poetry library (`theseus`, src layout, Python 3.12). Agents are composed from Theseus modules; the in-repo reference agent is `src/theseus/agents/alty_mcgee.py` (Alty). Production agents (e.g. Tam, at `/home/aldric/tam`) live in their own repos and depend on Theseus as a git dependency pinned to a release tag.

## Commands

```bash
poetry install               # library + dev deps (pytest)
poetry install --with eval   # also sentence-transformers/torch, needed by tests/test_fact_retention.py

make test                    # poetry run pytest -q tests/
poetry run pytest tests/test_cognitive_core.py -v   # single file
poetry run alty              # run the reference agent Alty (needs a local LLM endpoint)
```

The venv is in-project (`.venv`, via `poetry.toml`). Never activate it manually — use `poetry run`. `tests/test_fact_retention.py` needs a live LLM endpoint and the `eval` group; the rest of the suite is offline.

## Architecture

The README's Architecture section is the authoritative description. Short version:

- **`StimulusLog`** (`stimulus_log.py`) — append-only JSONL event stream; everything the agent experiences or does is a `StimulusEvent`. The Turing-machine "tape".
- **`CognitiveCore`** (`cognitive_core.py`) — the "head". Runs Orient → Decide → Act over the log. `orient(...)` is the entry point observers call. Decide uses `cognitive_prompts.py` and the configured `model_providers` (tried in priority order) to make one native tool-calling turn; Act executes the model's chosen tool(s) with the arguments it supplied and decides whether to loop again.
- **Observers** (`chat_observer.py`, `web_chat_ui_observer.py`) — feed stimuli in and trigger cognitive loops via the `orient` callback. The web observer is a FastAPI app serving the chat + debug UI from `web/templates` and `web/static` (packaged data files — they ship inside the wheel).
- **Tools** (`tools/tool.py` — the `Tool` protocol + `ToolResult`) — structured capabilities the model invokes with typed arguments; each `execute(...)` returns a `ToolResult` the loop feeds back. Concrete tools live in `tools/` (file/shell tools like `read.py`, `bash.py`; the terminal chat "mouth" `terminal_chat.py`; the web-UI reply tool `web_chat.py`).
- **Model providers** (`model_providers/`) — one class per backend (LM Studio, Ollama, llama.cpp, Claude...), all speaking an OpenAI-compatible interface.

An agent file (see `agents/alty_mcgee.py`, or `tam.py` in the Tam repo) just composes these: build a `StimulusLog`, a `CognitiveCore` with providers + tools, and observers wired to `core.orient`.

## Layout & conventions

```
pyproject.toml           # Poetry 2 / PEP 621; poetry.lock committed
src/theseus/             # the package — flat modules, no "modules" subdir
    model_providers/  web/  agents/
tests/                   # imports `theseus.*`; no sys.path hacks — ever
```

- Regular packages: every package dir has an `__init__.py`. The top-level `src/theseus/__init__.py` holds curated exports (`CognitiveCore`, `StimulusLog`, `StimulusEvent`, `Tool`) — add to it when a new module becomes part of the public API.
- Every module starts with `from __future__ import annotations`.
- Imports are absolute: `from theseus.cognitive_core import CognitiveCore`.
- New work goes through the spec → plan → implement flow under `docs/superpowers/`.

## Releasing / consumers

Deployed agents pin Theseus by tag: `theseus @ git+ssh://git@github.com/GeorgeLautenschlager/theseus.git@v0.1.0`. To ship a change to them, from an up-to-date `main` run `make release VERSION=X.Y.Z` — it verifies the tree is clean and the offline suite passes, then bumps `pyproject.toml`, commits, tags `vX.Y.Z`, and pushes the branch + tag (add `SKIP_TESTS=1` to skip the test gate). Then bump the pin in the consumer repo (`poetry add "theseus @ git+ssh://…@vX.Y.Z"`) and restart it. Merging to main does NOT affect deployed agents until their pin moves.

## Reference docs

- `docs/notes_on_organization.md` — the construction-kit vision.
- `docs/superpowers/specs/2026-07-13-theseus-packaging-design.md` — the Poetry packaging + deployment design (current structure).
- Older specs under `docs/superpowers/specs/` describe the pre-`CognitiveCore` architecture (`TheseusAgent`, OODA protocol slots) — historical context only; that code is gone.
