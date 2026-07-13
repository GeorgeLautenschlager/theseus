# Theseus Packaging & Deployment Design

**Date:** 2026-07-13
**Status:** Implemented

## Problem

Theseus used `from src.modules.X import ...` imports that only resolved with the repo root on `sys.path`, forcing a `sys.path.insert` hack in `tests/conftest.py`, fragile invocation of agent scripts, and manual `.venv` activation. The deployed Tam agent ran directly out of the development working tree, so development changes could break the running agent. Nothing outside the repo could depend on Theseus.

## Decisions

1. **Conventional Poetry package, src layout.** `src/theseus/` is the package; `pyproject.toml` (Poetry 2 / PEP 621) + committed `poetry.lock` replace `requirements.txt`. In-project venv (`poetry.toml`), never activated manually — everything goes through `poetry run`.
2. **Flatten the `modules` level.** `src/modules/cognitive_core.py` → `src/theseus/cognitive_core.py`; imports are `from theseus.cognitive_core import CognitiveCore`.
3. **Regular packages with `__init__.py`.** Retires the old "no `__init__.py`" namespace-package rule. `theseus/__init__.py` carries curated exports: `CognitiveCore`, `StimulusLog`, `StimulusEvent`, `Effector`.
4. **Tam moves out; reference agents stay.** `tam.py` lives in the Tam repo (`/home/aldric/tam/v3/`); `aldric.py` remains in-repo as the reference composition, exposed as the `aldric` console script.
5. **Deployed agents consume Theseus as a git dependency pinned to a tag** — `theseus @ git+ssh://git@github.com/GeorgeLautenschlager/theseus.git@v0.1.0`. Development on main never touches a deployment until its pin is bumped and `poetry update theseus` is run in the consumer repo.
6. **Dependency hygiene.** Runtime deps are only what's imported: `openai`, `fastapi`, `uvicorn`, `jinja2` (`textual` was dead and dropped). `pytest` is in the dev group; `sentence-transformers` (pulls torch) is in an optional `eval` group used only by `tests/test_fact_retention.py`.
7. **Web assets are package data.** `web/templates/` and `web/static/` live inside the package and are resolved via `Path(__file__)`, so they ship in the wheel and work from any install location.

## Consequences

- Any script anywhere can `from theseus import CognitiveCore` after `poetry add theseus @ git+ssh://...@<tag>`.
- Releases are tags: bump `version` in `pyproject.toml`, tag `vX.Y.Z`, push, bump consumer pins.
- The old `TheseusAgent`/OODA-protocol specs under `docs/superpowers/specs/` describe deleted code and are historical context only.
