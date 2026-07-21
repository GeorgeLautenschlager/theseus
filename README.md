# Theseus - Many Parts, Same Agent

Theseus is my Rotrod, the project car. I'm building it's capabilities just to see how capable I can build it. Out of this little hobby, I hope to produce nicely packaged modules that will eventually provide a "construction kit". That's planned to include:
- model providers (Ollama, LM Studio, OpenRouter, Claude, etc.)
- context assemblers
- reasoning engines
- memory modules of several types, including:
  - episodic
  - semantic
  - procedural
- reflection systems
- TUIs, GUIs, and other oooies
- "sensory organs", such as Gemma 4 E4B intelligently feeding keyframes to a larger model
- at least one OpenTelemetry tracer, possibly others
- a "gold set" and other agent evaluation tooling

The current plan is to build the main agent loop as an OODA loop because... I'm a big fan of the works of John Boyd.

## Getting started

Theseus is a conventional Poetry package (Python 3.12, src layout).

```bash
poetry install              # library + pytest
poetry install --with eval  # also installs sentence-transformers/torch for the eval tests
make test                   # poetry run pytest -q tests/
poetry run alty             # run the reference agent
```

Agents built on Theseus live in their own repos and depend on it as a pinned git dependency:

```toml
[project]
dependencies = [
    "theseus @ git+ssh://git@github.com/GeorgeLautenschlager/theseus.git@v0.1.0",
]
```

```python
from theseus import CognitiveCore, StimulusLog
```

`src/theseus/agents/alty_mcgee.py` (Alty) is the in-repo reference for composing an agent.

## Architecture

Theseus is built on a simple foundation that I call the `StimulusLog`. If you've read [Generative Agents]([url](https://arxiv.org/abs/2304.03442)), you'll recognize the memory stream from that paper. Everything the agent experiences, including in most cases the things it does, goes into the `StimulusLog`. The `StimulusLog` is an append-online JSONL file, with each line corresponding to exactly one `StimulusEvent`. Another way to think of it is like the tape in a Universal Turing Machine. That brings us to the `CognitiveCore`. If `StimulusLog` is the tape in our Turing Machine analogy, `CognitiveCore` is the head. It is a module like any other in Theseus, but it exposes an opinionated set of ports:
- **Observe**, where information enters the cognitive loop. These are implemented as callbacks so that asynchronous `Observer` modules of various types can trigger a cognitive loop or loops after they've appended stimuli to the log
- **Orient**, where cognition really begins for a Theseus agent. The `StimulusLog` is read, along with relevant memories, the configured tools, and of course the agent's constitution document. This is fed to a `ContextAssembler` which prepares it for the configured LLM's context window.
- **Decide**, where we translate context to action. Another invariant of a `CognitiveCore` is that they have a reasoning engine of some sort built in. I expect this will be an LLM in pretty much every case, but nevertheless I'm leaving the door open, at least for VLMs or even VLAMs. Decide makes a single native tool-calling turn: the reasoning engine chooses a tool and supplies its arguments in one shot. That tool call is then interpretted and executed by....
- **Act**, where the agent actually does stuff. Even in the simplest case — when **Decide** calls a "respond in chat" tool — execution stays separate: **Act** takes the tool call **Decide** produced and runs it, delivering the message the model already composed (there's no second LLM call). **Act** is also crucially in charge of terminating the cognitive loop or triggering another one, but again it *does not* decide that for itself. If the decision requires another loop, **Act** triggers one, otherwise execution terminates.
