# Claude tool-calling via the `claude` CLI

## Context

`CognitiveCore.decide()` drives one native tool-calling turn through
`ModelProvider.complete_with_tools(messages, tools)`, which returns an `AssistantTurn`
carrying `ToolCall`s. The OpenAI-compatible providers (LM Studio, Ollama, llama.cpp)
inherit a working implementation from the base `ModelProvider`. `ClaudeProvider` does not:
it runs the `claude` CLI as a plain completion (`claude -p … --tools ""`) and its
`complete_with_tools` raises `NotImplementedError`.

That gap blocks Tam. Tam lists `ClaudeProvider` first in its core `model_providers`, and
`ClaudeProvider.is_available()` is true whenever the `claude` CLI is on PATH — so the core
selects Claude, `decide()` calls `complete_with_tools`, and the turn raises. Tam cannot use
the v0.3.0 tool-based core until Claude can return tool calls.

Two ways to close the gap were weighed: rewrite `ClaudeProvider` on the Anthropic Messages
API (native `tool_use`, but adds the `anthropic` dependency and requires an
`ANTHROPIC_API_KEY` with per-token billing, separate from the Claude Code subscription), or
keep the existing `claude` CLI and have it emit a schema-constrained tool-call decision. We
chose the CLI: it preserves the subscription auth (no API key, no billing change), adds no
dependency, and reuses the CLI plumbing that `chat()` and `AgenticMemory` already rely on.
The trade-off — `arguments` is validated by the model against a prompt description rather
than by native `tool_use` — is accepted.

## Design

Implement `ClaudeProvider.complete_with_tools(self, messages, tools=None, max_tokens=8196,
temperature=0.7) -> AssistantTurn` on top of the existing `chat(..., json_schema=…)` path.

**Output schema.** Ask the CLI, via `--json-schema`, for one decision:

```json
{"type": "object",
 "properties": {
   "rationale": {"type": "string"},
   "action": {"type": "string", "enum": ["<tool name>", "…", "wait"]},
   "arguments": {"type": "object"}},
 "required": ["action"]}
```

One action per call matches the OODA "single next action" design and the multi-pass loop
(the core re-orients after each tool). `"wait"` is the no-action sentinel.

**Conveying argument shapes.** Native providers pass tool schemas through the API `tools`
field; the CLI cannot. So `complete_with_tools` appends a compact "tool argument schemas"
block — each `tool.parameters` rendered as JSON, keyed by tool name — to the system prompt,
so the model knows how to populate `arguments`. The `enum` constrains *which* tool; this
block informs *its* arguments.

**Messages → CLI.** `chat()` takes `(prompt, system_prompt)`; `complete_with_tools` receives
OpenAI-style `messages`. A helper splits them: `system`-role content becomes `system_prompt`;
the remaining turns render into the user `prompt`. For the core's `[system, user]` this is
exact. For the multi-turn `ToolRunner` path it is a best-effort labeled rendering of the
history — acceptable, since the core is the sole consumer today and nothing wires
`ClaudeProvider` to `ToolRunner`.

**Parse and map.** Delegate to `self.chat(prompt, system_prompt, json_schema=SCHEMA,
max_tokens=…, temperature=…)`, then `parse_json_response(raw)` (the same pair
`AgenticMemory._form` uses for CLI JSON). Map the decision:

- `action == "wait"`, or an `action` not among the tool names, or missing → `AssistantTurn(text=rationale, tool_calls=())`.
- otherwise → `AssistantTurn(text=rationale, tool_calls=(ToolCall(id=uuid4().hex, name=action, arguments=arguments or {}),))`.

The generated `id` fills the role native providers get from the wire (used in decision
logging and `ToolRunner`'s `tool_call_id`).

**No tools.** If `tools` is empty/None, skip the schema and return
`AssistantTurn(text=self.chat(prompt, system_prompt), tool_calls=())`.

**Unchanged.** `is_available()`, `chat()`, and `embed()` (still `NotImplementedError` —
Claude exposes no embeddings; Ollama covers memory) stay as they are. No change to
`cognitive_core.py`, `cognitive_prompts.py`, the other providers, or Tam.

## Testing

Offline, with `chat()` (or `subprocess.run`) mocked so no `claude` CLI is needed:

- A tool-call decision (`{"action":"ls","arguments":{"path":"."},"rationale":"…"}`) → an
  `AssistantTurn` with exactly one `ToolCall(name="ls", arguments={"path":"."})` and `text`
  equal to the rationale.
- `{"action":"wait","rationale":"…"}` → empty `tool_calls`.
- An `action` not in the tool set → treated as wait (empty `tool_calls`).
- The `json_schema` handed to `chat()` carries an `action` enum containing every tool name
  plus `"wait"`.
- The system prompt handed to `chat()` includes each tool's parameter schema.
- No tools → one `chat()` call, `tool_calls` empty, `text` equal to the raw reply.

Optional: one live smoke test invoking the real `claude` CLI (uses the subscription) to
confirm a real decision round-trips.

## Verification

1. `poetry run pytest tests/test_claude_provider.py -q` — the new unit tests pass.
2. `poetry run pytest tests/ -q --ignore=tests/test_fact_retention.py` — full offline suite
   still green.
3. Optional live: construct `ClaudeProvider`, call
   `complete_with_tools([...], [TerminalChat()])`, confirm a `ToolCall` (or wait) comes back.
