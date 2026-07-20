# Convert ChatEffector and WebChatUIEffector to the Tool protocol

## Context

Theseus grew a model-agnostic `Tool` abstraction (`src/theseus/tools/tool.py`,
`tools/tool_runner.py`) alongside the older `Effector` protocol (`src/theseus/effector.py`) that
`CognitiveCore`'s Decide/Act steps currently drive. The two shapes differ:

- `Effector`: `name`, `description`, `act_instruction` (freeform Act-phase task text), and
  `execute(payload: str) -> None`.
- `Tool`: `name`, `description`, `parameters` (JSON Schema), and `execute(**kwargs) -> ToolResult`.

`cognitive_core.py` is mid-refactor toward driving native tool-calling instead of the two-call
Decide-then-Act flow, but that file and `alty_mcgee.py` are explicitly **out of scope** for this
change — they're being handled separately. This change only converts the two concrete chat
effectors, `ChatEffector` (`src/theseus/chat_effector.py`) and `WebChatUIEffector`
(`src/theseus/web_chat_ui_effector.py`), so they're ready to be handed to a model as native tools
once the core catches up.

`aldric.py` (an older agent using a different, deprecated `ChatCognitiveCore`) calls
`chat_effector.respond_callback(response: str)` directly. Aldric is deprecated and does not
constrain this design, but `respond()`/`respond_callback()` are cheap to leave untouched, so they
are left as-is rather than removed.

## Design

For both `ChatEffector` and `WebChatUIEffector`:

- Drop `act_instruction` — no longer part of the `Tool` shape, and there's no second Act call left
  for these two classes to brief.
- Add `parameters` (class-level JSON Schema, matching the style in `tools/write.py` etc.):
  ```python
  parameters: dict[str, Any] = {
      "type": "object",
      "properties": {
          "message": {"type": "string", "description": "The chat message to send, delivered verbatim."},
      },
      "required": ["message"],
  }
  ```
- Reword `description` to drop the Decide-menu framing ("Choose this when...") in favor of a direct
  capability statement, since a tool-calling model reads descriptions to decide whether to call the
  tool, not to fill in a menu slot.
- Replace `execute(self, payload: str) -> None` with `execute(self, message: str) -> ToolResult`:
  calls `self.respond(message)`, then returns a `ToolResult` confirming delivery (short content
  string + `details={"message": message}`), matching the convention in `tools/write.py`.
- Leave `respond()` and `respond_callback()` exactly as they are.
- Update the "Satisfies the Effector protocol" docstring/comment on `execute` to reference the Tool
  protocol instead.
- Module-level docstrings that describe these classes in Effector/Decide-Act terms get a light
  wording pass so they don't misdescribe the class, without rewriting unrelated context (e.g. the
  SSE-streaming explanation on `WebChatUIEffector.respond` stays as-is).

No changes to `effector.py`, `cognitive_core.py`, `alty_mcgee.py`, `aldric.py`,
`chat_cognitive_core.py`, or `tools/registry.py`. Neither class is added to any tool registry as
part of this change — wiring them into a core is a separate step.

## Testing

No existing tests cover `ChatEffector`/`WebChatUIEffector` directly (checked: no
`test_chat_effector.py` / `test_web_chat_ui_effector.py`, and `tests/test_cognitive_core.py` uses
its own `StubEffector`, unaffected by this change). Add a small test file per class asserting:
`name`/`parameters`/`description` are present and shaped correctly, `execute(message=...)` calls
`respond` with that message and returns a non-error `ToolResult` whose `details["message"]` matches.

## Verification

1. `poetry run pytest tests/ -q` — full suite still passes (offline subset).
2. `to_openai_tool(ChatEffector())` / `to_openai_tool(WebChatUIEffector(...))` produce valid
   OpenAI-style tool entries (spot-check by hand or in the new tests).
