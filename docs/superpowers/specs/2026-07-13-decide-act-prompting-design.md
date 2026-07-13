# Decide vs Act: prompt redesign + effector refactor for CognitiveCore

## Context

Tam's `CognitiveCore` (`src/modules/cognitive_core.py`) runs the back half of an OODA loop with
interchangeable model providers (Claude, LM Studio, Ollama). The problem: the **Decide** prompt
should get the model to *only emit a decision*, and the **Act** prompt should get it to *do the
thing* — but the current prompts don't cleanly separate the two, and the language, content, and
structure needed clarification.

Diagnosis from reading the code:

1. **Act never sees the decision.** `decide()` logs the rationale to the StimulusLog but `act()`
   builds a fresh prompt from raw history — so Act isn't "executing a decision", it's a second
   independent pass. This is the structural cause of the blur.
2. **Decide's menu is opaque and forced.** Options are bare class names (`WebChatUIEffector`) with
   no description of what they do, no "do nothing" option, and nothing telling the model *not* to
   compose the reply inside its rationale.
3. **Format burden is on prose, not schema.** Every provider already supports `json_schema`
   (`model_provider.py`, `claude_provider.py`) but `decide()` doesn't use it — and the in-prompt
   example uses Python-style single quotes, which small models mimic and `json.loads` rejects.
4. **Tam's own replies never enter the StimulusLog.** The observer logs user messages; `decide()`
   logs decisions; nobody logs the act output. So "this is your recent history" is missing half
   the conversation — the model can't see what it already said.
5. Mechanical: missing spaces/newlines at concatenation seams; `action` before `rationale` in the
   JSON (model commits before reasoning); effector list repeated 3×; stray `from typer import
   prompt` import.

## Design principles (the decide-vs-act answer)

- **Decide is a selection task; Act is a generation task.** Decide's prompt gives a menu with
  affordance descriptions, states explicitly "choose only — do not carry it out; a later step
  will", and enforces shape with a JSON schema whose `action` is an **enum**. Act's prompt hands
  over the decision as a fait accompli ("you chose X, your rationale was Y — now carry it out")
  with the effector's own output contract ("output only the message text; it is delivered
  verbatim").
- **Rationale before action** in the JSON — the model reasons before committing (schema property
  order + prompt example both enforce this).
- **Static vs dynamic split:** constitution + role framing + rules + menu + output contract in the
  system prompt (stable per cycle); stimulus log + current time in the user prompt.
- **Semantic action names**, not class names: `respond_in_web_chat`, `wait` — legible to small
  local models and stable if the implementing class is renamed.
- **`wait` is a core-level decision option, not an effector** — it short-circuits Act with no
  second model call. A choice isn't a real decision if acting is mandatory.

## Architecture

- `src/modules/effector.py` — new `Effector` protocol: `name`, `description`, `act_instruction`,
  `execute(payload)`.
- `src/modules/cognitive_prompts.py` — new module of pure prompt-builder functions plus
  `decision_json_schema()` and the `WAIT_ACTION` constant. No side effects, fully unit-testable.
- `src/modules/web_chat_ui_effector.py` — `WebChatUIEffector` implements `Effector`.
- `src/modules/cognitive_core.py` — `decide()`/`act()` rewired: `effector_callbacks: dict` becomes
  `effectors: dict[str, Effector]`; `decide()` uses the schema-constrained call and validates the
  chosen action is in the menu; `act()` passes the effector its own `act_instruction` plus the
  rationale from Decide, and appends the resulting output back onto the StimulusLog as an
  `action` event (closing gap #4 above).
- `src/agents/tam.py` — wiring updated to the `effectors` dict keyed by `Effector.name`.

## Testing

- `tests/test_cognitive_prompts.py` — prompt builders are pure functions; assert the menu, the
  wait option, double-quoted JSON example with rationale-before-action, and that
  `decision_json_schema()`'s enum matches the supplied action names.
- `tests/test_cognitive_core.py` — stub provider + stub effector + tmp-path StimulusLog:
  `decide()` passes `json_schema`; a normal decision reaches `effector.execute()` and logs an
  `action` event; `wait` makes exactly one model call and no effector call; an unknown action
  doesn't raise; fenced/malformed JSON still parses via the existing fallback.

## Verification

1. `python -m pytest tests/ -q` — new and existing tests pass.
2. End-to-end: run `python src/agents/tam.py`, send a chat message, confirm in `/debug` that one
   cycle now logs `chat_message` → `decision` → `action`, and the reply renders in the chat UI.
