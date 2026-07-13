from __future__ import annotations

WAIT_ACTION = "wait"


def decision_json_schema(action_names: list[str]) -> dict:
    """JSON schema for the Decide-step response, passed as ModelProvider.chat(json_schema=...).

    `action_names` is the full menu the model may choose from — every effector's `name` plus
    the wait action. `rationale` is listed before `action` so the model reasons before it
    commits.
    """
    return {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "action": {"type": "string", "enum": action_names},
        },
        "required": ["rationale", "action"],
        "additionalProperties": False,
    }


def _render_stimulus_log_section(context: str) -> str:
    return (
        "Your recent history, oldest first, as recorded in your stimulus log "
        "(one JSON event per line):\n\n"
        "<stimulus_log>\n"
        f"{context}\n"
        "</stimulus_log>"
    )


def build_decide_system_prompt(constitution: str, options: list[tuple[str, str]]) -> str:
    menu_lines = "\n".join(f"- {name}: {description}" for name, description in options)

    return (
        f"{constitution}\n\n"
        "---\n\n"
        "# This step: DECIDE\n\n"
        "You are one step in a running Observe-Orient-Decide-Act loop. This call is the "
        "Decide step only. Your sole task is to choose your next action from the menu below, "
        "based on your recent history (provided in the user message), and to explain your "
        "choice.\n\n"
        "Do not carry out the action here. Do not draft a chat message or address anyone. If "
        "action is needed, a separate Act step will receive your decision and rationale and "
        "carry it out.\n\n"
        "# Available actions\n"
        f"{menu_lines}\n"
        f"- {WAIT_ACTION}: Take no action this cycle. Choose this when nothing in your recent "
        "history calls for a response.\n\n"
        "# Output\n"
        "Reply with a single JSON object and nothing else — no code fences, no commentary. "
        "Use double quotes:\n"
        '{"rationale": "<1-3 sentences: what in the history drives this choice>", '
        f'"action": "<{"|".join(name for name, _ in options)}|{WAIT_ACTION}>"}}'
    )


def build_decide_user_prompt(context: str, now: str) -> str:
    return (
        f"{_render_stimulus_log_section(context)}\n\n"
        f"Current system time: {now}\n\n"
        "Decide your next action."
    )


def build_act_system_prompt(constitution: str) -> str:
    return (
        f"{constitution}\n\n"
        "---\n\n"
        "# This step: ACT\n\n"
        "You are one step in a running Observe-Orient-Decide-Act loop. The decision was "
        "already made in the Decide step; this call carries it out. Follow the execution "
        "instructions in the user message exactly."
    )


def build_act_user_prompt(context: str, action: str, rationale: str, act_instruction: str) -> str:
    return (
        f"{_render_stimulus_log_section(context)}\n\n"
        f'In the Decide step you chose the action "{action}" with this rationale:\n'
        f"{rationale}\n\n"
        f"{act_instruction}"
    )
