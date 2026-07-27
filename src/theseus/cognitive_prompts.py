from __future__ import annotations

from typing import List

from theseus.tools.recall import RECALL_TOOL_NAME
from theseus.tools.tool import Tool


def _render_stimulus_log_section(context: str) -> str:
    return (
        "Your recent history, oldest first, as recorded in your stimulus log "
        "(one JSON event per line):\n\n"
        "<stimulus_log>\n"
        f"{context}\n"
        "</stimulus_log>"
    )


def _render_tools_section(tools: list[Tool]) -> str:
    tool_lines = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
    return f"# Available tools\n{tool_lines}\n"


def _render_recall_section(tools: list[Tool]) -> str:
    """Only rendered for agents that actually hold the recall tool — telling an agent
    without a memory module to remember things would just invite it to confabulate.

    Needed because the tool description alone does not get the tool called. Smaller local
    models treat "I don't know" as a complete answer and stop there; observed live, the
    agent asked to be reminded of a name that was sitting in its own note store. The
    trigger has to be named explicitly: not knowing is the cue to look.
    """
    if not any(tool.name == RECALL_TOOL_NAME for tool in tools):
        return ""
    return (
        "# Remembering\n\n"
        f"Your recent history below is a window, not everything you know. You have a "
        f"long-term memory reaching past it, and `{RECALL_TOOL_NAME}` is how you read it.\n\n"
        "If you don't know something you might plausibly have been told before — a name, "
        "a preference, a commitment, a decision already taken — search your memory before "
        "you answer. Answering that you don't know, or asking to be reminded, is a "
        "failure when you never looked. Recall does not end your turn: you will see what "
        "came back and can answer properly on the next pass.\n\n"
    )


def build_decide_system_prompt(constitution: str, persona: str, tools: List[Tool]) -> str:
    tools = list(tools)  # callers pass dict_values; it is walked more than once below
    return (
        f"{constitution}\n\n"
        "---\n\n"
        f"{persona}\n\n"
        "---\n\n"
        "# This step: DECIDE\n\n"
        "You are one step in a running Observe-Orient-Decide-Act loop, and this call is the "
        "Decide step. Read your recent history (in the user message) and choose your single "
        "next action by calling the appropriate tool. Supply complete arguments in the call: "
        "if you decide to reply to someone, write the full message as the tool's argument now — "
        "the call is carried out exactly as you make it, with no further chance to fill it in.\n\n"
        "Default to engaging. If someone has spoken to you, asked you something, or is waiting "
        "on a response from you, the right action is almost always to respond — call the tool "
        "that sends them a message. Only when your recent history genuinely contains nothing "
        "that calls for a response or action should you call no tool at all.\n\n"
        f"{_render_recall_section(tools)}"
        f"{_render_tools_section(tools)}"
    )


def build_decide_user_prompt(context: str, now: str) -> str:
    """The stimulus log is the only context channel. Long-term memories reach the model
    by being recalled — the `recall` tool's result is a stimulus event like any other —
    so there is no separate memories section to render."""
    return (
        f"{_render_stimulus_log_section(context)}\n\n"
        f"Current system time: {now}\n\n"
        "Decide your next action."
    )
