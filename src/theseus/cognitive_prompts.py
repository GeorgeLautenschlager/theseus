from __future__ import annotations

from typing import List

from theseus.tools.tool import Tool


def _render_stimulus_log_section(context: str) -> str:
    return (
        "Your recent history, oldest first, as recorded in your stimulus log "
        "(one JSON event per line):\n\n"
        "<stimulus_log>\n"
        f"{context}\n"
        "</stimulus_log>"
    )


def _render_memories_section(memories: str) -> str:
    return (
        "Distilled long-term memories relevant to the current situation, "
        "retrieved from your memory system:\n\n"
        "<memories>\n"
        f"{memories}\n"
        "</memories>"
    )


def _render_tools_section(tools: list[Tool]) -> str:
    tool_lines = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
    return f"# Available tools\n{tool_lines}\n"


def build_decide_system_prompt(constitution: str, tools: List[Tool]) -> str:
    return (
        f"{constitution}\n\n"
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
        f"{_render_tools_section(tools)}"
    )


def build_decide_user_prompt(context: str, now: str, memories: str = "") -> str:
    memories_section = f"{_render_memories_section(memories)}\n\n" if memories else ""
    return (
        f"{memories_section}"
        f"{_render_stimulus_log_section(context)}\n\n"
        f"Current system time: {now}\n\n"
        "Decide your next action."
    )
