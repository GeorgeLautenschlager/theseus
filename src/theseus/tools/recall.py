"""RecallTool — the agent's own act of remembering.

Deliberate retrieval, exposed to the cognitive core as a `Tool`. Non-terminal by
design: the recollection lands in the stimulus log as a `tool_result` like every
other tool's output, and the core re-enters Orient so the agent can act on what
it remembered. That is the whole reason recall is a tool rather than something
the context assembler does behind the agent's back — it makes remembering an
event the agent experiences, on the same path as everything else.

Depends only on the `Memory` protocol, so any memory module can back it.
"""

from __future__ import annotations

from typing import Any

from theseus.memory import Memory
from theseus.tools.tool import ToolResult

# Formation filters recall's own output back out of the stimulus log by this name
# (see `AgenticMemory.form`), so it lives here with the tool that produces it.
RECALL_TOOL_NAME = "recall"


class RecallTool:
    name = RECALL_TOOL_NAME
    description = (
        "Search your long-term memory for what you already know about something. Use it "
        "when the conversation touches on a person, preference, decision or event you may "
        "have recorded before, and you would rather check than guess. What you recall "
        "comes back as a tool result — decide your next action once you can see it."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What you are trying to remember, in your own words.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def execute(self, query: str) -> ToolResult:
        try:
            recollection = self.memory.retrieve(query)
        except Exception as exc:
            # Never raise into the loop: a memory outage should cost the agent its
            # recollection, not its turn.
            return ToolResult(
                f"You reached for your memory but it is unavailable right now: {exc}",
                is_error=True,
                details={"query": query, "found": False},
            )

        if not recollection:
            return ToolResult(
                f"Nothing came to mind about: {query}",
                details={"query": query, "found": False},
            )
        return ToolResult(recollection, details={"query": query, "found": True})
