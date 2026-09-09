"""RecallTool — the agent's own act of remembering.

Deliberate retrieval, exposed to the cognitive core as a `Tool`. Non-terminal by
design: the recollection lands in the stimulus log as a `tool_result` like every
other tool's output, and the core re-enters Orient so the agent can act on what
it remembered. That is the whole reason recall is a tool rather than something
the context assembler does behind the agent's back — it makes remembering an
event the agent experiences, on the same path as everything else.

Supports the legacy Memory protocol and MemoryModule’s budgeted recall boundary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from theseus.memory import Memory
from theseus.tools.tool import ToolResult

if TYPE_CHECKING:
    from theseus.memory_module import MemoryModule

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

    def __init__(self, memory: Memory | MemoryModule, *, budget_tokens: int = 2000) -> None:
        if type(budget_tokens) is not int or budget_tokens <= 0:
            raise ValueError("budget_tokens must be a positive integer")
        self.memory = memory
        self.budget_tokens = budget_tokens

    def execute(self, query: str) -> ToolResult:
        try:
            details = {"query": query}
            if hasattr(self.memory, "retrieve"):
                recollection = self.memory.retrieve(query)
            else:
                result = self.memory.recall(query, budget_tokens=self.budget_tokens)
                recollection = "\n\n".join(entry.text for entry in result.entries)
                details.update(misses=list(result.misses), total_tokens=result.total_tokens)
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
                details={**details, "found": False},
            )
        return ToolResult(recollection, details={**details, "found": True})
