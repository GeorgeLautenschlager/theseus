from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from theseus.cognitive_prompts import (
    build_decide_system_prompt,
    build_decide_user_prompt,
)
from theseus.tools.tool import ToolResult


@dataclass(frozen=True)
class FakeTool:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult("")


class TestBuildDecideSystemPrompt:
    def setup_method(self):
        self.constitution = "You are Tam, a machine intelligence."
        self.tools = [
            FakeTool(
                name="terminal_chat",
                description="Send a chat message through the terminal chat UI.",
                parameters={"type": "object", "properties": {"message": {"type": "string"}}},
            ),
            FakeTool(
                name="read",
                description="Read a file from disk.",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ]
        self.prompt = build_decide_system_prompt(self.constitution, self.tools)

    def test_includes_constitution(self):
        assert self.constitution in self.prompt

    def test_identifies_decide_step(self):
        assert "Decide" in self.prompt

    def test_instructs_calling_a_tool_with_arguments(self):
        # Native tool-calling: the model invokes the tool AND supplies the arguments in
        # the same turn — there is no separate Act model call to compose them later.
        lower = self.prompt.lower()
        assert "call" in lower
        assert "argument" in lower

    def test_lists_each_tool_as_a_readable_action(self):
        # A readable "- name: description" menu, in addition to the native tool schemas,
        # so small local models still reliably see what is on offer.
        for tool in self.tools:
            assert f"- {tool.name}: {tool.description}" in self.prompt

    def test_frames_not_acting_as_calling_no_tool(self):
        # The old explicit "wait" action is gone; declining to act is simply calling no tool.
        assert "no tool" in self.prompt.lower()

    def test_defaults_to_engaging(self):
        # Root-cause guard: the model was defaulting to inaction on a plain greeting. The
        # prompt must positively encourage responding and frame not-acting as the exception.
        lower = self.prompt.lower()
        assert "respond" in lower
        assert "only" in lower or "reserve" in lower

    def test_no_json_output_contract(self):
        # Regression guard: the old two-call flow asked for a {"rationale","action"} JSON
        # blob. Native tool-calling must not — it fights the tool-call response format.
        assert '{"rationale"' not in self.prompt
        assert '"action"' not in self.prompt


class TestBuildDecideUserPrompt:
    def test_includes_context_and_current_time(self):
        context = '{"id":"1","actor":"user","type":"chat_message","content":{}}'
        now = "2026-07-13 12:00:00"

        prompt = build_decide_user_prompt(context, now)

        assert context in prompt
        assert now in prompt

    def test_instructs_deciding_next_action(self):
        prompt = build_decide_user_prompt("", "now")

        assert "decide" in prompt.lower()

    def test_includes_memories_when_present(self):
        prompt = build_decide_user_prompt("ctx", "now", memories="George is the user.")

        assert "<memories>" in prompt
        assert "George is the user." in prompt

    def test_omits_memories_section_when_empty(self):
        prompt = build_decide_user_prompt("ctx", "now")

        assert "<memories>" not in prompt
