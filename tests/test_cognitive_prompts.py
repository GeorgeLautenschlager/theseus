from __future__ import annotations

from theseus.cognitive_prompts import (
    WAIT_ACTION,
    build_act_system_prompt,
    build_act_user_prompt,
    build_decide_system_prompt,
    build_decide_user_prompt,
    decision_json_schema,
)


class TestDecisionJsonSchema:
    def test_returns_exact_schema_shape(self):
        action_names = ["respond_in_web_chat", "wait"]

        schema = decision_json_schema(action_names)

        assert schema == {
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "action": {"type": "string", "enum": action_names},
            },
            "required": ["rationale", "action"],
            "additionalProperties": False,
        }

    def test_rationale_property_precedes_action(self):
        schema = decision_json_schema(["wait"])

        assert list(schema["properties"].keys()) == ["rationale", "action"]

    def test_enum_matches_supplied_action_names(self):
        action_names = ["respond_in_web_chat", "log_observation", "wait"]

        schema = decision_json_schema(action_names)

        assert schema["properties"]["action"]["enum"] == action_names


class TestBuildDecideSystemPrompt:
    def setup_method(self):
        self.constitution = "You are Tam, a machine intelligence."
        self.options = [
            ("respond_in_web_chat", "Send a chat message to George through the web chat UI."),
        ]
        self.prompt = build_decide_system_prompt(self.constitution, self.options)

    def test_includes_constitution(self):
        assert self.constitution in self.prompt

    def test_identifies_decide_step(self):
        assert "Decide" in self.prompt

    def test_instructs_not_to_carry_out_action(self):
        assert "do not carry out the action" in self.prompt.lower()

    def test_lists_each_effector_option(self):
        for name, description in self.options:
            assert name in self.prompt
            assert description in self.prompt

    def test_lists_wait_option(self):
        assert WAIT_ACTION in self.prompt
        assert "Take no action this cycle" in self.prompt

    def test_output_contract_is_double_quoted_json(self):
        assert "code fences" in self.prompt.lower() or "no commentary" in self.prompt.lower()
        assert '{"rationale"' in self.prompt
        assert '"action"' in self.prompt
        assert "'rationale'" not in self.prompt
        assert "'action'" not in self.prompt

    def test_example_places_rationale_before_action(self):
        rationale_index = self.prompt.index('"rationale"')
        action_index = self.prompt.index('"action"')

        assert rationale_index < action_index


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


class TestBuildActSystemPrompt:
    def test_includes_constitution_and_identifies_act_step(self):
        constitution = "You are Tam."

        prompt = build_act_system_prompt(constitution)

        assert constitution in prompt
        assert "Act" in prompt

    def test_states_decision_already_made(self):
        prompt = build_act_system_prompt("constitution")

        assert "already" in prompt.lower()


class TestBuildActUserPrompt:
    def setup_method(self):
        self.context = '{"id":"1","actor":"user","type":"chat_message","content":{}}'
        self.action = "respond_in_web_chat"
        self.rationale = "George just said hello and is waiting on a reply."
        self.act_instruction = "Compose your chat message to George now."
        self.prompt = build_act_user_prompt(
            self.context, self.action, self.rationale, self.act_instruction
        )

    def test_includes_context(self):
        assert self.context in self.prompt

    def test_includes_chosen_action(self):
        assert self.action in self.prompt

    def test_includes_rationale(self):
        assert self.rationale in self.prompt

    def test_includes_act_instruction_verbatim(self):
        assert self.act_instruction in self.prompt
