from __future__ import annotations

from sentence_transformers import CrossEncoder

from theseus.agents.alty_mcgee import AltyMcGee
from theseus.stimulus_log import StimulusLog

# ModernCE-base-nli emits its three scores in this fixed label order.
CORRECTED_LABELS_FOR_MODERN_CE = ["contradiction", "entailment", "neutral"]


class TestFactRetention:
    def setup_method(self):
        self.encoder = CrossEncoder("dleemiller/ModernCE-base-nli")

    def _say(self, agent: AltyMcGee, message: str) -> str | None:
        """Feed a user message the way TerminalChatObserver would, run one cognitive loop,
        and return whatever Alty said back through the terminal_chat tool (None if
        it chose not to reply this cycle)."""
        agent.stimulus_log.append(
            actor="user", type="chat_message", content={"message": message}
        )
        agent.terminal_chat.response = None  # so a stale prior reply can't pass the check
        agent.core.orient()
        return agent.terminal_chat.response

    def _entails_name_is_george(self, response: str) -> bool:
        scores = dict(
            zip(
                CORRECTED_LABELS_FOR_MODERN_CE,
                self.encoder.predict(
                    [(response, "The user's name is George")], apply_softmax=True
                )[0],
            )
        )
        return max(scores, key=scores.get) == "entailment"

    def test_remembers_users_name_within_a_session(self, tmp_path):
        agent = AltyMcGee(stimulus_log=StimulusLog(path=tmp_path / "log.jsonl"))

        self._say(agent, "Hello, my name is George.")
        response = self._say(agent, "What is my name?")

        assert response is not None
        assert self._entails_name_is_george(response)

    def test_remembers_users_name_across_sessions(self, tmp_path):
        # A "session" is one AltyMcGee instance. Alty has no memory module, so the
        # stimulus log persisted on disk is the only thing carrying the name from the
        # first session to the second — both instances point at the same log file.
        log_path = tmp_path / "log.jsonl"

        first_session = AltyMcGee(stimulus_log=StimulusLog(path=log_path))
        self._say(first_session, "Hello, my name is George.")

        second_session = AltyMcGee(stimulus_log=StimulusLog(path=log_path))
        response = self._say(second_session, "What is my name?")

        assert response is not None
        assert self._entails_name_is_george(response)
