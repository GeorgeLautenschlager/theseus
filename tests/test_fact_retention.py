from __future__ import annotations

from sentence_transformers import CrossEncoder

from theseus.agents.alty_mcgee import AltyMcGee
from theseus.memory_store import MemoryStore
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
        # Within one session the name is still in the stimulus-log window, so this passes
        # on recent events alone — no recall required.
        agent = AltyMcGee(
            stimulus_log=StimulusLog(path=tmp_path / "log.jsonl"),
            memory_store=MemoryStore(tmp_path / "a_mem.jsonl"),
        )

        self._say(agent, "Hello, my name is George.")
        response = self._say(agent, "What is my name?")

        assert response is not None
        assert self._entails_name_is_george(response)

    def test_remembers_users_name_across_sessions(self, tmp_path):
        # A "session" is one AltyMcGee instance. Both sessions share the log and the note
        # store on disk, so either path can carry the name across: the window if the log
        # is still short enough, or a note Alty formed and later chose to `recall`.
        log_path = tmp_path / "log.jsonl"
        store_path = tmp_path / "a_mem.jsonl"

        def session() -> AltyMcGee:
            return AltyMcGee(
                stimulus_log=StimulusLog(path=log_path),
                memory_store=MemoryStore(store_path),
            )

        self._say(session(), "Hello, my name is George.")
        response = self._say(session(), "What is my name?")

        assert response is not None
        assert self._entails_name_is_george(response)

    def test_recalls_a_name_from_an_entirely_separate_conversation(self, tmp_path):
        """The one that actually exercises recall. Shrinking `window_size` cannot isolate
        memory: the agent's own reply to the greeting says "George" too, so the name stays
        in any window wide enough to still hold the question. A second session with a
        fresh conversation log over the same note store is the only sound isolation —
        nothing in session two's log mentions the name, so recall is the only route."""
        store_path = tmp_path / "a_mem.jsonl"

        first = AltyMcGee(
            stimulus_log=StimulusLog(path=tmp_path / "session_one.jsonl"),
            memory_store=MemoryStore(store_path),
        )
        self._say(first, "Hello, my name is George.")
        assert first.memory.store.read_all(), "the first session should leave a note"

        second = AltyMcGee(
            stimulus_log=StimulusLog(path=tmp_path / "session_two.jsonl"),
            memory_store=MemoryStore(store_path),
        )
        response = self._say(second, "What is my name?")

        assert response is not None
        assert self._entails_name_is_george(response)
