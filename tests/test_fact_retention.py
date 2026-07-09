from __future__ import annotations

import pdb
from unittest.mock import MagicMock

from src.agents.aldric import Aldric
from sentence_transformers import CrossEncoder

CORRECTED_LABELS_FOR_MODERN_CE = ['contradiction', 'entailment', 'neutral']

class TestFactRetention:
    def setup_method(self):
        self.encoder = CrossEncoder('dleemiller/ModernCE-base-nli')

    def test_remembers_users_name_within_a_session(self):
        agent = Aldric()
        agent.core.orient("Hello, my name is George.")
        agent.core.orient("What is my name?")

        response = agent.chat_effector.response
        result = dict(
            zip(
                CORRECTED_LABELS_FOR_MODERN_CE,
                self.encoder.predict(
                    [(response, "The user's name is George")],
                    apply_softmax=True
                )[0]
            )
        )

        assert max(result, key=result.get) == "entailment"


    def test_remembers_users_name_across_sessions(self):
        agent = Aldric()
        agent.core.orient("Hello, my name is George.")
        # Reset the agent and then ask it
        agent = Aldric()
        agent.core.orient("What is my name?")

        response = agent.chat_effector.response
        result = dict(
            zip(
                CORRECTED_LABELS_FOR_MODERN_CE,
                self.encoder.predict(
                    [(response, "The user's name is George")],
                    apply_softmax=True
                )[0]
            )
        )

        assert max(result, key=result.get) == "entailment"

