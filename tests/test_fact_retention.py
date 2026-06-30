from __future__ import annotations

from unittest.mock import MagicMock

from src.agents.aldric import Aldric
from sentence_transformers import CrossEncoder


class TestFactRetention:
    def setup_method(self):
        self.encoder = CrossEncoder('dleemiller/ModernCE-base-nli')

    def test_remembers_users_name(self):
        agent = Aldric()
        agent.core.orient("Hello, my name is George.")
        agent.core.orient("What is my name?")

        response = agent.chat_effector.response

        self.encoder.predict([(response, "The user's name is George")])


