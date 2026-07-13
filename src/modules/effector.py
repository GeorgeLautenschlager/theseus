from __future__ import annotations

from typing import Protocol


class Effector(Protocol):
    """A pluggable action CognitiveCore's Decide step can choose and Act can carry out."""

    name: str             # semantic action name used as the dict key and in Decide's menu,
                           # e.g. "respond_in_web_chat" — NOT a class name
    description: str      # one line for Decide's menu: what it does + when to choose it
    act_instruction: str  # the Act-phase task instruction + output contract, handed to the
                           # model when this effector's action was chosen

    def execute(self, payload: str) -> None:
        """Carry out the effector's real-world effect with the model's Act-phase output."""
        ...
