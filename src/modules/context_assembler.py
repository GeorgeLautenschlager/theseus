from src.modules.stimulus_log import StimulusLog


class ContextAssembler:
    """assembles context for Decide from the following sources:
    - StimulusLog
    - persona, if present
    - constitution, if present
    - the agent's memory systems, if any are present
    """

    def __init__(self, stimulus_log: StimulusLog, persona: str | None = None, constitution: str | None = None):
        self.stimulus_log = stimulus_log
        self.constitution = constitution

    def assemble_context(self) -> dict:
        context = '\n'.join(event.to_json() for event in self.stimulus_log.read_all())

        return context

