from modules.cognitive_core import CognitiveCore


class Aldric:
    """Aldric is an agent built with the Theseus architecture.
    
    Aldric is build in concentric layers. In the centre is an LLM, surrounded by a 
    truncated OODA loop. Orient, Decide and Act make up what is called the Cognitive Core.
    Observers, are outside of that, along with memory systems and more complex subagents that
    function as sensory surrogates, feeding pre-processed information into the core.

    Args: 
        core: this is the cognition of the agent. It operates as a truncated OODA loop which can be 
        entered at any point, but loops until it decides to terminate.
        observers: One or more modules responsible for collecting and  where appropriate, pre-processing data
        Memories: One or more memory systems
        surrogates: One or more subagents that function as sensory surrogates, feeding pre-processed information into the core.
        model_providers: One or more model provider. These are listed in priority order, with the cognitive core ultimately
        deciding which model provider is used.
    """
    def __init__(
        self,
        core: CognitiveCore,
        observers: list[Observer],
        Memories: List[Memory],
        surrogates: List[Surrogate]
    ):
        self.core = core