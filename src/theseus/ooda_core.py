from __future__ import annotations

import threading
from datetime import datetime
from typing import List

from theseus.agentic_memory import AgenticMemory
from theseus.cognitive_prompts import (
    build_decide_system_prompt,
    build_decide_user_prompt,
)
from theseus.context_assembler import ContextAssembler
from theseus.memory import Memory
from theseus.stimulus_log import StimulusLog
from theseus.tools.tool import Tool

from .model_providers.model_provider import ModelProvider


class OODACore:
    """A truncated OODA loop that serves as the primary cognitive process for a Theseus agent.

    The core does Orient, Decide and Act, steps and internalizes the LLM. Steps never pass
    data to each other directly — everything a loop accumulates lives in `loop_memory`,
    because any step may kick back to Orient and re-enter the cycle.

    Args:
        model_providers: One or more model provider. These are listed in priority order, with the cognitive core ultimately
        deciding which model provider is used.
        tools: List of available Tool objects.
        memory: Optional memory module, seen only through the Memory protocol. The core's
        one remaining tie to it is the `form()` signal at loop termination; retrieval is
        the agent's own business, done by calling the `recall` tool, which reaches the
        module directly. What and how the module consolidates is its own business.
        name: The core's own actor name, used when logging its decisions and actions.
    """
    def __init__(
        self,
        name: str,
        constitution: str,
        persona: str,
        context_assembler: ContextAssembler,
        stimulus_log: StimulusLog,
        model_providers: List[ModelProvider],
        tools: dict[str, Tool],
        memory: AgenticMemory | None = None,
        max_loops: int = 10
    ):
        self.name = name
        self.stimulus_log = stimulus_log
        self.memory = memory
        self.constitution = constitution
        self.persona = persona
        self.context_assembler = context_assembler
        self.model_providers = model_providers
        self.loop_memory = {}
        self.tools = tools
        self.max_loops = max_loops
        # One gate guarding cognitive-cycle execution, shared across every Observer
        # regardless of concurrency model, so exactly one cycle runs at a time across
        # the whole process. threading.Lock (not asyncio.Lock) because it is held from
        # plain OS threads in every case: TerminalChatObserver's stdin thread, and
        # WebChatUIObserver's per-message background thread.
        self._cycle_lock = threading.Lock()

    def _select_model_provider(self) -> ModelProvider:
        """Selects the first available provider, in priority order."""
        for provider in self.model_providers:
            if provider.is_available():
                return provider
        raise RuntimeError("No model providers are currently available.")

    def try_orient(self) -> bool:
        """Skip-on-contention entry point. Non-blocking acquire of the cycle gate: run
        one cognitive cycle iff no other cycle is in flight, and report whether one
        actually ran. A False return is the ordinary outcome of contention, never an
        error. Observers that must not block (TimeObserver) call this."""
        if not self._cycle_lock.acquire(blocking=False):
            return False
        try:
            self.orient()
            return True
        finally:
            self._cycle_lock.release()

    def orient_and_wait(self) -> None:
        """Wait-on-contention entry point. Blocking acquire of the cycle gate: wait for
        any in-flight cycle to finish, then run one. Safe only from a thread with
        nothing else to do while it waits — TerminalChatObserver's dedicated stdin
        thread, or WebChatUIObserver's per-message background thread. Never call this
        from a coroutine on an event loop."""
        self._cycle_lock.acquire()
        try:
            self.orient()
        finally:
            self._cycle_lock.release()

    def orient(self):
        """Run one cognitive cycle. Un-gated on purpose: it assumes the caller already
        holds the cycle gate (via try_orient / orient_and_wait) or is a single-threaded
        test. Act() re-enters this method directly to process tool results, so it must
        never take the gate itself. Observers must NOT call orient() directly — they go
        through try_orient() or orient_and_wait()."""
        assembled = self.context_assembler.assemble_context()
        self.loop_memory["recent_events"] = assembled.recent_events
        self.loop_memory["window_chars"] = assembled.window_chars

        self.decide()

    def decide(self):
        model_provider = self._select_model_provider()

        system_prompt = build_decide_system_prompt(self.constitution, self.persona, self.tools.values())
        prompt = build_decide_user_prompt(
            self.loop_memory["recent_events"],
            str(datetime.now()),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # One native tool-calling turn: the model's chosen tool(s) and their arguments
        # come back together, so there is no separate Act model call.
        turn = model_provider.complete_with_tools(messages, list(self.tools.values()))
        self.loop_memory["decision"] = turn

        # Close the context-sizing loop: the backend just told us what this prompt really
        # cost, which is the only trustworthy signal about the window we get. Act() may
        # re-enter orient() on a non-terminal tool, so a multi-pass turn corrects itself
        # partway through rather than waiting for the next one.
        self.context_assembler.observe(
            prompt_tokens=turn.prompt_tokens,
            prompt_chars=sum(len(message["content"]) for message in messages),
            window_chars=self.loop_memory["window_chars"],
        )

        self.stimulus_log.append(
            actor=self.name,
            type="decision",
            content={
                "text": turn.text,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in turn.tool_calls
                ],
            },
        )

        self.act()

    def act(self):
        turn = self.loop_memory["decision"]

        # No tool call is the native-tools equivalent of "wait": the model chose to
        # act on nothing this cycle.
        if not turn.tool_calls:
            self.loop_termination()
            return

        ends_turn = False
        for call in turn.tool_calls:
            tool = self.tools.get(call.name)
            if tool is None:
                # Record the miss as a stimulus so the next pass can see it and recover.
                self.stimulus_log.append(
                    actor=self.name,
                    type="tool_result",
                    content={
                        "tool": call.name,
                        "output": f"Unknown tool: {call.name}",
                        "is_error": True,
                    },
                )
                continue

            result = tool.execute(**call.arguments)
            self.stimulus_log.append(
                actor=self.name,
                type="tool_result",
                content={
                    "tool": call.name,
                    "arguments": call.arguments,
                    "output": result.content,
                    "is_error": result.is_error,
                },
            )
            if getattr(tool, "ends_turn", False):
                ends_turn = True

        # The pass counter lives in loop_memory (reset by loop_termination), so it counts
        # passes within a single turn and resets when the turn ends.
        passes = self.loop_memory.get("passes", 0) + 1
        self.loop_memory["passes"] = passes

        # Terminate when a terminal tool ran (the agent responded) or the runaway cap is
        # hit; otherwise re-enter Orient so the model can act on the tool results, which
        # are now in the stimulus log.
        if ends_turn or passes >= self.max_loops:
            self.loop_termination()
        else:
            self.orient()

    def loop_termination(self):
        """Runs once per cognitive loop, only when Act is actually terminating it —
        skipped whenever a step kicks off another cycle. For now the memory-formation
        signal is hard coded; later this should accumulate callbacks and iterate over
        them."""
        if self.memory is not None:
            self.memory.form()
        self.loop_memory = {}
