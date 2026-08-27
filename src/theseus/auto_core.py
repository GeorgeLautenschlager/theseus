from __future__ import annotations

import hashlib
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Dict, List, Tuple
from uuid import uuid4

from theseus.cadence import DEFAULT_TICK_SECONDS, Cadence
from theseus.cognitive_prompts import render_tools_section
from theseus.context_assembler import ContextAssembler
from theseus.model_providers import PROVIDER_REGISTRY
from theseus.model_providers.model_provider import ModelProvider
from theseus.schedule import Schedule
from theseus.stimulus_log import StimulusEvent, StimulusLog
from theseus.tools.tool import AssistantTurn, Tool, ToolCall

SLEEP_FLOOR_SECONDS = 5

# Shown to the agent when a config line fails the grammar, and embedded in the seed
# files. The grammars are deterministic on purpose: interpreting these files costs
# zero model calls per turn; the agent itself rewrites any line that doesn't parse.
CONFIG_GRAMMAR_HELP = (
    "SCHEDULE.md task lines, one per line, times UTC: "
    "'- [ ] once @ YYYY-MM-DD HH:MM: task', '- [ ] daily @ HH:MM: task', "
    "'- [ ] weekly @ Weekday HH:MM: task', '- [ ] monthly @ D HH:MM: task', "
    "'- [ ] quarterly @ D HH:MM: task', '- [ ] annually @ MM-DD HH:MM: task', "
    "'- [ ] every N seconds|minutes|hours: task'. "
    "CADENCE.md rule lines, matched against server time: "
    "'- HH:MM-HH:MM: provider model[, context N[k]][, tick every N seconds|minutes|hours]' "
    "(windows may wrap midnight; first match wins) and "
    "'- default: provider model[, context N[k]][, tick every N ...]'. "
    "'context' is that model's context window in tokens ('context 128k'); omit it only "
    "if you don't know, since omitting it means assuming a very small window. "
    f"Providers: {', '.join(sorted(PROVIDER_REGISTRY))}."
)

# Grammar examples are backtick-quoted so the seeds themselves never parse (or lint)
# as live entries.
SCHEDULE_SEED = """\
# Schedule

One-time and repeated tasks, one per line. Times are UTC. The grammar (examples in
backticks so they stay inert — remove the backticks to activate a line):

`- [ ] once @ 2026-08-05 14:00: Water the plants`
`- [ ] daily @ 09:00: Check email`
`- [ ] weekly @ Monday 09:00: Submit timesheet`
`- [ ] monthly @ 1 09:00: Pay rent`
`- [ ] quarterly @ 1 09:00: File quarterly report`
`- [ ] annually @ 12-25 09:00: Send holiday cards`
`- [ ] every 30 minutes: Check queue depth`

Anything else in this file is prose and is ignored.
"""

CADENCE_SEED = """\
# Cadence

Which model to think with at which time of day, and how often to take an autonomous
turn. Rules are matched against server time; the first matching window wins and the
`default` line is both the off-hours rule and the fallback when a provider is down.
Example (backtick-quoted so it stays inert):

`- 22:00-08:00: lm_studio qwen/qwen3-32b, context 32k, tick every 15 minutes`

`context` declares that model's context window, and is what your prompt gets sized
against. Leave it off and a deliberately small window is assumed — safe, but you will
think with far less history than the model could actually hold.

- default: lm_studio local-model, tick every 5 minutes
"""


class Autocore:
    """A self-driving cognitive loop: think, act, sleep, repeat.

    Unlike OODACore — which is *driven*, sitting inert until an observer calls
    `orient` — Autocore owns its own thread and paces itself from CADENCE.md. Each
    turn assembles context from the StimulusLog, takes one native tool-calling turn,
    executes whatever the model chose, fires due SCHEDULE.md tasks, and sleeps until
    the next tick.

    ### Waking early

    Autonomy on its own makes an agent that acts on its own schedule but cannot be
    *reached*: a message arriving ten minutes into a thirty-minute sleep would wait out
    the other twenty. So the sleep is not `time.sleep`, which nothing can cancel — it is
    `threading.Event.wait`, which is a sleep another thread can cut short. Anything that
    wants the loop to come around now calls `wake()`; the core also subscribes to its own
    StimulusLog, so in practice any externally-authored event — a chat message from
    George, a fired reminder — wakes it without the appender needing to know Autocore
    exists at all.

    Only the *sleep* is interruptible. A turn already in flight runs to completion and
    the wake is consumed by the turn after it, so cognition never overlaps itself and —
    unlike OODACore — Autocore needs no cycle lock. Observers only ever flip a flag;
    every model call, tool execution and file read stays on the loop thread.

    Args:
        name: the core's own actor name. Its `decision` and `tool_result` events are
            logged under it, and events by this actor are exactly what the default wake
            filter ignores — a core that woke itself would never sleep again.
        home_directory: where the log, config and identity files live. Created and
            seeded on construction.
        tools: the tools the model may call, keyed by name.
        wake_on: predicate deciding which appended events cut a sleep short. Defaults to
            "anything this core did not write itself". Pass something narrower (say
            `lambda event: event.type == "chat_message"`) to be woken only by
            conversation.
    """

    def __init__(
        self,
        name: str,
        home_directory: Path,
        tools: Dict[str, Tool], #TODO: why not pull this from config as well?
        wake_on: Callable[[StimulusEvent], bool] | None = None,
    ):
        self.name: str = name
        self._initialize_home_directory(home_directory)
        self.stimulus_log: StimulusLog = StimulusLog(
            str(self.home_directory / "stimulus_log.jsonl")
        )
        self.tools: dict[str, Tool] = tools
        self.context_assembler: ContextAssembler = ContextAssembler(
            stimulus_log=self.stimulus_log
        )

        with (self.home_directory / "constitution.md").open("r") as file:
            self.constitution: str = file.read()

        with (self.home_directory / "persona.md").open("r") as file:
            self.persona: str = file.read()

        self.loop_memory: Dict[str, Any] = {}
        self.sleep_duration: float = DEFAULT_TICK_SECONDS
        self.model_providers: Dict[Tuple[str, str], ModelProvider] = {}
        self.unknown_providers: List[str] = []
        self._cadence_hash: str | None = None

        # The interrupt. `_wake` is the flag `_sleep` waits on; `_wake_trigger` is what
        # set it, kept for the prompt. Both are touched from other threads, so every
        # read-modify-write of the pair goes through `_wake_lock` — otherwise a wake
        # landing between "read trigger" and "clear flag" would be silently swallowed.
        self._wake_on: Callable[[StimulusEvent], bool] = wake_on or self._is_external
        self._wake: threading.Event = threading.Event()
        self._wake_trigger: StimulusEvent | str | None = None
        self._wake_lock: threading.Lock = threading.Lock()
        self.stimulus_log.subscribe(self._on_stimulus)

    def loop(self) -> None:
        while True:
            # Take the pending wake *before* reading any state. Everything this turn
            # goes on to see was appended before this line, so anything landing after
            # it re-arms the flag and is answered by the next turn instead of idling
            # out a full tick. The race costs at worst one redundant turn; the other
            # ordering costs a dropped message.
            self.loop_memory["wake_trigger"] = self._consume_wake()

            self.goals = self._read_goals()
            self.tasks = self._read_tasks()
            self.current_task = self._read_current_task()
            self.schedule = Schedule(self.home_directory / "SCHEDULE.md")

            # Pick the model *first*: the context budget is a property of whichever
            # model this turn's cadence rule selected, so there is nothing to fit the
            # window against until that is decided. _select_model_provider hands the
            # rule's declared window to the assembler on the way through.
            model = self._select_model_provider()

            # assemble system prompt
            system_prompt = self._assemble_system_prompt()

            # load history from stimulus log, fitted around everything else in the
            # prompt. Measuring the overhead beats inferring it from the previous turn:
            # the inferred value is zero on the first turn, which is precisely when a
            # mis-sized budget overruns.
            goals_and_tasks = self._current_goals_and_tasks()
            context = self.context_assembler.assemble_context(
                overhead_chars=len(system_prompt) + len(goals_and_tasks)
            )
            self.loop_memory["window_chars"] = context.window_chars

            # assemble autonomous prompt
            autonomous_prompt = (
                f"{goals_and_tasks}"
                f"<stimulus_log>\n{context.recent_events}\n</stimulus_log>\n\n"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": autonomous_prompt},
            ]
            turn = model.complete_with_tools(messages, list(self.tools.values()))

            # log the decision and execute whatever it chose, rescuing a reply the
            # model wrote as prose instead of as a call to its chat tool
            self._take_action(turn)

            # check schedule and append reminders
            self._append_reminders()

            # assess context length
            self.context_assembler.observe(
                prompt_tokens=turn.prompt_tokens,
                prompt_chars=sum(len(message["content"]) for message in messages),
                window_chars=self.loop_memory["window_chars"],
            )

            self._sleep()

    def wake(self, trigger: StimulusEvent | str | None = None) -> None:
        """Cut short any sleep in progress so the next turn starts now.

        Safe to call from any thread, at any time, as often as you like: it sets a flag,
        it does not run cognition. A wake raised while a turn is already running is not
        lost — the flag survives the turn, and the `_sleep` that follows returns without
        waiting at all.

        `trigger` is recorded so the prompt can tell the agent what pulled it out of its
        sleep. The first one after a turn boundary wins, since that is the one that
        actually did the waking; later ones are along for the ride and show up in
        context anyway.
        """
        with self._wake_lock:
            if self._wake_trigger is None:
                self._wake_trigger = trigger
            self._wake.set()

    def _consume_wake(self) -> StimulusEvent | str | None:
        """Take and clear the pending wake, returning whatever caused it (None if the
        loop simply came round on the clock)."""
        with self._wake_lock:
            trigger = self._wake_trigger
            self._wake_trigger = None
            self._wake.clear()
        return trigger

    def _on_stimulus(self, event: StimulusEvent) -> None:
        """StimulusLog listener: something landed — is it worth interrupting a sleep
        for? Runs on whichever thread did the append (an HTTP handler, the stdin
        reader, the loop itself), so it does the least work possible and touches
        nothing the loop thread owns."""
        if self._wake_on(event):
            self.wake(event)

    def _is_external(self, event: StimulusEvent) -> bool:
        """The default wake filter: anything this core did not write itself. Its own
        `decision` and `tool_result` events must not count — every turn writes several,
        so counting them would re-wake the loop that just finished and the agent would
        never sleep at all."""
        return event.actor != self.name

    def _initialize_home_directory(self, home_directory: Path) -> None:
        self.home_directory = home_directory
        self.home_directory.mkdir(parents=True, exist_ok=True)
        for name in (
            "stimulus_log.jsonl",
            "CONSTITUTION.md",
            "PERSONA.md",
            "GOALS.md",
            "TASKS.md",
            "CURRENT_TASK.md",
        ):
            (self.home_directory / name).touch(exist_ok=True)
        self._seed_config("SCHEDULE.md", SCHEDULE_SEED)
        self._seed_config("CADENCE.md", CADENCE_SEED)

    def _seed_config(self, name: str, seed: str) -> None:
        """Write the template only when the file is missing or empty — a config the
        agent (or George) has edited is never clobbered."""
        path = self.home_directory / name
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            path.write_text(seed, encoding="utf-8")

    # All of these files need to be converted over to JSONL. There's little reason to
    # introduce markdown lists when JSONL is working great in akk other cases.
    def _read_goals(self) -> List[str] :
        with (self.home_directory / "GOALS.md").open("r") as file:
            return [
                line.strip() for line in file.readlines() if line.strip()
            ]

    def _read_tasks(self) -> List[str] :
        with (self.home_directory / "TASKS.md").open("r") as file:
            return [
                line.strip() for line in file.readlines() if line.strip()
            ]

    def _read_current_task(self) -> List[str]:
        with (self.home_directory / "CURRENT_TASK.md").open("r") as file:
            return file.readlines()

    def _assemble_system_prompt(self) -> str:
        system_prompt = (
            f"{self.constitution}\n\n"
            "---\n\n"
            f"{self._automated_prompt_instructions()}\n\n"
            "---\n\n"
            f"{render_tools_section(list(self.tools.values()))}\n\n"
            "---\n\n"
            f"{self.persona}\n\n"
        )

        return system_prompt

    def _automated_prompt_instructions(self) -> str:
        return f"""### Autonomous Prompts
            You are being prompted autonomously with a gap of {self.sleep_duration} seconds
            between your response and the next prompt. That means you will have plenty of opportunity
            for self-directed work - make the most of that time. Communication with other minds
            (especially George) should be your top priority. Try to be proactive. Each prompt
            includes events from recent history such as tool calls and results, memories recalled,
            messages from George, etc. That history is how youperceive your environment and other minds
            so pay close attention to it, especially the most recent events which may require a timely
            response.{self._wake_notice()}

            ### Self Directed Action
             ---\n\n
            If you don't have a goal, form one or more goals (each with a UUID) in alignment with your telos. Put them in {self.home_directory}/GOALS.md ranked by priority.
            if you don't have any tasks form one or more tasks (each with a UUID) in alignment with your goals. Put them in {self.home_directory}/TASKS.md ranked by priority.
            If you have tasks, but not a current task, select one and put it in {self.home_directory}/CURRENT_TASK.md.
            If you have a current task, do something to advance it. If that task is complete, remove it from {self.home_directory}/TASKS.md and clear {self.home_directory}/CURRENT_TASK.md.
            Pay close attention to your recent history in <stimulus_log> as that's where you can see the results of your previous actions."""

    def _wake_notice(self) -> str:
        """The line that tells the agent this turn was asked for rather than scheduled.

        Without it the prompt insists it is on a fixed cadence with time to spare, which
        is exactly the wrong posture for a turn that exists because someone is waiting
        on a reply."""
        trigger = self.loop_memory.get("wake_trigger")
        if trigger is None:
            return ""
        if isinstance(trigger, StimulusEvent):
            source = f"a '{trigger.type}' event from '{trigger.actor}'"
        else:
            source = str(trigger)
        slept = self.loop_memory.get("slept_seconds")
        if self.loop_memory.get("woke_early") and isinstance(slept, (int, float)):
            cause = f"your sleep was cut short after {slept:.0f} seconds by {source}"
        else:
            # The wake landed in the gap between the tick elapsing and this turn
            # starting — nothing was actually interrupted, so don't claim it was.
            cause = f"it was triggered by {source}"
        return (
            f" That gap does not apply to this turn: {cause}. Something outside you"
            f" wants attention now, so read the end of <stimulus_log> first and deal"
            f" with it before returning to self-directed work."
        )

    def _current_goals_and_tasks(self) -> str:
        return (
            "---\n\n"
            "## Goals\n"
            f"{self.goals}\n\n"
            "## Tasks\n"
            f"{self.tasks}\n\n"
            "## Current Task\n"
            f"{self.current_task}\n\n"
            "---\n\n"
        )

    def _take_action(self, turn: AssistantTurn) -> None:
        """Commit one decision to the log and carry it out."""
        recovered = self._recover_stray_text(turn)
        self._log_decision(recovered or turn, recovered=recovered is not None)
        self._execute_tool_calls(recovered or turn)

    def _terminal_tool(self) -> Tool | None:
        """The agent's mouth: the tool whose call completes a turn (`WebChat`,
        `TerminalChat`). None for an agent composed without one."""
        for tool in self.tools.values():
            if getattr(tool, "ends_turn", False):
                return tool
        return None

    def _recover_stray_text(self, turn: AssistantTurn) -> AssistantTurn | None:
        """Turn a reply the model wrote as prose into the call it should have made.

        A model that answers George in `content` rather than by calling its chat tool
        has said something nobody will ever read: `_log_decision` files the text in the
        stimulus log and `_execute_tool_calls` walks `tool_calls`, which is empty. The
        text is visible in the debug view and nowhere else.

        Left alone it also compounds, which is the worse half. The orphaned decision
        stays in the context window as a worked example — `{"text": "...",
        "tool_calls": []}` — authored by the agent itself, so the next turn imitates it
        and the one after that imitates them both. Observed live: the same reply
        regenerated verbatim across four turns while George waited.

        So the prose is delivered rather than dropped, and what lands in the log is the
        call, not the pattern that caused this. `text_recovered` keeps the record
        honest about the substitution. Returns None when there is nothing to recover.

        Commentary alongside a real tool call is not a stray reply — the model chose an
        action and narrated it — and neither is empty text, which is how a native
        tool-calling model says "nothing to do this tick".
        """
        if turn.tool_calls or not (turn.text or "").strip():
            return None
        tool = self._terminal_tool()
        if tool is None:
            return None
        # The mouth takes one string; read its name off the schema rather than assuming
        # "message", so a differently-shaped chat tool still works.
        required = tool.parameters.get("required") or ["message"]
        call = ToolCall(
            id=f"recovered-{uuid4().hex}",
            name=tool.name,
            arguments={required[0]: turn.text},
        )
        return replace(turn, text=None, tool_calls=(call,))

    def _log_decision(self, turn: AssistantTurn, recovered: bool = False):
        content: Dict[str, Any] = {
            "text": turn.text,
            "tool_calls": [
                {"name": call.name, "arguments": call.arguments}
                for call in turn.tool_calls
            ],
        }
        if recovered:
            content["text_recovered"] = True
        self.stimulus_log.append(actor=self.name, type="decision", content=content)

    def _execute_tool_calls(self, turn: AssistantTurn):
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

    def _construct_model_providers(self) -> None:
        """Parse CADENCE.md and instantiate one provider per unique (provider, model).

        Content-hash guarded: instances (and their HTTP clients) are reused across
        ticks until the file actually changes. An unknown provider name must not
        crash the loop — it is recorded so _append_reminders can surface it to the
        agent as a lint."""
        text = (self.home_directory / "CADENCE.md").read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest == self._cadence_hash:
            return

        cadence = Cadence.parse(text)
        providers: Dict[Tuple[str, str], ModelProvider] = {}
        unknown: List[str] = []
        for rule in cadence.rules:
            key = (rule.provider_key, rule.model)
            if key in providers:
                continue
            provider_class = PROVIDER_REGISTRY.get(rule.provider_key)
            if provider_class is None:
                message = f"unknown provider '{rule.provider_key}' in CADENCE.md"
                if message not in unknown:
                    unknown.append(message)
                continue
            providers[key] = provider_class(model=rule.model)

        self.cadence = cadence
        self.model_providers = providers
        self.unknown_providers = unknown
        self._cadence_hash = digest

    def _select_model_provider(self) -> ModelProvider:
        """First available provider for this moment's cadence rules, in priority
        order: matching windows in file order, then the default rule. Cadence is
        matched against naive server time on purpose — it describes this machine's
        day, while SCHEDULE.md stays UTC."""
        self._construct_model_providers()
        for rule in self.cadence.candidates_for(datetime.now()):
            provider = self.model_providers.get((rule.provider_key, rule.model))
            if provider is not None and provider.is_available():
                self.loop_memory["tick_seconds"] = rule.tick_seconds
                self.loop_memory["context_tokens"] = rule.context_tokens
                # The budget travels with the rule, so falling back to a smaller model
                # shrinks the window in the same breath that selects it. A rule that
                # declares no `context` leaves the assembler on its own conservative
                # default rather than inheriting the previous rule's larger one.
                self.context_assembler.set_context_limit(rule.context_tokens)
                return provider
        raise RuntimeError("No model providers are currently available.")

    def _append_reminders(self) -> None:
        """Fire due SCHEDULE.md tasks into the stimulus log, then surface any config
        lines that failed the grammar so the agent rewrites them itself. The lint is
        appended only when the set of bad lines changes — a standing mistake nags
        once, not every tick."""
        self.schedule.fire_due(self.stimulus_log)

        bad = (
            self.schedule.lint_lines()
            + self.cadence.lint_lines()
            + self.unknown_providers
        )
        if not bad:
            self.loop_memory.pop("last_lint", None)
            return
        if bad == self.loop_memory.get("last_lint"):
            return
        self.stimulus_log.append(
            actor="schedule",
            type="schedule_lint",
            content={
                "message": (
                    "These lines in your SCHEDULE.md/CADENCE.md look like entries but "
                    "are not machine-readable, so they will never take effect. Rewrite "
                    "them with your edit tool to match the grammar. "
                    + CONFIG_GRAMMAR_HELP
                ),
                "lines": bad,
            },
        )
        self.loop_memory["last_lint"] = bad

    def _next_tick_seconds(self) -> float:
        """How long this turn's sleep should last: the current cadence tick, shortened
        if a scheduled task comes due sooner, and never busier than the floor."""
        duration = float(self.loop_memory.get("tick_seconds", DEFAULT_TICK_SECONDS))
        next_due = self.schedule.next_occurrence()
        if next_due is not None:
            until_due = (next_due - datetime.now(timezone.utc)).total_seconds()
            duration = min(duration, until_due)
        return max(SLEEP_FLOOR_SECONDS, duration)

    def _sleep(self) -> bool:
        """Sleep until the next tick or until something wakes us, whichever comes first.
        Returns True if the sleep was cut short.

        `Event.wait(timeout)` is the whole trick: a `time.sleep` another thread can
        cancel. It returns True the instant the flag is set and False when the full
        duration elapses, so the loop can tell "the clock came round" from "somebody
        wants me now" — and a wake raised during the preceding turn is still pending
        here, so it returns immediately rather than sleeping on an unanswered message.
        """
        duration = self._next_tick_seconds()
        self.sleep_duration = duration
        started = monotonic()
        woke_early = self._wake.wait(duration)
        self.loop_memory["slept_seconds"] = monotonic() - started
        self.loop_memory["woke_early"] = woke_early
        return woke_early
