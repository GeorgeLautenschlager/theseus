from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Tuple

from theseus.cadence import DEFAULT_TICK_SECONDS, Cadence
from theseus.cognitive_prompts import render_tools_section
from theseus.context_assembler import ContextAssembler
from theseus.model_providers import PROVIDER_REGISTRY
from theseus.model_providers.model_provider import ModelProvider
from theseus.schedule import Schedule
from theseus.stimulus_log import StimulusLog
from theseus.tools.tool import AssistantTurn, Tool

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
    "'- HH:MM-HH:MM: provider model[, tick every N seconds|minutes|hours]' "
    "(windows may wrap midnight; first match wins) and "
    "'- default: provider model[, tick every N ...]'. "
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

`- 22:00-08:00: lm_studio qwen/qwen3-32b, tick every 15 minutes`

- default: lm_studio local-model, tick every 5 minutes
"""


class Autocore:
    def __init__(
        self,
        name: str,
        home_directory: Path,
        tools: Dict[str, Tool], #TODO: why not pull this from config as well?
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

    def loop(self) -> None:
        while True:
            self.goals = self._read_goals()
            self.tasks = self._read_tasks()
            self.current_task = self._read_current_task()
            self.schedule = Schedule(self.home_directory / "SCHEDULE.md")

            # load history from stimulus log
            context = self.context_assembler.assemble_context()
            self.loop_memory["window_chars"] = context.window_chars

            # assemble system prompt
            system_prompt = self._assemble_system_prompt()

            # assemble autonomous prompt
            autonomous_prompt = (
                f"{self._current_goals_and_tasks()}"
                f"<stimulus_log>\n{context.recent_events}\n</stimulus_log>\n\n"
                f"{self._automated_prompt_instructions()}"
            )

            # prompt model with tool calling
            model = self._select_model_provider()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": autonomous_prompt},
            ]
            turn = model.complete_with_tools(messages, list(self.tools.values()))
            self._log_decision(turn)

            # execute tool calls, if any
            self._execute_tool_calls(turn)
                # commit results to stimulus log

            # check schedule and append reminders
            self._append_reminders()

            # assess context length
            self.context_assembler.observe(
                prompt_tokens=turn.prompt_tokens,
                prompt_chars=sum(len(message["content"]) for message in messages),
                window_chars=self.loop_memory["window_chars"],
            )

            self._sleep()

    def _initialize_home_directory(self, home_directory: Path) -> None:
        self.home_directory = home_directory
        self.home_directory.mkdir(parents=True, exist_ok=True)
        for name in (
            "stimulus_log.jsonl",
            "constitution.md",
            "persona.md",
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
            response.

            ### Self Directed Action
             ---\n\n
            If you don't have a goal, form one or more goals (each with a UUID) in alignment with your telos. Put them in {self.home_directory}/GOALS.md ranked by priority.
            if you don't have any tasks form one or more tasks (each with a UUID) in alignment with your goals. Put them in {self.home_directory}/TASKS.md ranked by priority.
            If you have tasks, but not a current task, select one and put it in {self.home_directory}/CURRENT_TASK.md.
            If you have a current task, do something to advance it. If that task is complete, remove it from {self.home_directory}/TASKS.md and clear {self.home_directory}/CURRENT_TASK.md.
            Pay close attention to your recent history in <stimulus_log> as that's where you can see the results of your previous actions."""

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

    def _log_decision(self, turn: AssistantTurn):
        self.stimulus_log.append(
            actor="autobot",
            type="decision",
            content={
                "text": turn.text,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in turn.tool_calls
                ],
            },
        )

    def _execute_tool_calls(self, turn: AssistantTurn):
        for call in turn.tool_calls:
          tool = self.tools.get(call.name)
          if tool is None:
              # Record the miss as a stimulus so the next pass can see it and recover.
              self.stimulus_log.append(
                  actor="autobot",
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
              actor="autobot",
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

    def _sleep(self) -> None:
        """Sleep the current cadence tick, waking early if a scheduled task comes due
        sooner, and never busier than the floor."""
        duration = float(self.loop_memory.get("tick_seconds", DEFAULT_TICK_SECONDS))
        next_due = self.schedule.next_occurrence()
        if next_due is not None:
            until_due = (next_due - datetime.now(timezone.utc)).total_seconds()
            duration = min(duration, until_due)
        duration = max(SLEEP_FLOOR_SECONDS, duration)
        self.sleep_duration = duration
        sleep(duration)
