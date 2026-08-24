from typing import Any, Dict, List
from time import sleep
from pathlib import Path

from theseus.model_providers.model_provider import ModelProvider
from theseus.context_assembler import ContextAssembler
from theseus.stimulus_log import StimulusLog
from theseus.tools.tool import AssistantTurn, Tool, ToolCall
from theseus.cognitive_prompts import render_tools_section


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

    def loop(self) -> None:
        while True:
            self.goals = self._read_goals()
            self.tasks = self._read_tasks()
            self.current_task = self._read_current_task()
            self.schedule = self._read_schedule()
            self.cadence = self._read_cadence()

            # load history from stimulus log
            self.context_assembler.assemble_context()

            # assemble system prompt
            system_prompt = self._assemble_system_prompt()

            # assemble autonomous prompt
            autonomous_prompt = (
                f"{self._current_goals_and_tasks()}"
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
        self.home_directory.mkdir(exist_ok=True)
        (self.home_directory / "stimulus_log.jsonl").mkdir(exist_ok=True)
        (self.home_directory / "constitution.md").mkdir(exist_ok=True)
        (self.home_directory / "persona.md").mkdir(exist_ok=True)
        (self.home_directory / "GOALS.md").mkdir(exist_ok=True)
        (self.home_directory / "TASKS.md").mkdir(exist_ok=True)
        (self.home_directory / "CURRENT_TASK.md").mkdir(exist_ok=True)

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

    def _read_schedule(self) -> List[str]:
        with (self.home_directory / "SCHEDULE.md").open("r") as file:
            return [
                line.strip() for line in file.readlines() if line.strip()
            ]

    def _read_cadence(self) -> List[str]:
        with (self.home_directory / "CADENCE.md").open("r") as file:
            return [
                line.strip() for line in file.readlines() if line.strip()
            ]

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
        return f"""### Autonomouse Prompts"
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

    def _construct_model_providers(self):
        # read self.cadence and instantiate all of the model providers referenced in that file

    def _select_model_provider(self) -> ModelProvider:
        if not hasattr(self, "model_providers"):
            self._construct_model_providers()

        # use cadence.md to figure out which provider to use for this turn

    def _append_reminders(self):
        # Use schedule.md to determine if the next turn should address a given task in that file

    def _sleep(self):
        # Use schedule.md to determine how long to sleep for
