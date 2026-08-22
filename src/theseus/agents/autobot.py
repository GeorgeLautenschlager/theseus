from __future__ import annotations
from typing import List
from pathlib import Path
from time import sleep

from theseus.model_providers.lm_studio_provider import LmStudioProvider
from theseus.model_providers.ollama_provider import OllamaProvider
from theseus.model_providers.claude_provider import ClaudeProvider
from theseus.model_providers.unsloth_provider import UnslothProvider
from theseus.tools.registry import all_tools
from theseus.stimulus_log import StimulusLog
from theseus.cognitive_prompts import render_tools_section
from theseus.context_assembler import ContextAssembler

stimulus_log = StimulusLog('stimulus_log.jsonl')
context_assembler = ContextAssembler(stimulus_log=stimulus_log)
# model = LmStudioProvider(model="gemma-4-26b-a4b-it-qat")
# model = OllamaProvider(model="gemma4:e4b")
# model = ClaudeProvider(model="claude-sonnet-4-6")
# model = UnslothProvider(model="unsloth/Qwen3.8-27B-GGUF:UD-IQ3_S-1.0.0")
model = UnslothProvider(model="Gemma-4-E4B-it-QAT-NVFP4-GGUF:gemma-4-E4B-it-qat-nvfp4")
tools = all_tools()

# TODO: load these from markdown
tasks: List[str] = []
current_task_uuid: str = ""
goals: List[str] = []
telos: str = "Your purpose is to act as a steward for this Linux server. Keep it running, secure, healthy and organized."

def main() -> None:
  while True:
    events = stimulus_log.read_all()[-50:]
    lines = [event.to_json() for event in events]
    recent_events = "\n".join(lines)
    history = (
        "Your recent history, oldest first, as recorded in your stimulus log "
        "(one JSON event per line):\n\n"
        "<stimulus_log>\n"
        f"{recent_events}\n"
        "</stimulus_log>"
    )

    system_prompt = (
      "## Telos\n"
      f"{telos}\n\n"
      "---\n\n"
      "## Goals\n"
      f"{goals}\n\n"
      "## Tasks\n"
      f"{tasks}\n\n"
      "## Current Task\n"
      f"{current_task_uuid}\n\n"
      "---\n\n"
      f"{render_tools_section(tools.values())}"
      "---\n\n"
    )

    prompt = (
      f"{history}"
      "---\n\n"
      "If you don't have a goal form one or more goals (each with a UUID) in alignment with your telos. Put them in GOALS.md ranked by priority."
      "if you don't have any tasks form one or more tasks (each with a UUID) in alignment with your top prority goal. Put them in TASKS.md ranked by priority."
      "If you have tasks, but not a current task, select one and put it in CURRENT_TASK.md."
      "If you have a current task, do something to advance it. If that task is complete, remove it from TASKS.md and clear CURRENT_TASK.md."
      "Pay close attention to your recent history in <stimulus_log> as that's where you can see the results of your previous actions."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # One native tool-calling turn: the model's chosen tool(s) and their arguments
    # come back together, so there is no separate Act model call.
    print("Prompting")
    turn = model.complete_with_tools(messages, list(tools.values()))
    print("Decision: ", turn.text)
    stimulus_log.append(
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

    for call in turn.tool_calls:
      tool = tools.get(call.name)
      if tool is None:
          # Record the miss as a stimulus so the next pass can see it and recover.
          stimulus_log.append(
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
      stimulus_log.append(
          actor="autobot",
          type="tool_result",
          content={
              "tool": call.name,
              "arguments": call.arguments,
              "output": result.content,
              "is_error": result.is_error,
          },
      )

    print("Sleeping\n")
    # message = input("Press enter for the next turn")
    sleep(5)

if __name__ == "__main__":
    main()
