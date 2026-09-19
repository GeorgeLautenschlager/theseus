"""The reference surrogate HOST: accepts replicated stimuli, streams commands back.

This is the door the surrogate talks to — the other half of the Phase 1
surrogate↔host loop (issue #94/#102). It mounts the replication ingress
(`POST /replicate`) and the command feed (`GET /commands/{target}`) on the SAME
FastAPI app that serves the chat UI, so the surrogate and the person share one
port. `build_surrogate_host` wires the HTTP surface without touching any model;
`main` is the thin live entry point that adds the cognitive core.
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from theseus.auto_core import Autocore
from theseus.command_feed import CommandFeed
from theseus.high_water import HighWaterMarks
from theseus.replication_ingress import ReplicationIngress
from theseus.stimulus_log import StimulusLog
from theseus.tools.registry import all_tools
from theseus.tools.surrogate_voice import NotifySurrogate, SaySurrogate
from theseus.tools.web_chat import WebChat
from theseus.web_chat_ui_observer import WebChatUIObserver


@dataclass(frozen=True)
class SurrogateHost:
    """Everything the host mounted, so callers can wire a core and tests can assert."""

    observer: WebChatUIObserver
    ingress: ReplicationIngress
    command_feed: CommandFeed
    say_tool: SaySurrogate
    notify_tool: NotifySurrogate
    target: str


def build_surrogate_host(
    log: StimulusLog,
    marks: HighWaterMarks,
    orient_callback: Callable[[], None],
    *,
    target: str = "windows-desktop",
) -> SurrogateHost:
    """Wire the host's HTTP surface. No cognitive core, no model — offline-testable.

    `orient_callback` is invoked with no arguments by both the ingress (after a burst)
    and the chat observer (`orient_chat_message_callback`, `Callable[[], None]`), so one
    zero-arg callback — e.g. `Autocore.wake`, which tolerates being called bare — serves both.
    """
    observer = WebChatUIObserver(
        stimulus_log=log, orient_chat_message_callback=orient_callback
    )
    ingress = ReplicationIngress(log, marks, on_arrival=orient_callback)
    ingress.add_routes(observer.app)
    command_feed = CommandFeed(log)
    command_feed.add_routes(observer.app)
    return SurrogateHost(
        observer=observer,
        ingress=ingress,
        command_feed=command_feed,
        say_tool=SaySurrogate(target, log),
        notify_tool=NotifySurrogate(target, log),
        target=target,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reference surrogate host")
    parser.add_argument("--data-dir", type=Path, required=True, help="Home directory (state, log, model config)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--target", default="windows-desktop", help="Surrogate name commands are addressed to")
    args = parser.parse_args()

    home = args.data_dir
    home.mkdir(parents=True, exist_ok=True)
    log = StimulusLog(home / "stimulus_log.jsonl")
    marks = HighWaterMarks(log)

    tools = dict(all_tools(cwd=home))
    core = Autocore(
        name="surrogate-host", home_directory=home, tools=tools, stimulus_log=log
    )
    host = build_surrogate_host(log, marks, core.wake, target=args.target)
    core.tools[host.say_tool.name] = host.say_tool
    core.tools[host.notify_tool.name] = host.notify_tool
    mouth = WebChat(web_observer=host.observer)
    core.tools[mouth.name] = mouth
    host.ingress.start()
    threading.Thread(target=core.loop, name="agent-core", daemon=True).start()
    host.observer.serve(args.host, args.port)


if __name__ == "__main__":
    main()
