from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog


class TextualUI:
    def __init__(self) -> None:
        self._log: RichLog | None = None

    def start(self, agent) -> None:
        AgentApp(agent, self).run()

    def set_log(self, log: RichLog) -> None:
        self._log = log

    def render(self, content: str) -> None:
        if self._log is not None:
            self._log.write(f"[bold]agent:[/bold] {content}")


class AgentApp(App):
    def __init__(self, agent, ui: TextualUI) -> None:
        super().__init__()
        self._agent = agent
        self._ui = ui

    def compose(self) -> ComposeResult:
        yield RichLog(highlight=True, markup=True)
        yield Input(placeholder="words go here")

    def on_mount(self) -> None:
        self._ui.set_log(self.query_one(RichLog))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        log = self.query_one(RichLog)
        log.write(f"[bold]you:[/bold] {event.value}")
        self._agent.process(event.value)
        event.input.clear()
