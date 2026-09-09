"""The surrogate's command execution reporter (issue #35).

Commands stay fire-and-forget on the wire; the confirmation arrives as experience.
So every delivered command produces exactly one report event — `executed`,
`partial`, `failed`, or `barged_in` — appended to the local log, which is what
makes it replicate through the ordinary drain like any own-origin event. A
command that vanishes without a report is indistinguishable from one the
surrogate never received, and the host has no way to tell the difference.

The renderer is injected: real speaking, VAD, and playback control live
elsewhere and hand back one of the `Outcome` value objects in
`command_reports`. What happened is a return value, not an exception — barge-in
and muted output are not crashes — but anything that raises on the way to a
report (the renderer, the outcome mapping, the content construction) becomes a
`failed` report rather than escaping. The only thing that may propagate past
`execute_one` is the append itself raising: the report never became durable, and
swallowing that would be the false memory this issue exists to prevent.
"""

from __future__ import annotations

from typing import Callable

import theseus.command_reports as reports
from theseus.command_reports import BargedIn, Executed, Failed, Outcome, Partial
from theseus.stimulus_log import StimulusEvent, StimulusLog
from theseus.surrogates.command_channel import CommandChannel
from theseus.surrogates.cursor import AckedCursor

Renderer = Callable[[StimulusEvent], Outcome]


class CommandExecutor:
    """Executes host commands via an injected renderer and reports each one.

    `execute_one` renders one command and appends exactly one report to the
    local log, returning the appended event. The cursor is not advanced here —
    that belongs to the driver loop (`run`), which reports first and advances
    second, so a crash in between replays rather than loses.
    """

    def __init__(
        self,
        log: StimulusLog,
        render: Renderer,
        cursor: AckedCursor,
        *,
        actor: str = "surrogate",
    ) -> None:
        self._log = log
        self._render = render
        self._cursor = cursor
        self._actor = actor

    def execute_one(self, command: StimulusEvent) -> StimulusEvent:
        """Render one command and append exactly one report to the local log.

        A violation of the command contract (seq/origin/id) raises `ValueError`
        before any append — an out-of-contract event is a seam bug, not a
        reportable command. The only thing that may propagate past here is the
        append itself raising: the report never became durable.
        """
        _check_command(command)
        try:
            report_type, content = self._report_for(self._render(command), command)
        except Exception as exc:  # a failed report, not an escape
            content = reports.failed(
                command_seq=command.seq,
                command_origin=command.origin,
                command_id=command.id,
                reason=f"{type(exc).__name__}: {exc}",
            )
            report_type = reports.FAILED
        return self._log.append(self._actor, report_type, content)

    def run(self, channel: CommandChannel) -> None:
        """Drain `channel`, reporting each command and then advancing the cursor.

        Report-then-advance is at-least-once: a crash between the two leaves the
        cursor behind, so a reconnect replays the command and produces a second
        (visible) report rather than losing it. Returns when `stream()` ends;
        shipping the reports upstream is `Replicator.drain()`'s job, unchanged.
        """
        for command in channel.stream():
            self.execute_one(command)
            self._cursor.advance(command.seq)

    def _report_for(self, outcome: Outcome, command: StimulusEvent) -> tuple[str, dict]:
        """Map a renderer outcome to (report type, report content)."""
        ref = {
            "command_seq": command.seq,
            "command_origin": command.origin,
            "command_id": command.id,
        }
        if isinstance(outcome, Executed):
            return reports.EXECUTED, reports.executed(**ref)
        if isinstance(outcome, Partial):
            return reports.PARTIAL, reports.partial(progress=outcome.progress, **ref)
        if isinstance(outcome, BargedIn):
            return reports.BARGED_IN, reports.barged_in(
                playback_position=outcome.playback_position, **ref
            )
        if isinstance(outcome, Failed):
            return reports.FAILED, reports.failed(reason=outcome.reason, **ref)
        raise TypeError(f"renderer returned {outcome!r}; expected an Outcome")


def _check_command(command: StimulusEvent) -> None:
    """The command's contract, checked before anything is appended."""
    if isinstance(command.seq, bool) or not isinstance(command.seq, int):
        raise ValueError(f"command seq must be an integer (got {command.seq!r})")
    if command.seq < 1:
        raise ValueError(f"command seq must be 1 or greater (got {command.seq!r})")
    if not command.origin:
        raise ValueError("command origin must be non-empty")
    if not command.id:
        raise ValueError("command id must be non-empty")
