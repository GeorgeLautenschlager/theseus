"""What a command execution report is (issue #35): the outcome vocabulary, the value
objects an injected renderer returns, and the content constructors.

Commands stay fire-and-forget on the wire; confirmation arrives as experience. That
asymmetry is deliberate — there's no ack on the command channel, because the honest
report of what happened is itself an observation, and observations already have a path
home. So every delivered command produces exactly one report event: `executed`,
`partial`, `failed`, or `barged_in`. The reports replicate upstream through the ordinary
path — no special casing, no side channel; they are just more events on the tape.

The reports live under their own prefix, not `command.`: `is_command` recognises anything
under `command.` with a verb, so a report typed `command.report` would read back as a
command and be re-served to the surrogate. The separate namespace keeps reports off that
gate on both logs — a command is never a report, and a report is never a command.

A report references its command by cross-node identity `(origin, seq)`, never `id` (the
id changes when an event lands on another log). The referenced seq/origin are the host's,
since a command is a host-origin event on the host's own log; `command_id` rides along for
human tracing only.

**Validation is write-side only**, as in `commands` and `replication_events`: the
constructors refuse to build a malformed report at the mistake, raising `ValueError`. The
predicates are the read side and never raise — a report comes off a log that may hold
anything, and a feed that throws while filtering is a feed that stops serving.
`report_outcome` answers `None` instead.

The renderer itself — real speaking, VAD, playback control — is injected by the caller
and returns one of the `Outcome` value objects below; this module only names its shape.
Nothing here emits, appends or transports anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from theseus.replication_events import clean_reason
from theseus.stimulus_log import StimulusEvent

REPORT_PREFIX = "command_report."

# One event type per outcome. These strings are wire protocol — a surrogate and a host
# built from different releases still have to agree on them, so they are pinned by a test.
EXECUTED = "command_report.executed"
PARTIAL = "command_report.partial"
FAILED = "command_report.failed"
BARGED_IN = "command_report.barged_in"

OUTCOMES = (EXECUTED, PARTIAL, FAILED, BARGED_IN)


# --- Outcome value objects --------------------------------------------------------
# What an injected renderer returns. The payloads are validated in the content
# constructors below, not here: a renderer may hand back whatever it observed, and the
# constructor is where that observation becomes a permanent log line.


@dataclass(frozen=True, slots=True)
class Executed:
    """The command ran to completion. Nothing else to report."""


@dataclass(frozen=True, slots=True)
class Partial:
    """The command got partway and stopped; `progress` says how far, in the surrogate's
    own words."""

    progress: str


@dataclass(frozen=True, slots=True)
class BargedIn:
    """A user interrupted playback mid-command. Not a failure — the report just says
    where playback was when it happened, in whatever marker the reflex layer keeps."""

    playback_position: str | int | float


@dataclass(frozen=True, slots=True)
class Failed:
    """The command could not be executed; `reason` is why (bounded on the content side)."""

    reason: str


Outcome = Executed | Partial | BargedIn | Failed


# --- Content constructors ----------------------------------------------------------


def executed(*, command_seq: int, command_origin: str, command_id: str) -> dict[str, Any]:
    """Content for a `command_report.executed`."""
    _check_reference(command_seq, command_origin, command_id)
    return {
        "command_seq": command_seq,
        "command_origin": command_origin,
        "command_id": command_id,
    }


def partial(
    *, command_seq: int, command_origin: str, command_id: str, progress: str
) -> dict[str, Any]:
    """Content for a `command_report.partial`."""
    _check_reference(command_seq, command_origin, command_id)
    if not isinstance(progress, str) or not progress:
        raise ValueError(f"progress must be a non-empty string (got {progress!r})")
    return {
        "command_seq": command_seq,
        "command_origin": command_origin,
        "command_id": command_id,
        "progress": progress,
    }


def failed(
    *, command_seq: int, command_origin: str, command_id: str, reason: str
) -> dict[str, Any]:
    """Content for a `command_report.failed`.

    `reason` goes through `clean_reason`: stripped, non-empty, and bounded per
    `MAX_REASON_CHARS` — a renderer exception can stringify to an arbitrarily long
    traceback going onto a permanent tape.
    """
    _check_reference(command_seq, command_origin, command_id)
    return {
        "command_seq": command_seq,
        "command_origin": command_origin,
        "command_id": command_id,
        "reason": clean_reason(reason),
    }


def barged_in(
    *,
    command_seq: int,
    command_origin: str,
    command_id: str,
    playback_position: str | int | float,
) -> dict[str, Any]:
    """Content for a `command_report.barged_in`.

    `playback_position` is whatever marker the reflex layer keeps — a millisecond offset,
    a wall-clock time, a token index. A real number must be ≥ 0 and a string non-empty:
    freedom of form, not of meaning.
    """
    _check_reference(command_seq, command_origin, command_id)
    _check_playback_position(playback_position)
    return {
        "command_seq": command_seq,
        "command_origin": command_origin,
        "command_id": command_id,
        "playback_position": playback_position,
    }


def _check_reference(command_seq: int, command_origin: str, command_id: str) -> None:
    """The shared reference fields, validated once for every constructor.

    The report points at its command by cross-node identity `(origin, seq)` — the host's
    own, since a command is a host-origin event on the host's log. A wrong reference is a
    report that cannot be matched to anything, which is worse than no report.
    """
    if isinstance(command_seq, bool) or not isinstance(command_seq, int):
        raise ValueError(f"command_seq must be an integer (got {command_seq!r})")
    if command_seq < 1:
        raise ValueError(f"command_seq must be 1 or greater (got {command_seq!r})")
    _check_name("command_origin", command_origin)
    _check_name("command_id", command_id)


def _check_name(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string (got {value!r})")


def _check_playback_position(playback_position: str | int | float) -> None:
    if isinstance(playback_position, bool) or not isinstance(
        playback_position, (str, int, float)
    ):
        raise ValueError(
            f"playback_position must be a non-empty string or a number >= 0 "
            f"(got {playback_position!r})"
        )
    if isinstance(playback_position, str):
        if not playback_position:
            raise ValueError("playback_position must be a non-empty string (got '')")
    elif not (playback_position >= 0):
        # `not (x >= 0)` rather than `x < 0`: NaN is a float that is neither, and a NaN
        # marker is not a position.
        raise ValueError(f"playback_position must be >= 0 (got {playback_position!r})")


# --- Read-side predicates ----------------------------------------------------------


def is_report(event: StimulusEvent) -> bool:
    """True for an event whose type is namespaced under `REPORT_PREFIX` *with an outcome
    after it*. The bare prefix is not a report: it names the namespace, not anything in
    it — mirroring `is_command`."""
    return event.type.startswith(REPORT_PREFIX) and len(event.type) > len(REPORT_PREFIX)


def report_outcome(event: StimulusEvent) -> str | None:
    """The event's type when it is a report, else `None`.

    Returns rather than raises because these events come off a log that may hold anything:
    a report line whose content is a list or is missing fields is somebody else's bug, not
    a reason for a filtering feed to stop serving. The predicate answers about the type,
    never the payload.
    """
    if not is_report(event):
        return None
    # Non-dict content happens: `from_json` does not check the shape, so a log line like
    # this reaches here by type prefix alone (see the module docstring).
    if not isinstance(event.content, dict):
        return None
    return event.type
