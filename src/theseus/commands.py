"""What a command is (issue #34, Task 1): the type namespace, content shape, and
predicates. Transport — how a command reaches its surrogate — lives elsewhere; nothing
here does I/O or imports a web framework.

A command is a `StimulusEvent` whose `type` is namespaced under `COMMAND_PREFIX`, with the
verb in the type and the arguments in the content's `payload`. The event is on the host's
own log like any other stimulus, which is why this module is pure: the same commands a
transport streams are ordinary log entries the host can inspect, and the surrogate
executes on arrival without evaluating them.

This module defines the shape, not the vocabulary — no verbs are named here.

**Validation is write-side only**, as in `replication_events`: the constructors refuse to
build a malformed command at the mistake, raising `ValueError`. The predicates are the
read side and never raise — a command comes off a log that may hold anything, and a feed
that throws while filtering is a feed that stops serving. `command_target` answers `None`
instead. An event arriving over the wire has been through none of these constructors;
the transport that receives one must apply these same rules rather than inventing a
second definition of a well-formed command.
"""

from __future__ import annotations

from typing import Any

from theseus.stimulus_log import StimulusEvent

COMMAND_PREFIX = "command."


def command_type(verb: str) -> str:
    """`"say"` -> `"command.say"`.

    An empty verb would produce the bare prefix, which `is_command` deliberately does not
    recognise — refusing it here keeps that rule from being reachable by accident. An
    already-prefixed verb is a caller trying to be helpful with `command_type(command_type(v))`;
    double-prefixing is a bug, not a convention to paper over. Whitespace is refused because
    a verb is a vocabulary token, not prose, and `"say something".replace(" ", "")` would
    silently fuse two verbs into one.
    """
    if not isinstance(verb, str) or not verb:
        raise ValueError(f"verb must be a non-empty string (got {verb!r})")
    if verb.startswith(COMMAND_PREFIX):
        raise ValueError(
            f"verb {verb!r} already carries the {COMMAND_PREFIX!r} prefix — pass the bare verb"
        )
    if any(c.isspace() for c in verb):
        raise ValueError(f"verb must not contain whitespace (got {verb!r})")
    return COMMAND_PREFIX + verb


def command_content(*, target: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Content for a command addressed to one surrogate.

    An unaddressed command is ambiguous in the worst way: either every surrogate executes
    it or none does, and both readings are wrong for a channel whose contract is one
    host, one surrogate. So an empty `target` raises. An empty `payload` is fine — a verb
    with no arguments is legitimate.
    """
    if not isinstance(target, str) or not target:
        raise ValueError(f"target must be a non-empty surrogate name (got {target!r})")
    return {"target": target, "payload": payload}


def is_command(event: StimulusEvent) -> bool:
    """True for an event whose type is namespaced under `COMMAND_PREFIX` *with a verb
    after it*. The bare prefix is not a command: it names the namespace, not anything in
    it, and treating it as one would hand the executor an unnamed verb.
    """
    return event.type.startswith(COMMAND_PREFIX) and len(event.type) > len(COMMAND_PREFIX)


def command_target(event: StimulusEvent) -> str | None:
    """The surrogate a command is for, or `None` for anything malformed.

    Returns rather than raises because these events come off a log that may hold anything:
    a well-typed command with a missing or non-string `target` is somebody else's bug, not
    a reason for a filtering feed to stop serving.
    """
    target = event.content.get("target")
    if not isinstance(target, str):
        return None
    return target
