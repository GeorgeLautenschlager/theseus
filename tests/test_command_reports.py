"""Claims about the command execution report vocabulary (issue #35, Task 1).

The four outcome type strings are wire protocol between a surrogate and a host that may
be built from different releases, so they are pinned literally. The constructors validate
write-side; the predicates never raise — they run over logs that may hold anything."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone

import pytest

from theseus.command_reports import (
    BARGED_IN,
    EXECUTED,
    FAILED,
    PARTIAL,
    REPORT_PREFIX,
    OUTCOMES,
    BargedIn,
    Executed,
    Failed,
    Partial,
    barged_in,
    executed,
    failed,
    is_report,
    partial,
    report_outcome,
)
from theseus.commands import command_content, command_type, is_command
from theseus.replication_events import MAX_REASON_CHARS
from theseus.stimulus_log import StimulusEvent


def _event(type_: str, content) -> StimulusEvent:
    return StimulusEvent(
        id="e1",
        ts=datetime.now(timezone.utc),
        actor="surrogate",
        type=type_,
        content=content,
    )


def _ref(**over):
    base = {"command_seq": 7, "command_origin": "host-a", "command_id": "cmd-1"}
    base.update(over)
    return base


# --- Item 1: the outcome type strings are wire protocol --------------------------

def test_outcome_type_strings_are_pinned_wire_protocol():
    assert EXECUTED == "command_report.executed"
    assert PARTIAL == "command_report.partial"
    assert FAILED == "command_report.failed"
    assert BARGED_IN == "command_report.barged_in"
    assert OUTCOMES == (EXECUTED, PARTIAL, FAILED, BARGED_IN)


# --- Items 2–5: the constructors validate write-side ------------------------------

@pytest.mark.parametrize(
    "build",
    [
        lambda **kw: executed(**kw),
        lambda **kw: partial(progress="3 of 5", **kw),
        lambda **kw: failed(reason="muted", **kw),
        lambda **kw: barged_in(playback_position=1200, **kw),
    ],
)
def test_shared_reference_fields_are_validated_by_every_constructor(build):
    # `bool` is an `int` in Python and would serialise as `true` onto a numeric wire
    # field — so it is refused alongside every other non-int.
    for bad_seq in (True, 0, -1, "7", None):
        with pytest.raises(ValueError, match="command_seq"):
            build(**_ref(command_seq=bad_seq))
    with pytest.raises(ValueError, match="command_origin"):
        build(**_ref(command_origin=""))
    with pytest.raises(ValueError, match="command_id"):
        build(**_ref(command_id=""))


def test_executed_carries_only_the_shared_reference_fields():
    content = executed(**_ref())
    assert content == {"command_seq": 7, "command_origin": "host-a", "command_id": "cmd-1"}


def test_partial_requires_a_non_empty_progress_string():
    content = partial(progress="3 of 5", **_ref())
    assert content["progress"] == "3 of 5"
    for bad in ("", None, 42):
        with pytest.raises(ValueError, match="progress"):
            partial(progress=bad, **_ref())


def test_failed_binds_reason_through_clean_reason():
    content = failed(reason="  muted by the observer  ", **_ref())
    assert content["reason"] == "muted by the observer"
    for bad in ("", "   ", None, 42):
        with pytest.raises(ValueError, match="reason"):
            failed(reason=bad, **_ref())


def test_failed_truncates_an_over_long_reason_and_marks_the_cut():
    content = failed(reason="x" * (MAX_REASON_CHARS + 50), **_ref())
    assert len(content["reason"]) == MAX_REASON_CHARS
    assert content["reason"].endswith("…")


def test_barged_in_accepts_any_usable_playback_marker():
    # Freedom for the reflex layer: a millisecond offset, a wall-clock time, a token index.
    for position in (1200, 3.5, "00:03:12"):
        content = barged_in(playback_position=position, **_ref())
        assert content["playback_position"] == position
    # An empty string, a negative number, None or a bool is not a marker.
    for bad in ("", -1, -0.5, None, True, False):
        with pytest.raises(ValueError, match="playback_position"):
            barged_in(playback_position=bad, **_ref())


# --- Item 6: content is JSON-native so the log can encode it ----------------------

@pytest.mark.parametrize(
    "content",
    [
        executed(**_ref()),
        partial(progress="3 of 5", **_ref()),
        failed(reason="muted", **_ref()),
        barged_in(playback_position=1200, **_ref()),
    ],
)
def test_content_is_json_native_so_the_log_can_encode_it(content):
    event = _event(FAILED, content)
    assert json.loads(event.to_json())["content"] == content


# --- Items 7–8: the read-side predicates never raise ------------------------------

@pytest.mark.parametrize("type_", OUTCOMES)
def test_is_report_recognizes_each_outcome(type_):
    assert is_report(_event(type_, {}))


def test_bare_prefix_names_the_namespace_not_a_report():
    assert not is_report(_event(REPORT_PREFIX, {"command_seq": 1}))


def test_non_report_events_are_not_reports():
    # Carries a real "command_seq" value: the assertion must fail if is_report stops
    # gating on the type prefix, not merely because the key happens to be absent.
    event = _event("observation", {"command_seq": 1})
    assert not is_report(event)
    assert report_outcome(event) is None


@pytest.mark.parametrize("type_", OUTCOMES)
def test_report_outcome_returns_the_event_type(type_):
    assert report_outcome(_event(type_, {})) == type_


def test_report_outcome_never_raises_on_malformed_content():
    # A log line whose content is a list reaches the predicate by prefix alone; a
    # filtering reader must not be taken down by it.
    assert report_outcome(_event(FAILED, ["not", "a", "dict"])) is None
    # Missing fields is fine too: the predicate answers about the type, not the payload.
    assert report_outcome(_event(PARTIAL, {})) == PARTIAL


# --- Item 9: reports and commands live in separate namespaces ----------------------

def test_a_command_is_not_a_report_and_vice_versa():
    # The exact collision the separate prefix exists to prevent, driven through the real
    # predicate pair from both modules.
    command = _event(command_type("say"), command_content(target="tam", payload={}))
    assert not is_report(command)
    assert report_outcome(command) is None
    report = _event(FAILED, failed(reason="muted", **_ref()))
    assert not is_command(report)


# --- The Outcome value objects -----------------------------------------------------

def test_outcomes_are_frozen_value_objects():
    assert Executed() == Executed()
    assert Partial(progress="3 of 5") == Partial(progress="3 of 5")
    assert BargedIn(playback_position=1200) == BargedIn(playback_position=1200)
    assert Failed(reason="muted") == Failed(reason="muted")
    outcome = Partial(progress="3 of 5")
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.progress = "4 of 5"
