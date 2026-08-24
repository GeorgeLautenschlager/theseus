from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from theseus.auto_core import SLEEP_FLOOR_SECONDS, Autocore
from theseus.model_providers import PROVIDER_REGISTRY
from theseus.schedule import Schedule


class FakeUpProvider:
    def __init__(self, model):
        self.model = model

    def is_available(self):
        return True


class FakeDownProvider:
    def __init__(self, model):
        self.model = model

    def is_available(self):
        return False


def make(tmp_path, cadence_text=None, schedule_text=None):
    home = tmp_path / "home"
    if cadence_text is not None:
        home.mkdir(parents=True, exist_ok=True)
        (home / "CADENCE.md").write_text(cadence_text, encoding="utf-8")
    if schedule_text is not None:
        home.mkdir(parents=True, exist_ok=True)
        (home / "SCHEDULE.md").write_text(schedule_text, encoding="utf-8")
    core = Autocore(name="testbot", home_directory=home, tools={})
    core.schedule = Schedule(home / "SCHEDULE.md")
    return core, home


# --- _initialize_home_directory ---


def test_initialize_creates_files_not_directories(tmp_path):
    _, home = make(tmp_path)
    for name in [
        "stimulus_log.jsonl",
        "constitution.md",
        "persona.md",
        "GOALS.md",
        "TASKS.md",
        "CURRENT_TASK.md",
        "SCHEDULE.md",
        "CADENCE.md",
    ]:
        assert (home / name).is_file(), name


def test_initialize_seeds_cadence_with_default_rule(tmp_path):
    core, home = make(tmp_path)
    core._construct_model_providers()
    assert core.cadence.rule_for(datetime.now()) is not None
    assert core.cadence.lint_lines() == []


def test_initialize_does_not_clobber_existing_configs(tmp_path):
    cadence_text = "- default: ollama gemma3\n"
    schedule_text = "- [ ] daily @ 09:00: Check email\n"
    _, home = make(tmp_path, cadence_text=cadence_text, schedule_text=schedule_text)
    assert (home / "CADENCE.md").read_text(encoding="utf-8") == cadence_text
    assert (home / "SCHEDULE.md").read_text(encoding="utf-8") == schedule_text


# --- _construct_model_providers ---


def test_construct_builds_one_instance_per_unique_provider_model(tmp_path, monkeypatch):
    monkeypatch.setitem(PROVIDER_REGISTRY, "fake_up", FakeUpProvider)
    core, _ = make(
        tmp_path,
        cadence_text=(
            "- 00:00-12:00: fake_up m1\n"
            "- 12:00-00:00: fake_up m1\n"
            "- default: fake_up m2\n"
        ),
    )
    core._construct_model_providers()
    assert set(core.model_providers) == {("fake_up", "m1"), ("fake_up", "m2")}
    assert all(isinstance(p, FakeUpProvider) for p in core.model_providers.values())


def test_construct_caches_until_file_changes(tmp_path, monkeypatch):
    monkeypatch.setitem(PROVIDER_REGISTRY, "fake_up", FakeUpProvider)
    core, home = make(tmp_path, cadence_text="- default: fake_up m1\n")
    core._construct_model_providers()
    first = core.model_providers[("fake_up", "m1")]
    core._construct_model_providers()
    assert core.model_providers[("fake_up", "m1")] is first

    (home / "CADENCE.md").write_text("- default: fake_up m2\n", encoding="utf-8")
    core._construct_model_providers()
    assert set(core.model_providers) == {("fake_up", "m2")}


def test_construct_records_unknown_providers_without_crashing(tmp_path):
    core, _ = make(tmp_path, cadence_text="- default: nosuch m1\n")
    core._construct_model_providers()
    assert core.model_providers == {}
    assert core.unknown_providers and "nosuch" in core.unknown_providers[0]


# --- _select_model_provider ---


def test_select_falls_through_to_available_and_records_tick(tmp_path, monkeypatch):
    monkeypatch.setitem(PROVIDER_REGISTRY, "fake_up", FakeUpProvider)
    monkeypatch.setitem(PROVIDER_REGISTRY, "fake_down", FakeDownProvider)
    core, _ = make(
        tmp_path,
        cadence_text=(
            "- 00:00-00:00: fake_down m1, tick every 10 minutes\n"
            "- 00:00-00:00: fake_up m2, tick every 7 minutes\n"
        ),
    )
    provider = core._select_model_provider()
    assert isinstance(provider, FakeUpProvider)
    assert provider.model == "m2"
    assert core.loop_memory["tick_seconds"] == 420


def test_select_raises_when_nothing_available(tmp_path, monkeypatch):
    monkeypatch.setitem(PROVIDER_REGISTRY, "fake_down", FakeDownProvider)
    core, _ = make(tmp_path, cadence_text="- default: fake_down m1\n")
    with pytest.raises(RuntimeError):
        core._select_model_provider()


# --- _append_reminders ---


def test_append_reminders_fires_due_task(tmp_path):
    core, _ = make(
        tmp_path,
        cadence_text="- default: nosuch m1\n",
        schedule_text="- [ ] every 30 minutes: Check queue\n",
    )
    core._construct_model_providers()
    core.unknown_providers = []  # isolate: only the schedule task here
    core._append_reminders()
    events = [e for e in core.stimulus_log.read_all() if e.type == "scheduled_task"]
    assert len(events) == 1
    assert events[0].content == {"message": "Time to Check queue"}


def test_append_reminders_lints_once_until_the_bad_set_changes(tmp_path):
    core, home = make(
        tmp_path,
        cadence_text="- default: lm_studio local-model\n",
        schedule_text="- [ ] every other tuesday: water plants\n",
    )
    core._construct_model_providers()
    core._append_reminders()
    core._append_reminders()
    lints = [e for e in core.stimulus_log.read_all() if e.type == "schedule_lint"]
    assert len(lints) == 1
    assert lints[0].actor == "schedule"
    assert lints[0].content["lines"] == ["- [ ] every other tuesday: water plants"]

    (home / "SCHEDULE.md").write_text(
        "- [ ] some thursdays maybe: water plants\n", encoding="utf-8"
    )
    core._append_reminders()
    lints = [e for e in core.stimulus_log.read_all() if e.type == "schedule_lint"]
    assert len(lints) == 2


# --- _sleep ---


def sleep_calls(monkeypatch):
    calls = []
    monkeypatch.setattr("theseus.auto_core.sleep", lambda seconds: calls.append(seconds))
    return calls


def test_sleep_uses_tick_when_no_schedule(tmp_path, monkeypatch):
    calls = sleep_calls(monkeypatch)
    core, _ = make(tmp_path)
    core.loop_memory["tick_seconds"] = 900
    core._sleep()
    assert calls == [900]
    assert core.sleep_duration == 900


def test_sleep_wakes_early_for_next_due_task(tmp_path, monkeypatch):
    calls = sleep_calls(monkeypatch)
    last_fired = datetime.now(timezone.utc) - timedelta(minutes=29)
    core, _ = make(
        tmp_path,
        schedule_text=(
            f"- [ ] every 30 minutes: Check queue"
            f" <!-- last-fired: {last_fired.isoformat()} -->\n"
        ),
    )
    core.loop_memory["tick_seconds"] = 900
    core._sleep()
    assert len(calls) == 1
    assert 50 <= calls[0] <= 62
    assert core.sleep_duration == calls[0]


def test_sleep_clamps_to_floor(tmp_path, monkeypatch):
    calls = sleep_calls(monkeypatch)
    core, _ = make(tmp_path, schedule_text="- [ ] every 30 minutes: Check queue\n")
    core.loop_memory["tick_seconds"] = 900
    core._sleep()
    assert calls == [SLEEP_FLOOR_SECONDS]
