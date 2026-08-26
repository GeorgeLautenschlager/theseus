from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from time import monotonic

import pytest

from theseus.auto_core import (
    CONFIG_GRAMMAR_HELP,
    SLEEP_FLOOR_SECONDS,
    Autocore,
)
from theseus.context_assembler import (
    DEFAULT_RESERVED_OUTPUT_TOKENS,
    DEFAULT_TOKEN_BUDGET,
)
from theseus.model_providers import PROVIDER_REGISTRY
from theseus.schedule import Schedule
from theseus.tools.tool import AssistantTurn


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


# --- _next_tick_seconds ---


def test_tick_uses_cadence_when_no_schedule(tmp_path):
    core, _ = make(tmp_path)
    core.loop_memory["tick_seconds"] = 900
    assert core._next_tick_seconds() == 900


def test_tick_shortens_for_next_due_task(tmp_path):
    last_fired = datetime.now(timezone.utc) - timedelta(minutes=29)
    core, _ = make(
        tmp_path,
        schedule_text=(
            f"- [ ] every 30 minutes: Check queue"
            f" <!-- last-fired: {last_fired.isoformat()} -->\n"
        ),
    )
    core.loop_memory["tick_seconds"] = 900
    assert 50 <= core._next_tick_seconds() <= 62


def test_tick_clamps_to_floor(tmp_path):
    core, _ = make(tmp_path, schedule_text="- [ ] every 30 minutes: Check queue\n")
    core.loop_memory["tick_seconds"] = 900
    assert core._next_tick_seconds() == SLEEP_FLOOR_SECONDS


# --- _sleep ---


def test_sleep_waits_out_the_tick_when_nothing_arrives(tmp_path, monkeypatch):
    monkeypatch.setattr("theseus.auto_core.SLEEP_FLOOR_SECONDS", 0.0)
    core, _ = make(tmp_path)
    core.loop_memory["tick_seconds"] = 0.05
    started = monotonic()
    assert core._sleep() is False
    assert monotonic() - started >= 0.05
    assert core.sleep_duration == 0.05


def test_sleep_returns_at_once_for_a_wake_raised_during_the_turn(tmp_path):
    """A message that lands while the model is still thinking must not then be slept
    on — the flag survives the turn and the next sleep is a no-op."""
    core, _ = make(tmp_path)
    core.loop_memory["tick_seconds"] = 3600
    core.wake("landed mid-turn")
    started = monotonic()
    assert core._sleep() is True
    assert monotonic() - started < 1


def test_sleep_is_cut_short_by_a_message_from_another_thread(tmp_path):
    core, _ = make(tmp_path)
    core.loop_memory["tick_seconds"] = 3600
    core._consume_wake()
    threading.Timer(
        0.05,
        lambda: core.stimulus_log.append(
            actor="user", type="chat_message", content={"message": "you awake?"}
        ),
    ).start()
    started = monotonic()
    assert core._sleep() is True
    assert monotonic() - started < 30


def test_sleep_is_not_cut_short_by_the_cores_own_events(tmp_path, monkeypatch):
    """Every turn logs a decision and its tool results. If those woke the loop it
    would never sleep again."""
    monkeypatch.setattr("theseus.auto_core.SLEEP_FLOOR_SECONDS", 0.0)
    core, _ = make(tmp_path)
    core.loop_memory["tick_seconds"] = 0.05
    core.stimulus_log.append(actor=core.name, type="decision", content={"text": "hm"})
    core.stimulus_log.append(actor=core.name, type="tool_result", content={"tool": "ls"})
    assert core._sleep() is False


def test_wake_on_predicate_narrows_what_interrupts(tmp_path, monkeypatch):
    monkeypatch.setattr("theseus.auto_core.SLEEP_FLOOR_SECONDS", 0.0)
    home = tmp_path / "home"
    core = Autocore(
        name="testbot",
        home_directory=home,
        tools={},
        wake_on=lambda event: event.type == "chat_message",
    )
    core.schedule = Schedule(home / "SCHEDULE.md")
    core.loop_memory["tick_seconds"] = 0.05

    core.stimulus_log.append(actor="schedule", type="scheduled_task", content={})
    assert core._sleep() is False

    core.stimulus_log.append(actor="user", type="chat_message", content={"message": "?"})
    assert core._sleep() is True


# --- wake bookkeeping ---


def test_consume_wake_reports_the_first_trigger_then_disarms(tmp_path):
    core, _ = make(tmp_path)
    first = core.stimulus_log.append(
        actor="user", type="chat_message", content={"message": "one"}
    )
    core.stimulus_log.append(actor="user", type="chat_message", content={"message": "two"})

    assert core._consume_wake() is first
    assert core._consume_wake() is None
    assert core._wake.is_set() is False


def test_wake_notice_names_what_interrupted_the_sleep(tmp_path):
    core, _ = make(tmp_path)
    assert core._wake_notice() == ""

    core.stimulus_log.append(actor="george", type="chat_message", content={"message": "hi"})
    core.loop_memory["wake_trigger"] = core._consume_wake()
    core.loop_memory["slept_seconds"] = 12.4
    core.loop_memory["woke_early"] = True
    notice = core._wake_notice()
    assert "chat_message" in notice and "george" in notice
    assert "12 seconds" in notice
    assert notice in core._automated_prompt_instructions()


def test_wake_notice_does_not_claim_a_sleep_was_cut_short_when_it_was_not(tmp_path):
    """A wake landing in the gap between the tick elapsing and the turn starting still
    names its trigger, but nothing was interrupted and the prompt shouldn't say so."""
    core, _ = make(tmp_path)
    core.stimulus_log.append(actor="george", type="chat_message", content={"message": "hi"})
    core.loop_memory["wake_trigger"] = core._consume_wake()
    core.loop_memory["slept_seconds"] = 300.0
    core.loop_memory["woke_early"] = False

    notice = core._wake_notice()
    assert "cut short" not in notice
    assert "chat_message" in notice and "george" in notice


# --- loop ---
#
# The unit tests above prove `_sleep` returns early. These run a real `loop()` on its
# own thread and prove the loop actually comes back round — the thing that was broken.


class TurnRecorder:
    """Shared between the loop thread and the test: every turn the fake provider takes
    lands here and trips `took_a_turn`, so tests wait on an event instead of guessing at
    wall-clock sleeps."""

    def __init__(self):
        self.prompts: list[str] = []
        self.took_a_turn = threading.Event()
        self._lock = threading.Lock()

    def record(self, prompt: str) -> None:
        with self._lock:
            self.prompts.append(prompt)
        self.took_a_turn.set()

    def wait_for_turn(self, timeout: float = 5.0) -> bool:
        took = self.took_a_turn.wait(timeout)
        self.took_a_turn.clear()
        return took

    def count(self) -> int:
        with self._lock:
            return len(self.prompts)


def recording_provider(recorder: TurnRecorder):
    """A PROVIDER_REGISTRY entry whose turns are instant and land in `recorder`."""

    class _Provider:
        def __init__(self, model):
            self.model = model

        def is_available(self):
            return True

        def complete_with_tools(self, messages, tools):
            recorder.record(messages[1]["content"])
            return AssistantTurn(text="thinking...", tool_calls=[], prompt_tokens=10)

    return _Provider


def start_loop(tmp_path, monkeypatch, tick="30 minutes") -> tuple[Autocore, TurnRecorder]:
    recorder = TurnRecorder()
    monkeypatch.setitem(PROVIDER_REGISTRY, "fake_loop", recording_provider(recorder))
    core, _ = make(
        tmp_path, cadence_text=f"- default: fake_loop m1, tick every {tick}\n"
    )
    threading.Thread(target=core.loop, daemon=True).start()
    return core, recorder


def test_loop_wakes_from_a_long_sleep_when_a_message_lands(tmp_path, monkeypatch):
    """The whole point. Half an hour into a thirty-minute tick, a message from George
    must start the next turn now rather than in twenty more minutes."""
    core, recorder = start_loop(tmp_path, monkeypatch)
    assert recorder.wait_for_turn(), "the loop never took its first turn"

    core.stimulus_log.append(
        actor="george", type="chat_message", content={"message": "you there?"}
    )

    assert recorder.wait_for_turn(), "the loop slept through a message from George"
    assert recorder.count() == 2


def test_the_woken_turn_sees_the_message_and_is_told_why_it_woke(tmp_path, monkeypatch):
    core, recorder = start_loop(tmp_path, monkeypatch)
    assert recorder.wait_for_turn()

    core.stimulus_log.append(
        actor="george", type="chat_message", content={"message": "you there?"}
    )
    assert recorder.wait_for_turn()

    prompt = recorder.prompts[-1]
    assert "you there?" in prompt, "the waking message never reached the context"
    assert "george" in prompt and "chat_message" in prompt, (
        "the prompt does not tell the agent what pulled it out of its sleep"
    )


def test_loop_settles_back_to_sleep_rather_than_spinning_on_its_own_events(
    tmp_path, monkeypatch
):
    """Each turn appends its own `decision` and `tool_result` events. If those counted
    as wakes the loop would never sleep again — it would run flat out against the
    model."""
    _, recorder = start_loop(tmp_path, monkeypatch)
    assert recorder.wait_for_turn()

    assert recorder.wait_for_turn(timeout=1.0) is False
    assert recorder.count() == 1


class TestContextBudgetFollowsTheModel:
    """Cadence changes models by time of day, so a budget fixed once at startup is a
    budget that is wrong for most of the day. The window must be sized against whichever
    model this turn's rule actually selected."""

    def test_selection_hands_the_rules_window_to_the_assembler(self, tmp_path, monkeypatch):
        monkeypatch.setitem(PROVIDER_REGISTRY, "unsloth", FakeUpProvider)
        core, _ = make(
            tmp_path,
            cadence_text="- default: unsloth qwen3-27b, context 128k, tick every 5 minutes\n",
        )

        core._select_model_provider()

        assert core.context_assembler.context_tokens == 131072
        assert core.loop_memory["context_tokens"] == 131072

    def test_falling_back_to_a_smaller_model_shrinks_the_window(self, tmp_path, monkeypatch):
        """The availability fallback and the budget must move together — otherwise a
        dropped provider silently keeps the previous model's much larger budget."""
        monkeypatch.setitem(PROVIDER_REGISTRY, "unsloth", FakeDownProvider)
        monkeypatch.setitem(PROVIDER_REGISTRY, "ollama", FakeUpProvider)
        core, _ = make(
            tmp_path,
            cadence_text=(
                "- 00:00-00:00: unsloth qwen3-27b, context 128k\n"
                "- default: ollama gemma3:12b, context 8k\n"
            ),
        )

        provider = core._select_model_provider()

        assert isinstance(provider, FakeUpProvider)
        assert core.context_assembler.context_tokens == 8192

    def test_rule_without_context_leaves_the_conservative_default(self, tmp_path, monkeypatch):
        monkeypatch.setitem(PROVIDER_REGISTRY, "ollama", FakeUpProvider)
        core, _ = make(tmp_path, cadence_text="- default: ollama gemma3:12b\n")

        core._select_model_provider()

        assert core.context_assembler.context_tokens is None
        assert core.context_assembler.token_budget == DEFAULT_TOKEN_BUDGET

    def test_grammar_help_documents_the_context_fragment(self):
        """The lint stimulus is how the agent learns to rewrite its own config; if it
        omitted `context`, Tam would 'repair' valid lines into unbudgeted ones."""
        assert "context N[k]" in CONFIG_GRAMMAR_HELP

    def test_instructions_are_not_rendered_twice_per_turn(self, tmp_path):
        """They used to go into both the system prompt and the user prompt — the same
        several hundred tokens, paid on every single turn."""
        core, _ = make(tmp_path)
        core.goals, core.tasks, core.current_task = [], [], []

        system_prompt = core._assemble_system_prompt()
        goals_and_tasks = core._current_goals_and_tasks()

        assert "### Self Directed Action" in system_prompt
        assert "### Self Directed Action" not in goals_and_tasks


class _CapturingProvider:
    """Records the prompt of the first turn, then breaks the loop."""

    class Done(Exception):
        pass

    captured: list = []

    def __init__(self, model):
        self.model = model

    def is_available(self):
        return True

    def complete_with_tools(self, messages, tools=None, **kwargs):
        _CapturingProvider.captured.append(messages)
        raise _CapturingProvider.Done()


def test_one_real_turn_fits_the_declared_window(tmp_path, monkeypatch):
    """The whole point, end to end: a turn against a 32k-declared model must produce a
    prompt that fits 32k, with room left for the reply.

    The regression it guards is not arithmetic in isolation — it is the *ordering*. The
    budget is a property of the selected model, so selection has to happen before
    assembly; when it did not, the window was fitted against whatever budget happened to
    be left over from construction.
    """
    _CapturingProvider.captured = []
    monkeypatch.setitem(PROVIDER_REGISTRY, "unsloth", _CapturingProvider)
    core, home = make(
        tmp_path, cadence_text="- default: unsloth qwen3-27b, context 32k\n"
    )
    (home / "constitution.md").write_text("You are a test agent. " * 200)
    core.constitution = (home / "constitution.md").read_text()
    core.context_assembler.window_size = 1000
    for i in range(400):
        core.stimulus_log.append(
            actor="george", type="exchange", content={"message": f"msg {i} " + "x" * 600}
        )

    with pytest.raises(_CapturingProvider.Done):
        core.loop()

    messages = _CapturingProvider.captured[0]
    prompt_chars = sum(len(m["content"]) for m in messages)
    prompt_tokens = prompt_chars / core.context_assembler.chars_per_token

    assert prompt_tokens + DEFAULT_RESERVED_OUTPUT_TOKENS < 32_768
    # ...and it did not solve the problem by sending nothing.
    assert "msg 399" in messages[1]["content"]
