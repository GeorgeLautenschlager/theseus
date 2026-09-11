from __future__ import annotations

from dataclasses import replace

import pytest

from theseus.assembly import AgentSpec, InterfaceSpec, ModelSpec, assemble, build_agent, render
from theseus.durable_delivery import (
    DeliveryJournal, DeliveryOutcome, DurableInbox, DurableOutbox,
)
from theseus.stimulus_log import StimulusLog
from theseus.telegram_api import TelegramAPIError, TelegramSender
from theseus.telegram_observer import TELEGRAM_TRANSPORT, TelegramObserver
from theseus.tools.telegram import TelegramTool, _utf16_units, split_telegram_message


def update(update_id=42, *, user_id=10, chat_id=20):
    return {
        "update_id": update_id,
        "message": {
            "message_id": 300,
            "date": 1,
            "from": {"id": user_id, "first_name": "George", "username": "george"},
            "chat": {"id": chat_id, "type": "private"},
            "caption": "look",
            "photo": [
                {"file_id": "small", "file_unique_id": "same", "width": 90, "height": 90},
                {"file_id": "large", "file_unique_id": "same", "width": 900, "height": 900},
            ],
            "document": {
                "file_id": "document-id", "file_unique_id": "doc-unique",
                "file_name": "notes.txt", "mime_type": "text/plain",
            },
            "reply_to_message": {
                "message_id": 299,
                "from": {"id": 11},
                "text": "earlier",
            },
        },
    }


class PollAPI:
    def __init__(self, batches):
        self.batches = list(batches)
        self.calls = []
        self.before_call = None

    def get_updates(self, *, offset, timeout):
        self.calls.append((offset, timeout))
        if self.before_call:
            self.before_call(len(self.calls), offset)
        return self.batches.pop(0)


class ReceiptSender:
    def __init__(self, outcomes=None):
        self.items = []
        self.outcomes = list(outcomes or [])

    def send(self, item):
        self.items.append(item)
        if self.outcomes:
            return self.outcomes.pop(0)
        return DeliveryOutcome.delivered(str(1000 + len(self.items)))


def observer(tmp_path, api, callback=lambda: None, *, users=(10,), chats=(20,)):
    journal = DeliveryJournal(tmp_path / "delivery.sqlite3")
    return TelegramObserver(
        StimulusLog(tmp_path / "stimulus.jsonl"), callback, api,
        DurableInbox(journal, TELEGRAM_TRANSPORT),
        allowed_user_ids=users, allowed_chat_ids=chats, poll_timeout_seconds=1,
    )


def test_update_is_persisted_and_processed_before_next_poll_acknowledges_it(tmp_path):
    api = PollAPI([[update()], []])
    telegram = observer(tmp_path, api)

    def check_second_call(call_number, offset):
        if call_number == 2:
            saved = telegram.inbox.get("42")
            assert saved is not None and saved.state == "processed"
            assert offset == 43

    api.before_call = check_second_call
    assert telegram.poll_once() == 1
    assert telegram.poll_once() == 0
    assert [event.content["telegram_update_id"] for event in telegram.stimulus_log] == [42]


def test_restart_recovers_callback_failure_without_duplicate_stimulus(tmp_path):
    calls = []

    def fail():
        calls.append("failed")
        raise RuntimeError("model unavailable")

    first = observer(tmp_path, PollAPI([[update()]]), fail)
    with pytest.raises(RuntimeError, match="model unavailable"):
        first.poll_once()
    assert first.inbox.get("42").state == "stimulus_persisted"
    assert len(first.stimulus_log.read_all()) == 1

    restarted = observer(tmp_path, PollAPI([[]]), lambda: calls.append("recovered"))
    assert restarted.next_offset == 43
    assert restarted.recover_pending() == 1
    assert len(restarted.stimulus_log.read_all()) == 1
    assert restarted.inbox.get("42").state == "processed"
    assert calls == ["failed", "recovered"]


def test_restart_closes_crash_window_after_log_append_before_inbox_mark(tmp_path):
    path = tmp_path / "delivery.sqlite3"
    inbox = DurableInbox(DeliveryJournal(path), TELEGRAM_TRANSPORT)
    inbox.store([("42", update())])
    log = StimulusLog(tmp_path / "stimulus.jsonl")
    log.append("telegram_user:10", "chat_message", {
        "message": "look", "telegram_update_id": 42,
    })
    calls = []
    restarted = TelegramObserver(
        log, lambda: calls.append("called"), PollAPI([[]]), inbox,
        allowed_user_ids=(10,), allowed_chat_ids=(20,), poll_timeout_seconds=1,
    )

    restarted.recover_pending()
    assert len(log.read_all()) == 1
    assert calls == ["called"]
    assert inbox.get("42").state == "processed"


def test_attachments_reply_metadata_and_allowlists_are_projected(tmp_path):
    api = PollAPI([[update()], [update(43, user_id=99)], [update(44, chat_id=99)]])
    telegram = observer(tmp_path, api)
    telegram.poll_once()
    telegram.poll_once()
    telegram.poll_once()

    events = telegram.stimulus_log.read_all()
    assert len(events) == 1
    content = events[0].content
    assert content["message"] == "look"
    assert content["reply_to_message_id"] == 299
    assert content["reply_to"]["text"] == "earlier"
    assert [(item["kind"], item["file_id"]) for item in content["attachments"]] == [
        ("photo", "large"), ("document", "document-id")
    ]
    assert telegram.inbox.get("43").state == "ignored"
    assert telegram.inbox.get("44").state == "ignored"


def test_long_reply_and_attachment_have_per_part_receipts(tmp_path):
    journal = DeliveryJournal(tmp_path / "delivery.sqlite3")
    sender = ReceiptSender()
    outbox = DurableOutbox(journal, TELEGRAM_TRANSPORT, sender)
    tool = TelegramTool(outbox, allowed_chat_ids=(20,))
    message = ("🙂 telegram words " * 400) + "tail"

    result = tool.execute(
        message=message,
        chat_id=20,
        reply_to_message_id=300,
        attachments=[{"kind": "document", "source": "telegram-file-id", "caption": "notes"}],
    )
    parts = outbox.group(result.details["group_id"])
    text_parts = [part for part in parts if part.payload["kind"] == "text"]
    assert not result.is_error
    assert "".join(part.payload["text"] for part in text_parts) == message
    assert all(_utf16_units(part.payload["text"]) <= 4096 for part in text_parts)
    assert parts[-1].payload["attachment_kind"] == "document"
    assert parts[0].payload["reply_to_message_id"] == 300
    assert all(part.status == "delivered" for part in parts)
    assert [part.external_message_id for part in parts] == [
        str(1000 + index) for index in range(1, len(parts) + 1)
    ]


def test_splitter_preserves_whitespace_and_counts_emoji_as_two_units():
    text = "a b\n" + "🙂" * 8 + " end"
    chunks = split_telegram_message(text, limit=6)
    assert "".join(chunks) == text
    assert all(_utf16_units(chunk) <= 6 for chunk in chunks)


def test_telegram_sender_classifies_rate_limit_for_durable_retry(tmp_path):
    class API:
        def request(self, *args, **kwargs):
            raise TelegramAPIError(
                "Too Many Requests", error_code=429, retry_after=17, temporary=True
            )

    sender = TelegramSender(API())
    journal = DeliveryJournal(tmp_path / "delivery.sqlite3")
    # A file-backed temporary item is simpler than manufacturing a storage dataclass.
    outbox = DurableOutbox(journal, TELEGRAM_TRANSPORT, sender)
    item = outbox.enqueue("20", [{"kind": "text", "text": "hello"}])[0]
    outcome = sender.send(item)
    assert outcome.status == "retry"
    assert outcome.retry_after == 17


def telegram_spec():
    return AgentSpec(
        name="Tam", constitution="Be helpful", core="auto",
        models=(ModelSpec("ollama", "test"),),
        interface=InterfaceSpec(
            "telegram", bot_token_env="TAM_TELEGRAM_TOKEN",
            allowed_user_ids=(10,), allowed_chat_ids=(20,), poll_timeout_seconds=10,
        ),
    )


def test_telegram_dsl_uses_secret_env_and_reassembly_preserves_delivery_state(
    tmp_path, monkeypatch
):
    definition = telegram_spec()
    assert "super-secret" not in render(definition)
    monkeypatch.setenv("TAM_TELEGRAM_TOKEN", "super-secret")
    launcher = assemble(definition, tmp_path / "build")
    home = tmp_path / "tam-state"
    first = build_agent(definition, home)
    assert isinstance(first.observer, TelegramObserver)
    assert isinstance(first.core.tools[TelegramTool.name], TelegramTool)
    queued = first.core.tools[TelegramTool.name].outbox.enqueue(
        "20", [{"kind": "text", "text": "survive upgrade"}]
    )[0]

    assemble(replace(definition, name="Tam upgraded"), tmp_path / "build")
    restarted = build_agent(replace(definition, name="Tam upgraded"), home)
    recovered = restarted.core.tools[TelegramTool.name].outbox.group(queued.group_id)
    assert recovered[0].payload["text"] == "survive upgrade"
    assert launcher.exists()


def test_telegram_dsl_requires_allowlist_and_runtime_secret(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="allowed user or chat"):
        replace(
            telegram_spec(),
            interface=InterfaceSpec("telegram", bot_token_env="TOKEN"),
        ).validate()
    monkeypatch.delenv("TAM_TELEGRAM_TOKEN", raising=False)
    with pytest.raises(ValueError, match="TAM_TELEGRAM_TOKEN"):
        build_agent(telegram_spec(), tmp_path)
