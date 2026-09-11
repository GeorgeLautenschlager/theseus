from __future__ import annotations

from theseus.durable_delivery import (
    DeliveryJournal, DeliveryOutcome, DurableInbox, DurableOutbox,
)


class Clock:
    def __init__(self, now=100.0):
        self.value = now

    def __call__(self):
        return self.value


class Sender:
    def __init__(self, outcomes, inspect=None):
        self.outcomes = list(outcomes)
        self.calls = []
        self.inspect = inspect

    def send(self, item):
        self.calls.append(item)
        if self.inspect:
            self.inspect(item)
        return self.outcomes.pop(0)


def test_inbox_batch_is_deduplicated_and_survives_reopen(tmp_path):
    path = tmp_path / "delivery.sqlite3"
    inbox = DurableInbox(DeliveryJournal(path), "example")
    assert inbox.store([("7", {"value": "first"}), ("8", {"value": "second"})]) == 2
    assert inbox.store([("7", {"value": "replacement"})]) == 0

    reopened = DurableInbox(DeliveryJournal(path), "example")
    assert [item.external_id for item in reopened.pending()] == ["7", "8"]
    assert reopened.get("7").payload == {"value": "first"}
    assert reopened.max_numeric_external_id == 8


def test_outbox_is_committed_before_sender_sees_it_and_records_receipt(tmp_path):
    journal = DeliveryJournal(tmp_path / "delivery.sqlite3")
    inspected = []

    def inspect(item):
        persisted = DeliveryJournal(journal.path).outbox_group(item.group_id)
        inspected.append((persisted[0].status, persisted[0].attempts))

    sender = Sender([DeliveryOutcome.delivered("remote-42")], inspect=inspect)
    outbox = DurableOutbox(journal, "example", sender)
    group = outbox.enqueue("destination", [{"body": "hello"}])
    assert group[0].status == "pending"

    assert outbox.drain() == 1
    delivered = outbox.group(group[0].group_id)[0]
    assert inspected == [("sending", 1)]
    assert delivered.status == "delivered"
    assert delivered.external_message_id == "remote-42"
    assert delivered.delivered_at is not None


def test_rate_limit_defers_head_without_overtaking(tmp_path):
    clock = Clock()
    sender = Sender([
        DeliveryOutcome.retry("too many requests", retry_after=30),
        DeliveryOutcome.delivered("one"),
        DeliveryOutcome.delivered("two"),
    ])
    outbox = DurableOutbox(
        DeliveryJournal(tmp_path / "delivery.sqlite3"), "example", sender, now=clock
    )
    first = outbox.enqueue("chat", [{"text": "one"}])[0]
    second = outbox.enqueue("chat", [{"text": "two"}])[0]

    assert outbox.drain() == 1
    assert outbox.group(first.group_id)[0].available_at == 130
    assert outbox.drain() == 0
    assert sender.calls == [sender.calls[0]]
    clock.value = 130
    assert outbox.drain() == 2
    assert outbox.group(first.group_id)[0].external_message_id == "one"
    assert outbox.group(second.group_id)[0].external_message_id == "two"


def test_interrupted_send_is_recovered_after_restart(tmp_path):
    path = tmp_path / "delivery.sqlite3"
    journal = DeliveryJournal(path)
    first_process = DurableOutbox(journal, "example", Sender([]))
    item = first_process.enqueue("chat", [{"text": "recover me"}])[0]
    journal.mark_sending(item.id, now=100)

    sender = Sender([DeliveryOutcome.delivered("after-restart")])
    restarted = DurableOutbox(DeliveryJournal(path), "example", sender, now=Clock(200))
    assert restarted.recover() == 1
    assert restarted.drain() == 1
    recovered = restarted.group(item.group_id)[0]
    assert recovered.status == "delivered"
    assert recovered.attempts == 2
    assert recovered.external_message_id == "after-restart"
