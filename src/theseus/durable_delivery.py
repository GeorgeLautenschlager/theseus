"""Transport-neutral durable inbox and outbox primitives.

The journal owns persistence and retry bookkeeping; transport adapters only translate a
payload into one remote attempt. SQLite is used here because inbox state transitions and
multi-part outbox enqueue both need atomic updates that an append-only line cannot express.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence


_FINAL_INBOX_STATES = ("processed", "ignored")
_FINAL_OUTBOX_STATES = ("delivered", "failed")


@dataclass(frozen=True, slots=True)
class InboxItem:
    transport: str
    external_id: str
    payload: dict[str, Any]
    state: str
    received_at: float
    updated_at: float
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class OutboxItem:
    id: str
    transport: str
    group_id: str
    destination: str
    payload: dict[str, Any]
    part_index: int
    part_count: int
    status: str
    attempts: int
    available_at: float
    created_at: float
    delivered_at: float | None = None
    external_message_id: str | None = None
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class DeliveryOutcome:
    """One transport attempt, classified without exposing transport policy to storage."""

    status: str
    external_message_id: str | None = None
    error: str | None = None
    retry_after: float | None = None

    @classmethod
    def delivered(cls, external_message_id: str) -> "DeliveryOutcome":
        return cls("delivered", external_message_id=external_message_id)

    @classmethod
    def retry(cls, error: str, retry_after: float | None = None) -> "DeliveryOutcome":
        return cls("retry", error=error, retry_after=retry_after)

    @classmethod
    def failed(cls, error: str) -> "DeliveryOutcome":
        return cls("failed", error=error)


class DeliverySender(Protocol):
    def send(self, item: OutboxItem) -> DeliveryOutcome:
        """Attempt one already-persisted delivery part."""
        ...


class DeliveryJournal:
    """SQLite storage shared by durable inboxes and outboxes.

    Every method opens a short-lived connection. This is slightly more work than retaining
    one connection, but lets an observer thread and a cognitive/tool thread safely share the
    journal without leaking SQLite's connection-thread rules into either caller.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS delivery_inbox (
                    transport TEXT NOT NULL,
                    external_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    state TEXT NOT NULL,
                    received_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_error TEXT,
                    PRIMARY KEY (transport, external_id)
                );
                CREATE TABLE IF NOT EXISTS delivery_outbox (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    id TEXT NOT NULL UNIQUE,
                    transport TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    part_index INTEGER NOT NULL,
                    part_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    available_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    delivered_at REAL,
                    external_message_id TEXT,
                    last_error TEXT
                );
                CREATE INDEX IF NOT EXISTS delivery_outbox_pending
                    ON delivery_outbox (transport, status, sequence);
                CREATE INDEX IF NOT EXISTS delivery_outbox_group
                    ON delivery_outbox (group_id, part_index);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _json(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def store_incoming(
        self,
        transport: str,
        items: Sequence[tuple[str, dict[str, Any]]],
        *,
        now: float,
    ) -> int:
        """Commit a received batch atomically; return the number of new rows."""
        if not items:
            return 0
        with self._lock, self._connect() as connection:
            before = connection.total_changes
            connection.executemany(
                """INSERT OR IGNORE INTO delivery_inbox
                   (transport, external_id, payload, state, received_at, updated_at)
                   VALUES (?, ?, ?, 'received', ?, ?)""",
                [
                    (transport, external_id, self._json(payload), now, now)
                    for external_id, payload in items
                ],
            )
            return connection.total_changes - before

    def pending_incoming(self, transport: str) -> list[InboxItem]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM delivery_inbox
                   WHERE transport = ? AND state NOT IN (?, ?)
                   ORDER BY CAST(external_id AS INTEGER), received_at""",
                (transport, *_FINAL_INBOX_STATES),
            ).fetchall()
        return [self._inbox_item(row) for row in rows]

    def incoming(self, transport: str, external_id: str) -> InboxItem | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM delivery_inbox WHERE transport = ? AND external_id = ?",
                (transport, external_id),
            ).fetchone()
        return self._inbox_item(row) if row is not None else None

    def max_numeric_external_id(self, transport: str) -> int | None:
        with self._lock, self._connect() as connection:
            values = connection.execute(
                "SELECT external_id FROM delivery_inbox WHERE transport = ?", (transport,)
            ).fetchall()
        numeric = [int(row[0]) for row in values if str(row[0]).lstrip("-").isdigit()]
        return max(numeric) if numeric else None

    def mark_incoming(
        self,
        transport: str,
        external_id: str,
        state: str,
        *,
        now: float,
        error: str | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            changed = connection.execute(
                """UPDATE delivery_inbox SET state = ?, updated_at = ?, last_error = ?
                   WHERE transport = ? AND external_id = ?""",
                (state, now, error, transport, external_id),
            ).rowcount
            if changed != 1:
                raise KeyError(f"unknown inbox item {transport}:{external_id}")

    def enqueue_outgoing(
        self,
        transport: str,
        destination: str,
        payloads: Sequence[dict[str, Any]],
        *,
        now: float,
        group_id: str | None = None,
    ) -> list[OutboxItem]:
        """Atomically enqueue every part of one logical message before any can send."""
        if not payloads:
            raise ValueError("an outbox group must contain at least one part")
        group_id = group_id or uuid.uuid4().hex
        part_count = len(payloads)
        values = [
            (
                uuid.uuid4().hex,
                transport,
                group_id,
                destination,
                self._json(payload),
                index,
                part_count,
                now,
                now,
            )
            for index, payload in enumerate(payloads, start=1)
        ]
        with self._lock, self._connect() as connection:
            connection.executemany(
                """INSERT INTO delivery_outbox
                   (id, transport, group_id, destination, payload, part_index, part_count,
                    status, attempts, available_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)""",
                values,
            )
        return self.outbox_group(group_id)

    def recover_sending(self, transport: str, *, now: float) -> int:
        """Put attempts interrupted between claim and receipt recording back in the queue."""
        with self._lock, self._connect() as connection:
            return connection.execute(
                """UPDATE delivery_outbox
                   SET status = 'retry', available_at = ?,
                       last_error = 'process stopped during delivery attempt'
                   WHERE transport = ? AND status = 'sending'""",
                (now, transport),
            ).rowcount

    def next_outbox(self, transport: str) -> OutboxItem | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM delivery_outbox
                   WHERE transport = ? AND status NOT IN (?, ?)
                   ORDER BY sequence LIMIT 1""",
                (transport, *_FINAL_OUTBOX_STATES),
            ).fetchone()
        return self._outbox_item(row) if row is not None else None

    def mark_sending(self, item_id: str, *, now: float) -> OutboxItem:
        with self._lock, self._connect() as connection:
            changed = connection.execute(
                """UPDATE delivery_outbox
                   SET status = 'sending', attempts = attempts + 1, last_error = NULL
                   WHERE id = ? AND status NOT IN (?, ?)""",
                (item_id, *_FINAL_OUTBOX_STATES),
            ).rowcount
            if changed != 1:
                raise KeyError(f"unknown or completed outbox item {item_id}")
            row = connection.execute(
                "SELECT * FROM delivery_outbox WHERE id = ?", (item_id,)
            ).fetchone()
        return self._outbox_item(row)

    def mark_delivered(
        self, item_id: str, external_message_id: str, *, now: float
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """UPDATE delivery_outbox
                   SET status = 'delivered', delivered_at = ?, external_message_id = ?,
                       last_error = NULL
                   WHERE id = ?""",
                (now, external_message_id, item_id),
            )

    def mark_retry(self, item_id: str, error: str, *, available_at: float) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """UPDATE delivery_outbox
                   SET status = 'retry', available_at = ?, last_error = ? WHERE id = ?""",
                (available_at, error, item_id),
            )

    def mark_failed(self, item_id: str, error: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """UPDATE delivery_outbox SET status = 'failed', last_error = ? WHERE id = ?""",
                (error, item_id),
            )

    def outbox_group(self, group_id: str) -> list[OutboxItem]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM delivery_outbox WHERE group_id = ? ORDER BY part_index""",
                (group_id,),
            ).fetchall()
        return [self._outbox_item(row) for row in rows]

    @staticmethod
    def _inbox_item(row: sqlite3.Row) -> InboxItem:
        return InboxItem(
            transport=row["transport"], external_id=row["external_id"],
            payload=json.loads(row["payload"]), state=row["state"],
            received_at=row["received_at"], updated_at=row["updated_at"],
            last_error=row["last_error"],
        )

    @staticmethod
    def _outbox_item(row: sqlite3.Row) -> OutboxItem:
        return OutboxItem(
            id=row["id"], transport=row["transport"], group_id=row["group_id"],
            destination=row["destination"], payload=json.loads(row["payload"]),
            part_index=row["part_index"], part_count=row["part_count"],
            status=row["status"], attempts=row["attempts"],
            available_at=row["available_at"], created_at=row["created_at"],
            delivered_at=row["delivered_at"], external_message_id=row["external_message_id"],
            last_error=row["last_error"],
        )


class DurableInbox:
    def __init__(
        self,
        journal: DeliveryJournal,
        transport: str,
        *,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.journal = journal
        self.transport = transport
        self._now = now

    def store(self, items: Sequence[tuple[str, dict[str, Any]]]) -> int:
        return self.journal.store_incoming(self.transport, items, now=self._now())

    def pending(self) -> list[InboxItem]:
        return self.journal.pending_incoming(self.transport)

    def get(self, external_id: str) -> InboxItem | None:
        return self.journal.incoming(self.transport, external_id)

    def mark(self, external_id: str, state: str, error: str | None = None) -> None:
        self.journal.mark_incoming(
            self.transport, external_id, state, now=self._now(), error=error
        )

    @property
    def max_numeric_external_id(self) -> int | None:
        return self.journal.max_numeric_external_id(self.transport)


class DurableOutbox:
    """Ordered durable dispatch with transport-supplied result classification."""

    def __init__(
        self,
        journal: DeliveryJournal,
        transport: str,
        sender: DeliverySender,
        *,
        now: Callable[[], float] = time.time,
        base_retry_seconds: float = 2.0,
        max_retry_seconds: float = 300.0,
    ) -> None:
        self.journal = journal
        self.transport = transport
        self.sender = sender
        self._now = now
        self._base_retry_seconds = base_retry_seconds
        self._max_retry_seconds = max_retry_seconds
        self._drain_lock = threading.Lock()

    def enqueue(
        self,
        destination: str,
        payloads: Sequence[dict[str, Any]],
        *,
        group_id: str | None = None,
    ) -> list[OutboxItem]:
        return self.journal.enqueue_outgoing(
            self.transport, destination, payloads, now=self._now(), group_id=group_id
        )

    def recover(self) -> int:
        return self.journal.recover_sending(self.transport, now=self._now())

    def group(self, group_id: str) -> list[OutboxItem]:
        return self.journal.outbox_group(group_id)

    def drain(self, *, limit: int = 100) -> int:
        """Attempt due parts oldest-first, stopping at the first deferred retry."""
        attempted = 0
        with self._drain_lock:
            while attempted < limit:
                item = self.journal.next_outbox(self.transport)
                if item is None or item.available_at > self._now():
                    break
                item = self.journal.mark_sending(item.id, now=self._now())
                attempted += 1
                try:
                    outcome = self.sender.send(item)
                except Exception as exc:
                    outcome = DeliveryOutcome.retry(
                        f"{type(exc).__name__}: {exc}"[:1000]
                    )
                now = self._now()
                if outcome.status == "delivered" and outcome.external_message_id is not None:
                    self.journal.mark_delivered(
                        item.id, outcome.external_message_id, now=now
                    )
                    continue
                if outcome.status == "failed":
                    self.journal.mark_failed(item.id, outcome.error or "permanent delivery failure")
                    continue
                if outcome.status != "retry":
                    self.journal.mark_retry(
                        item.id,
                        outcome.error or f"invalid delivery outcome {outcome.status!r}",
                        available_at=now + self._retry_delay(item.attempts),
                    )
                    break
                delay = (
                    max(0.0, outcome.retry_after)
                    if outcome.retry_after is not None
                    else self._retry_delay(item.attempts)
                )
                self.journal.mark_retry(
                    item.id, outcome.error or "temporary delivery failure",
                    available_at=now + delay,
                )
                break
        return attempted

    def _retry_delay(self, attempts: int) -> float:
        return min(
            self._max_retry_seconds,
            self._base_retry_seconds * (2 ** max(0, attempts - 1)),
        )
