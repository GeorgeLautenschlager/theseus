"""Durable Telegram long-polling observer."""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

from theseus.durable_delivery import DurableInbox, DurableOutbox, InboxItem
from theseus.stimulus_log import StimulusLog
from theseus.telegram_api import TelegramAPIError, TelegramBotAPI

logger = logging.getLogger(__name__)

TELEGRAM_TRANSPORT = "telegram"


class TelegramObserver:
    """Persist, project, and dispatch Telegram updates using long polling.

    Calling `getUpdates` with an offset acknowledges every lower update. `_offset` is
    therefore advanced only after the complete returned batch has committed to the inbox.
    Pending inbox work is recovered before every subsequent poll.
    """

    def __init__(
        self,
        stimulus_log: StimulusLog,
        orient_chat_message_callback: Callable[[], None],
        api: TelegramBotAPI,
        inbox: DurableInbox,
        *,
        allowed_user_ids: tuple[int, ...] = (),
        allowed_chat_ids: tuple[int, ...] = (),
        outbox: DurableOutbox | None = None,
        poll_timeout_seconds: int = 25,
        failure_delay_seconds: float = 2.0,
    ) -> None:
        if not allowed_user_ids and not allowed_chat_ids:
            raise ValueError("Telegram requires at least one allowed user or chat ID")
        self.stimulus_log = stimulus_log
        self.orient_chat_message_callback = orient_chat_message_callback
        self.api = api
        self.inbox = inbox
        self.outbox = outbox
        self.allowed_user_ids = frozenset(allowed_user_ids)
        self.allowed_chat_ids = frozenset(allowed_chat_ids)
        self.poll_timeout_seconds = poll_timeout_seconds
        self.failure_delay_seconds = failure_delay_seconds
        self._stop = threading.Event()
        highest = inbox.max_numeric_external_id
        self._offset = highest + 1 if highest is not None else None
        self._logged_update_ids = {
            str(event.content["telegram_update_id"])
            for event in stimulus_log.read_all()
            if event.type == "chat_message"
            and isinstance(event.content, dict)
            and "telegram_update_id" in event.content
        }

    @property
    def next_offset(self) -> int | None:
        return self._offset

    def stop(self) -> None:
        self._stop.set()

    def poll_once(self) -> int:
        """Recover local work, then make one long-poll request and durably ingest it."""
        self.recover_pending()
        updates = self.api.get_updates(
            offset=self._offset, timeout=self.poll_timeout_seconds
        )
        if not updates:
            return 0
        ids: list[int] = []
        rows: list[tuple[str, dict[str, Any]]] = []
        for update in updates:
            update_id = update.get("update_id")
            if isinstance(update_id, bool) or not isinstance(update_id, int):
                raise TelegramAPIError(
                    "Telegram update is missing an integer update_id", temporary=True
                )
            ids.append(update_id)
            rows.append((str(update_id), update))

        # The commit is the acknowledgement boundary. No code may update `_offset` or
        # make the next request until this returns successfully.
        self.inbox.store(rows)
        self._offset = max(ids) + 1
        self.recover_pending()
        return len(updates)

    def recover_pending(self) -> int:
        processed = 0
        for item in self.inbox.pending():
            self._process(item)
            processed += 1
        return processed

    def _process(self, item: InboxItem) -> None:
        state = item.state
        if state == "received":
            projected = self._project(item.payload)
            if projected is None:
                self.inbox.mark(item.external_id, "ignored")
                return
            actor, content = projected
            if item.external_id not in self._logged_update_ids:
                self.stimulus_log.append(
                    actor=actor, type="chat_message", content=content,
                )
                self._logged_update_ids.add(item.external_id)
            # Separate state records the JSONL append before cognition begins. On a crash,
            # recovery can re-trigger cognition without appending the input twice.
            self.inbox.mark(item.external_id, "stimulus_persisted")
            state = "stimulus_persisted"
        if state == "stimulus_persisted":
            try:
                self.orient_chat_message_callback()
            except Exception as exc:
                self.inbox.mark(
                    item.external_id,
                    "stimulus_persisted",
                    f"{type(exc).__name__}: {exc}"[:1000],
                )
                raise
            self.inbox.mark(item.external_id, "processed")

    def _project(self, update: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
        kind = next(
            (
                name for name in (
                    "message", "edited_message", "channel_post", "edited_channel_post"
                )
                if isinstance(update.get(name), dict)
            ),
            None,
        )
        if kind is None:
            return None
        message = update[kind]
        chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
        sender = message.get("from") if isinstance(message.get("from"), dict) else {}
        chat_id = chat.get("id")
        user_id = sender.get("id")
        if self.allowed_user_ids and user_id not in self.allowed_user_ids:
            return None
        if self.allowed_chat_ids and chat_id not in self.allowed_chat_ids:
            return None
        message_id = message.get("message_id")
        if isinstance(message_id, bool) or not isinstance(message_id, int):
            return None
        text = message.get("text")
        if not isinstance(text, str):
            text = message.get("caption") if isinstance(message.get("caption"), str) else ""
        attachments = _attachments(message)
        if not text and not attachments:
            return None
        reply = message.get("reply_to_message")
        reply_summary = _reply_summary(reply) if isinstance(reply, dict) else None
        actor_id = user_id if isinstance(user_id, int) else chat_id
        actor_kind = "user" if isinstance(user_id, int) else "chat"
        content: dict[str, Any] = {
            "message": text,
            "transport": TELEGRAM_TRANSPORT,
            "telegram_update_id": update["update_id"],
            "telegram_message_id": message_id,
            "chat_id": chat_id,
            "user_id": user_id,
            "update_kind": kind,
            "edited": kind.startswith("edited_"),
            "chat": _selected(chat, "id", "type", "title", "username", "first_name", "last_name"),
            "sender": _selected(sender, "id", "is_bot", "username", "first_name", "last_name"),
            "attachments": attachments,
        }
        if reply_summary is not None:
            content["reply_to"] = reply_summary
            content["reply_to_message_id"] = reply_summary["message_id"]
        return f"telegram_{actor_kind}:{actor_id}", content

    def run(self) -> None:
        """Recover both queues, then poll until stopped."""
        if self.outbox is not None:
            self.outbox.recover()
        delay = self.failure_delay_seconds
        while not self._stop.is_set():
            try:
                if self.outbox is not None:
                    self.outbox.drain()
                self.poll_once()
                if self.outbox is not None:
                    self.outbox.drain()
                delay = self.failure_delay_seconds
            except TelegramAPIError as exc:
                wait = exc.retry_after if exc.retry_after is not None else delay
                logger.warning("Telegram polling deferred: %s", exc)
                self._stop.wait(max(0.0, wait))
                delay = min(60.0, max(self.failure_delay_seconds, delay * 2))
            except Exception:
                logger.exception("Telegram observer iteration failed; pending work is durable")
                self._stop.wait(delay)
                delay = min(60.0, max(self.failure_delay_seconds, delay * 2))


def _selected(source: dict[str, Any], *names: str) -> dict[str, Any]:
    return {name: source[name] for name in names if name in source}


def _reply_summary(message: dict[str, Any]) -> dict[str, Any] | None:
    message_id = message.get("message_id")
    if isinstance(message_id, bool) or not isinstance(message_id, int):
        return None
    sender = message.get("from") if isinstance(message.get("from"), dict) else {}
    text = message.get("text")
    if not isinstance(text, str):
        text = message.get("caption") if isinstance(message.get("caption"), str) else ""
    return {
        "message_id": message_id,
        "user_id": sender.get("id"),
        "text": text,
    }


def _attachments(message: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    photos = message.get("photo")
    if isinstance(photos, list):
        usable = [photo for photo in photos if isinstance(photo, dict) and photo.get("file_id")]
        if usable:
            photo = usable[-1]  # Telegram supplies sizes smallest to largest.
            found.append({"kind": "photo", **_selected(
                photo, "file_id", "file_unique_id", "width", "height", "file_size"
            )})
    for kind in ("document", "audio", "video", "animation", "voice", "sticker"):
        value = message.get(kind)
        if isinstance(value, dict) and value.get("file_id"):
            found.append({"kind": kind, **_selected(
                value, "file_id", "file_unique_id", "file_name", "mime_type",
                "file_size", "duration", "width", "height", "emoji"
            )})
    return found
