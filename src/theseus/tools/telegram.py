"""Telegram reply tool backed by the reusable durable outbox."""

from __future__ import annotations

from typing import Any

from theseus.durable_delivery import DurableOutbox
from theseus.tools.tool import ToolResult

TELEGRAM_TEXT_UNITS = 4096
TELEGRAM_CAPTION_UNITS = 1024
_ATTACHMENT_KINDS = ("photo", "document", "audio", "video", "animation", "voice")


class TelegramTool:
    name = "respond_in_telegram"
    ends_turn = True
    description = (
        "Send a Telegram reply. Use the chat_id and telegram_message_id from the incoming "
        "Telegram stimulus; pass telegram_message_id as reply_to_message_id when replying "
        "to that message. Text and attachments are durably queued before delivery."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Text to send verbatim. Long text is split safely.",
            },
            "chat_id": {
                "type": "integer",
                "description": "Destination Telegram chat ID from the incoming stimulus.",
            },
            "reply_to_message_id": {
                "type": "integer",
                "description": "Telegram message ID this response replies to.",
            },
            "attachments": {
                "type": "array",
                "description": "Optional files, Telegram file IDs, or HTTP URLs to send.",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": list(_ATTACHMENT_KINDS)},
                        "source": {
                            "type": "string",
                            "description": "Runtime-home path, Telegram file ID, or HTTP URL.",
                        },
                        "caption": {"type": "string"},
                    },
                    "required": ["kind", "source"],
                },
            },
        },
        "anyOf": [{"required": ["message"]}, {"required": ["attachments"]}],
    }

    def __init__(
        self,
        outbox: DurableOutbox,
        *,
        allowed_chat_ids: tuple[int, ...],
        allowed_user_ids: tuple[int, ...] = (),
    ) -> None:
        self.outbox = outbox
        # In a private Telegram conversation chat_id == user_id. Including allowed users
        # therefore gives a safe destination when a deployment only configures user IDs;
        # group destinations still have to appear explicitly in allowed_chat_ids.
        self.allowed_destinations = frozenset(allowed_chat_ids) | frozenset(allowed_user_ids)

    def execute(
        self,
        message: str = "",
        chat_id: int | None = None,
        reply_to_message_id: int | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> ToolResult:
        if not isinstance(message, str):
            return ToolResult("Telegram message must be text.", is_error=True)
        destination = self._destination(chat_id)
        if isinstance(destination, str):
            return ToolResult(destination, is_error=True)
        if (
            reply_to_message_id is not None
            and (isinstance(reply_to_message_id, bool) or not isinstance(reply_to_message_id, int))
        ):
            return ToolResult("reply_to_message_id must be an integer.", is_error=True)
        attachments = [] if attachments is None else attachments
        if not isinstance(attachments, list):
            return ToolResult("Telegram attachments must be a list.", is_error=True)

        payloads: list[dict[str, Any]] = [
            {"kind": "text", "text": chunk}
            for chunk in split_telegram_message(message)
        ]
        for attachment in attachments:
            validated = self._attachment(attachment)
            if isinstance(validated, str):
                return ToolResult(validated, is_error=True)
            payloads.append(validated)
        if not payloads:
            return ToolResult("Telegram reply needs text or at least one attachment.", is_error=True)
        if reply_to_message_id is not None:
            payloads[0]["reply_to_message_id"] = reply_to_message_id

        queued = self.outbox.enqueue(str(destination), payloads)
        group_id = queued[0].group_id
        # Best effort now; a temporary failure remains in the durable queue and the
        # observer retries it. Draining may first finish an older queued message, which is
        # intentional: per-chat conversation order is safer than overtaking a retry.
        self.outbox.drain()
        parts = self.outbox.group(group_id)
        failed = [part for part in parts if part.status == "failed"]
        delivered = [part for part in parts if part.status == "delivered"]
        status = "delivered" if len(delivered) == len(parts) else "queued"
        if failed:
            status = "partially_failed" if delivered else "failed"
        return ToolResult(
            (
                f"Telegram message {status}: {len(delivered)}/{len(parts)} part(s) delivered; "
                "undelivered parts remain recorded."
            ),
            is_error=bool(failed),
            details={
                "group_id": group_id,
                "status": status,
                "parts": [
                    {
                        "id": part.id,
                        "index": part.part_index,
                        "status": part.status,
                        "attempts": part.attempts,
                        "telegram_message_id": part.external_message_id,
                        "error": part.last_error,
                    }
                    for part in parts
                ],
            },
        )

    def _destination(self, chat_id: int | None) -> int | str:
        if chat_id is None:
            if len(self.allowed_destinations) == 1:
                return next(iter(self.allowed_destinations))
            return "chat_id is required when more than one Telegram destination is allowed."
        if isinstance(chat_id, bool) or not isinstance(chat_id, int):
            return "chat_id must be an integer."
        if chat_id not in self.allowed_destinations:
            return f"Telegram chat {chat_id} is not allowed."
        return chat_id

    @staticmethod
    def _attachment(value: Any) -> dict[str, Any] | str:
        if not isinstance(value, dict):
            return "Each Telegram attachment must be an object."
        kind = value.get("kind")
        source = value.get("source")
        caption = value.get("caption")
        if kind not in _ATTACHMENT_KINDS:
            return f"Unsupported Telegram attachment kind {kind!r}."
        if not isinstance(source, str) or not source:
            return "Telegram attachment source must be nonempty text."
        if caption is not None and not isinstance(caption, str):
            return "Telegram attachment caption must be text."
        if caption is not None and _utf16_units(caption) > TELEGRAM_CAPTION_UNITS:
            return f"Telegram attachment captions may not exceed {TELEGRAM_CAPTION_UNITS} UTF-16 units."
        payload: dict[str, Any] = {
            "kind": "attachment", "attachment_kind": kind, "source": source,
        }
        if caption:
            payload["caption"] = caption
        return payload


def _utf16_units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def split_telegram_message(text: str, limit: int = TELEGRAM_TEXT_UNITS) -> list[str]:
    """Losslessly split on Telegram's UTF-16 limit, preferring nearby whitespace."""
    if not text:
        return []
    if limit < 1:
        raise ValueError("limit must be positive")
    chunks: list[str] = []
    remaining = text
    while _utf16_units(remaining) > limit:
        units = 0
        end = 0
        for index, character in enumerate(remaining):
            character_units = _utf16_units(character)
            if units + character_units > limit:
                break
            units += character_units
            end = index + 1
        if end == 0:
            raise ValueError("limit is smaller than one Unicode character")
        candidate = remaining[:end]
        split_at = max(candidate.rfind("\n\n"), candidate.rfind("\n"), candidate.rfind(" "))
        # Do not create pathologically tiny chunks just because an old whitespace occurs
        # near the beginning of this window.
        if split_at < end // 2:
            split_at = end
        else:
            split_at += 1  # Preserve the boundary character in exactly one chunk.
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:]
    if remaining:
        chunks.append(remaining)
    return chunks
