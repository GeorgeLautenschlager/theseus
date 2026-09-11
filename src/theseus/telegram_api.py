"""Small Telegram Bot API boundary used by the observer and durable sender."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from theseus.durable_delivery import DeliveryOutcome, OutboxItem


class TelegramAPIError(RuntimeError):
    def __init__(
        self,
        description: str,
        *,
        error_code: int | None = None,
        retry_after: float | None = None,
        temporary: bool = False,
    ) -> None:
        super().__init__(description)
        self.error_code = error_code
        self.retry_after = retry_after
        self.temporary = temporary


class TelegramBotAPI:
    """Synchronous Bot API client.

    The token remains private to this object. Callers and stored payloads use method names
    and arguments only, so neither the generated DSL snapshot nor the delivery database
    contains the secret.
    """

    def __init__(
        self,
        bot_token: str,
        *,
        client: Any | None = None,
        request_timeout: float = 30.0,
    ) -> None:
        if not isinstance(bot_token, str) or not bot_token.strip():
            raise ValueError("Telegram bot token must be nonempty")
        self.__base_url = f"https://api.telegram.org/bot{bot_token.strip()}"
        self._client = client
        self._request_timeout = request_timeout

    def get_updates(self, *, offset: int | None, timeout: int) -> list[dict[str, Any]]:
        data: dict[str, Any] = {
            "timeout": timeout,
            "allowed_updates": [
                "message", "edited_message", "channel_post", "edited_channel_post"
            ],
        }
        if offset is not None:
            data["offset"] = offset
        result = self.request("getUpdates", json_body=data, timeout=timeout + 10)
        if not isinstance(result, list) or any(not isinstance(item, dict) for item in result):
            raise TelegramAPIError("Telegram returned a malformed update list", temporary=True)
        return result

    def request(
        self,
        method: str,
        *,
        json_body: dict[str, Any] | None = None,
        form: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        own_client = self._client is None
        client = self._client or httpx.Client(follow_redirects=False)
        try:
            try:
                response = client.post(
                    f"{self.__base_url}/{method}",
                    json=json_body,
                    data=form,
                    files=files,
                    timeout=timeout or self._request_timeout,
                )
            except httpx.HTTPError as exc:
                # httpx includes the URL in many exception strings. The URL contains the
                # bot token, so replace it at the API boundary before an observer logs it.
                raise TelegramAPIError(
                    f"Telegram request failed: {type(exc).__name__}", temporary=True
                ) from exc
        finally:
            if own_client:
                client.close()
        try:
            body = response.json()
        except Exception as exc:
            raise TelegramAPIError(
                "Telegram returned a non-JSON response",
                error_code=getattr(response, "status_code", None),
                temporary=getattr(response, "status_code", 500) >= 500,
            ) from exc
        if isinstance(body, dict) and body.get("ok") is True:
            return body.get("result")
        error_code = body.get("error_code") if isinstance(body, dict) else None
        description = body.get("description") if isinstance(body, dict) else None
        parameters = body.get("parameters") if isinstance(body, dict) else None
        retry_after = parameters.get("retry_after") if isinstance(parameters, dict) else None
        status = getattr(response, "status_code", None)
        code = error_code if isinstance(error_code, int) else status
        raise TelegramAPIError(
            str(description or "Telegram rejected the request")[:1000],
            error_code=code,
            retry_after=float(retry_after) if isinstance(retry_after, (int, float)) else None,
            temporary=code == 429 or (isinstance(code, int) and code >= 500),
        )


class TelegramSender:
    """Translate stored Telegram payloads into classified delivery attempts."""

    _ATTACHMENT_METHODS = {
        "photo": ("sendPhoto", "photo"),
        "document": ("sendDocument", "document"),
        "audio": ("sendAudio", "audio"),
        "video": ("sendVideo", "video"),
        "animation": ("sendAnimation", "animation"),
        "voice": ("sendVoice", "voice"),
    }

    def __init__(self, api: TelegramBotAPI, *, cwd: str | Path | None = None) -> None:
        self.api = api
        self.cwd = Path(cwd or ".").resolve()

    def send(self, item: OutboxItem) -> DeliveryOutcome:
        try:
            result = self._send(item)
        except TelegramAPIError as exc:
            if exc.temporary:
                return DeliveryOutcome.retry(str(exc), retry_after=exc.retry_after)
            return DeliveryOutcome.failed(str(exc))
        except (OSError, httpx.HTTPError) as exc:
            # HTTP exception strings can contain the request URL (and therefore the bot
            # token), so persist only the exception class at this boundary.
            return DeliveryOutcome.retry(f"Telegram request failed: {type(exc).__name__}")
        except Exception as exc:
            return DeliveryOutcome.retry(f"Telegram request failed: {type(exc).__name__}")
        if not isinstance(result, dict) or not isinstance(result.get("message_id"), int):
            return DeliveryOutcome.retry("Telegram returned no message ID")
        return DeliveryOutcome.delivered(str(result["message_id"]))

    def _send(self, item: OutboxItem) -> Any:
        payload = item.payload
        kind = payload.get("kind")
        form: dict[str, Any] = {"chat_id": item.destination}
        reply_to = payload.get("reply_to_message_id")
        if reply_to is not None:
            form["reply_parameters"] = json.dumps(
                {"message_id": reply_to, "allow_sending_without_reply": True},
                separators=(",", ":"),
            )
        if kind == "text":
            form["text"] = payload["text"]
            return self.api.request("sendMessage", form=form)
        if kind != "attachment":
            raise TelegramAPIError(f"Unsupported outbox payload kind {kind!r}")
        attachment_kind = payload.get("attachment_kind")
        if attachment_kind not in self._ATTACHMENT_METHODS:
            raise TelegramAPIError(f"Unsupported attachment kind {attachment_kind!r}")
        method, field = self._ATTACHMENT_METHODS[attachment_kind]
        source = payload.get("source")
        if not isinstance(source, str) or not source:
            raise TelegramAPIError("Attachment source must be nonempty")
        caption = payload.get("caption")
        if caption:
            form["caption"] = caption
        path = Path(source).expanduser()
        if not path.is_absolute():
            path = self.cwd / path
        if path.is_file():
            with path.open("rb") as stream:
                return self.api.request(
                    method, form=form, files={field: (path.name, stream)}
                )
        form[field] = source
        return self.api.request(method, form=form)
