# Telegram integration

Theseus gains a Telegram interface made of a `TelegramObserver` and a terminal
`TelegramTool`. The interface uses Telegram long polling and the Bot API, with the bot
token read from a named environment variable at runtime. Agent definitions contain the
environment-variable name and allowlists, never the token value.

## Durable boundary

A reusable SQLite delivery journal lives in the agent runtime home. Its inbox stores the
complete Telegram update before another `getUpdates` request can acknowledge it. Inbox
identity is `(transport, external_id)`, so redelivery is harmless. Processing appends one
`chat_message` stimulus carrying the Telegram update, chat, sender, message, reply, and
attachment metadata. Recovery checks the stimulus log for that update ID before appending,
closing the crash window between the JSONL append and the inbox state transition. The
cognitive callback is at-least-once: a crash after cognition but before marking the inbox
row processed can trigger another turn, but never duplicate the incoming stimulus.

The same journal's transport-neutral outbox persists a group of ordered delivery parts
before any send. Each part records its payload, destination, position, attempts, next
attempt time, status, error, delivered time, and remote message ID. A reusable durable
outbox dispatcher sends oldest-first. Network errors, Telegram 429 responses, and 5xx
responses remain queued; `retry_after` wins over exponential backoff. Permanent Telegram
errors are recorded and do not silently disappear. An interrupted `sending` row returns
to the retry queue on boot. The Bot API has no idempotency key, so a crash after Telegram
accepted a message but before the local delivered commit can produce a duplicate on retry;
the journal deliberately chooses possible duplication over silent loss.

## Messages and access

Incoming text and captions become chat stimuli. Photo, document, audio, video, animation,
voice, and sticker metadata are included without downloading files. Reply relationships
are retained using Telegram message IDs. Both configured allowlists are restrictive: when
user IDs are configured, the sender must match; when chat IDs are configured, the chat
must match. At least one allowlist is required.

The reply tool accepts text, Telegram/local/HTTP attachment sources, a destination chat,
and an optional Telegram message ID to reply to. Text is split without data loss at the
Telegram 4096 UTF-16-unit limit, preferring whitespace boundaries. Every chunk and every
attachment is its own ordered outbox part and therefore has independent delivery status
and Telegram message ID.

## Assembly and operation

`InterfaceSpec("telegram", ...)` selects the interface. Assembly snapshots only safe
configuration. The delivery database stays under the runtime home, so upgrading Theseus
and reassembling Tam preserves pending updates, pending sends, delivery receipts, and all
other state. The observer recovers the inbox and outbox before beginning long polling.
