# Telegram integration implementation plan

1. Add a transport-neutral SQLite delivery journal plus durable inbox/outbox facades and
   retry outcomes.
2. Add a small Bot API client and sender adapter with temporary/permanent error
   classification, rate-limit handling, file uploads, and remote message IDs.
3. Add `TelegramObserver` polling, allowlist enforcement, message/attachment projection,
   inbox recovery, stimulus dedupe, and orderly outbox draining.
4. Add `TelegramTool` destination validation, replies, attachment planning, lossless
   long-message splitting, atomic group enqueue, and immediate best-effort delivery.
5. Extend the assembly DSL, runtime wiring, docs, and public exports without placing bot
   secrets in generated launchers.
6. Test persistence-before-side-effect, restart recovery on both sides, dedupe, rate-limit
   retries, chunk receipts, attachments/replies, allowlists, and reassembly state retention.
