# Web Chat Notifications — Design

Date: 2026-07-26
Status: approved, not yet implemented

## Problem

An agent can already speak into the web chat UI at any moment: `WebChat.respond`
calls `WebChatUIObserver.publish_assistant_chunk`, which fans HTML fragments out
over SSE to every connected browser tab. A message that no user asked for renders
correctly today — the `is_first` branch of `_assistant_reply_fragment.html`
creates the bubble, and the `is_done` branch re-enables a composer that was never
disabled.

What is missing is any signal the operator can perceive when they are not looking
at the tab. A spontaneous message lands silently in a background tab and is
noticed minutes or hours later.

This design adds that signal: a desktop notification plus an unread count in the
document title, raised whenever a completed agent reply arrives while the page is
unfocused.

## Scope

In scope: delivery of the perceptible signal.

Out of scope: anything that wakes the agent. Nothing in this design calls
`orient()`. A heartbeat tick, an agent-scheduled wakeup, and an external trigger
endpoint were all considered and deliberately deferred — until one of them exists,
agent-initiated messages only happen when something outside this design orients
the core.

Also out of scope: Web Push (service worker + VAPID + subscription persistence),
which would reach the operator with no tab open at all. The chosen tier requires a
tab to be open somewhere; it just does not need to be focused.

## Constraint: secure context

The Notification API is only available in a secure context. It works on
`http://localhost:<port>` and over HTTPS. It is `undefined` when the same server
is reached at a LAN address over plain HTTP.

This matters concretely: Tam serves on `0.0.0.0:1337`, so the UI is reachable both
as `http://localhost:1337` (notifications available) and as
`http://<lan-ip>:1337` (notifications unavailable). The design therefore treats
the notification as a capability that may be absent, and keeps the title unread
count — which works in every context — as the baseline signal.

## Approach

Three ways to get notification data from server to client were considered:

1. **A hidden out-of-band element inside the existing reply fragment.** Chosen.
2. A separate SSE event type carrying JSON. Cleaner separation, but
   `_sse_stream` and `_format_sse_event` currently assume every queued item is an
   HTML fragment, and htmx's SSE extension only swaps HTML — so it needs either a
   typed listener queue or a second `EventSource` per tab. More machinery than the
   payload justifies.
3. A client-side `MutationObserver` over `#messages` with no server change.
   Rejected: replies stream in word by word, so "the reply is finished" has to be
   inferred with a debounce, which risks firing mid-sentence or twice.

Approach 1 also lets the notification preview be computed server-side from the
**raw** reply text, before markdown rendering, so the client never has to strip
HTML out of a rendered bubble to find a plain-text summary.

## Server contract

### `src/theseus/web/preview.py` (new)

One pure function, no dependencies on the observer or FastAPI:

```python
def notification_preview(text: str, limit: int = 140) -> str: ...
```

Behavior:

- Collapses all runs of whitespace (including newlines) to single spaces.
- Strips leading markdown noise from lines: `#` heading markers, `-`/`*` list
  bullets, and backticks.
- If the result exceeds `limit` characters, truncates at the last word boundary
  at or before `limit` and appends `…`.
- Returns `"New message"` for empty or whitespace-only input.

It lives in its own module because it is the one piece of this feature with
interesting behavior worth testing directly.

### `WebChatUIObserver.publish_assistant_chunk`

Unchanged in signature and in every existing behavior. One addition: when
rendering `_assistant_reply_fragment.html`, pass
`notify_preview=notification_preview(text)`. Computing it on every chunk is
harmless; only the `is_done` render uses it.

### `_assistant_reply_fragment.html`

Inside the existing `{% if is_done %}` block, alongside the status and composer
out-of-band swaps, emit:

```html
<span id="chat-notify" hx-swap-oob="true"
      data-msg-id="{{ bubble_id }}"
      data-preview="{{ notify_preview }}"></span>
```

Jinja's autoescaping handles the attribute values; quotes and angle brackets in a
reply cannot break out of the attribute.

### `chat.html`

Carries an empty `<span id="chat-notify"></span>` so the out-of-band swap always
has a target on a freshly loaded page, and a bell button in the header (see
below).

No new endpoint, no change to `_sse_stream` or `_format_sse_event`.

## Client behavior

All of the following lives in `src/theseus/web/static/chat.js`, feature-detected
so that a missing `Notification` global never throws.

### Bell toggle

A button in the header, immediately left of the Debug link, styled to match
`.debug-link`. Its state is derived from `Notification.permission` and a
preference stored in `localStorage` under `theseus.notifications`:

| Condition | Appearance | Click behavior |
| --- | --- | --- |
| `Notification` undefined (insecure context) | dimmed, `title` explains notifications need localhost or HTTPS | no-op |
| permission `"default"` | outline bell | calls `Notification.requestPermission()`; on grant, sets the preference to on |
| permission `"granted"`, preference on | filled bell | mutes (preference off) |
| permission `"granted"`, preference off | muted outline bell | unmutes (preference on) |
| permission `"denied"` | struck-through bell, `title` says to unblock in browser settings | no-op |

The bell glyph is an inline SVG in `chat.html` — no external asset, no icon font,
consistent with the page's existing zero-image styling. The five states above are
expressed with CSS classes toggled on the button (`is-on`, `is-off`, `is-denied`,
`is-unavailable`), not by swapping markup.

The preference defaults to on the first time permission is granted. Requesting
permission from a click satisfies the user-gesture requirement that makes Chrome
auto-deny bare on-load requests.

### Notification on arrival

The existing `htmx:oobAfterSwap` listener gains a check: if `#chat-notify` carries
a `data-msg-id` that has not been seen before (tracked in a module-level `Set`,
which also guards against a repeated swap double-firing), then:

- If `document.hasFocus()` — do nothing. The operator is looking at the page.
- Otherwise: increment the unread counter, set
  `document.title = "(" + unread + ") Theseus Chat"`, and — only if
  `Notification` exists, permission is `"granted"`, and the preference is on —
  raise `new Notification("Theseus", { body: preview, tag: msgId })`.

`tag` set to the message id is what keeps multiple open tabs from stacking
duplicate notifications: every unfocused tab raises one, and the browser collapses
same-tag notifications into a single visible entry.

Clicking the notification calls `window.focus()` and closes it.

### Clearing

On `window` `focus` and on `visibilitychange` to visible: reset the unread counter
to 0 and restore `document.title` to `"Theseus Chat"`.

### Timing

Only the `is_done` fragment carries `#chat-notify`, so the word-by-word streaming
in `WebChat.respond` never raises a notification mid-sentence.

## Edge cases

- **Insecure context / no API** — bell renders unavailable, title counter still
  works, nothing throws.
- **Multiple tabs** — every unfocused tab notifies; `tag` collapses them to one.
  Each tab counts its own unread total, which is correct since each has its own
  title.
- **Permission denied** — notification skipped, title counting continues.
- **SSE reconnect** — the server replays nothing on reconnect, so no stale
  notification; the seen-id set covers any duplicate swap.
- **No tabs open when the agent speaks** — unchanged: the reply is appended to
  `self.transcript` and appears on the next page load. That transcript is
  in-memory only and is lost on agent restart; out of scope here.
- **Empty or whitespace-only reply** — preview falls back to `"New message"`.

## Testing

### Automated (pytest)

- `tests/test_notification_preview.py` — the pure helper: whitespace collapse,
  markdown-noise stripping, truncation at a word boundary with `…`, short text
  passing through untouched, empty and whitespace-only input returning
  `"New message"`.
- `tests/test_web_notify_fragment.py` — Jinja fragment rendering, following the
  idiom in `tests/test_debug_row_rendering.py` (load `Jinja2Templates` directly
  against the packaged templates directory):
  - `#chat-notify` is absent when `is_done=False`.
  - `#chat-notify` is present with the correct `data-msg-id` when `is_done=True`.
  - A preview containing `"` and `<script>` is escaped in the attribute and does
    not appear raw.

Both are offline and join the default `make test` suite.

### Manual (no JS test infrastructure exists in this repo)

Run an agent with the web observer on `http://localhost:<port>` and confirm:

1. Bell starts as outline; clicking it produces the browser permission prompt;
   granting turns it filled.
2. Send a message, switch to another application before the reply lands — a
   desktop notification appears with the reply's opening words, and the tab title
   shows `(1) Theseus Chat`.
3. Focus the tab — the title resets to `Theseus Chat`.
4. Reply while the tab is focused — no notification, no title change.
5. Two tabs open, both unfocused — exactly one OS notification appears.
6. Click the bell to mute — a subsequent unfocused reply updates the title but
   raises no notification.
7. Load the same server at `http://<lan-ip>:<port>` — the bell shows unavailable,
   nothing throws in the console, and the title counter still increments.

## Files touched

| File | Change |
| --- | --- |
| `src/theseus/web/preview.py` | new — `notification_preview` |
| `src/theseus/web_chat_ui_observer.py` | pass `notify_preview` into the reply fragment render |
| `src/theseus/web/templates/_assistant_reply_fragment.html` | emit `#chat-notify` when done |
| `src/theseus/web/templates/chat.html` | empty `#chat-notify` target, header bell button |
| `src/theseus/web/static/chat.js` | permission/bell state, notification on arrival, title unread count |
| `src/theseus/web/static/chat.css` | bell button styling |
| `tests/test_notification_preview.py` | new |
| `tests/test_web_notify_fragment.py` | new |
