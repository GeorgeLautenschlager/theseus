# BRIEF: Windows Surrogate — Phase 1 (chat, debug, notifications)

**Project:** Theseus Agent Construction Kit
**Date:** 2026-09-18
**Status:** Specified — not yet implemented
**Builds on:** `docs/superpowers/specs/2026-09-04-surrogate-replication-protocol.md`

---

## Problem

The surrogate *replication protocol* — how a surrogate's stimuli reach a host and how a
host's commands reach a surrogate — is built and tested (`src/theseus/surrogates/`:
`Replicator`, `HttpTransport`, `AckedCursor`, `SseCommandChannel`, `CommandExecutor`,
`Buffer`; host side `replication_ingress.py`, `command_feed.py`, `commands.py`). But there
is **no running surrogate**: no composition, no `main`, no renderer, no UI. And the host
has **no agent-facing tool that emits a command** (`command.say`/`command.notify` are
unimplemented), and `CommandFeed`/`ReplicationIngress` are mounted onto no assembled agent.

Phase 1 makes a Theseus agent a *real presence* on a Windows machine, at the smallest
useful scope: the user chats with the agent, can watch the surrogate's own stimulus log,
and receives native OS notifications. It is necessarily **end-to-end** — the host needs a
mouth for this surface before any of it can be seen.

## Goals

- A runnable **Windows surrogate app**: local `StimulusLog`, upstream replication, a
  command channel, and a renderer that turns host commands into on-screen chat and native
  toasts, plus a text input that produces user stimuli.
- A **debug surface**: a live view of the surrogate's local stimulus log.
- A **host mouth**: agent tools that emit `command.say` / `command.notify` to a target
  surrogate, and a reference host that mounts the ingress + command feed.
- OS integration behind **injectable organ interfaces** so the runtime is testable on Linux
  and the offline suite needs no Windows dependencies.

## Non-goals (later phases)

Screen "vision" organ (the superseded tiered VLM pipeline in
`docs/windows_surrogate_plan.md`); audio capture; STT/TTS + barge-in/VAD; `assembly.py`
integration so any agent (Alty/Tam) can host a surrogate; an Android native app; OS
installer packaging (MSIX/PyInstaller).

## Decisions

- **App shell:** Python edge + a local FastAPI chat/debug UI, wrapped in a **pywebview**
  window + **pystray** tray icon (real Windows app, not "a browser tab"); **native Windows
  toast** notifications. Chosen over a Flutter/.NET rewrite because Python (ctypes +
  mature wrappers) reaches every Win32/WinRT surface the surrogate needs, and the entire
  tested edge is already Python; a UI-framework rewrite would duplicate it for polish, not
  capability. Android is a separate native app under the same wire contract, not shared
  code — so a cross-platform UI framework buys nothing here.
- **OS capabilities are organs behind interfaces** (`Notifier`, `ChatSurface`) with
  console/no-op fallbacks. The runtime core depends only on existing main deps (httpx,
  fastapi, uvicorn, jinja2); pywebview/pystray/windows-toasts live in an optional
  `windows` group, imported lazily.
- **Text-only.** No voice in Phase 1.
- **Location:** in-repo under `src/theseus/surrogates/`, with a `windows-surrogate` entry
  point. A reusable runtime + renderer/organ seams are construction-kit material; the
  surrogate protocol already lives there.
- **Auth:** none app-level — the surrogate↔host link is a Tailscale peer (protocol
  decision #40). The reference host binds to its tailnet interface.
- **User-input stimulus shape** matches the existing chat observers
  (`actor`, `type="chat_message"`, `content={"message": ...}`) so the host's
  `ContextAssembler` reads a replicated surrogate message unchanged.

## Components

### Surrogate side (`src/theseus/surrogates/`)

- **`runtime.py` — `SurrogateRuntime`**: composes a local `StimulusLog(origin)`, two
  distinct `AckedCursor`s (upstream position; command-execution position), a `Replicator`,
  an `SseCommandChannel`, and a `CommandExecutor`. Runs a **drain loop** (an
  `Event`-signalled worker rung by `log.subscribe` on every local append + a periodic
  flush — the `CoalescingTrigger` shape, so drains never overlap and a burst coalesces) and
  the **command loop** (`CommandExecutor.run(channel)`). `submit_user_message(text)`
  appends the user stimulus and rings the drain. `start()`/`stop()` manage threads and
  close the channel.
- **`presence.py`** — pure renderer + seams: `ChatSurface`
  (`publish_agent_message`, `is_focused`), `Notifier` (`notify(title, body)`) +
  `ConsoleNotifier`, and `WindowsPresence(chat, notifier)` — the
  `Renderer = Callable[[StimulusEvent], Outcome]`. Dispatches on `event.type`:
  `command.say` → publish + toast when unfocused; `command.notify` → `notifier.notify`;
  unknown/missing payload → `Failed`. Returns `Executed`/`Failed` from `command_reports`.
- **`web_ui.py`** — local FastAPI app (implements `ChatSurface`), reusing
  `web_chat_ui_observer.py` patterns: `GET /`, `POST /chat` → `runtime.submit_user_message`,
  `GET /events` (SSE agent bubbles), and the `/debug*` routes rendering the **local** log
  (reuse `debug.html`, `_debug_macros.html`, `web/debug_pagination.py`, the 1s poll loop,
  and shared templates under `src/theseus/web/`).
- **`windows/`** — optional organs: `toast.py` (`ToastNotifier` via `windows-toasts`),
  `shell.py` (pywebview window + pystray tray + autostart).
- **`windows_surrogate.py`** — `main()`/argparse: `--host-url`, `--origin`
  (default `windows-desktop`, also the command target), `--ui-port`, log/cursor paths,
  `--headless`. Wires the runtime + transports + `WindowsPresence`; serves the UI (uvicorn
  thread) and runs pywebview on the main thread (Windows requirement); console/no-op organs
  when headless/non-Windows.

### Host side

- **`tools/surrogate_voice.py`** — `SaySurrogate` (`ends_turn=True`, arg `text`) and
  `NotifySurrogate` (`ends_turn=False`, args `title`, `body`), each constructed with the
  target name + the host `StimulusLog`; they append host-origin commands via
  `commands.command_type`/`command_content`. (Multi-surrogate later: add a `target` arg.)
- **`agents/surrogate_host.py`** + `surrogate-host` script — a reference host: a
  `CognitiveCore` + `WebChatUIObserver` app onto which `ReplicationIngress.add_routes` and
  `CommandFeed(log).add_routes` are mounted (one port), with the surrogate tools in the
  tool set and the ingress trigger wired to the core via `CoalescingTrigger`.

### Packaging

`[project.scripts]`: `windows-surrogate`, `surrogate-host`. New optional group `windows`
(pywebview, pystray, windows-toasts). `poetry lock`.

## Data flow

User types → local append (`chat_message`) → `Replicator.drain` → host `/replicate` →
ingress append + coalesced `orient` → agent calls `SaySurrogate` → `command.say` on the
host log → `CommandFeed` SSE → `SseCommandChannel` → `CommandExecutor` → `WindowsPresence`
(bubble + toast) → `Executed` → `command_report.executed` appended → `Replicator.drain`
carries the confirmation home.

## Acceptance scenarios

1. User message in the UI appears on the host log with the surrogate's `origin` and
   triggers exactly one coalesced `orient`.
2. Agent `SaySurrogate` reply reaches the UI as an agent bubble; a
   `command_report.executed` replicates back to the host log.
3. `command.notify` fires the `Notifier` (toast on Windows; console in dev).
4. `command.say` while the window is unfocused also raises a toast.
5. The debug view renders the local stimulus log (user messages, received commands,
   execution reports, any gap markers) and updates live.
6. An unknown command verb or a malformed payload yields a `command_report.failed`, not a
   crash; the command loop continues.
7. Host unreachable: the drain stops cleanly and resumes on reconnect; the command channel
   reconnects from its cursor — no duplicate execution, no lost user message.

## Consequences

- The first real surrogate exists, and the Phase-2/3 organs (screen, audio, STT/TTS) plug
  into the same `SurrogateRuntime` as additional stimulus producers and renderer branches.
- `assembly.py` still does not mount the surrogate endpoints; the reference host is the E2E
  target until that integration lands (follow-up), after which Tam can host a surrogate by
  moving its pin.
- The OS-organ seam keeps the offline suite Windows-free and lets the whole loop be
  exercised headless on Linux.
