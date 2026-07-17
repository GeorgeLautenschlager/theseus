# Tiered Windows Surrogate — screen awareness as a visual organ for the primary agent

## Context

The `windows_surrogate.py` PoC (repo root) proved a Windows box can watch its own screen
with a local VLM in near-real-time. Measured envelope (LM Studio box over direct
Tailscale LAN): client capture+encode ~30 ms; server fixed overhead ~70–90 ms/request;
vision encoder ~100 ms (e2b) / ~190 ms (e4b) regardless of image size; decode 9–17
ms/token; sustained 7–8 fps with 4 in flight. The PoC sends every frame to the VLM and
only prints. This work turns it into the promised "sensory organ" module family
(README: "Gemma 4 E4B intelligently feeding keyframes to a larger model"): a tiered
pipeline where cheap signals gate VLM calls, a bigger model escalates locally, and
qualifying escalations reach the remote primary agent (Tam-style, Linux LAN server) as
StimulusLog events. PC-side action-taking is out of scope this session.

**Decisions locked with the user:** Tier 1 = Win32 foreground-window metadata (ctypes,
no new deps) + dhash frame-diff gating (PIL only) + tuned e2b constrained-label VLM on
change/heartbeat. Tier 3 payload = text in the event + keyframe JPEG spooled to disk,
referenced by relative path (ContextAssembler feeds verbatim event JSON into prompts →
**never base64 in event content**). Scope = all three tiers + FastAPI ingress observer +
in-repo demo primary for E2E verification.

**Architecture seams (verified in source):** `CognitiveCore.orient()` is zero-arg —
observers `stimulus_log.append(...)` then trigger orient; only one orient in flight
(loop_memory not concurrency-safe). `Memory` protocol ([memory.py](c:\dev\theseus\src\theseus\memory.py))
is the pluggability template. `WebChatUIObserver` is the HTTP-ingress/thread template.
`src/theseus/surrogates/__init__.py` exists empty; `[tool.poetry.group.surrogates]`
(Pillow) exists. Demo primary builds on `CognitiveCore` (per `tests/test_cognitive_core.py`),
NOT the legacy `ChatCognitiveCore` Aldric still uses.

## Design overview

Client half (Windows PC) and server half (primary's ingress), both in
`src/theseus/surrogates/`. Providers are injected everywhere; concrete model choices
(e2b tier 1, e4b/26b tier 2) live only in the two `main()` entry points.

```
tick @ target_fps (default 2)
  → WindowMetadataSource.sample()          (ctypes, sub-ms)
  → ScreenSource.capture()                 (PIL, ~30ms → Frame: jpeg, data_uri, dhash)
  → FrameGate.should_process()             (dhash distance / heartbeat; process-change forces open)
  fired → TierOneLabeler.label()           (e2b, constrained label, max_tokens=16, temp 0)
        → EscalationPolicy.evaluate()      (pure) → TierTwoTrigger?
  trigger → TierTwoEscalator.describe()    (e4b/26b, json_schema structured report)
          → ForwardingPolicy.evaluate()    (pure) → forward?
  forward → EscalationSink.deliver(Escalation)   (protocol; HTTP sink POSTs to ingress)
ingress → spool JPEG → append text-only event → coalesced core.orient()
```

## Modules (`src/theseus/surrogates/`)

All start `from __future__ import annotations`; heavy deps (PIL, httpx) imported lazily
inside methods so the package imports on a bare Linux install.

- **`frame_gate.py`** (pure, zero deps): `dhash(gray_pixels, width=9, height=8) -> int`
  (64-bit row-gradient), `hamming_distance(a, b)`, `GateDecision(fire, reason, distance)`
  with reasons `first_frame|changed|heartbeat|quiet`, `FrameGate(distance_threshold=6,
  heartbeat_seconds=30.0).should_process(frame_hash, now)` — clock injected, stateful,
  deterministic.
- **`screen_source.py`**: `Frame(captured_at, wall_time, jpeg_bytes, data_uri, dhash)`
  dataclass; `ScreenSource(image_long_side=448, jpeg_quality=70).capture()` absorbs the
  PoC's `_grab_screen` + computes dhash from a 9×8 grayscale thumbnail.
- **`window_metadata.py`**: `ForegroundWindow(title, process_name, pid)`;
  `WindowMetadataSource.sample()` via ctypes (`GetForegroundWindow`, `GetWindowTextW`,
  `GetWindowThreadProcessId`, `OpenProcess` + `QueryFullProcessImageNameW`). All failure
  paths degrade to `""`, never raise; `ctypes.windll` touched only inside `sample()`.
- **`tier_one_labeler.py`**: `TierOneLabeler(model_provider, max_tokens=16)`; the PoC's
  tuned constrained-label system prompt; user prompt includes window process+title (near-
  free prefill, stabilizes labels); `temperature=0.0`, `images=[frame.data_uri]`.
- **`tier_two_escalator.py`**: `ESCALATION_JSON_SCHEMA` — six required fields, ordered
  rationale-before-commit: `description`, `activity`, `user_goal`,
  `assistance_rationale`, `assistance_opportunity` (bool), `salience` (0–10 int).
  `TierTwoReport` frozen dataclass; pure `parse_tier_two_response(raw)` (uses
  `theseus.json_utils.parse_json_response`, `ValueError` on bad shape);
  `TierTwoEscalator(model_provider, max_tokens=1024).describe(frame, window, trigger)` —
  prompt carries trigger reason + previous/current labels + window metadata. (max_tokens
  1024 measured, not guessed: server models reason ~250–770 hidden tokens before the JSON;
  512 truncates on busy frames. See findings.)
- **`escalation_policy.py`** (pure, clockless — the criteria brain):
  - `EscalationPolicy(label_change_confirmations=2, dwell_seconds=600, big_change_distance=24,
    min_escalation_interval_seconds=20).evaluate(TierOneSnapshot) -> TierTwoTrigger | None`.
    Priority order, one reason per fire, interval floor applies, any fire resets dwell:
    (1) `window_changed` — process name differs (immediate; title-only rides along in the
    tier-2 prompt); (2) `label_changed` — normalized label differs for N consecutive
    tier-1 results (debounces VLM phrasing jitter); (3) `big_visual_change` — gate
    distance ≥ threshold with unchanged label (dialogs/popups); (4) `dwell` — ambient
    keep-alive after silence.
  - `ForwardingPolicy(salience_threshold=7, min_forward_interval_seconds=30,
    activity_change_cooldown_seconds=180).evaluate(report, now) -> str | None`:
    forward on `assistance_opportunity`, on `salience >= threshold`, or on changed
    activity after cooldown; global interval floor.
- **`escalation_sink.py`**: `Escalation` frozen dataclass (captured_at, trigger,
  tier1_label, window title/process, `TierTwoReport`, `frame_jpeg: bytes`) with
  `to_wire()/from_wire()` — the single wire codec; `EscalationSink` Protocol (Memory
  template: "the surrogate only signals the sink; transport is the sink's business;
  never raise into the loop"); `ConsoleEscalationSink` for standalone runs.
- **`http_escalation_sink.py`**: `HttpEscalationSink(url, auth_token=None,
  timeout_seconds=10)` — lazy `import httpx`, `X-Theseus-Token` header when configured,
  catches everything, `_send(...)` as the test override point.
- **`ingress_observer.py`**: `SurrogateIngressObserver(orient_callback, stimulus_log,
  frame_spool_dir, auth_token=None, actor="windows_surrogate")` + `.serve(host, port=8800)`.
  Routes: `POST /surrogate/escalation` (token check via `secrets.compare_digest` → 401;
  `Escalation.from_wire` → 422 on bad payload; JPEG → `frames/<new_id()>.jpg` using the
  ULID helper from `stimulus_log.py` so filenames time-sort; append event; request
  orient; 202 with event_id + frame path) and `GET /health`. Appended event:
  `actor="windows_surrogate"`, `type="surrogate_observation"`, content = the six report
  fields + trigger + tier1_label + window {title, process} + `"frame": "frames/<id>.jpg"`
  + captured_at ISO. **Orient coalescing:** lock + running/pending flags; worker thread
  loops while pending was set during a run — events always append, triggers coalesce,
  orients never overlap (matches the zero-arg log-driven contract).
- **`windows_surrogate.py`** (orchestrator + entry point): `WindowsSurrogate(tier_one,
  tier_two, sink, screen_source=None, window_source=None, frame_gate=None,
  escalation_policy=None, forwarding_policy=None, target_fps=2.0, max_in_flight=2)`.
  Threading (PoC discipline extended): paced main loop samples window + captures +
  gates; process-change forces gate open. Tier-1 calls run on a
  `ThreadPoolExecutor(max_in_flight)` behind a non-blocking semaphore — saturated ⇒
  DROP frame, count it. `_label_and_evaluate` discards stale out-of-order completions
  (`frame.captured_at <= last_labeled_at` under a state lock), else snapshots + runs
  policy. Tier-2 guarded by a `Semaphore(1)` — busy ⇒ drop trigger (next change
  re-fires). `_escalate` runs describe → forwarding policy → `sink.deliver` inline;
  `ValueError` from bad JSON ⇒ print + drop; always releases the slot. Console lines in
  the PoC style (latency, fps, drops, label; distinct tier-2 and `FORWARDED (<reason>)`
  lines). `main()`: argparse — `--fps --duration --ingress-url --tier-one-model
  --tier-two-model --console-sink`; wires `LmStudioProvider(e2b)` /
  `LmStudioProvider(e4b)` / `HttpEscalationSink(url, token from THESEUS_INGRESS_TOKEN)`.
- **`__init__.py`**: curated exports of all public names above. Top-level
  `theseus/__init__.py` unchanged (precedent: `WebChatUIObserver` isn't top-level either).

## Wire format (deliberate: NOT `StimulusEvent.to_json()`)

Versioned POST body `{"v": 1, "source", "captured_at", "trigger", "tier1_label",
"window": {...}, "report": {...six fields}, "frame_jpeg_base64"}`, encoded/decoded only
by `Escalation.to_wire()/from_wire()`. Rationale: (1) event ids/ts must be minted by the
receiving log (ULIDs are its ordering primitive — remote minting imports PC clock skew);
(2) the wire must carry the frame, the event must not; (3) the ingress owns the event
shape.

## Demo primary (`src/theseus/agents/demo_primary.py`)

`CognitiveCore` + `SurrogateIngressObserver` + one `AcknowledgeEffector`
(`name="acknowledge_surrogate"`, execute prints `[demo-primary] <payload>`). argparse:
`--data-dir` (default `./demo_primary_data`), `--port` 8800, `--host`, `--model`
(default e4b). Constitution: brief "you observe George's PC through a surrogate" text.
No web chat UI.

## pyproject / repo changes

- `[project.scripts]`: `windows-surrogate = "theseus.surrogates.windows_surrogate:main"`,
  `demo-primary = "theseus.agents.demo_primary:main"`.
- httpx `>=0.27,<1` added to `[tool.poetry.group.surrogates.dependencies]` and dev group
  (starlette TestClient needs it). Verified already in poetry.lock transitively via
  openai — nothing new installs; declaring direct use is hygiene. NOT in main deps.
- Root `windows_surrogate.py`: absorbed and deleted in the landing commit (capture →
  screen_source, prompt → tier_one_labeler, loop discipline → orchestrator; git history
  preserves the PoC).

## Tests (all offline — no PIL, no network, no win32 required)

`tests/test_frame_gate.py` (dhash on hand-built grids, gate fire/quiet/heartbeat with
injected clock) · `test_escalation_policy.py` (debounce, jitter A→B→A no-fire,
big-change, dwell, interval floors; ForwardingPolicy thresholds/cooldowns) ·
`test_tier_one_labeler.py` + `test_tier_two_escalator.py` (MagicMock provider stubs per
`test_cognitive_core.py`'s `make_provider` pattern; assert kwargs, prompt contents,
parse errors) · `test_escalation_sink.py` (wire round-trip; `_send`-override sink:
token header, swallow-never-raise) · `test_surrogate_ingress_observer.py` (FastAPI
TestClient + tmp_path log: 202/401/422, spool file lands, event content small with no
base64, coalescing under two rapid POSTs with a slow orient — runs exactly twice, never
concurrent) · `test_windows_surrogate.py` (orchestrator internals synchronously with
fakes: gate-quiet ⇒ no label call, label change ⇒ escalate, busy tier-2 slot ⇒ drop,
forward ⇒ sink got jpeg bytes, bad JSON ⇒ slot released) · optional skipif-win32
`test_window_metadata.py` smoke.

## Docs

- `docs/superpowers/specs/2026-07-17-windows-surrogate-design.md` (Status: Implemented,
  lands with code; agentic-memory spec shape): Problem / The tiered pipeline (with the
  measured envelope as rationale) / Decisions (ctypes not psutil; dhash not numpy;
  drop-not-queue; injected providers; pure clockless policies; wire ≠ StimulusEvent;
  spool + no-base64 invariant; sink as Memory-template seam; orient coalescing;
  shared-token auth) / Components / Wiring example / Non-goals (PC-side effectors,
  spool GC, multi-monitor, audio, cross-observer orient serialization) / Consequences.
- `docs/notes_on_organization.md` — create it (CLAUDE.md references it; it doesn't
  exist): construction-kit vision + vocabulary (observer / surrogate–organ / effector /
  memory), citing Aldric's "sensory surrogates feeding pre-processed information into
  the core".

## Verification

1. Offline: `poetry install --with surrogates` then `make test` — entire new suite runs
   without endpoints.
2. Live E2E (LM Studio box with e2b + e4b loaded, thinking disabled, full GPU offload):
   Terminal A `poetry run demo-primary --port 8800`; Terminal B
   `poetry run windows-surrogate --ingress-url http://127.0.0.1:8800/surrogate/escalation --fps 2`.
   Working = labels print only on change/heartbeat; switching apps produces a tier-2
   report and `FORWARDED (activity_changed)`; demo-primary prints a decision rationale
   referencing the on-screen activity and an `[demo-primary]` acknowledgement;
   `demo_primary_data/frames/*.jpg` accumulates; every stimulus-log line is small JSON
   with a `frames/...jpg` reference and zero base64.

## Implementation order

1. `docs/notes_on_organization.md` + spec skeleton.
2. `frame_gate.py` + test (pure foundation).
3. `window_metadata.py` (+ skipif smoke test).
4. `screen_source.py`.
5. `tier_one_labeler.py` + test; 6. `tier_two_escalator.py` + test;
7. `escalation_policy.py` + test; 8. `escalation_sink.py` + `http_escalation_sink.py` + test.
9. `ingress_observer.py` + test. 10. `windows_surrogate.py` orchestrator + test.
11. `agents/demo_primary.py`; pyproject scripts + httpx; `surrogates/__init__.py`
    exports; delete root PoC; `poetry lock && poetry install --with surrogates`.
12. Finalize spec; offline suite; live E2E.

## Derisking experiment findings (this session, against the live LM Studio box)

- **Tier-1 label jitter — RESOLVED, validates design.** e2b at temp 0 is fully
  deterministic (12/12 identical labels on one frame), so the debounce protects against
  small *frame* changes, not sampling noise. Window metadata in the prompt **corrected a
  wrong label** (without it e2b called VS Code "Browsing computer files"; with it,
  "Coding and optimizing code"). Keep window metadata in the tier-1 prompt — confirmed.
- **Tier-2 structured output — MAJOR FINDING, changes defaults.** All vision models on
  the server currently run with **thinking ON**, and it dominates: e4b spends 400–770
  hidden reasoning tokens per call. Consequences measured: with `max_tokens=512` (the
  original default) reasoning eats the whole budget → truncated/empty JSON (the observed
  failure); `max_tokens=1024` yields valid, correctly-ordered six-field JSON but costs
  **~7–8.6s**. The `enable_thinking:false` chat-template kwarg had **no effect** (server
  must toggle it). Model comparison at 1024 budget, image + strict schema: e4b ~8s but
  semantically confused (salience 0, thought the screen was "the target of the
  surrogate's automation"); 26b-a4b MoE accurate ("VS Code running windows_surrogate.py…
  several lines of Python") but **~26s** and it **evicted e4b on load** (co-residency
  risk confirmed); 12b never stops thinking (1021 reasoning tokens, JSON never emitted).
  → **Plan change:** the original `max_tokens=512` tier-2 default is wrong — reasoning
  eats it. Two re-measurements after server-side toggle attempts still showed reasoning
  tokens essentially unchanged (247–289), so **e4b's thinking is not effectively
  disable-able on this server build**; treat tier-2 models here as thinking-on and give
  the budget headroom rather than fighting it. Usable regime found:
  `max_tokens=512` yields valid, correctly-ordered six-field JSON in ~4.8s — BUT reasoning
  length varies with screen content (measured 247–770 tokens), so 512 truncates on busy
  frames. **Final tier-2 defaults (thinking-tolerant):** `TierTwoEscalator(max_tokens=1024)`
  (headroom above the ~800-token reasoning spikes), tier-2 latency budget ~5–8s,
  `min_escalation_interval_seconds=20`, `min_forward_interval_seconds=30` (one call in
  flight via the tier-2 `Semaphore(1)`, comfortably longer than an 8s call). e4b's
  structured descriptions stay marginal (observed a hallucinated panel); the accurate
  describer was the 26b-a4b MoE but at ~26s. Because the tier-2 provider is **injected**,
  the model is a wiring choice — the demo uses e4b; production can swap in the 26b (or a
  better-configured thinking-off endpoint) without touching library code.
- **dhash calibration — RESOLVED, thresholds validated.** 60s active capture (2fps,
  real typing/scrolling/app-switching) sweeping 9×8 (64-bit) and 17×16 (256-bit):
  the 9×8 hash separates the bands cleanly on its own, so 256-bit is unnecessary — keep
  9×8. Distribution (9×8): 44% of frames exactly 0 even during active use (idle/between
  keystrokes → **heartbeat is essential**); typing/minor edits at distance 1–4; real
  content changes 6–21; app switches / full repaints cluster 26–42 (p90=26, p95=28,
  p99=37). → `FrameGate(distance_threshold=6, heartbeat_seconds=30)` and
  `big_change_distance=24` are all validated by the measured p-values (6 sits above the
  1–4 typing-noise floor; 24 sits at p90 for genuine app-switch events).

## Known risks (accepted, flagged in spec)

Frame spool unbounded (GC later) · `ImageGrab` is primary-monitor-only · process name
blank for elevated windows · a primary composing this ingress alongside the web chat
observer has two independent orient serializers (pre-existing gap, non-goal) · sustained
motion (video) keeps the gate open — bounded by `target_fps=2` and the in-flight ceiling.

**Resolved by this session's experiments (no longer risks):** tier-2 schema compliance
(proven — valid ordered JSON at `max_tokens=1024`); label jitter (e2b deterministic at
temp 0; window metadata corrects labels); dhash thresholds (calibrated from active data).
**Confirmed real, now designed-for:** tier-2 thinking tax (~5–8s, budget 1024) and
model co-residency eviction (loading a second model evicts the first → keep the tier-1
and tier-2 models both resident, or accept a multi-second reload on first tier-2 call).
