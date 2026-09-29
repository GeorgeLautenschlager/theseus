# Ambient Windows desktop awareness

`windows-surrogate` now observes the current Windows session by default. No new
packages are required. Add `--no-observe-desktop` to disable observation (independent
of `--headless`, which only controls the shell).

The observer samples once per second and publishes a full `observation.desktop`
snapshot when the foreground window, title, process, activity, or lock state changes.
It also publishes a heartbeat every 60 seconds. Activity becomes idle after 60
seconds without input. These defaults can be changed when constructing
`DesktopObserver` and `WindowsDesktopSource` in Python.

Snapshots include UTC capture time, origin sequence, source/scope, and a reason.
Unknown fields are null; activity can be `unknown`. Locked snapshots omit window
metadata. Permission failures and unavailable native APIs do not imply a locked or
unlocked session. Non-Windows sources return unknown fields. Sampling can miss
transitions shorter than one second. This increment does not capture screenshots,
keystrokes, or accessibility trees, and does not call a model.

Snapshots use the existing durable log and replication retry path. Both replication
ingress and Autocore's default wake predicate suppress ambient wakes. Chat, command
reports, and replication gap events retain their existing wake behavior. Custom
`wake_on` predicates override the default on each component independently.
Observations remain visible in the log/debug UI and ordinary later-turn context.

Both the reference `surrogate-host` and managed surrogate service register the
`desktop_state` agent tool. It reads the latest snapshot for the configured origin,
including after host restart. It does not request a fresh sample. Results include
sample age and `stale` (older than 120 seconds, unknown timestamp, or future source
clock). A fresh snapshot describes when it was captured; it does not prove the
surrogate is still connected. No observation is returned for other origins.

## Windows smoke check

1. Start the updated host and `windows-surrogate` using the existing connection
   settings. Check the local debug log for an initial `observation.desktop`.
2. Switch between applications and documents. Confirm process/title changes appear
   locally and on the host, without a model turn for each update.
3. Stop input for over a minute, then resume; confirm idle then active snapshots.
4. Lock Windows, wait a few seconds, unlock; confirm locked/unlocked transitions.
5. Ask the agent to use `desktop_state`; check title, age, and lock state. Stop the
   surrogate and check again after two minutes; the stored observation must be stale.
6. Restart the surrogate; confirm a new initial snapshot. Restart with
   `--no-observe-desktop`; confirm chat still works without new observations.

Native calls target Windows 10/11. Linux tests cover the pipeline and simulated
Win32 responses, including resource release and structure layout. Actual desktop,
RDP, elevated-window, and lock/unlock behavior still requires a Windows smoke run.
