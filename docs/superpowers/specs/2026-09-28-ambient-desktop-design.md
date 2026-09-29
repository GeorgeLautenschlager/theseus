# Ambient desktop awareness

Observe foreground window (handle, PID, executable name, title), session idle/active,
and locked/unlocked/unknown. Start with one-second polling of the current Windows
session, a 60-second idle threshold, and a 60-second heartbeat. Polling bounds
observation latency but may miss transitions shorter than a sample interval. Sources
are injectable; Win32 libraries load only when sampling Windows. No screenshots or
model calls in this increment.

Publish full `observation.desktop` snapshots on changes and heartbeat, including
version, source, scope, capture time, reason, and field availability (null means
unknown). Compare categorical activity rather than continuously changing idle time.
Failures publish unknown state rather than retaining apparently fresh old state.
Stop observation before stopping replication. `--no-observe-desktop` disables it.

A shared host wake predicate treats this exact event type as ambient. Apply it both
to replication ingress and Autocore's default log wake predicate. Existing custom
wake predicates remain authoritative. Mixed batches still wake for nonambient events,
and duplicates never wake. Ambient events remain durable and available in ordinary
context on subsequent turns.

Register `desktop_state` on both surrogate hosts. It reads the latest snapshot by
origin sequence from the durable log, including after restart, and reports sample
age/staleness (120 seconds) rather than claiming continuous connectivity. Source
clock skew is reported explicitly. No additional mutable state cache is needed yet.

Native lock detection uses WTSSessionInfoEx SessionFlags for the process session on
Windows 10/11; API failure is unknown, not locked. Idle uses GetLastInputInfo.
References: https://learn.microsoft.com/en-us/windows/win32/api/wtsapi32/ns-wtsapi32-wtsinfoex_level1_w
and https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getlastinputinfo.
