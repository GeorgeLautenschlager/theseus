# Tam's Windows surrogate

Windows data directory: `C:\dev\surrogate` (confirmed by George).
Origin/command target: `windows-desktop`.

The managed agent uses `SurrogateSpec(host="0.0.0.0", port=8800)` alongside
Telegram. Both transports share Tam's core and active stimulus log. The generated
Compose configuration publishes port 8800 only on the droplet's 127.0.0.1.
The protocol has no authentication of its own; access it through SSH.

In Windows PowerShell, keep this tunnel running (use your authorized SSH key):

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 `
  -L 127.0.0.1:8800:127.0.0.1:8800 root@159.203.30.50
```

From the Windows Theseus source checkout, with Python 3.12 and Poetry installed:

```powershell
poetry env use 3.12
poetry install --with windows
Invoke-RestMethod http://127.0.0.1:8800/health
poetry run windows-surrogate `
  --host-url http://127.0.0.1:8800 `
  --origin windows-desktop `
  --data-dir C:\dev\surrogate
```

The data directory is separate from the source checkout; preserve its log and
cursor files between runs. No OpenRouter credentials are needed on Windows.
The current surrogate supports chat, tray presence, and notifications.

The managed surrogate service starts only after deployment activation and state
preflight. Shutdown signals the HTTP server, bounds SSE draining, stops the
replication trigger, and joins the worker before acknowledging a clean stop.
