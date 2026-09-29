# Implementation plan

1. Add shared ambient wake policy and apply to ingress and default Autocore filter.
2. Add injected desktop snapshot producer, change/heartbeat gating, and lifecycle.
3. Add lazy Win32 source with explicit ctypes signatures and unknown fallbacks.
4. Wire observation into Windows entry point and desktop state tool into both hosts.
5. Verify gating, failures, stop/restart, replication, both wake paths, freshness,
   import safety, and existing surrogate/core tests. Document Windows smoke checks.

## Verification

Implemented all five steps. Offline suite: 1,299 passed, 4 skipped (excluding the
live model fact-retention suite). Final focused tests after wiring assertions and
wake-filter cleanup: 61 passed, 1 skipped. FastAPI tests required execution outside
the sandbox because its async thread bridge stalled inside it. `git diff --check`
passed. Real Windows smoke checks remain manual; see `docs/ambient-desktop.md`.
