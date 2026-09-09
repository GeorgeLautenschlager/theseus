# Headless Autocore for scheduled agents

Knope runs on the Linux server without chat and checks the community forum four
times daily. Support `InterfaceSpec("none")` for Autocore, with no observer or
reply tool. Continuous headless execution runs the core on the main thread.

Expose the existing single-turn body as `Autocore.step()` without sleeping;
`loop()` remains step + sleep forever. An agent's external scheduler can invoke
bounded turns without a custom core subclass. OODA still requires an observer.

Unsloth availability must check the requested model ID, not merely server
reachability. Knope's requested model is not currently the one advertised by
the configured server, so this is necessary for her explicit fallback to work.
