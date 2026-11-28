# RoboDojo Environment Server

See [benchmark documentation](../../benchmarks/robodojo/README.md) for runtime
inputs, configuration, grading records, deadline semantics, and evaluation
commands.

This Environment Server (`BaseEnvironmentServer`) binds frozen RoboDojo panel
cases, runs EmberBridge + Nanoverse episodes (single-case or multi-case native
simulator pool), hosts the aiohttp OpenCode Responses compatibility relay, and
records audited native grades via `observe_verifier`.
