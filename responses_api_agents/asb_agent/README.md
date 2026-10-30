# Agent Security Bench agent

This Responses API agent reproduces ASB's plan-then-execute loop. It asks the policy model
for a JSON workflow, executes the named simulated tools against the ASB resources server,
records tool observations and memory retrieval, and sends the completed trajectory to the
verifier.

The adapter records whether a plan used upstream's strict JSON parser or a disclosed salvage
path. It also normalizes parameterless tool schemas for the Responses API and merges ASB's
two consecutive system messages so supported model templates receive the same text.

See [`benchmarks/asb/README.md`](../../benchmarks/asb/README.md) for benchmark preparation
and [`benchmarks/asb/METRICS.md`](../../benchmarks/asb/METRICS.md) for protocol deviations.
