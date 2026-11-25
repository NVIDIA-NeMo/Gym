# Description
Runs SynClaw's multi-turn retrosynthesis planning loop using a Gym model server and
the SynClaw resources server.

Data links: [SynClaw](https://github.com/NVIDIA-BioNeMo/SynClaw),
see the *Quick start* and *Task sets* sections of `README.md`.

# Observability
Enable `observability_enabled=true` and set an absolute `model_call_capture_dir`
for both the Gym servers and the rollout collector. Use a separate capture directory
for each concurrent evaluation shard.

The saved traces include model turns, tool calls, execution timings and verification
results. Inspect them with
`python scripts/inspect_harness_conformance.py --bundle <rollouts.jsonl>` alongside
the rollout coverage, scoring and quality reports.

# Licensing information
Code: Apache 2.0
Data: N/A

Dependencies
- nemo_gym: Apache 2.0
- synclaw: Apache 2.0
