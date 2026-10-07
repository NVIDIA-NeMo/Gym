# NeMo User Sim environment

This environment runs population-grounded, multi-turn user simulation with
NeMo User Sim. Its generated validation set includes one task for each
first-party User Sim probe.

## Generate validation tasks

Preparation runs the pinned User Sim package in an isolated dependency
environment and asks its canonical sampler to materialize one fully resolved
input for every registered probe:

```bash
gym eval prepare --environment nemo_user_sim
```

The resulting `environments/nemo_user_sim/data/nemo_user_sim.jsonl` contains 14 rows. Each
generated row contains a
stable string `task_id` and the complete User Sim row under `resolved_row`.
Gym adds the remaining task metadata when it materializes the dataset for a
run. User Sim owns persona, probe, theme, toolset, locale, configuration,
trajectory identity, and provenance selection; Gym does not maintain sampling
templates or persona panels.

The preparation and runtime environments install public revision
`a5f676bf6dc5a73914c8a0860f97c10dd2c214ee` from
`github.com/NVIDIA-NeMo/UserSim`.
New rows store `usersim_provenance` as JSON text, including when
`bank_version` is empty, so the rows remain Arrow- and Parquet-safe.

## Runtime ownership

The Environment Server creates the sole
`ProbeEpisodeRuntime.from_resolved_row(...)` for each episode. The runtime
owns User Sim's conversation lifecycle, probe state, tools, semantic
indices, and evidence.

The User Agent and Assistant Agent each perform one model activation per
request. For `tool_calling`, `safety_agentic`, and `financial_services`, the
Assistant returns function calls without executing them. User Sim validates and
executes those calls inside the Environment, adds tool results, and
requests another Assistant activation when necessary. Internal
`api_response_model` calls route through `tool_simulation_model`.

The Resources Server validates the materialized row, stores its episode
identity and digest, and verifies the episode result with User Sim's
trajectory evaluator and registered probe scorer. It owns no second runtime
and exposes no probe tool endpoints.

The ordered top-level `invocations` ledger retains User and Assistant Agent
activations plus Judge, Summary, and tool-simulation support-model calls.
`usersim_result` remains authoritative for the complete transcript,
tool calls, tool results, and simulation outcome.

## Models

The User Agent uses `user_model`, the Assistant Agent uses `policy_model`, and
Judge, Summary, tool simulation, and verification use `support_model`. These
are independently configured Model Servers, although their upstream endpoints
may match when one provider satisfies multiple roles.

## Run

```bash
gym eval run \
  --environment nemo_user_sim \
  --split validation \
  --output results/nemo_user_sim.jsonl \
  +user_model_uses_reasoning_parser=false \
  +policy_uses_reasoning_parser=false \
  +support_model_uses_reasoning_parser=false \
  ++observability_enabled=true \
  ++model_call_capture_dir=/absolute/path/to/model-calls
```

For participant-specific SFT or custom collation, filter
top-level `invocations` by the `assistant` or `user` role and use each selected
invocation's exact `request` and `response`.
