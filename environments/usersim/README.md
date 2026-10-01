# NeMo UserSim environment

This environment runs population-grounded, multi-turn user simulation with
NeMo UserSim. Its example suite contains one prepared task for every
first-party UserSim probe.

## Prepare tasks

Preparation runs the pinned UserSim package in an isolated dependency
environment and asks its canonical sampler to materialize one fully resolved
input for every registered probe:

```bash
gym eval prepare --config environments/usersim/config.yaml
```

The generated `environments/usersim/data/example.jsonl` wraps each UserSim row
in a Gym task envelope without changing its content. UserSim owns persona,
probe, theme, toolset, locale, configuration, trajectory identity, and
provenance selection.

Preparation and runtime pin revision
`8fd3ff4798af4037e164364f6c59ad14b4eabdc0` over Git+SSH from
`github.com/NVIDIA-NeMo/UserSim`. The host therefore needs GitHub SSH access.
New rows store `usersim_provenance` as JSON text, including when
`bank_version` is empty, so the rows remain Arrow- and Parquet-safe.

## Runtime ownership

The Environment Server creates the sole
`ProbeEpisodeRuntime.from_resolved_row(...)` for each episode. The runtime
owns UserSim's conversation lifecycle, probe state, tools, native semantic
indices, and evidence.

The User Agent and Assistant Agent each perform one model activation per
request. For `tool_calling`, `safety_agentic`, and `financial_services`, the
Assistant returns function calls without executing them. UserSim validates and
executes those calls inside the Environment, adds native tool results, and
requests another Assistant activation when necessary. Internal
`api_response_model` calls route through `tool_simulation_model`.

The Resources Server only validates the immutable prepared row, stores its
episode identity, and verifies the authoritative native result with UserSim's
trajectory evaluator and registered probe scorer. It owns no second runtime
and exposes no probe tool endpoints.

The ordered `result.invocations` ledger retains User and Assistant Agent
activations plus Judge, Summary, and tool-simulation support-model calls.
UserSim's native result remains authoritative for the complete transcript,
tool calls, tool results, and simulation outcome.

## Models

The User Agent uses `user_model`, the Assistant Agent uses `policy_model`, and
Judge, Summary, tool simulation, and verification use `support_model`. These
are independently configured Model Servers, although their upstream endpoints
may match when one provider satisfies multiple roles.

## Run

```bash
gym eval run \
  --environment usersim \
  --split example \
  --output results/usersim.jsonl \
  +user_model_uses_reasoning_parser=false \
  +policy_uses_reasoning_parser=true \
  +support_model_uses_reasoning_parser=false \
  ++observability_enabled=true \
  ++model_call_capture_dir=/absolute/path/to/model-calls
```

For participant-specific SFT or custom collation, filter
`result.invocations` by the `assistant` or `user` role and use each selected
invocation's exact `request` and `response`.
