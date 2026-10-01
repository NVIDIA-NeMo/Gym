# NeMo UserSim environment

This environment runs population-grounded, multi-turn user simulation with
NeMo UserSim. It includes one example task for each first-party UserSim probe;
the examples exercise the environment's supported interaction shapes.

Environment preparation runs the pinned UserSim package in an isolated
dependency environment and asks its canonical sampler to materialize one fully
resolved input for every registered probe:

```bash
gym eval prepare --config environments/usersim/config.yaml
```

The generated `environments/usersim/data/example.jsonl` wraps each UserSim row
in a Gym task envelope without changing the resolved content. UserSim owns
persona, probe, theme, toolset, locale, configuration, trajectory identity, and
provenance selection; Gym does not maintain sampling templates or persona
panels.

While UserSim is private, the preparation and runtime environments install
revision `b3381ae021baac2a6fb314b5f08a55845243017d` over Git+SSH from
`github.com/NVIDIA-NeMo/UserSim`; the host therefore needs GitHub SSH access.

Each task carries UserSim's resolved row unchanged. The Environment constructs
`ConversationRuntime.from_resolved_row`; Resources independently constructs
`ProbeToolSession.from_resolved_row`. Both receive the same immutable row,
preserving the materialized configuration, trajectory identity, and
provenance.

`tool_calling`, `safety_agentic`, and `financial_services` expose
probe-selected tool schemas to the Assistant Agent. Once per assistant turn,
the Environment invokes the Assistant Agent once. The Agent autonomously
performs every model → named Resources tool endpoint → model round, preserving
raw calls and opaque string payloads. Resources lazily initializes native tool
state, assigns semantic turn/call indices, and returns hidden cumulative
evidence separately from the model-visible payload. The Agent includes that
evidence in its completed trace, and the Environment submits one
`CompletedAssistantTurn` to its local `ConversationRuntime`. There are no
UserSim-specific tool lifecycle endpoints, record-before-tool callbacks, or
Environment orchestration between Assistant model calls. All other probes execute their native UserSim
conversation shape without Assistant tools.
For `identity_disclosure`, the integration gives UserSim the Assistant's
configured upstream model ID so the native probe can resolve and score the
expected developer identity.

The resulting ordered `result.invocations` retain User and Assistant Agent
activations, any Environment-owned Judge and Summary support-model calls,
tool calls and results, post-activation state, and observations. Probe tool
simulation and verification support calls remain Resources-owned evidence.

The User Agent uses `user_model`, while the Assistant Agent uses
`policy_model`. The Environment Server routes UserSim's Judge and Summary
calls directly to `support_model`, without creating support Agent sessions.
Resources-owned tool-result synthesis and native probe scoring retain their
purpose-specific `tool_simulation_model` and `probe_scorer_model` fields, but
both reference the same support server. All three Model Servers have explicit,
independently configurable endpoint settings.

After preparation:

```bash
gym eval run \
  --environment usersim \
  --split example \
  --output results/usersim.jsonl \
  ++observability_enabled=true \
  ++model_call_capture_dir=/absolute/path/to/model-calls
```

Configure the explicit `user_model_*`, `policy_*`, and `support_model_*`
settings for OpenAI-compatible endpoints. The aliases may use the same
upstream service only when it satisfies each role's requirements.

For participant-specific SFT or custom collation, filter
`result.invocations` by the `assistant` or `user` role and use each selected
invocation's exact `request` and `response`. Judge and Summary support-model
calls retain their own roles and cannot be mistaken for participant training
data.
