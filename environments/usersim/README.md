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
revision `4fd4c800bbef8883329543df632f328860fc6429` over Git+SSH from
`github.com/NVIDIA-NeMo/UserSim`; the host therefore needs GitHub SSH access.

Each task carries UserSim's resolved row unchanged. Resources constructs the
runtime only with `ProbeEpisodeRuntime.from_resolved_row`, preserving the
materialized configuration, trajectory identity, and provenance.

`tool_calling`, `safety_agentic`, and `financial_services` expose
probe-selected tool schemas to the Assistant Agent. The Agent owns model/tool
iteration and invokes standard Resources Server `POST /{tool_name}` endpoints.
Ordered parallel calls use `POST /runtime/tool_calls`; UserSim assigns semantic
turn/call indices. Single tool responses remain opaque text.
The Resources Server owns the episode-scoped tool implementation, mutable
state, and native verification evidence. All other probes execute their native
UserSim conversation shape without Assistant tools.
For `identity_disclosure`, the integration gives UserSim the Assistant's
configured upstream model ID so the native probe can resolve and score the
expected developer identity.

The resulting ordered `result.invocations` retain User and Assistant Agent
activations, Judge and Summary support-model calls, tool calls and results,
post-activation state, and observations.

The User and Assistant Agents share `policy_model`. The Environment Server
routes UserSim's Judge and Summary calls directly to `support_model`, without
creating support Agent sessions. Resources-owned tool-result synthesis and
native probe scoring retain their purpose-specific `tool_simulation_model` and
`probe_scorer_model` fields, but both reference the same support server. Policy
and support are distinct model aliases, and all three support endpoint settings
must be configured explicitly.

After preparation:

```bash
gym eval run \
  --environment usersim \
  --split benchmark \
  --output results/usersim.jsonl \
  ++observability_enabled=true \
  ++model_call_capture_dir=/absolute/path/to/model-calls
```

Configure `policy_base_url`, `policy_api_key`, and `policy_model_name`, plus
the corresponding explicit `support_model_*` settings, for OpenAI-compatible
endpoints. The two aliases may use the same upstream service only when it
satisfies both participant and structured support-call requirements.

For participant-specific SFT or custom collation, filter
`result.invocations` by the `assistant` or `user` role and use each selected
invocation's exact `request` and `response`. Judge and Summary support-model
calls retain their own roles and cannot be mistaken for participant training
data.
