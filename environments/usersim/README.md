# NeMo UserSim environment

This environment runs population-grounded, multi-turn user simulation with
NeMo UserSim. It includes one example task for each first-party UserSim probe;
the examples exercise the environment's supported interaction shapes.

Environment preparation delegates persona sampling to NeMo UserSim and treats
the resulting panel Parquet as the immutable prepared artifact. Preparation
runs the pinned UserSim package in its own isolated dependency environment, so
it does not alter the Gym or Resources Server environments:

```bash
gym eval prepare --config environments/usersim/config.yaml
```

Preparation invokes `usersim panel`, validates the resulting Parquet, and
writes its checksum manifest under:

```text
environments/usersim/data/personas/
└── 0.0.2/
    └── panels/
        ├── en_US.parquet
        └── en_US.manifest.json
```

The 14 tracked templates in `resources_servers/usersim/data/example_source.jsonl`
provide one example for every first-party UserSim probe. Preparation resolves
each template against the generated panel and writes the ignored
`environments/usersim/data/example.jsonl` consumed by the dataset config.
Every prepared row contains its persona, locale, seed, probe, theme, goal,
probe data, and panel/UserSim provenance. The Resources Server validates that
resolved row against the prepared panel and uses it unchanged.

While UserSim is private, the Environment Server installs the pinned source
revision over Git+SSH from `github.com/NVIDIA-NeMo/UserSim`; the host therefore
needs GitHub SSH access. This temporary source dependency should become a normal
published-package dependency when UserSim is open-sourced.

NeMo UserSim behavior is pinned once in the Resources Server's typed
`protocol_config`; it is not repeated in task rows. The Resources Server also
owns `max_turns`, fixes internal column names, and passes the complete protocol
configuration into the canonical runtime. `context_compression` remains
disabled and finance retrieval uses UserSim's `hybrid` default.

`tool_calling`, `safety_agentic`, and `financial_services` expose
probe-selected tool schemas to the Assistant Agent. The Agent owns model/tool
iteration and invokes standard Resources Server `POST /{tool_name}` endpoints.
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
  --split example \
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
