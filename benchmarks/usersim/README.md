# NeMo UserSim benchmark preparation

The benchmark preparation step delegates persona sampling to NeMo UserSim and
treats the resulting panel Parquet as the immutable prepared artifact. Install
the pinned NeMo UserSim package so the `usersim` executable is on `PATH`, then
prepare the benchmark:

```bash
gym eval prepare --benchmark usersim
```

Preparation invokes `usersim panel`, validates the resulting Parquet, and
writes its checksum manifest under:

```text
benchmarks/usersim/data/personas/
└── 0.0.2/
    └── panels/
        ├── en_US.parquet
        └── en_US.manifest.json
```

It also materializes `benchmarks/usersim/data/usersim.jsonl`, the lightweight
Gym task dataset. Its 13 rows provide one example for every first-party UserSim
probe. The Resources Server validates and loads the prepared panel; it does not
duplicate UserSim's person-sampling logic.

While UserSim is private, the Environment Server installs the pinned source
revision over Git+SSH from `github.com/NVIDIA-NeMo/UserSim`; the host therefore
needs GitHub SSH access. This temporary source dependency should become a normal
published-package dependency when UserSim is open-sourced.

NeMo UserSim behavior is pinned once in the benchmark's typed
`UserSimEnvironmentServerConfig.protocol_config`; it is not repeated in task
rows. The Environment Server separately owns `max_turns`, fixes the Data
Designer output-column name internally, and supplies each scenario's locale
after `/seed_session`.
Dataset rows contain only per-task sampling inputs and optional model-call
parameter overrides. Resolved scenarios are output-only and cannot be supplied
by a dataset row.

`tool_calling`, `safety_agentic`, and `financial_services` expose
probe-selected tool schemas to the Assistant Agent. The Agent owns model/tool
iteration and invokes standard Resources Server `POST /{tool_name}` endpoints.
The Resources Server owns the episode-scoped tool implementation, mutable
state, and native verification evidence. All other probes execute their native
UserSim conversation shape without Assistant tools.

The resulting ordered `result.invocations` retain User, Assistant, Judge, and
Summary Agent activations, tool calls and results, post-activation state, and
observations.

The four Agents have explicit policy Model Servers. Resources-owned tool-result
synthesis uses `tool_simulation_model`, while `probe_scorer_model` reuses
`judge_model` by default. All Model Servers inherit the standard
`policy_base_url`, `policy_api_key`, and `policy_model_name` settings unless
their role-specific settings override them.

After preparation:

```bash
gym eval run \
  --benchmark usersim \
  --agent usersim_assistant \
  --output results/usersim.jsonl \
  ++observability_enabled=true \
  ++model_call_capture_dir=/absolute/path/to/model-calls
```

Configure `policy_base_url`, `policy_api_key`, and `policy_model_name` for any
OpenAI-compatible endpoint. The role-specific settings can override that
endpoint independently without embedding provider-specific credentials or
model names in the repository.

For participant-specific SFT or custom collation, filter
`result.invocations` by the `assistant` or `user` role and use each selected
invocation's exact `request` and `response`. Judge and Summary Agent calls
retain their own roles and cannot be mistaken for participant training data.
