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
Gym task dataset. The Resources Server validates and loads this prepared panel;
it does not duplicate Data Designer's person-sampling logic.

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

The included example independently configures User tools
(`record_user_context`, `finish_episode`) and the Assistant tool
(`read_user_context`). Their calls share one task-scoped Resources session.
The resulting ordered `agent_turns` retain both participants, tool calls and
results, post-turn state, observations, and the final termination reason.

The Assistant, simulated User, and NeMo UserSim support calls use three explicit
model-server references: `assistant_policy_model`, `user_policy_model`, and
`simulation_support_model`. By default, all three inherit the standard
`policy_base_url`, `policy_api_key`, and `policy_model_name` settings. Override
their corresponding `assistant_policy_*`, `user_policy_*`, or
`simulation_support_*` settings to run them on different models or endpoints.

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
