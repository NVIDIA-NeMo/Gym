# NeMo UserSim benchmark preparation

The benchmark preparation step downloads the pinned Nemotron Personas source.
The Resources Server never accesses NGC and fails fast when the prepared source
is missing.

Configure NGC credentials in the repository-root `env.yaml`:

```yaml
ngc_cli_api_key: <NGC API key>
ngc_cli_org: <NGC organization>
```

An existing `~/.ngc/config` or exported `NGC_CLI_API_KEY` and `NGC_CLI_ORG`
remain valid alternatives. Then prepare the benchmark:

```bash
gym eval prepare --benchmark usersim
```

Preparation downloads the immutable NGC resource version configured by
`prepare.py`, validates the Parquet, and writes its checksum manifest under:

```text
benchmarks/usersim/data/personas/
└── 0.0.2/
    └── source/
        ├── en_US.parquet
        └── en_US.manifest.json
```

It also materializes `benchmarks/usersim/data/usersim.jsonl`, the lightweight
Gym task dataset. At Resources Server startup, the source is validated and a
bounded deterministic panel is created or reused beside it under `panels/`.

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
`result.invocations` by the `assistant_model` or `user_model` alias and use
each selected invocation's exact `request` and `response`. Support-model calls
retain separate aliases and cannot be mistaken for participant training data.
