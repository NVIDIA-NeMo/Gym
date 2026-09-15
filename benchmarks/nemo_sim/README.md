# NeMo-Sim benchmark preparation

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
gym eval prepare --benchmark nemo_sim
```

Preparation downloads the immutable NGC resource version configured by
`prepare.py`, validates the Parquet, and writes its checksum manifest under:

```text
benchmarks/nemo_sim/data/personas/
└── 0.0.2/
    └── source/
        ├── en_US.parquet
        └── en_US.manifest.json
```

It also materializes `benchmarks/nemo_sim/data/nemo_sim.jsonl`, the lightweight
Gym task dataset. At Resources Server startup, the source is validated and a
bounded deterministic panel is created or reused beside it under `panels/`.

NeMo-Sim behavior is pinned once in the benchmark's typed
`NeMoSimProcessorConfig.protocol_config`; it is not repeated in task rows. The
Processor separately owns `max_turns`, fixes the Data Designer output-column
name internally, and supplies each scenario's locale after `/seed_session`.
Dataset rows contain only per-task sampling inputs and optional model-call
parameter overrides. Resolved scenarios are output-only and cannot be supplied
by a dataset row.

The included example independently configures User tools
(`record_user_context`, `finish_episode`) and the Assistant tool
(`read_user_context`). Their calls share one task-scoped Resources session.
The resulting ordered `agent_turns` retain both participants, tool calls and
results, post-turn state, observations, and the final termination reason.

After preparation:

```bash
gym eval run \
  --benchmark nemo_sim \
  --agent nemo_sim_processor \
  --output results/nemo_sim.jsonl \
  ++observability_enabled=true \
  ++model_call_capture_dir=/absolute/path/to/model-calls
```

For evaluation and existing RL consumers, top-level `response` remains the
final Assistant response. For participant-specific SFT or custom collation,
filter `agent_turns` by `participant == "assistant"` or
`participant == "user"` and use each selected turn's exact `request` and
`response`. Selecting both produces separate participant-labelled examples;
User outputs are never folded into the Assistant response.
