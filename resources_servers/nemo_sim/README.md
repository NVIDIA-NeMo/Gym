# NeMo-Sim Resources Server

This environment initializes one deterministic NeMo-Sim scenario at the
beginning of each `NeMoSimProcessor` episode. The Resources Server
prepares a deterministic panel from a previously downloaded, versioned
Nemotron Personas source. It does not access NGC, generate personas with an
LLM, or run a complete Data Designer pipeline.

## Environment initialization

The benchmark configuration pins the NGC resource version to `0.0.2` and uses:

```text
benchmarks/nemo_sim/data/personas/
└── 0.0.2/
    ├── source/
    │   ├── en_US.parquet
    │   └── en_US.manifest.json
    ├── panels/
    │   ├── en_US-n1000-seed42.parquet
    │   └── en_US-n1000-seed42.manifest.json
    └── locks/
```

For every configured locale, server startup:

1. Requires the pinned source Parquet prepared by the benchmark recipe.
2. Validates the Parquet and records its row count, size, and SHA-256.
3. Reuses a matching deterministic panel when present; otherwise streams the
   source dataset once and materializes a bounded panel.
4. Loads only the panel into memory for episode sampling.

File locks and atomic replacement prevent concurrent server processes sharing
the prepared assets from publishing partial panels. A matching source manifest
avoids hashing or scanning the full source again.

NGC and its credentials are preparation-time concerns. Run
`gym eval prepare --benchmark nemo_sim` before starting the Resources Server.
Startup fails with that instruction when the pinned source is absent. See
[`benchmarks/nemo_sim`](../../benchmarks/nemo_sim/) for credential and artifact
details.

## Episode initialization

Each Gym row supplies a compact sampling request:

```json
{
  "nemo_sim_sampling": {
    "locale": "en_US",
    "seed": 1042,
    "probe_type": "general_open_ended"
  }
}
```

`probe_type` is optional. When omitted, the resources server selects it from
the configured `probe_mix`. The same locale and seed always resolve to the same
persona, probe, and theme for an unchanged persona dataset and server config.

At `/seed_session`, the server:

1. Selects one persona from the prepared panel, plus one probe and theme,
   deterministically.
2. Stores the resolved context in task-scoped session state.
3. Records the source version, SHA-256, and panel seed for replay.
4. Returns a `NeMoSimScenario` to the Processor before its first participant
   invocation.

The Processor gives the scenario to NeMo-Sim's conversation generator, routes
its participant and support-model calls through Gym, and submits the completed
trajectory to `/verify`. The verifier includes `nemo_sim_context` for replay
and auditing.

## Static and dynamic configuration

The YAML config owns static population and probe policy:

- `personas_cache_dir`
- `personas_dataset_version`
- `personas_locales`
- `personas_panel_size` and `personas_panel_seed`
- `probe_mix`
- `probe_themes`
- agent, model, and resources-server references
- turn limits

Each dataset row owns dynamic task identity:

- `nemo_sim_sampling.locale`
- `nemo_sim_sampling.seed`
- optional `nemo_sim_sampling.probe_type`
- default and per-model Responses API parameters

Changing the dataset version or panel configuration creates a different cache
path rather than silently overwriting an existing panel.

## Supported probes

This first implementation supports:

- `general_open_ended`
- `general_educational`

These probes resolve a theme and user goal but do not require dynamic tools.
Tool-calling, safety, sovereign-AI, finance, health, and trajectory-evaluator
semantics require probe-specific resources-server adapters.

The current reward is an integration signal: `1.0` when both assistant and
simulated-user trajectories contain at least one turn, otherwise `0.0`. It is
not an assistant-quality benchmark score.

## Run

Configure `policy_base_url`, `policy_api_key`, and `policy_model_name`, then
prepare the pinned persona source before collecting rollouts:

```bash
gym eval prepare --benchmark nemo_sim

.venv/bin/gym eval run \
  --benchmark nemo_sim \
  --agent nemo_sim_processor \
  --output results/nemo_sim.jsonl
```
