# NeMo-Sim Resources Server

This environment initializes one deterministic NeMo-Sim scenario at the
beginning of each `UserAssistantProcessor` episode. The Resources Server
prepares a versioned Nemotron persona panel once at startup; it does not
generate personas with an LLM or run a complete Data Designer pipeline.

## Environment initialization

The default configuration pins the NGC resource version to `0.0.2` and uses:

```text
~/.cache/nemo-gym/nemo-sim/personas/
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

1. Reuses the pinned source Parquet when it already exists in the cache.
2. On a cache miss, downloads the explicit NGC resource version, for example
   `nvidia/nemotron-personas/nemotron-personas-dataset-en_us:0.0.2`.
3. Validates the Parquet and records its row count, size, and SHA-256.
4. Reuses a matching deterministic panel when present; otherwise streams the
   source dataset once and materializes a bounded panel.
5. Loads only the panel into memory for episode sampling.

File locks and atomic replacement prevent concurrent server processes sharing
the cache from publishing partial artifacts. A cache hit does not invoke NGC
and does not hash or scan the full source again.

The NGC CLI and its authentication are required only when a pinned source is
absent. Set `NGC_CLI_API_KEY` and `NGC_CLI_ORG`, or configure NGC once with
`ngc config set`. Set `download_missing_personas: false` for air-gapped runs;
startup will then fail clearly if the pinned artifact was not pre-populated.

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
4. Adds a JSON-encoded `metadata.nemo_sim` value to the existing
   `user_responses_create_params`.
5. Returns those resolved user parameters to the processor before its first
   participant turn.

The processor includes the resolved user parameters in the rollout, and the
verifier includes `nemo_sim_context` for replay and auditing.

## Static and dynamic configuration

The YAML config owns static population and probe policy:

- `personas_cache_dir`
- `personas_dataset_version`
- `personas_locales`
- `personas_panel_size` and `personas_panel_seed`
- `download_missing_personas`
- `probe_mix`
- `probe_themes`
- agent, model, and resources-server references
- turn limits

Each dataset row owns dynamic task identity:

- `nemo_sim_sampling.locale`
- `nemo_sim_sampling.seed`
- optional `nemo_sim_sampling.probe_type`
- assistant and user Responses API inputs

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

Configure `policy_base_url`, `policy_api_key`, and `policy_model_name`. On the
first run, also make the NGC CLI and credentials available so initialization
can fill the pinned cache:

```bash
.venv/bin/gym eval run \
  --config resources_servers/nemo_sim/configs/nemo_sim.yaml \
  --agent nemo_sim_processor \
  --split validation \
  --output results/nemo_sim.jsonl
```
