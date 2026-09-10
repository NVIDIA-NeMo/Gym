# NeMo-Sim Resources Server

This environment initializes one deterministic NeMo-Sim scenario at the
beginning of each `UserAssistantProcessor` episode. It samples from an existing
Nemotron persona dataset; it does not generate a persona with an LLM or run a
complete Data Designer pipeline.

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

1. Reads `~/.data-designer/managed-assets/datasets/<locale>.parquet` by
   default.
2. Selects one persona, probe, and theme deterministically.
3. Stores the resolved context in task-scoped session state.
4. Adds a JSON-encoded `metadata.nemo_sim` value to the existing
   `user_responses_create_params`.
5. Returns those resolved user parameters to the processor before its first
   participant turn.

The processor includes the resolved user parameters in the rollout, and the
verifier includes `nemo_sim_context` for replay and auditing.

## Static and dynamic configuration

The YAML config owns static population and probe policy:

- `personas_dir`
- `probe_mix`
- `probe_themes`
- agent, model, and resources-server references
- turn limits

Each dataset row owns dynamic task identity:

- `nemo_sim_sampling.locale`
- `nemo_sim_sampling.seed`
- optional `nemo_sim_sampling.probe_type`
- assistant and user Responses API inputs

Set `personas_dir` in the resources-server YAML when the managed Data Designer
assets are installed somewhere other than the default path.

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

Configure `policy_base_url`, `policy_api_key`, and `policy_model_name`, ensure
the `en_US.parquet` managed persona asset exists, then run:

```bash
.venv/bin/gym eval run \
  --config resources_servers/nemo_sim/configs/nemo_sim.yaml \
  --agent nemo_sim_processor \
  --split validation \
  --output results/nemo_sim.jsonl
```
