# Polar Agent

Gym adapter for ProRL-Agent-Server Polar inference on SWE-Gym rows.

The agent converts each Gym `/run` request into the single-row JSONL format used by
`ProRL-Agent-Server/examples/swegym_slime_grpo/simple_inference.py`, runs Polar
inference, then converts the result JSON back into a Gym `NeMoGymResponse`.

## Modes

- `direct`: assumes an OpenAI-compatible SGLang server is already running at
  `base_url`, then calls `simple_inference.py` directly.
- `slurm`: submits
  `ProRL-Agent-Server/examples/swegym_slime_grpo/submit_simple_inference.sh` and
  waits for the script's result JSON. Use this mode when you want the same SLURM
  path as the standalone inference script.

## Example

Start Gym with the Polar config:

```bash
ng_run "+config_paths=[responses_api_agents/polar/configs/polar.yaml]"
```

Collect one rollout:

```bash
ng_collect_rollouts \
  +agent_name=polar \
  +input_jsonl_fpath=responses_api_agents/polar/data/example.jsonl \
  +output_jsonl_fpath=outputs/polar/example_rollouts.jsonl \
  +limit=1 \
  +upload_rollouts_to_wandb=false
```

For SLURM-backed inference, override the mode and any submit-script environment
variables:

```bash
ng_run "+config_paths=[responses_api_agents/polar/configs/polar.yaml]" \
  ++polar.responses_api_agents.polar.mode=slurm \
  ++polar.responses_api_agents.polar.slurm_env.POLR_TRAIN_SQSH=/path/to/polr_swegym_slime_grpo_train.sqsh \
  ++polar.responses_api_agents.polar.slurm_env.HF_CHECKPOINT=/path/to/hf/checkpoint
```

`/run` returns `reward=0.0` by default because this adapter is inference-only and
does not run a SWE-bench evaluator.
