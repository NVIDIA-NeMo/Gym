# OpenClaw Reasoning Gym

Runs Reasoning Gym tasks through the OpenClaw CLI agent harness. The checked-in configuration uses the same pinned OpenClaw version and NVIDIA OpenAI-compatible provider shape as `responses_api_agents/openclaw_agent`.

Configure `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`. The default agent model is `nvidia/qwen/qwen3-next-80b-a3b-instruct`. override both `model` and `openclaw_config` when selecting another model.

```bash
gym env start --config environments/openclaw_reasoning_gym/config.yaml
```

```bash
gym eval run --no-serve \
  --agent openclaw_reasoning_gym_agent \
  --input environments/openclaw_reasoning_gym/data/example.jsonl \
  --output results/openclaw_reasoning_gym_rollouts.jsonl
```

Generate training rows:

```bash
python environments/openclaw_reasoning_gym/prepare.py \
  --task knights_knaves \
  --size 1000
```
