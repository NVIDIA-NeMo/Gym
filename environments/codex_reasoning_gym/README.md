# Codex Reasoning Gym

Runs Reasoning Gym tasks through the Codex CLI agent harness.

Configure `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`. The endpoint must expose the streaming OpenAI Responses API. Codex appends `/responses` to a base URL ending in `/v1`.

```bash
gym env start --config environments/codex_reasoning_gym/config.yaml
```

```bash
gym eval run --no-serve \
  --agent codex_reasoning_gym_agent \
  --input environments/codex_reasoning_gym/data/example.jsonl \
  --output results/codex_reasoning_gym_rollouts.jsonl
```

Generate training rows:

```bash
python environments/codex_reasoning_gym/prepare.py \
  --task knights_knaves \
  --size 1000
```
