# Kiro Reasoning Gym

Runs reasoning gym tasks through Kiro.

Set `KIRO_API_KEY` in the environment. Set `KIRO_MODEL` to select a Kiro model.

```bash
gym env start --environment kiro_reasoning_gym

gym eval run --no-serve \
  --agent kiro_reasoning_gym_agent \
  --input environments/kiro_reasoning_gym/data/example.jsonl \
  --output outputs/kiro_reasoning_gym_rollouts.jsonl \
  --limit 5
```

```bash
python environments/kiro_reasoning_gym/prepare.py --task knights_knaves --size 1000
```
