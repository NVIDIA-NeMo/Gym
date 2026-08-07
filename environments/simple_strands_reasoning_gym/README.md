# Simple Strands Reasoning Gym

Simple Strands Agent with reasoning gym tasks using [benchmark-harnesses](https://github.com/strands-labs/benchmark-harnesses).

```bash
gym env start --environment simple_strands_reasoning_gym --model-type openai_model

gym eval run --no-serve \
  --agent simple_strands_reasoning_gym_agent \
  --input environments/simple_strands_reasoning_gym/data/example.jsonl \
  --output environments/simple_strands_reasoning_gym/data/example_rollouts.jsonl \
  --limit 5
```

```bash
uv run --no-project --python python3 --with reasoning-gym==0.1.25 python \
  environments/simple_strands_reasoning_gym/prepare.py --task knights_knaves --size 1000
```
