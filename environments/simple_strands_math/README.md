# Simple Strands Math

Simple Strands Agent with math tasks using [benchmark-harnesses](https://github.com/strands-labs/benchmark-harnesses).

```bash
gym env start --environment simple_strands_math --model-type openai_model

gym eval run --no-serve \
  --agent simple_strands_math_agent \
  --input environments/simple_strands_math/data/example.jsonl \
  --output environments/simple_strands_math/data/example_rollouts.jsonl \
  --limit 5
```

```bash
python environments/simple_strands_math/prepare.py --split train
```
