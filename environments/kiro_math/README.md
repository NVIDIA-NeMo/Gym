# Kiro Math

Runs math tasks with Kiro and verifies boxed answers with `math-verify`.

Set `KIRO_API_KEY` in the environment. Set `KIRO_MODEL` to select a Kiro model.

```bash
gym env start --environment kiro_math --model-type openai_model

gym eval run --no-serve \
  --agent kiro_math_agent \
  --input environments/kiro_math/data/example.jsonl \
  --output outputs/kiro_math_rollouts.jsonl \
  --limit 5
```

```bash
python environments/kiro_math/prepare.py --split train
```
