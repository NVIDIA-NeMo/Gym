# OpenClaw Math

Runs DAPO-Math-17k with OpenClaw. Answers are graded by `math_with_judge` using `math-verify`. The LLM judge is disabled.

Set `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

## Prepare

```bash
python environments/openclaw_math/prepare.py
```

## Start

```bash
gym env start \
  --config environments/openclaw_math/config.yaml \
  --model-type openai_model
```

## Run

```bash
gym eval run --no-serve \
  --agent openclaw_math_agent \
  --input environments/openclaw_math/data/example.jsonl \
  --output results/openclaw_math_rollouts.jsonl
```
