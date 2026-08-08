# Hermes Math

Runs DAPO-Math-17k with Hermes. Answers are graded by `math_with_judge` using `math-verify`. The LLM judge is disabled.

Set `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

## Prepare

```bash
python environments/hermes_math/prepare.py
```

## Start

```bash
gym env start \
  --config environments/hermes_math/config.yaml \
  --model-type openai_model
```

## Run

```bash
gym eval run --no-serve \
  --agent hermes_math_agent \
  --input environments/hermes_math/data/example.jsonl \
  --output results/hermes_math_rollouts.jsonl
```
