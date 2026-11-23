# Cline Math

Runs DAPO-Math-17k with Cline. Answers are graded by `math_with_judge` using `math-verify`. The LLM judge is disabled.

Set `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

## Prepare

```bash
python environments/cline_math/prepare.py
```

## Start

```bash
gym env start \
  --config environments/cline_math/config.yaml \
  --model-type openai_model
```

## Run

```bash
gym eval run --no-serve \
  --agent cline_math_agent \
  --input environments/cline_math/data/example.jsonl \
  --output results/cline_math_rollouts.jsonl
```
