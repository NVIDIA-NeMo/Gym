# Codex Math

Runs DAPO-Math-17k with Codex. Answers are graded by `math_with_judge` using `math-verify`. The LLM judge is disabled.

Set `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

## Prepare

```bash
python environments/codex_math/prepare.py
```

## Start

```bash
gym env start \
  --config environments/codex_math/config.yaml \
  --model-type openai_model
```

## Run

```bash
gym eval run --no-serve \
  --agent codex_math_agent \
  --input environments/codex_math/data/example.jsonl \
  --output results/codex_math_rollouts.jsonl
```
