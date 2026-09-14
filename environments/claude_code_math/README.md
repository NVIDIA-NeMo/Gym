# Claude Code Math

Runs DAPO-Math-17k with Claude Code. Answers are graded by `math_with_judge` using `math-verify`. The LLM judge is disabled.

Set `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

## Prepare

```bash
python environments/claude_code_math/prepare.py
```

## Start

```bash
gym env start \
  --config environments/claude_code_math/config.yaml \
  --model-type openai_model
```

## Run

```bash
gym eval run --no-serve \
  --agent claude_code_math_agent \
  --input environments/claude_code_math/data/example.jsonl \
  --output results/claude_code_math_rollouts.jsonl
```
