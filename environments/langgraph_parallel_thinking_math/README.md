# LangGraph Parallel Thinking Math

Runs DAPO-Math-17k with the LangGraph parallel thinking agent. Answers are graded by `math_with_judge` using `math-verify`. The LLM judge is disabled.

Set `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

## Prepare

```bash
python environments/langgraph_parallel_thinking_math/prepare.py
```

## Start

```bash
gym env start \
  --config environments/langgraph_parallel_thinking_math/config.yaml \
  --model-type openai_model
```

## Run

```bash
gym eval run --no-serve \
  --agent langgraph_parallel_thinking_math_agent \
  --input environments/langgraph_parallel_thinking_math/data/example.jsonl \
  --output results/langgraph_parallel_thinking_math_rollouts.jsonl
```
