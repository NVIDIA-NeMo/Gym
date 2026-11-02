# LangGraph Parallel Thinking Math

Runs DAPO-Math-17k with the LangGraph parallel thinking agent. Answers are graded by `math_with_judge` using `math-verify`. The LLM judge is disabled.

Set `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

## Prepare

```bash
python environments/langgraph_parallel_thinking_math/prepare.py
```

## Start

The graph runs inside a task sandbox, so load a sandbox provider config (Docker below) and give the sandbox a model endpoint it can reach. `++use_absolute_ip=true` advertises the Model Server on a non-loopback address. See the [LangGraph agent README](../../responses_api_agents/langgraph_agent/README.md).

```bash
gym env start \
  --config environments/langgraph_parallel_thinking_math/config.yaml \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
  --model-type openai_model \
  ++use_absolute_ip=true
```

## Run

```bash
gym eval run --no-serve \
  --agent langgraph_parallel_thinking_math_agent \
  --input environments/langgraph_parallel_thinking_math/data/example.jsonl \
  --output results/langgraph_parallel_thinking_math_rollouts.jsonl
```
