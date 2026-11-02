# LangGraph ReWOO Reasoning Gym

LangGraph ReWOO agent compatible with resource servers that do not use tools. It enables diverse agent training data and test time scaling vs a simple agent, and it can be extended to use tools or other agent architectures.

Source benchmark: https://github.com/open-thought/reasoning-gym
The instructions below assume you have a vLLM server running and `policy_base_url`, `policy_model_name`, and `policy_api_key` configured in your `env.yaml` file. See [documentation](https://docs.nvidia.com/nemo/gym/reference/configuration#local-configuration-envyaml) for details.
## Start

The graph runs inside a task sandbox, so load a sandbox provider config (Docker below) and give the sandbox a model endpoint it can reach. `++use_absolute_ip=true` advertises the Model Server on a non-loopback address. See the [LangGraph agent README](../../responses_api_agents/langgraph_agent/README.md).

```bash
gym env start \
  --config environments/langgraph_rewoo_reasoning_gym/config.yaml \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
  --model-type openai_model \
  ++use_absolute_ip=true
```

## Run

```bash
gym eval run --no-serve \
  --agent langgraph_rewoo_reasoning_gym_agent \
  --input environments/langgraph_rewoo_reasoning_gym/data/example.jsonl \
  --output results/langgraph_rewoo_reasoning_gym_rollouts.jsonl
```

## Prepare training data

```bash
python environments/langgraph_rewoo_reasoning_gym/prepare.py --task knights_knaves --size 1000 --output environments/langgraph_rewoo_reasoning_gym/data/train_knights_knaves.jsonl
```

See `prepare.py` for all available tasks, categories, and config options.

Alternatively, a pre-built dataset is hosted on HuggingFace at [nvidia/Nemotron-RL-ReasoningGym-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1).
