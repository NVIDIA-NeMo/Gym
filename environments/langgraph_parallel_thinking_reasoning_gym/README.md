# LangGraph Parallel Thinking Reasoning Gym

LangGraph parallel thinking agent compatible with resource servers that do not use tools. It enables diverse agent training data and test time scaling vs a simple agent, and it can be extended to use tools or other agent architectures.

Source benchmark: https://github.com/open-thought/reasoning-gym
The instructions below assume you have a vLLM server running and `policy_base_url`, `policy_model_name`, and `policy_api_key` configured in your `env.yaml` file. See [documentation](https://docs.nvidia.com/nemo/gym/reference/configuration#local-configuration-envyaml) for details.
## Quick start

```bash
ng_run "+config_paths=[environments/langgraph_parallel_thinking_reasoning_gym/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

```bash
ng_collect_rollouts \
    +agent_name=langgraph_parallel_thinking_reasoning_gym_agent \
    +input_jsonl_fpath=environments/langgraph_parallel_thinking_reasoning_gym/data/example.jsonl \
    +output_jsonl_fpath=results/langgraph_parallel_thinking_reasoning_gym_rollouts.jsonl
```

## Prepare training data

```bash
python environments/langgraph_parallel_thinking_reasoning_gym/prepare.py --task knights_knaves --size 1000 --output environments/langgraph_parallel_thinking_reasoning_gym/data/train_knights_knaves.jsonl
```

See `prepare.py` for all available tasks, categories, and config options.

Alternatively, a pre-built dataset is hosted on HuggingFace at [nvidia/Nemotron-RL-ReasoningGym-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1).
