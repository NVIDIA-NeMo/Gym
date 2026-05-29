# Reasoning Gym — LangGraph ReWOO Agent

LangGraph ReWOO agent compatible with resource servers that do not use tools; enables diverse agent training data and test time scaling vs a simple agent, extensible to use tools or other agent architectures.

Source benchmark: https://github.com/open-thought/reasoning-gym

## Quick start

```bash
ng_run "+config_paths=[environments/reasoning_gym_rewoo/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

```bash
ng_collect_rollouts \
    +agent_name=reasoning_gym_rewoo_agent \
    +input_jsonl_fpath=environments/reasoning_gym_rewoo/data/example.jsonl \
    +output_jsonl_fpath=results/reasoning_gym_rewoo_rollouts.jsonl
```

## Prepare training data

```bash
python environments/reasoning_gym_rewoo/prepare.py --task knights_knaves --size 1000 --output environments/reasoning_gym_rewoo/data/train_knights_knaves.jsonl
```

See `prepare.py` for all available tasks, categories, and config options.

Alternatively, a pre-built dataset is hosted on HuggingFace at [nvidia/Nemotron-RL-ReasoningGym-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1).
