# Reasoning Gym — Hermes Agent

Hermes Agent with terminal, file, code_execution, skills, todo toolsets on reasoning_gym tasks.

Source benchmark: https://github.com/open-thought/reasoning-gym

## Quick start

```bash
ng_run "+config_paths=[environments/reasoning_gym_hermes/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

```bash
ng_collect_rollouts \
    +agent_name=reasoning_gym_hermes_agent \
    +input_jsonl_fpath=environments/reasoning_gym_hermes/data/example.jsonl \
    +output_jsonl_fpath=results/reasoning_gym_hermes_rollouts.jsonl
```

## Prepare training data

```bash
python environments/reasoning_gym_hermes/prepare.py --task knights_knaves --size 1000 --output environments/reasoning_gym_hermes/data/train_knights_knaves.jsonl
```

See `prepare.py` for all available tasks, categories, and config options.

Alternatively, a pre-built dataset is hosted on HuggingFace at [nvidia/Nemotron-RL-ReasoningGym-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1).
