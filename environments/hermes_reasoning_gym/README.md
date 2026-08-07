# Hermes Reasoning Gym

Hermes Agent with terminal, file, code_execution, skills, todo toolsets on reasoning_gym tasks.

Source benchmark: https://github.com/open-thought/reasoning-gym
The instructions below assume you have a vLLM server running and `policy_base_url`, `policy_model_name`, and `policy_api_key` configured in your `env.yaml` file. See [documentation](https://docs.nvidia.com/nemo/gym/reference/configuration#local-configuration-envyaml) for details.
## Start

```bash
gym env start \
  --config environments/hermes_reasoning_gym/config.yaml \
  --model-type openai_model
```

## Run

```bash
gym eval run --no-serve \
  --agent hermes_reasoning_gym_agent \
  --input environments/hermes_reasoning_gym/data/example.jsonl \
  --output results/hermes_reasoning_gym_rollouts.jsonl
```

## Prepare training data

```bash
python environments/hermes_reasoning_gym/prepare.py --task knights_knaves --size 1000 --output environments/hermes_reasoning_gym/data/train_knights_knaves.jsonl
```

See `prepare.py` for all available tasks, categories, and config options.

Alternatively, a pre-built dataset is hosted on HuggingFace at [nvidia/Nemotron-RL-ReasoningGym-v1](https://huggingface.co/datasets/nvidia/Nemotron-RL-ReasoningGym-v1).
