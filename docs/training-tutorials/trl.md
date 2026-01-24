(training-trl)=

# RL Training with TRL

This tutorial demonstrates how to use [Hugging Face TRL](https://github.com/huggingface/trl) to train models in NeMo Gym environments using GRPO.

**TRL (Transformer Reinforcement Learning)** is Hugging Face's library for post-training foundation models. It provides implementations of algorithms like SFT, GRPO, and DPO, with support for distributed training such as with Accelerate and DeepSpeed.

TRL integrates with NeMo Gym to support both single-step and multi-step verifiable environments including math tasks, tool calling, reasoning, coding, and more.

:::{card}

**Goal**: Train models on NeMo Gym environments using TRL's GRPO trainer.

^^^

**In this tutorial, you will**:

1. Set up TRL with NeMo Gym environments
2. Prepare NeMo-Gym datasets and configure training
3. Train models using GRPO with vLLM for optimized inference
4. Scale to multi-node training with Slurm

:::

## Overview

The TRL integration with NeMo Gym enables:

- **Multi-step and multi-turn environments**: Full support for tool calling and complex agentic tasks
- **Multi-environment training**: Train on multiple environments simultaneously
- **Production-scale training**: Multi-node distributed training with DeepSpeed
- **Flexible verification**: Algorithmic verification, LLM-as-a-judge, and custom reward functions


## Installation

Install TRL with vLLM support:

```bash
pip install trl[vllm]
```

Install NeMo Gym:

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git
cd Gym
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev
```

## Quick Start

### 1. Prepare Dataset

Download and prepare a NeMo Gym dataset:

```bash
cd Gym
source .venv/bin/activate

# Set HuggingFace token in env.yaml
echo "hf_token: your_token_here" >> env.yaml

# Prepare workplace assistant dataset
config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,\
resources_servers/workplace_assistant/configs/workplace_assistant.yaml"

ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface
```

### 2. Configure Training

Create a `config.yaml` file:

```yaml
model_name: "Qwen/Qwen2.5-1.5B-Instruct"

dataset_path: "data/workplace_assistant/train.jsonl"
eval_dataset_path: "data/workplace_assistant/validation.jsonl"

output_dir: "outputs/trl_nemo_gym"
report_to: "wandb"
project_name: "trl-nemo-gym"

learning_rate: 1.0e-5
max_steps: 1000
num_generations: 8
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
max_seq_length: 16384

temperature: 1.0
top_p: 0.999

save_steps: 100
eval_strategy: "steps"
eval_steps: 50
```

### 3. Start NeMo Gym Servers

In Terminal 1:

```bash
cd Gym
source .venv/bin/activate

# Update env.yaml with vLLM endpoint
cat > env.yaml << EOF
policy_base_url: http://127.0.0.1:8000/v1
policy_api_key: EMPTY
policy_model_name: Qwen/Qwen2.5-1.5B-Instruct
hf_token: your_token_here
EOF

config_paths="resources_servers/workplace_assistant/configs/workplace_assistant.yaml,\
responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"

ng_run "+config_paths=[${config_paths}]"
```

### 4. Start vLLM Server

In Terminal 2:

```bash
cd trl
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 16384 \
  --host 0.0.0.0 \
  --port 8000
```

### 5. Run Training

In Terminal 3:

```bash
cd trl
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=1 python examples/scripts/nemo_gym/run_grpo_nemo_gym.py \
  --config config.yaml \
  --vllm_server_host 127.0.0.1 \
  --head_server_host 127.0.0.1
```

## Multi-Environment Training

Train on multiple environments simultaneously. We downloaded the workplace assistant dataset above. Now, lets generate a reasoning gym dataset:

```bash
cd Gym
source .venv/bin/activate
uv add reasoning-gym
cd resources_servers/reasoning_gym
python scripts/create_dataset.py \
    --task mini_sudoku \
    --size 2000 \
    --seed 42 \
    --output data/reasoning_gym/train_mini_sudoku.jsonl

python scripts/create_dataset.py \
    --task mini_sudoku \
    --size 50 \
    --seed 24 \
    --output data/reasoning_gym/val_mini_sudoku.jsonl
```

Next, shuffle the datasets into one file for training:
```bash
cat data/workplace_assistant/train.jsonl data/reasoning_gym/train_mini_sudoku.jsonl | shuf > train_multi_env.jsonl
```

Start NeMo Gym servers:
```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,\
resources_servers/workplace_assistant/configs/workplace_assistant.yaml,\
resources_servers/reasoning_gym/configs/reasoning_gym.yaml"

ng_run "+config_paths=[${config_paths}]"
```

Run multi environment training!
```bash
python run_grpo_nemo_gym.py --config config_multi_env.yaml
```

The training script automatically routes each example to the correct environment based on the `agent_ref` field in the dataset.

## Multi-Node Training

For training at scale, TRL supports multi-node distributed training with Slurm. See the example slurm script in https://github.com/huggingface/trl/blob/7669c006aa39199dda58d7bb9d166d2c411d7cac/examples/scripts/nemo_gym/submit.sh which serves vLLM on one node, and runs the training backend on 4 nodes.


## Documentation

Visit the huggingface integration docs for more details:

:::{button-link} https://huggingface.co/docs/trl/main/nemo_gym_integration
:color: primary
:class: sd-rounded-pill

TRL NeMo-Gym Integration Guide
:::

---

## What's Next?

After completing this tutorial, explore these options:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Use Other Training Environments
:link: https://github.com/NVIDIA-NeMo/Gym#-available-resource-servers

Browse available resource servers on GitHub to find other training environments.
+++
{bdg-secondary}`github` {bdg-secondary}`resource-servers`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build a Custom Training Environment
:link: creating-resource-server
:link-type: doc

Create your own resource server with custom tools and verification logic.
+++
{bdg-secondary}`tutorial` {bdg-secondary}`custom-tools`
:::

::::

## Resources

- [TRL GitHub Repository](https://github.com/huggingface/trl)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [TRL NeMo-Gym Integration](https://huggingface.co/docs/trl/main/nemo_gym_integration)
- [Training Script](https://github.com/huggingface/trl/blob/main/examples/scripts/nemo_gym/run_grpo_nemo_gym.py)
- [GRPO Trainer Documentation](https://huggingface.co/docs/trl/main/grpo_trainer)
