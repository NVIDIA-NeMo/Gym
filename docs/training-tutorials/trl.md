(training-trl)=

# RL Training with TRL

This tutorial demonstrates how to use [Hugging Face TRL](https://github.com/huggingface/trl) to train models in NeMo Gym environments using GRPO.

**TRL (Transformer Reinforcement Learning)** is Hugging Face's library for post-training foundation models. It provides implementations of algorithms like SFT, GRPO, and DPO, with support for distributed training with vLLM inference.

The integration supports multi-step and multi-turn rollouts, multi-environment training, and any NeMo Gym environment (thoroughly tested: workplace assistant, reasoning gym, MCQA, and math with judge).

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


---

## Steps to get started

### Install TRL and NeMo Gym

1. **Install TRL with vLLM support**

   ```bash
   cd trl/
   uv venv
   source .venv/bin/activate
   uv sync --extra vllm
   ```

2. **Install NeMo Gym in a separate virtual environment**

   ```bash
   # deactivate trl venv
   deactivate
   git clone https://github.com/NVIDIA-NeMo/Gym.git
   cd Gym
   uv venv --python 3.12
   source .venv/bin/activate
   uv sync
   ```


### Download a Dataset

1. **Prepare Dataset**

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

#### Dataset Format

NeMo Gym datasets are stored as JSONL. Each line contains a task with input messages, tool definitions, metadata such as ground truth for verification, and an agent server reference. The following example shows the workplace dataset structure. Metadata fields can differ between datasets, as long as the corresponding resources server uses the fields appropriately.

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "Move any of jinsoo's tasks that are in review to completed"}
    ],
    "tools": [...],
    "parallel_tool_calls": false,
    "temperature": 1
  },
  "ground_truth": [
    {"name": "project_management_update_task", "arguments": "{...}"},
    ...
  ],
  "category": "workbench_project_management",
  "environment_name": "workbench",
  "agent_ref": {
    "type": "responses_api_agents",
    "name": "workplace_assistant_simple_agent"
  }
}
```

2. **Update Environment Config**

   Update `env.yaml` in `Gym/` to include model information:

   ```yaml
   policy_base_url: http://127.0.0.1:8000/v1
   policy_api_key: EMPTY
   policy_model_name: Qwen/Qwen2.5-1.5B-Instruct
   hf_token: ...
   ```

3. **Update Training Config**

   Update `examples/scripts/nemo_gym/config.yaml` to point to the dataset generated above, and any other optional modifications.

### Start vLLM and NeMo Gym Servers and Run Training

The following steps run in 3 terminals. It can also be ran with processes in the background, or using tmux.

1. **Start NeMo Gym Servers** (Terminal 1)

   ```bash
   cd Gym/
   source .venv/bin/activate

   config_paths="resources_servers/workplace_assistant/configs/workplace_assistant.yaml,\
   responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"

   ng_run "+config_paths=[${config_paths}]"
   ```

   This starts:
   - **Agent server**: Orchestrates rollouts using resource servers and model servers
   - **Resources server**: Supports environment logic such as state-management, tool implementations, and task verification
   - **Model server**: Adapts vLLM server requests to support NeMo Gym agents and on-policy RL training while ensuring OpenAI API compatibility
   - **Head server**: Manages servers used in training enabling their discovery

2. **Start TRL vLLM Server on GPU 0** (Terminal 2)

   ```bash
   cd trl/
   source .venv/bin/activate
   CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
     --model Qwen/Qwen2.5-1.5B-Instruct \
     --max-model-len 16384 \
     --host 0.0.0.0 \
     --port 8000
   ```

3. **Run Training on GPU 1** (Terminal 3)

   ```bash
   source trl/.venv/bin/activate
   cd trl/examples/scripts/nemo_gym
   export WANDB_API_KEY=...
   uv add omegaconf

   CUDA_VISIBLE_DEVICES=1 python train_multi_environment.py --config config.yaml
   ```

## Multi-Environment Training

Train on multiple environments simultaneously. This example combines the workplace assistant dataset from the previous section with a reasoning gym dataset.

1. **Generate Reasoning Gym Dataset**

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

2. **Shuffle Datasets**

   Combine datasets into one file for training:

   ```bash
   cat data/workplace_assistant/train.jsonl data/reasoning_gym/train_mini_sudoku.jsonl | shuf > train_multi_env.jsonl
   ```

3. **Start NeMo Gym Servers**

   ```bash
   config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,\
   resources_servers/workplace_assistant/configs/workplace_assistant.yaml,\
   resources_servers/reasoning_gym/configs/reasoning_gym.yaml"

   ng_run "+config_paths=[${config_paths}]"
   ```

4. **Run Training**

Update the config to point at the new dataset, then run training:

   ```bash
   python train_multi_environment.py --config config.yaml
   ```

The training script automatically routes each example to the correct environment based on the `agent_ref` field in the dataset.

## Multi-Node Training with Slurm

An example five-node training script is provided in `submit.sh`. Nodes one through four run the training algorithm, while node five runs vLLM inference for NeMo Gym agent rollouts.

1. **Configure the Script**

   Update `submit.sh` with your Slurm account, partition, paths to your project directory, and updated training configs.

1. **Submit the Job**

   ```bash
   sbatch submit.sh
   ```

1. **Monitor Training**

   ```bash
   tail -f logs/<job_id>/*
   ```

> **Tip**: Set up wandb logging for detailed training metrics. For more details on TRL's vLLM integration, refer to the vLLM integration page.


## Documentation

Visit the Hugging Face integration docs for more details:

:::{button-link} https://huggingface.co/docs/trl/main/nemo_gym_integration
:color: primary
:class: sd-rounded-pill

TRL NeMo-Gym Integration Guide
:::

---

## Next Steps

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
