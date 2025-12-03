(rl-training-with-nemo-rl)=

# RL Training with NeMo RL - Experimental

:::{warning}
This tutorial is **experimental** and may contain bugs. Proceed with caution.
:::

**Goal**: Train a model with NeMo RL. Learn how to set up NeMo Gym + NeMo RL training environment, run tests, prepare data, and launch single and multi-node training runs!

## Overview

In this tutorial, you will train a language model using {term}`reinforcement learning <RL (Reinforcement Learning)>` with NeMo RL and NeMo Gym. Specifically, you will:

- **Model**: Train the **Qwen 3 4B Instruct** {term}`policy model`, which is the smallest model that provides meaningful experimental signal for this task
- **Dataset**: Use the **DAPO 17K math dataset** ([ByteDTsinghua/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)), a solid baseline dataset curated by the DAPO team containing 17,000 mathematical reasoning problems
- **Verification**: Leverage the **library judge math {term}`verifier`** to automatically score mathematical reasoning outputs
- **Training Method**: Use {term}`GRPO (Group Relative Policy Optimization)` for reinforcement learning
- **Expected Outcome**: Achieve a {term}`reward` score of **0.8 or greater**, indicating improved mathematical reasoning capabilities

This tutorial covers both single-node setup (for testing and verification) and multi-node training (for production-scale training runs on Slurm clusters).

:::{note}
Throughout this tutorial, you may see mentions of "Penguin". This refers to NeMo Gym's codename before it was fully open-sourced.
:::

## Prerequisites

Before starting this tutorial, ensure you have the following hardware and software requirements:

### Hardware Requirements

- **GPU**: Multi-GPU setup required for RL training
  - **Single-node testing**: 1 node with 8 GPUs (e.g., 8x A100 or H100 GPUs)
  - **Multi-node training**: 8+ nodes with 8 GPUs each recommended for production training
- **CPU**: Modern x86_64 processor
- **RAM**: 64 GB+ recommended per node
- **Storage**: 100 GB+ free disk space for:
  - NeMo RL repository and dependencies
  - Model checkpoints and training artifacts
  - Dataset storage (DAPO 17K and prepared data)

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+ or equivalent)
- **Python**: 3.12 or higher
- **NeMo RL Container**: Pre-built container image with NeMo RL dependencies
  - Path example: `/path/to/nemo-rl/container` (update with your actual container path)
- **Slurm**: For multi-node training on GPU clusters
- **Git**: For cloning repositories
- **UV Package Manager**: Python package manager (installed during setup)

### Optional API Keys and Accounts

- **Weights & Biases (W&B) API Key** (optional): For experiment tracking and visualization
  - Sign up at [wandb.ai](https://wandb.ai) if you don't have an account
  - Find your API key at [wandb.ai/authorize](https://wandb.ai/authorize)
  - If not provided, training will proceed without W&B logging
- **HuggingFace Token** (optional): For downloading models and datasets
  - Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  - Recommended to avoid rate limits when downloading models and datasets
  - Ensure you have accepted the model license for Qwen 3 4B Instruct

### Filesystem Access

- **Shared Filesystem**: Required for multi-node training
  - Example: `/shared/filesystem` mounted and accessible from all compute nodes
  - Used for storing code, data, checkpoints, and results

:::{important}
**Do the single node setup first. Do not skip it.** The single-node setup validates that your environment is configured correctly before attempting multi-node training.
:::

## Single GPU node setup to ensure correctness

### SSH or enter into a GPU node

Here is an example command to enter into a GPU node hosted on a Slurm cluster.
```bash
srun \
    --no-container-mount-home \
    --container-mounts=/shared/filesystem:/shared/filesystem \
    --container-image=/path/to/nemo-rl/container \
    --gres=gpu:8 \
    --nodes=1 --ntasks=1 --time 04:00:00 \
    --pty /bin/bash
```

### Setup NeMo RL and NeMo Gym

```bash
# CD into your preferred workspace
# cd /shared/filesystem/$USER

# Clone NeMo RL
git clone https://github.com/NVIDIA-NeMo/RL
cd RL
git checkout d2e9ef71f71a7afcf01eb809cb0bf1af2c7ed82a  # specific commit for compatibility

# Clone NeMo Gym
git clone https://github.com/NVIDIA-NeMo/Gym.git 3rdparty/Penguin-workspace/Penguin

# Pull necessary submodules (for example, megatron, automodel, and so on). Nothing Gym-specific.
git submodule update --init --recursive

# This will take 10 to 15 minutes
# We add the HF token here to avoid HF rate limits
NRL_FORCE_REBUILD_VENVS=true \
HF_HOME=.cache/ \
HF_TOKEN={your HF token} \
    ./examples/penguin/run_penguin_single_node_sanity_tests.sh
```

### Prepare NeMo Gym data

Before training, you need to prepare the DAPO 17K math dataset for use with NeMo Gym's RL training pipeline.

#### What is the DAPO 17K Math Dataset?

The **DAPO 17K math dataset** ([ByteDTsinghua/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) on HuggingFace) is a collection of approximately 17,000 mathematics problems spanning diverse mathematical topics and difficulty levels, including competition-level mathematics from benchmarks like AIME and AMC. Each problem is paired with a single correct integer answer label, making it well-suited for verification-based learning approaches where models learn through correctness feedback rather than step-by-step supervision. The dataset was designed with minimal annotation beyond ground-truth labels, enabling compatibility with binary reward signals and preference optimization frameworks commonly used in RL training.

#### Why Data Preparation is Needed

The `ng_prepare_data` command performs several critical steps:

1. **Downloads the dataset** from HuggingFace (when `should_download=true`)
2. **Adds routing metadata**: Adds an `agent_ref` property to each example that tells NeMo Gym which agent server to route that example to during training
3. **Formats for training**: Structures the data for compatibility with the NeMo RL training loop
4. **Outputs prepared data**: Saves processed data to `data/bytedtsinghua_dapo17k` for use in training

The configuration files you provide to `ng_prepare_data` are the same ones used during training, ensuring consistency.

#### Data Preparation Commands

```bash
# Setup Penguin local venv
cd 3rdparty/Penguin-workspace/Penguin
uv venv --python 3.12 --allow-existing
uv sync --active --extra dev

# Prepare data - downloads DAPO 17K from HuggingFace and processes it
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml"
ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/bytedtsinghua_dapo17k \
    +mode=train_preparation \
    +should_download=true

# Return to NeMo RL directory and Python env
cd ../../.. && source /opt/nemo_rl_venv/bin/activate
```

**What this command does**:
- Downloads the DAPO 17K dataset from HuggingFace (requires HuggingFace token set earlier)
- Processes it according to the `math_with_judge` resource server configuration
- Outputs the prepared dataset to `data/bytedtsinghua_dapo17k/`
- This output directory is referenced in the training configuration files

:::{tip}
If you've already downloaded the dataset or are re-running preparation, you can set `+should_download=false` to skip the download step and use your local copy.
:::

### Single node training

Now you're ready to launch a single-node training run to verify your setup works correctly before scaling to multi-node training.

**Prerequisites for this step**:

1. Completed the `ng_prepare_data` command above (data should be in `data/bytedtsinghua_dapo17k/`)
2. W&B API key (see [Prerequisites](#prerequisites) section above)
3. HuggingFace token configured (see setup steps above)


```bash
# Run example training config for single node
pkill -f VllmAsyncGenerationWorker
ray stop --force
python -c "import ray; ray.shutdown()"
EXP_NAME="$(date +%Y%m%d)/penguin_grpo/qwen3_4binstruct/dapo17k_bytedtsinghua_test_001"
CONFIG_PATH=examples/penguin/grpo_dapo17k_bytedtsinghua_qwen3_4binstruct_nf.yaml
HF_HOME=.cache/ \
WANDB_API_KEY={your W&B API key} \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="{your username}-nemo-gym-rl-integration" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    grpo.val_at_start=false \
    ++grpo.num_prompts_per_step=4 \
    ++grpo.max_num_steps=3 \
    ++policy.dtensor_cfg.clear_cache_every_n_steps=1 \
    ++cluster.num_nodes=1 \
    checkpointing.checkpoint_dir=results/$EXP_NAME &
```

## Multi node

We will run a multi-node training job on a Slurm cluster. First, we will write our Slurm job launch script and then run it.

### Submit script

Place this script (named, for example, `temp_penguin_submit.sh`) in the root NeMo RL dir.

```bash
# ----- PARAMETERS -----
# WANDB_API_KEY, EXP_NAME, NUM_ACTOR_NODES, REPO_LOCATION

# ----- CONSTANTS -----
CONTAINER_IMAGE_PATH=/path/to/nemo-rl/container

read -r -d '' COMMAND <<EOF
cd ${REPO_LOCATION}

HF_HOME=.cache/ \
HF_HUB_OFFLINE=1 \
WANDB_API_KEY=$WANDB_API_KEY \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    cluster.num_nodes=$NUM_ACTOR_NODES \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    checkpointing.checkpoint_dir=results/$EXP_NAME \
    $@
EOF

echo -e "Running command:\n$COMMAND"

# ray.sub needs to be launched from the NeMo-RL root directory
cd $REPO_LOCATION
COMMAND=$COMMAND \
CONTAINER=$CONTAINER_IMAGE_PATH \
MOUNTS="/shared/filesystem:/shared/filesystem" \
sbatch \
    --nodes=$NUM_ACTOR_NODES \
    --time=4:0:0 \
    --job-name=$EXP_NAME \
    --gres=gpu:8 \
    ray.sub
```

### Submit script run

Run this command to launch the training job! This uses the same configuration as the single node setup, just with a larger batch size for actual training purposes.

```bash
WANDB_API_KEY={your W&B API key} \
EXP_NAME=penguin_grpo/qwen3_4binstruct/8nodes/dapo17k_bytedtsinghua_nf_001 \
NUM_ACTOR_NODES=8 \
REPO_LOCATION={your NeMo RL dir}\
    ./temp_penguin_submit.sh \
    --config=examples/penguin/grpo_dapo17k_bytedtsinghua_qwen3_4binstruct_nf.yaml \
    logger.wandb.project="$USER-nemo-gym-rl-integration"
```
