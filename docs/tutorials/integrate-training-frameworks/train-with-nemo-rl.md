(tutorial-train-with-nemo-rl)=

# Train with NeMo RL

In [Collecting Rollouts](../../get-started/rollout-collection.md), you generated scored interactions between your agent and environment. Now you'll use NeMo RL to run on-policy reinforcement learning training with your Gym environment.

:::{card}

**Goal**: Train a model with NeMo RL using Gym as the reward environment.

^^^

**In this tutorial, you will**:

1. Set up NeMo RL and NeMo Gym environments
2. Prepare training data with `ng_prepare_data`
3. Run single-node GRPO training
4. Scale to multi-node training on Slurm

:::

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← Get Started
:::

---

:::{warning}
This tutorial is **experimental** and may contain bugs. Proceed with caution.
:::

---

## Before You Begin

Make sure you have:

- ✅ Access to a GPU node (8× GPUs recommended)
- ✅ A Slurm cluster (for multi-node training)
- ✅ Hugging Face token (to download models)
- ✅ Weights & Biases API key (for logging)
- ✅ NeMo RL container image

**What you'll build**: An on-policy GRPO training loop that trains Qwen 3 4B Instruct on the DAPO 17K math dataset, using Gym's math verifier for rewards.

---

## 1. Access a GPU Node

Enter a GPU node on your Slurm cluster:

```bash
srun \
    --no-container-mount-home \
    --container-mounts=/shared/filesystem:/shared/filesystem \
    --container-image=/path/to/nemo-rl/container \
    --gres=gpu:8 \
    --nodes=1 --ntasks=1 --time 04:00:00 \
    --pty /bin/bash
```

```{list-table} Slurm Parameters
:header-rows: 1
:widths: 30 70

* - Parameter
  - Description
* - `--container-image`
  - Path to your NeMo RL container
* - `--container-mounts`
  - Mount your shared filesystem
* - `--gres=gpu:8`
  - Request 8 GPUs
* - `--time`
  - Job time limit
```

**✅ Success Check**: You should have a shell prompt inside the container on a GPU node.

---

## 2. Set Up NeMo RL and Gym

Clone and configure both repositories:

```bash
# Navigate to your workspace
cd /shared/filesystem/$USER

# Clone NeMo RL
git clone https://github.com/NVIDIA-NeMo/RL
cd RL

# Clone NeMo Gym into the expected location
git clone https://github.com/NVIDIA-NeMo/Gym.git 3rdparty/Penguin-workspace/Penguin

# Pull necessary submodules (Megatron, AutoModel, etc.)
git submodule update --init --recursive

# Activate NeMo RL environment and install dependencies
source /opt/nemo_rl_venv/bin/activate
uv sync --group={build,docs,dev,test} --extra penguin
```

Run sanity tests to verify the setup:

```bash
HF_HOME=.cache/ \
HF_TOKEN={your HF token} \
    ./examples/penguin/run_penguin_single_node_sanity_tests.sh
```

:::{tip}
- Tests take 10–15 minutes on first run
- If you've run Gym before, set `NRL_FORCE_REBUILD_VENVS=true`
- For HF rate limit errors, add `HF_HUB_OFFLINE=1`
:::

**✅ Success Check**: All sanity tests pass without errors.

---

## 3. Prepare Training Data

Use Gym's `ng_prepare_data` command to prepare your dataset. This adds an `agent_ref` property that tells Gym which agent server handles each example.

```bash
# Set up Gym's local environment
cd 3rdparty/Penguin-workspace/Penguin
uv venv --python 3.12 --allow-existing
source .venv/bin/activate
uv sync --active --extra dev

# Prepare the DAPO 17K dataset
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml"

ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/bytedtsinghua_dapo17k \
    +mode=train_preparation \
    +should_download=true

# Return to NeMo RL directory
cd ../../.. && source /opt/nemo_rl_venv/bin/activate
```

```{list-table} ng_prepare_data Parameters
:header-rows: 1
:widths: 30 70

* - Parameter
  - Description
* - `+config_paths`
  - Comma-separated list of config files (model + resource server)
* - `+output_dirpath`
  - Where to save prepared data
* - `+mode=train_preparation`
  - Preparation mode for training
* - `+should_download=true`
  - Download dataset if not present
```

**✅ Success Check**: Data files appear in `data/bytedtsinghua_dapo17k/`.

---

## 4. Run Single-Node Training

Launch GRPO training on a single node:

```bash
# Clean up any previous processes
pkill -f VllmAsyncGenerationWorker
ray stop --force
python -c "import ray; ray.shutdown()"

# Set experiment name
EXP_NAME="$(date +%Y%m%d)/penguin_grpo/qwen3_4binstruct/dapo17k_bytedtsinghua_test_001"
CONFIG_PATH=examples/penguin/grpo_dapo17k_bytedtsinghua_qwen3_4binstruct_nf.yaml

# Launch training
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

```{list-table} Key Training Parameters
:header-rows: 1
:widths: 35 65

* - Parameter
  - Description
* - `--config`
  - Path to training configuration YAML
* - `logger.wandb.project`
  - W&B project name for logging
* - `grpo.num_prompts_per_step`
  - Batch size (prompts per training step)
* - `grpo.max_num_steps`
  - Total training steps
* - `cluster.num_nodes`
  - Number of nodes (1 for single-node)
```

**✅ Success Check**: Training starts and reward reaches ~0.8 or higher.

---

## 5. Scale to Multi-Node Training

For larger-scale training, use Slurm to orchestrate multiple nodes.

### Create the Submit Script

Save this as `temp_penguin_submit.sh` in the NeMo RL root directory:

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

### Launch Multi-Node Training

```bash
WANDB_API_KEY={your W&B API key} \
EXP_NAME=penguin_grpo/qwen3_4binstruct/8nodes/dapo17k_bytedtsinghua_nf_001 \
NUM_ACTOR_NODES=8 \
REPO_LOCATION={your NeMo RL dir} \
    ./temp_penguin_submit.sh \
    --config=examples/penguin/grpo_dapo17k_bytedtsinghua_qwen3_4binstruct_nf.yaml \
    logger.wandb.project="$USER-nemo-gym-rl-integration"
```

**✅ Success Check**: Job submitted to Slurm queue; check with `squeue -u $USER`.

---

## Troubleshooting

:::{dropdown} HF rate limit errors
Add `HF_HUB_OFFLINE=1` to your command if you've already downloaded the models.
:::

:::{dropdown} Ray/vLLM processes not cleaning up
Run the cleanup commands before starting training:
```bash
pkill -f VllmAsyncGenerationWorker
ray stop --force
python -c "import ray; ray.shutdown()"
```
:::

:::{dropdown} Tests failing on re-run
Set `NRL_FORCE_REBUILD_VENVS=true` to rebuild virtual environments.
:::

:::{dropdown} "Penguin" references in logs
"Penguin" was Gym's internal codename before open-sourcing. These references are normal.
:::

---

## Learn More

For deeper understanding of the concepts used in this tutorial:

- {doc}`/training/datasets/prepare-data` — Details on `ng_prepare_data` and dataset configuration
- {doc}`/training/integrate/index` — HTTP integration architecture for custom frameworks
- {doc}`/about/concepts/training-integration-architecture` — Request lifecycle and token alignment

---

## What's Next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build Custom Environments
:link: /tutorials/creating-resource-server
:link-type: doc

Create a resource server with domain-specific tools and verification logic.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Understand the Architecture
:link: /about/concepts/training-integration-architecture
:link-type: doc

Deep dive into how Gym integrates with training frameworks.
:::

::::
