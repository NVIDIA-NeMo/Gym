(training-nemo-rl-grpo-multi-node-training)=

# Multi-Node Training

This section guides you through running a multi-node GRPO training job for production-scale training.

:::{card}

**Goal**: Scale training to multiple nodes using Slurm batch submission.

^^^

**In this section, you will**:

1. Create a Slurm submit script
2. Launch a multi-node training job

:::

:::{button-ref} training-nemo-rl-grpo-single-node-training
:color: secondary
:outline:
:ref-type: ref

← Previous: Single Node Training
:::

---

## Before You Begin

:::{important}
**Complete {doc}`Single Node Training <single-node-training>` first.** The single-node setup validates that your environment is configured correctly before attempting multi-node training.
:::

Make sure you have:

- ✅ Completed {doc}`Single Node Training <single-node-training>` successfully
- ✅ Access to 8+ GPU nodes for production training
- ✅ (Optional) W&B API key for logging

:::{note}
Run these commands from the **Slurm login/head node**, not from inside the interactive container. This submits a batch job that runs independently.
:::

---

## 1. Create the Submit Script

Place this script (for example, `temp_nemo_gym_submit.sh`) in the NeMo RL root directory:

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
uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
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

**✅ Success Check**: Script saved to the NeMo RL root directory.

---

## 2. Submit the Training Job

Run this command to launch the training job:

```bash
# Replace placeholders with your actual values
HF_TOKEN={your HF token} \
WANDB_API_KEY={your W&B API key} \
EXP_NAME=nemo_gym_grpo/qwen3_4binstruct/8nodes/dapo17k_bytedtsinghua_nf_001 \
NUM_ACTOR_NODES=8 \
REPO_LOCATION={your NeMo RL dir}\
    ./temp_nemo_gym_submit.sh \
    --config=examples/nemo_gym/grpo_dapo17k_bytedtsinghua_qwen3_4binstruct_nf.yaml \
    logger.wandb.project="$USER-nemo-gym-rl-integration"
```

**✅ Success Check**: Job submitted to Slurm and running across 8 nodes.

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
:link: ../creating-resource-server
:link-type: doc

Create your own resource server with custom tools and verification logic.
+++
{bdg-secondary}`tutorial` {bdg-secondary}`custom-tools`
:::

::::
