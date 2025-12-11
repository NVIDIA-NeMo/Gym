(training-nemo-rl-grpo-multi-node-training)=

# Multi-Node Training

TODO STILL

:::{important}
**Do the {doc}`Single Node Training <single-node-training>` first. Do not skip it.** The single-node setup validates that your environment is configured correctly before attempting multi-node training.
:::



### Multi-Node Training

For production training, scale to multiple nodes by changing `cluster.num_nodes`. This example uses **batch mode** (the `COMMAND` variable specifies what to run automatically when the job starts).

> **Note**: Run this command from the **Slurm login/head node**, not from inside the interactive container from Step 1. This submits a new batch job that will run independently.

```bash
# Set experiment name
EXP_NAME="penguin_grpo/nemotron_nano_v2_9b/2nodes/workplace_assistant_001"
CONFIG_PATH=examples/penguin/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml

# Submit multi-node job
# Set these environment variables before running:
#   HF_TOKEN: Your Hugging Face token for model downloads
#   WANDB_API_KEY: Your Weights & Biases API key for logging
#   NUM_NODES: Number of GPU nodes to use (2, 4, 8, etc.)
NUM_NODES=2
COMMAND="TORCH_CUDA_ARCH_LIST='9.0 10.0' HF_HOME=.cache/ HF_TOKEN='your_hf_token' WANDB_API_KEY='your_wandb_api_key' uv run python examples/penguin/run_grpo_penguin.py --config=$CONFIG_PATH cluster.num_nodes=$NUM_NODES logger.wandb.project=${USER}-nemo-gym-rl-integration logger.wandb.name=$EXP_NAME logger.log_dir=results/$EXP_NAME checkpointing.checkpoint_dir=results/$EXP_NAME" \
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.4.0 \
MOUNTS="/shared/filesystem:/shared/filesystem" \
sbatch \
    --nodes=$NUM_NODES \
    --time=4:0:0 \
    --job-name=grpo-workplace-assistant \
    --gres=gpu:8 \
    ray.sub
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

