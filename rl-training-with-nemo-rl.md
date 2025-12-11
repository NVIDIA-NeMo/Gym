(rl-training-with-nemo-rl)=

# RL Training with NeMo RL - Experimental

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
