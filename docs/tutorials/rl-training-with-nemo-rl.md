(rl-training-with-nemo-rl)=

# RL Training with NeMo RL

**Goal**: Train a model with NeMo RL. Learn how to set up NeMo Gym + NeMo RL training environment, run tests, prepare data, and launch single and multi-node training runs!

Multinode Slurm script and run command are found at the bottom of this document. Please do the single node setup first! Do NOT skip it. Throughout this tutorial, you may see mentions of "Penguin". This refers to Gym's codename before it was fully open-sourced.

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

# Clone NeMo Gym
git clone https://github.com/NVIDIA-NeMo/Gym.git 3rdparty/Penguin-workspace/Penguin

# Pull necessary submodules (e.g. megatron, automodel, etc). Nothing Gym-specific.
git submodule update --init --recursive

# Initial setup
source /opt/nemo_rl_venv/bin/activate
uv sync --group={build,docs,dev,test} --extra penguin

# This will take 10-15 mins
# We add the HF token here to avoid HF rate limits
HF_HOME=.cache/ \
HF_TOKEN={your HF token} \
    ./examples/penguin/run_penguin_single_node_sanity_tests.sh

# If you used Gym previously, to run these tests properly, you may need to set `NRL_FORCE_REBUILD_VENVS=true` on an initial run or something.
# If you've run these tests before and are getting HF rate limit errors, you can add `HF_HUB_OFFLINE=1`
```


### Prepare NeMo Gym data

You will need to use Gym's data preparation command `ng_prepare_data` to prepare the data you intend to train on, including data that you already have locally. The `ng_prepare_data` command will add an `agent_ref` property to each example that tells NeMo Gym which agent server to route that example to!

Note: The `ng_prepare_data` command below includes the full set of configuration yaml paths (including the model yaml path). The configs you use to prepare data are exactly the same configs you use for training.

This command will output the data into the `data/bytedtsinghua_dapo17k`, that subsequent configs will point to.

```bash
# Setup Penguin local venv
cd 3rdparty/Penguin-workspace/Penguin
uv venv --python 3.12 --allow-existing
source .venv/bin/activate
uv sync --active --extra dev

# Prepare data
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml"
ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/bytedtsinghua_dapo17k \
    +mode=train_preparation +should_download=true

# Return to NeMo RL directory and Python env
cd ../../.. && source /opt/nemo_rl_venv/bin/activate
```

### Single node training

Launch a single node training job training Qwen 3 4B Instruct using the library judge math verifier on the DAPO 17K math dataset. We find that Qwen 3 4B Instruct is the smallest model that still provides experimental signal. We use the DAPO 17K math dataset since it is a very solid baseline set by the DAPO team. You should see training start and for the reward to end up roughly around 0.8 or greater.

Prerequisites for the command below:
1. A W&B API key
2. The above `ng_prepare_data` command has been run.


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

Place this script (named e.g. `temp_penguin_submit.sh`) in the root NeMo RL dir.

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

# Not sure why this is necessary, but ray.sub needs to be launched from the NeMo-RL root directory
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
