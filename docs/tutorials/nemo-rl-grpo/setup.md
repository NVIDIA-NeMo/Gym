(training-nemo-rl-grpo-setup)=

# Setup

This section guides you through setting up NeMo RL and NeMo Gym for GRPO training.

:::{card}

**Goal**: Set up your environment for GRPO training with NeMo RL and NeMo Gym.

^^^

**In this section, you will**:

1. Launch an interactive GPU session
2. Clone and install NeMo RL and NeMo Gym
3. Prepare the Workplace Assistant dataset
4. Run sanity tests to validate the setup

:::

:::{button-ref} training-nemo-rl-grpo-nemo-rl-configuration
:color: secondary
:outline:
:ref-type: ref

← Previous: NeMo RL Configuration
:::

---

## Before You Begin

Make sure you have:

- ✅ Access to a Slurm cluster with GPU nodes
- ✅ A shared filesystem accessible from all nodes
- ✅ (Optional) HuggingFace token for downloading models

---

## 1. Enter a GPU Node

**Estimated time**: ~5 minutes

Launch an interactive Slurm session to run training commands. Refer to the [NeMo RL Cluster Setup documentation](https://docs.nvidia.com/nemo/rl/latest/cluster.html#interactive-launching) for more details.

```bash
NUM_ACTOR_NODES=1
ACCOUNT=<ACCOUNT_NAME>
JOB_NAME=<JOB_NAME>
PARTITION=<PARTITION>

# Use the official NeMo RL container from NGC
# See: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-rl
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.4.0
CONTAINER_WORKDIR=$PWD
MOUNTS="$PWD:$PWD"
srun \
    --nodes=${NUM_ACTOR_NODES} \
    --ntasks=1 \
    --account=${ACCOUNT} \
    --job-name=${JOB_NAME} \
    --partition=${PARTITION} \
    --time=04:00:00 \
    --gres=gpu:8 \
    --no-container-mount-home \
    --container-name=nemo-gym \
    --container-mounts="${MOUNTS}" \
    --container-image="${CONTAINER}" \
    --container-workdir=$CONTAINER_WORKDIR \
    --pty /bin/bash
```

**✅ Success Check**: You should be inside the container with a bash prompt.

---

## 2. Clone and Setup NeMo RL + NeMo Gym

**Estimated time**: ~15-20 minutes

```bash
# Clone NeMo RL repository
git clone https://github.com/NVIDIA-NeMo/RL
cd RL

# Clone NeMo Gym as a submodule
git clone https://github.com/NVIDIA-NeMo/Gym.git 3rdparty/Gym-workspace/Gym

# Initialize all submodules (Megatron, AutoModel, etc.)
git submodule update --init --recursive

# Activate the NeMo RL virtual environment
source /opt/nemo_rl_venv/bin/activate

# Install dependencies
uv sync --group={build,docs,dev,test} --extra nemo_gym
```

**✅ Success Check**: No errors during installation and `uv sync` completes successfully.

---

## 3. Prepare NeMo Gym Data

**Estimated time**: ~5-10 minutes

The Workplace Assistant dataset must be downloaded from HuggingFace and prepared for training. This is a two-step process:

This runs `ng_prepare_data` to download and validate the dataset, and to add an `agent_ref` property to each example that tells NeMo Gym which agent server should handle that example.

```bash
# Setup Gym local venv
cd 3rdparty/Gym-workspace/Gym
uv venv --python 3.12 --allow-existing
source .venv/bin/activate
uv sync --active --extra dev

config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,\
resources_servers/workplace_assistant/configs/workplace_assistant.yaml"

ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=resources_servers/workplace_assistant/data \
    +mode=train_preparation \
    +should_download=true

# Return to NeMo RL directory and Python env
cd ../../.. && source /opt/nemo_rl_venv/bin/activate
```

**✅ Success Check**: Dataset files are created in `3rdparty/Gym-workspace/Gym/data/workplace_assistant/`.

---

## 4. Run Sanity Tests

**Estimated time**: ~10-15 minutes

Validate your setup before training:

```bash
HF_HOME=.cache/ \
HF_TOKEN=${HF_TOKEN} \
    ./examples/nemo_gym/run_nemo_gym_single_node_sanity_tests.sh
```

**✅ Success Check**: All tests pass without errors.

:::{tip}
If you've run these tests before and encounter HuggingFace rate limit errors, add `HF_HUB_OFFLINE=1` to the command.
:::

---

:::{button-ref} training-nemo-rl-grpo-single-node-training
:color: primary
:ref-type: ref

Next: Single Node Training →
:::
