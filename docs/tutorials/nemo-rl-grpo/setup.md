(training-nemo-rl-grpo-setup)=

# Setup

## Setup Instructions

### Step 1: Enter a GPU Node

**Estimated Time:** ~5 minutes

Launch an interactive Slurm session to run training commands. See the [NeMo RL Cluster Setup documentation](https://docs.nvidia.com/nemo/rl/latest/cluster.html#interactive-launching) for more details.

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

### Step 2: Clone and Setup NeMo RL + NeMo Gym

**Estimated Time:** ~15-20 minutes

```bash
# Clone NeMo RL repository
git clone https://github.com/NVIDIA-NeMo/RL
cd RL

# Clone NeMo Gym as a submodule
git clone https://github.com/NVIDIA-NeMo/Gym.git 3rdparty/Penguin-workspace/Penguin

# Initialize all submodules (Megatron, AutoModel, etc.)
git submodule update --init --recursive

# This will remove any stale cached Ray venv and rebuilt it
# TODO: This is a WAR. Need a formal fix.
rm -rf /opt/ray_venvs/*

# Activate the NeMo RL virtual environment
source /opt/nemo_rl_venv/bin/activate

# Install dependencies
uv sync --group={build,docs,dev,test} --extra nemo_gym
```

### Step 3: Prepare NeMo Gym Data

**Estimated Time:** ~5-10 minutes

The Workplace Assistant dataset must be downloaded from HuggingFace and prepared for training. This is a two-step process:

This runs `ng_prepare_data` to download and validate the dataset, and to add an `agent_ref` property to each example that tells NeMo Gym which agent server should handle that example.

```bash
HF_TOKEN=SPECIFY_HF_TOKEN

# Setup Penguin local venv
cd 3rdparty/Penguin-workspace/Penguin
uv venv --python 3.12 --allow-existing
source .venv/bin/activate
uv sync --active --extra dev

config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,\
resources_servers/workplace_assistant/configs/workplace_assistant.yaml"

ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=resources_servers/workplace_assistant/data \
    +mode=train_preparation \
    +hf_token=$HF_TOKEN \
    +should_download=true

# Return to NeMo RL directory and Python env
cd ../../.. && source /opt/nemo_rl_venv/bin/activate
```

### Step 4: Run Sanity Tests (optional but recommended)

**Estimated Time:** ~10-15 minutes

Validate your setup before training:

```bash
HF_HOME=.cache/ \
HF_TOKEN=${HF_TOKEN} \
    ./examples/nemo_gym/run_nemo_gym_single_node_sanity_tests.sh
```

> **Note**: If you've run these tests before and encounter HuggingFace rate limit errors, add `HF_HUB_OFFLINE=1` to the command.

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
