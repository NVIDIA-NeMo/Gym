(training-nemo-rl-grpo-setup)=

# Setup

## Enter a GPU Node

**Estimated Time:** ~5 minutes

Launch an interactive Slurm session to run training commands. Refer to the [NeMo RL Cluster Setup documentation](https://docs.nvidia.com/nemo/rl/latest/cluster.html#interactive-launching) for more details.

If this is your first time downloading this Docker image, the srun command below will take 5-10 mins.

:::{tip}
If you are using enroot as a containerization framework, you can pull the container after defining `$CONTAINER_IMAGE_PATH`
```bash
mkdir -p "$(dirname "$CONTAINER_IMAGE_PATH")"
enroot import -o "$CONTAINER_IMAGE_PATH" "docker://${CONTAINER_IMAGE_PATH}"
# Swap to local container path
CONTAINER_IMAGE_PATH=./$CONTAINER_IMAGE_PATH
```
:::

```bash
# Use the official NeMo RL container from NGC
# See: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-rl
CONTAINER_IMAGE_PATH=nvcr.io/nvidia/nemo-rl:v0.4.0

NUM_ACTOR_NODES=1
ACCOUNT=<ACCOUNT_NAME>
PARTITION=<PARTITION>

CONTAINER_WORKDIR=$PWD
MOUNTS="$PWD:$PWD"
srun \
    --nodes=${NUM_ACTOR_NODES} \
    --ntasks=1 \
    --account=${ACCOUNT} \
    --partition=${PARTITION} \
    --time=04:00:00 \
    --gres=gpu:8 \
    --no-container-mount-home \
    --container-name=nemo-gym \
    --container-mounts="${MOUNTS}" \
    --container-image="${CONTAINER_IMAGE_PATH}" \
    --container-workdir=$CONTAINER_WORKDIR \
    --pty /bin/bash
```

## Clone and Setup NeMo RL + NeMo Gym

**Estimated Time:** ~5-10 minutes

For the first setup on your local filesystem:
```bash
# Clone NeMo RL repository
git clone https://github.com/NVIDIA-NeMo/RL
cd RL

# Clone NeMo Gym as a submodule
git clone https://github.com/NVIDIA-NeMo/Gym.git 3rdparty/Gym-workspace/Gym
```

Every time you enter a new container:
```bash
# CD into your NeMo RL folder
cd /path/to/nemo/rl

# Initialize all submodules (Megatron, AutoModel, etc.)
git submodule update --init --recursive

# Activate the NeMo RL virtual environment
source /opt/nemo_rl_venv/bin/activate

# Install dependencies. This may take may take 5-10 minutes!
uv sync --group={build,docs,dev,test} --extra nemo_gym
uv run nemo_rl/utils/prefetch_venvs.py
```

## Prepare NeMo Gym Data

**Estimated Time:** ~5 minutes

The Workplace Assistant dataset must be downloaded from HuggingFace and prepared for training. This is a two-step process:

This runs `ng_prepare_data` to download and validate the dataset, and to add an `agent_ref` property to each example that tells NeMo Gym which agent server should handle that example.

Clone and setup Gym Python environment
```bash
# Setup Gym local venv
cd 3rdparty/Gym-workspace/Gym
uv venv --python 3.12 --allow-existing .venv
source .venv/bin/activate
uv sync --active --extra dev
```

Add your HuggingFace token to download Gym datasets from HuggingFace. This command will store your HF token in a file that is excluded from Git, so it will never be committed or pushed!
```bash
echo "hf_token: {your HF token}" >> env.yaml
```

Prepare data
```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,\
resources_servers/workplace_assistant/configs/workplace_assistant.yaml"

ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface
```

Return to NeMo RL directory and Python env
```bash
cd ../../.. && source /opt/nemo_rl_venv/bin/activate
```

## Run Sanity Tests

**Estimated Time:** ~5-10 minutes

Download the model used in the following tests:
```bash
HF_HOME=$PWD/.cache/ \
HF_TOKEN={your HF token} \
    hf download Qwen/Qwen3-0.6B
```

Validate your setup before training.
```bash
HF_HOME=$PWD/.cache/ \
    ./examples/nemo_gym/run_nemo_gym_single_node_sanity_tests.sh
```

> **Note**: If you've run these tests before and encounter HuggingFace rate limit errors, try adding `HF_HUB_OFFLINE=1` to the command.
