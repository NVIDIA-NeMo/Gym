(training-nemo-rl-grpo-setup)=

# Setup

Now that you understand the configuration parameters for GRPO training, it's time to set up your environment. This involves launching containers, installing dependencies, and preparing your training data—the foundation for everything that follows.

:::{card}

**Goal**: Set up your environment for GRPO training with NeMo RL and NeMo Gym.

^^^

**In this section, you will**:

1. Launch an interactive GPU session
2. Clone and install NeMo RL and NeMo Gym
3. Run sanity tests to validate the setup
4. Prepare the Workplace Assistant dataset

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
- ✅ HuggingFace token for downloading models

---

## 1. Enter a GPU Node

**Estimated time**: ~5 minutes

Launch an interactive Slurm session to run training commands. Refer to the [NeMo RL Cluster Setup documentation](https://docs.nvidia.com/nemo/rl/latest/cluster.html#interactive-launching) for more details.

If this is your first time downloading this Docker image, the `srun` command below will take 5-10 minutes.

:::{tip}
If you are using enroot as a containerization framework, you can pull the container after defining `$CONTAINER_IMAGE_PATH`:

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
CONTAINER_IMAGE_PATH=nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano

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

**✅ Success Check**: You should be inside the container with a bash prompt.

---

## 2. Clone and Setup NeMo RL + NeMo Gym

**Estimated time**: ~5-10 minutes

For the first setup on your local filesystem:

```bash
# Clone NeMo RL repository
git clone https://github.com/NVIDIA-NeMo/RL
cd RL

# Initialize all submodules (Gym, Megatron, AutoModel, etc.)
git submodule update --init --recursive
```

**✅ Success Check**: No errors during installation and `uv sync` completes successfully.

---

## 3. Run Sanity Tests

**Estimated time**: ~5-10 minutes

Download the model used in the following tests:

```bash
HF_HOME=$PWD/.cache/ \
HF_TOKEN={your HF token} \
    hf download Qwen/Qwen3-0.6B
```

Validate your setup before training:

```bash
HF_HOME=$PWD/.cache/ \
    ./examples/nemo_gym/run_nemo_gym_single_node_sanity_tests.sh
```

**✅ Success Check**: All tests pass without errors.

:::{tip}
You can clean up any existing or leftover Ray/vLLM processes using the following commands:
```bash
pkill -f VLLM
ray stop --force
uv run python -c "import ray; ray.shutdown()"
```
:::

---

## 4. Prepare NeMo Gym Data

**Estimated time**: ~5 minutes

The Workplace Assistant dataset must be downloaded from HuggingFace and prepared for training. This runs `ng_prepare_data` to download and validate the dataset, and to add an `agent_ref` property to each example that tells NeMo Gym which agent server should handle that example.

Clone and setup the Gym Python environment:

```bash
# Setup Gym local venv
cd 3rdparty/Gym-workspace/Gym
uv venv --python 3.12 --allow-existing .venv
source .venv/bin/activate
uv sync --active --extra dev
```

Add your HuggingFace token to download Gym datasets from HuggingFace. This command will store your HF token in a file that is excluded from Git, so it will never be committed or pushed:

```bash
echo "hf_token: {your HF token}" >> env.yaml
```

Prepare the data:

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,\
resources_servers/workplace_assistant/configs/workplace_assistant.yaml"

ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface
```

Return to the NeMo RL Python environment and directory:

```bash
deactivate
cd ../../..
```

**✅ Success Check**: Dataset files are created in `3rdparty/Gym-workspace/Gym/data/workplace_assistant/`.

---

:::{button-ref} training-nemo-rl-grpo-single-node-training
:color: primary
:ref-type: ref

Next: Single Node Training →
:::
