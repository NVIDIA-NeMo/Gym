# GRPO Training with NeMo RL: Multi-step tool calling  on Nemotron Nano v2 9B

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

---

## Training Configuration

---

## Running Training

### Single Node Training (interactive mode)

**Estimated Time:** ~2-4 hours

Run these commands **from inside the container** after attaching via the interactive session from Step 1:

```bash
# Clean up any existing Ray/vLLM processes
pkill -f VllmAsyncGenerationWorker
ray stop --force
python -c "import ray; ray.shutdown()"

# Set experiment name with timestamp
EXP_NAME="$(date +%Y%m%d)/penguin_grpo/nemotron_nano_v2_9b/workplace_assistant_001"

# Configuration file path
CONFIG_PATH=examples/penguin/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml

# Launch training
# Set these environment variables before running:
#   HF_TOKEN: Your Hugging Face token for model downloads
#   WANDB_API_KEY: Your Weights & Biases API key for logging
#   TORCH_CUDA_ARCH_LIST: CUDA architectures compute capability
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

#   NRL_FORCE_REBUILD_VENVS: Set to true on first run to rebuild venvs
TORCH_CUDA_ARCH_LIST="9.0 10.0" \
HF_HOME=.cache/ \
HF_TOKEN="your_hugging_face_token" \
WANDB_API_KEY="your_wandb_api_key" \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="${USER}-nemo-gym-rl-integration" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME
```

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

---

## Expected Results

### Training Metrics

Monitor these metrics in W&B to track progress:

| Metric | Initial | After 1 Epoch | Description |
|--------|---------|---------------|-------------|
| `train:reward_mean` | ~0.1-0.2 | ~0.5-0.7 | Average reward per batch |
| `val:accuracy` | ~0.15 | ~0.5-0.6 | Validation task completion rate |
| `train:loss` | ~0.5 | ~0.2-0.3 | GRPO policy loss |

### Checkpoint Outputs

Checkpoints are saved to:
```
results/<EXP_NAME>/
├── step_6/
├── step_12/
├── step_18/
└── ...
```

The best checkpoint (highest `val:accuracy`) is retained based on `checkpointing.keep_top_k: 3`.

### Success Criteria

Training is successful when:
- Reward mean increases consistently over steps
- Validation accuracy improves from baseline (~15%) to 50%+
- No OOM (Out of Memory) errors
- Checkpoints are saved at specified intervals

### Validation Reward Plot

<!-- TODO: Add validation reward plot showing improvement over training steps -->
![Validation Reward Plot](images/val_reward_placeholder.png)
*Expected: Validation reward increasing from ~0.15 to ~0.5+ over the course of training.*

### Measuring Real-World Improvement

The Workplace Assistant environment's tool-calling tasks correlate with performance on the [Berkeley Function Calling Leaderboard (BFCL) v3](https://gorilla.cs.berkeley.edu/leaderboard.html) benchmark. To measure improvement, evaluate the Nemotron Nano v2 9B model on BFCL v3 before and after training and compare.  You should observe measurable improvement in tool-calling accuracy

You can run BFCL v3 evaluations using [NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator), which supports BFCL v3. See the [NeMo Evaluator docs](https://github.com/NVIDIA-NeMo/Evaluator#-supported-benchmarks-and-evaluation-harnesses) for full setup instructions and supported benchmarks.

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| HuggingFace rate limits | Specify your HF API token and/or add `HF_HUB_OFFLINE=1` after the initial download |
| vLLM process not shutting down | Run `pkill -f VllmAsyncGenerationWorker` before training |
| Ray cluster issues | Run `ray stop --force` before training |
| CUDA OOM | Increase `tensor_parallel_size`, lower batch sizes |
| Slow initial startup | Set `NRL_FORCE_REBUILD_VENVS=true` on first run only; if `uv` gets rate limited, set this back to `false` |

### Log Locations

```
logs/grpo-workplace-assistant-nemotron-nano-v2-9b/  # Training logs
results/<EXP_NAME>/                                  # Checkpoints and metrics
.cache/                                              # HuggingFace model cache
```

---

## Next Steps

After completing this tutorial, explore:

1. **Scale Up**: Try multi-node training for faster convergence and larger batch sizes
2. **Hyperparameter Tuning**: Adjust learning rate, number of generations, or reward normalization 
3. **Deploy Your Agent**: Export the trained checkpoint and deploy it with vLLM or NVIDIA NIM to build a production workplace assistant that integrates with real calendar, email, and file management APIs

### Related Tutorials

- [RL Training with NeMo RL](./rl-training-with-nemo-rl.md) - General RL training guide
- [GRPO Loss Configuration](../../docs/guides/grpo.md) - Advanced loss function customization
- [Sequence Packing](../../docs/design-docs/sequence-packing-and-dynamic-batching.md) - Optimize training throughput

---

## References

- **NeMo RL Repository**: [github.com/NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL)
- **NeMo Gym Repository**: [github.com/NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym)

---

## Appendix: Full Configuration Reference

The complete training configuration is available at:

[`examples/penguin/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/penguin/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml)
