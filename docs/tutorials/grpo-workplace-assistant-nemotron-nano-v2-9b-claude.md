# GRPO Training with NeMo RL: Workplace Assistant on Nemotron Nano v2 9B

## Overview

This tutorial demonstrates how to train NVIDIA's Nemotron Nano v2 9B model using **GRPO (Group Relative Policy Optimization)** with the **Workplace Assistant** environment in NeMo Gym. By the end of this tutorial, you will have trained a language model to perform multi-step tool-calling tasks in a realistic office productivity setting.

**Why this matters:** Tool-calling is a critical capability for enterprise AI assistants. The Workplace Assistant environment provides a realistic simulation with 26 different tools (calendar, email, file management, etc.), enabling you to train models that can complete complex multi-step office tasks—a key differentiator for production-ready AI systems.

---

## Time Estimate

| Phase | Duration |
|-------|----------|
| Environment Setup | 15-20 minutes |
| Data Preparation (`ng_prepare_data`) | 5-10 minutes |
| Sanity Tests (optional) | 10-15 minutes |
| Single Node Training | 2-4 hours (depending on steps) |
| **Total** | **~3-5 hours** |

---

## Objectives

By completing this tutorial, you will:

1. ✅ Set up NeMo RL and NeMo Gym for Reinforcement Learning (RL) training
2. ✅ Understand the Workplace Assistant environment and its 26 tool-calling capabilities
3. ✅ Configure and run GRPO training on Nemotron Nano v2 9B
4. ✅ Monitor training progress via Weights & Biases (W&B)
5. ✅ Achieve measurable improvements in tool-calling accuracy

---

## Prerequisites

### Technical Level & Required Knowledge
- **Intermediate to Advanced**: You should be comfortable with Python, LLM fine-tuning, and basic reinforcement learning concepts such as policy optimization, rewards, and rollouts. While detailed knowledge of the GRPO algorithm is not required, a general understanding is helpful.
- Some basic knowledge of Slurm (for multi-node runs) is helpful. Example commands are provided below as well.

### Hardware Requirements
- **Minimum**: 8× NVIDIA GPUs with 80GB or more VRAM each (e.g., H100, A100). Note: NeMo Gym does not require GPUs; GPUs are only necessary for GRPO training with NeMo RL.

### Required Accounts & Tokens
| Service                | Purpose                  | How to Obtain                         |
|------------------------|--------------------------|---------------------------------------|
| Hugging Face (HF)      | Model and data downloads | [Create account](https://huggingface.co/join) |
| Weights & Biases (W&B) | Training metrics logging | [Create account](https://wandb.ai/signup)      |

---

## Dataset Description

### Workplace Assistant Dataset

The Workplace Assistant dataset simulates realistic office productivity scenarios requiring multi-step tool usage.

| Property | Value |
|----------|-------|
| **HuggingFace Dataset** | [`nvidia/Nemotron-RL-agent-workplace_assistant`](https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant) |
| **Original Split** | `train` only (1,255 samples) |
| **After Local Split** | 1,129 train / 126 validation (90/10) |

### Available Tools (26 Total)

The environment includes tools for:
- 📅 **Calendar Management**: Schedule, modify, cancel meetings
- 📧 **Email Operations**: Send, read, search emails
- 📁 **File Management**: Create, read, modify documents
- 📋 **Task Management**: Create and track to-do items
- 🔍 **Search**: Query information across systems

### Example Tasks

Each task is a natural language request that the model must complete using the available tools. The environment allows up to 6 tool-calling steps per task.

**Single-Step Task** (1 tool call needed):

```json
{
  "input": [
    {
      "role": "system",
      "content": "Today's date is Thursday, 2023-11-30 and the current time is 23:59:00. Remember the current date and time when answering queries. Meetings must not start before 9am or end after 6pm."
    },
    {
      "role": "user", 
      "content": "Reply to carlos's last email about 'Task Update on Develop prototype for report generation' with 'Thanks for the update - I will get back to you tomorrow.'"
    }
  ],
  "tools": [
    {"type": "function", "name": "email_reply_email", "description": "Replies to an email by its ID.", "parameters": {"type": "object", "properties": {"email_id": {"type": "string"}, "body": {"type": "string"}}, "required": ["email_id", "body"]}},
    {"type": "function", "name": "email_search_emails", "description": "Searches for emails matching the given query...", "parameters": {...}},
    {"type": "function", "name": "email_send_email", "...": "..."},
    // ... 23 more tools (calendar, analytics, project_management, CRM)
  ],
  "parallel_tool_calls": false,
  "temperature": 1.0
}
```

**Expected output:** `email_reply_email(email_id="00000057", body="Thanks for the update - I will get back to you tomorrow.")`

---

**Multi-Step Task** (3 tool calls needed):

```json
{
  "input": [
    {
      "role": "system",
      "content": "Today's date is Thursday, 2023-11-30 and the current time is 23:59:00. Remember the current date and time when answering queries. Meetings must not start before 9am or end after 6pm."
    },
    {
      "role": "user",
      "content": "John is taking over all of Akira's leads that are interested in software. Can you reassign them in the crm?"
    }
  ],
  "tools": [
    {"type": "function", "name": "customer_relationship_manager_search_customers", "description": "Searches for customers based on the given parameters with pagination support.", "parameters": {"type": "object", "properties": {"assigned_to_email": {"type": "string"}, "product_interest": {"type": "string"}, "status": {"type": "string"}, ...}}},
    {"type": "function", "name": "customer_relationship_manager_update_customer", "description": "Updates a customer record by ID.", "parameters": {"type": "object", "properties": {"customer_id": {"type": "string"}, "field": {"type": "string"}, "new_value": {"type": "string"}}, "required": ["customer_id", "field", "new_value"]}},
    {"type": "function", "name": "company_directory_find_email_address", "description": "Finds all email addresses containing the given name...", "parameters": {...}},
    // ... 23 more tools
  ],
  "parallel_tool_calls": false,
  "temperature": 1.0
}
```

**Expected output sequence:**
1. `customer_relationship_manager_update_customer(customer_id="00000095", field="assigned_to_email", new_value="john.smith@atlas.com")`
2. `customer_relationship_manager_update_customer(customer_id="00000080", field="assigned_to_email", new_value="john.smith@atlas.com")`
3. `customer_relationship_manager_update_customer(customer_id="00000035", field="assigned_to_email", new_value="john.smith@atlas.com")`

---

The model must:
1. Understand the user's intent from natural language
2. Determine which tools to call and in what order
3. Infer correct parameters (e.g., look up email addresses, find matching customer records)
4. Execute all necessary steps to complete the task

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GRPO Training Pipeline                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐  │
│  │ 1. Setup │───▶│ 2. Prepare   │───▶│ 3. Configure  │───▶│ 4. Training  │  │
│  │   Env    │    │    Data      │    │    GRPO       │    │    Loop      │  │
│  └──────────┘    └──────────────┘    └───────────────┘    └──────────────┘  │
│       │                │                    │                    │          │
│       ▼                ▼                    ▼                    ▼          │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐  │
│  │ NeMo RL  │    │  Workplace   │    │  Model: 9B    │    │  Generate    │  │
│  │ NeMo Gym │    │  Assistant   │    │  TP=8, 32K    │    │  4 samples   │  │
│  │ venv     │    │  1,129 samples │    │  context      │    │  per prompt  │  │
│  └──────────┘    └──────────────┘    └───────────────┘    └──────────────┘  │
│                                                                    │        │
│                                      ┌─────────────────────────────┘        │
│                                      ▼                                      │
│                               ┌──────────────┐    ┌──────────────┐          │
│                               │ 5. Evaluate  │───▶│ 6. Update    │          │
│                               │    Rewards   │    │    Policy    │          │
│                               └──────────────┘    └──────────────┘          │
│                                      │                    │                 │
│                                      ▼                    ▼                 │
│                               ┌──────────────┐    ┌──────────────┐          │
│                               │ Environment  │    │ GRPO Loss    │          │
│                               │ verifies     │    │ Leave-one-   │          │
│                               │ tool calls   │    │ out baseline │          │
│                               └──────────────┘    └──────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Workflow Steps Explained

1. **Setup Environment**: Install NeMo RL and NeMo Gym, activate the Python virtual environment, and verify GPU access.

2. **Prepare Data**: Load the Workplace Assistant dataset containing 1,129 training tasks that require multi-step tool calling.

3. **Configure GRPO**: Set model parameters (Nemotron Nano v2 9B with tensor parallelism of 8), training hyperparameters, and environment-specific settings.

4. **Training Loop**: For each batch, generate 4 completions per prompt using vLLM for fast inference.

5. **Evaluate Rewards**: The Workplace Assistant environment verifies tool calls and task completion, assigning rewards (0 or 1) based on success.

6. **Update Policy**: Apply GRPO loss using leave-one-out baseline to reduce variance and update model weights.

---

## Setup Instructions

### Step 1: Enter a GPU Node

SSH into a GPU node or use Slurm to allocate resources.

**Option A: Interactive Slurm Session**

```bash
#!/bin/bash
NUM_ACTOR_NODES=1
ACCOUNT=your_slurm_account
JOB_NAME=nemo-gym-training
PARTITION=interactive
CONTAINER="/path/to/nemo-rl-container.sqsh"
MOUNTS="/shared/filesystem:/shared/filesystem"

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
    --container-workdir=/path/to/workspace \
    --container-mounts="${MOUNTS}" \
    --container-image="${CONTAINER}" \
    --pty /bin/bash
```

**Option B: Standard Slurm Allocation**

```bash
srun \
    --no-container-mount-home \
    --container-mounts=/shared/filesystem:/shared/filesystem \
    --container-image=/path/to/nemo-rl/container \
    --gres=gpu:8 \
    --nodes=1 --ntasks=1 --time=04:00:00 \
    --pty /bin/bash
```

### Step 2: Clone and Setup NeMo RL + NeMo Gym

```bash
# Navigate to your workspace
cd /shared/filesystem/$USER

# Clone NeMo RL repository
git clone https://github.com/NVIDIA-NeMo/RL
cd RL

# Clone NeMo Gym as a submodule
git clone https://github.com/NVIDIA-NeMo/Gym.git 3rdparty/Penguin-workspace/Penguin

# Initialize all submodules (Megatron, AutoModel, etc.)
git submodule update --init --recursive

# Activate the NeMo RL virtual environment
source /opt/nemo_rl_venv/bin/activate

# Install dependencies
uv sync --group={build,docs,dev,test} --extra penguin
```

### Step 3: Prepare NeMo Gym Data

The Workplace Assistant dataset must be downloaded from HuggingFace and prepared for training. This is a two-step process:

1. **Download & Split**: Download the dataset from HuggingFace and split into train/validation sets
2. **Prepare Data**: Run `ng_prepare_data` to validate and add agent references

```bash
# Setup Penguin local venv
cd 3rdparty/Penguin-workspace/Penguin
uv venv --python 3.12 --allow-existing
source .venv/bin/activate
uv sync --active --extra dev

# Step 1: Download and split the dataset from HuggingFace (90/10 train/val split)
python download_workplace_assistant.py

# Step 2: Prepare data for training (validates format and adds agent_ref to each sample)
config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,\
resources_servers/workplace_assistant/configs/workplace_assistant.yaml"

ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=resources_servers/workplace_assistant/data \
    +mode=train_preparation \
    +should_download=false

# Return to NeMo RL directory and Python env
cd ../../.. && source /opt/nemo_rl_venv/bin/activate
```

> **Note**: The `download_workplace_assistant.py` script downloads the dataset from HuggingFace (`nvidia/Nemotron-RL-agent-workplace_assistant`) and splits it into training (1,129 samples) and validation (126 samples) sets with a 90/10 ratio. The `ng_prepare_data` command then validates the data format and adds an `agent_ref` property to each example that tells NeMo Gym which agent server to route that example to.

### Step 4: Set Environment Variables

Create or export the following environment variables:

```bash
# Required: Hugging Face token for model downloads
export HF_TOKEN="your_hugging_face_token"

# Required: Weights & Biases API key for logging
export WANDB_API_KEY="your_wandb_api_key"

# Recommended: Local cache for HuggingFace models
export HF_HOME=".cache/"

# Optional: Force rebuild of virtual environments on first run
export NRL_FORCE_REBUILD_VENVS=true

# Optional: Set the CUDA architectures for GPU compilation.
# The values "9.0" and "10.0" refer to NVIDIA H100 (SM90) and GH200 or B100 (SM100) GPUs, respectively.
# Set if you are using H100, GH200, or B100 GPUs:
export TORCH_CUDA_ARCH_LIST="9.0 10.0"
```

### Step 5: Run Sanity Tests (Optional but Recommended)

Validate your setup before training:

```bash
# This will take 10-15 minutes
HF_HOME=.cache/ \
HF_TOKEN=${HF_TOKEN} \
    ./examples/penguin/run_penguin_single_node_sanity_tests.sh
```

> **Note**: If you've run these tests before and encounter HuggingFace rate limit errors, add `HF_HUB_OFFLINE=1` to the command.

---

## Training Configuration

### Key Configuration Parameters

The training configuration file is located at:
`examples/penguin/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml`

#### Environment Configuration

```yaml
env:
  should_use_penguin: true
  penguin:
    config_paths:
    - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
    - resources_servers/workplace_assistant/configs/workplace_assistant.yaml
    workplace_assistant_simple_agent:
      responses_api_agents:
        simple_agent:
          max_steps: 6  # Maximum tool-calling steps per task
```

#### GRPO Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_prompts_per_step` | 4 | Number of prompts per training step |
| `num_generations_per_prompt` | 4 | Rollouts generated per prompt |
| `max_rollout_turns` | 1 | Turns per rollout (1 turn, up to 6 tool steps) |
| `max_num_steps` | 10 | Total training steps |
| `use_leave_one_out_baseline` | true | Variance reduction technique |
| `normalize_rewards` | true | Normalize rewards across batch |

#### Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_name` | nvidia/NVIDIA-Nemotron-Nano-9B-v2 | Base model |
| `max_total_sequence_length` | 32768 | Maximum context length |
| `precision` | bfloat16 | Training precision |
| `tensor_model_parallel_size` | 8 | Tensor parallelism across GPUs |

#### Optimizer Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `optimizer` | Adam | Optimizer type |
| `lr` | 5.0e-6 | Learning rate |
| `min_lr` | 5.0e-7 | Minimum learning rate |
| `weight_decay` | 0.01 | Weight decay |
| `adam_beta1` / `adam_beta2` | 0.9 / 0.999 | Adam hyperparameters |
| `clip_grad` | 1.0 | Gradient clipping threshold |

---

## Running Training

### Single Node Training

Launch training on a single 8-GPU node:

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
TORCH_CUDA_ARCH_LIST="9.0 10.0" \
HF_HOME=.cache/ \
HF_TOKEN=${HF_TOKEN} \
WANDB_API_KEY=${WANDB_API_KEY} \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="${USER}-nemo-gym-rl-integration" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    grpo.val_at_start=true \
    grpo.val_period=6 \
    grpo.num_prompts_per_step=8 \
    grpo.num_generations_per_prompt=4 \
    grpo.max_num_steps=36 \
    grpo.max_num_epochs=1 \
    ++cluster.num_nodes=1 \
    checkpointing.enabled=true \
    checkpointing.save_period=6
```

### Training Parameters Explained

| Override | Value | Purpose |
|----------|-------|---------|
| `grpo.val_at_start=true` | true | Run validation before training starts |
| `grpo.val_period=6` | 6 | Validate every 6 steps |
| `grpo.num_prompts_per_step=8` | 8 | Prompts per step (batch) |
| `grpo.num_generations_per_prompt=4` | 4 | Generations per prompt (32 total rollouts/step) |
| `grpo.max_num_steps=36` | 36 | Total steps (~1 epoch over 1,129 samples) |
| `checkpointing.save_period=6` | 6 | Save checkpoint every 6 steps |

### Multi-Node Training

For production training, scale to multiple nodes:

```bash
# Create submit script: temp_penguin_submit.sh
cat << 'EOF' > temp_penguin_submit.sh
# ----- PARAMETERS -----
# WANDB_API_KEY, EXP_NAME, NUM_ACTOR_NODES, REPO_LOCATION

# ----- CONSTANTS -----
CONTAINER_IMAGE_PATH=/path/to/nemo-rl/container

read -r -d '' COMMAND <<CMDEOF
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
CMDEOF

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
EOF

chmod +x temp_penguin_submit.sh
```

**Submit the multi-node job:**

```bash
WANDB_API_KEY=${WANDB_API_KEY} \
EXP_NAME="penguin_grpo/nemotron_nano_v2_9b/2nodes/workplace_assistant_001" \
NUM_ACTOR_NODES=2 \
REPO_LOCATION="/path/to/nemo-rl" \
    ./temp_penguin_submit.sh \
    --config=examples/penguin/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml \
    logger.wandb.project="${USER}-nemo-gym-rl-integration"
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
- ✅ Reward mean increases consistently over steps
- ✅ Validation accuracy improves from baseline (~15%) to 50%+
- ✅ No OOM (Out of Memory) errors
- ✅ Checkpoints are saved at specified intervals

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| HuggingFace rate limits | Add `HF_HUB_OFFLINE=1` after initial download |
| vLLM process not shutting down | Run `pkill -f VllmAsyncGenerationWorker` before training |
| Ray cluster issues | Run `ray stop --force` before training |
| CUDA OOM | Reduce `num_prompts_per_step` or increase `tensor_parallel_size` |
| Slow initial startup | Set `NRL_FORCE_REBUILD_VENVS=true` on first run only |

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

- **NeMo RL Repository**: [github.com/NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL){:target="_blank"}
- **NeMo Gym Repository**: [github.com/NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym){:target="_blank"}
- **Nemotron Nano v2 9B Model**: [huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2){:target="_blank"}
- **GRPO Paper**: [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300){:target="_blank"}
- **Weights & Biases**: [wandb.ai](https://wandb.ai){:target="_blank"}

---

## Appendix: Full Configuration Reference

The complete training configuration is available at:

```
examples/penguin/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml
```
