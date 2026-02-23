# GRPO Training Guide for Workplace Assistant

This guide walks through the complete setup and training process for GRPO (Generalized Reinforcement Policy Optimization) with the Workplace Assistant environment.

## Overview

**Environment**: Multi-step agentic tool-use environment for business tasks
- **Tools**: 26 tools across 5 categories (Email, Calendar, CRM, Analytics, Project Management)
- **Tasks**: 1255 workplace scenarios (meetings, emails, data analysis, etc.)
- **Domain**: Business activities
- **Max Steps**: Up to 6 tool-calling steps per task

**Dataset**: `nvidia/Nemotron-RL-agent-workplace_assistant` on HuggingFace
- Full dataset: 1255 samples
- Default split: 90% train (1129 samples) / 10% validation (126 samples)

## Prerequisites

1. **NeMo-Gym RL Repository**: Cloned and set up
2. **Penguin Environment**: Located at `RL/3rdparty/Penguin-workspace/Penguin`
3. **Python 3.12+**: With `uv` package manager
4. **CUDA GPUs**: Minimum 8 GPUs recommended for full training

## Step-by-Step Setup

### 1. Setup Penguin Virtual Environment

```bash
cd RL/3rdparty/Penguin-workspace/Penguin
uv venv --python 3.12 --allow-existing
source .venv/bin/activate
uv sync --active --extra dev
```

### 2. Download and Prepare Dataset

The dataset needs to be downloaded from HuggingFace and split into train/validation sets.

**Option A: Using the download script (Recommended)**

```bash
# Run the download script (creates 90/10 stratified split)
uv run python download_workplace_assistant.py
```

This script will:
- Download the dataset from HuggingFace
- Perform a stratified 90/10 split based on task categories
- Save `train.jsonl` (1129 samples) and `validation.jsonl` (126 samples)
- Skip download if files already exist

**Option B: Manual download with HuggingFace**

```python
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json

dataset = load_dataset("nvidia/Nemotron-RL-agent-workplace_assistant")
full_train_data = list(dataset['train'])

categories = [item['category'] for item in full_train_data]
train_data, val_data = train_test_split(
    full_train_data, test_size=0.10, random_state=42, stratify=categories
)

# Save train.jsonl
with open('resources_servers/workplace_assistant/data/train.jsonl', 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\\n')

# Save validation.jsonl
with open('resources_servers/workplace_assistant/data/validation.jsonl', 'w') as f:
    for item in val_data:
        f.write(json.dumps(item) + '\\n')
```

### 3. Prepare Data with ng_prepare_data

Once the JSONL files are downloaded, prepare them for training:

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/workplace_assistant/configs/workplace_assistant.yaml"

ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=resources_servers/workplace_assistant/data \
    +mode=train_preparation \
    +should_download=false
```

This command:
- Validates the dataset samples
- Computes aggregate metrics
- Adds agent references to each sample
- Creates collated train/validation datasets ready for GRPO

**Expected output:**
```
✓ Train: 1129 samples
✓ Validation: 126 samples
✓ Metrics validated and saved
```

### 4. Test the Environment (Optional)

Before full training, test that the environment works correctly:

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/workplace_assistant/configs/workplace_assistant.yaml"

# Start the Penguin server
ng_run "+config_paths=[$config_paths]"
```

In another terminal, collect a test trajectory:

```bash
ng_collect_rollouts +agent_name=workplace_assistant_simple_agent \
    +input_jsonl_fpath=resources_servers/workplace_assistant/data/train.jsonl \
    +output_jsonl_fpath=results/workplace_assistant_test_trajectory.jsonl \
    +limit=1
```

### 5. Return to NeMo-Gym RL Directory

```bash
cd ../../../../  # Back to RL directory
source /opt/nemo_rl_venv/bin/activate  # Activate NeMo RL environment
```

## Running GRPO Training

### Quick Sanity Check (test_001)

Run a quick 5-step test to verify everything works:

```bash
pkill -f VllmAsyncGenerationWorker
ray stop --force
python -c "import ray; ray.shutdown()"

EXP_NAME="$(date +%Y%m%d)/penguin_grpo/qwen3_4binstruct/workplace_assistant_test_001"
CONFIG_PATH=examples/penguin/grpo_workplace_assistant_qwen3_4binstruct.yaml

HF_HOME=.cache/ \
WANDB_API_KEY=<your-wandb-key> \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="workplace-assistant-grpo" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    grpo.val_at_start=false \
    ++grpo.num_prompts_per_step=4 \
    ++grpo.num_generations_per_prompt=4 \
    ++grpo.max_num_steps=5 \
    ++policy.dtensor_cfg.clear_cache_every_n_steps=1 \
    ++cluster.num_nodes=1 \
    checkpointing.enabled=false
```

**Expected duration**: ~5-10 minutes on 8 GPUs

### Full Epoch Training (test_002)

Train for 1 complete epoch through the dataset:

```bash
pkill -f VllmAsyncGenerationWorker
ray stop --force
python -c "import ray; ray.shutdown()"

EXP_NAME="$(date +%Y%m%d)/penguin_grpo/qwen3_4binstruct/workplace_assistant_test_002"
CONFIG_PATH=examples/penguin/grpo_workplace_assistant_qwen3_4binstruct.yaml

HF_HOME=.cache/ \
WANDB_API_KEY=<your-wandb-key> \
NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/penguin/run_grpo_penguin.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="workplace-assistant-grpo" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME \
    grpo.val_at_start=false \
    grpo.val_period=36 \
    grpo.num_prompts_per_step=32 \
    grpo.num_generations_per_prompt=8 \
    grpo.max_num_steps=36 \
    grpo.max_num_epochs=1 \
    ++cluster.num_nodes=1 \
    checkpointing.enabled=true \
    checkpointing.save_period=36
```

**Training details:**
- **Prompts per step**: 32
- **Generations per prompt**: 8
- **Total rollouts per step**: 256
- **Steps per epoch**: 36 (ceil(1129/32))
- **Total rollouts**: 9,216
- **Expected duration**: ~2-3 hours on 8 GPUs

### Multi-Node Full Training

For production training with the default configuration:

```bash
# Use default config values (64 prompts/step × 16 generations = 1024 rollouts/step)
EXP_NAME="$(date +%Y%m%d)/penguin_grpo/qwen3_4binstruct/workplace_assistant_full"
CONFIG_PATH=examples/penguin/grpo_workplace_assistant_qwen3_4binstruct.yaml

HF_HOME=.cache/ \
WANDB_API_KEY=<your-wandb-key> \
uv run python examples/penguin/run_grpo_penguin.py \
    --config=$CONFIG_PATH \
    logger.wandb.project="workplace-assistant-grpo" \
    logger.wandb.name=$EXP_NAME \
    logger.log_dir=results/$EXP_NAME
```

**Default configuration:**
- **Nodes**: 8 nodes × 8 GPUs = 64 GPUs
- **Prompts per step**: 64
- **Generations per prompt**: 16
- **Total rollouts per step**: 1024
- **Validation**: Every 10 steps
- **Checkpointing**: Top-3 by validation accuracy

## Configuration Details

### Model Configuration
- **Model**: `Qwen/Qwen3-4B-Instruct-2507`
- **Tool Parser**: Hermes (for multi-tool handling)
- **Context Length**: 32,768 tokens
- **Precision**: BFloat16
- **Tensor Parallel**: 2 (for single node with 8 GPUs)

### GRPO Parameters
- **Max Rollout Turns**: 1 (single-turn with up to 6 tool steps)
- **Reward Normalization**: Enabled
- **Leave-One-Out Baseline**: Enabled
- **Reference Policy KL**: Disabled (skip_reference_policy_logprobs_calculation=true)

### Environment Configuration
- **Agent**: `workplace_assistant_simple_agent`
- **Max Tool Steps**: 6 per task
- **Tools**: 26 tools (Email, Calendar, CRM, Analytics, Project Management)
- **Reward**: State-based verification (compares final database state to ground truth)

## Monitoring Training

### WandB Metrics

Key metrics to monitor during training:

1. **Accuracy Metrics**:
   - `train:accuracy` - Training task success rate
   - `val:accuracy` - Validation task success rate

2. **Reward Metrics**:
   - `train:reward_mean` - Average reward per rollout
   - `train:advantage_mean` - Advantage values for policy gradient

3. **Performance Metrics**:
   - `E2E (Samples/sec)` - End-to-end throughput
   - `Training FLOPS` - Computational efficiency
   - `Generation (Tokens/sec)` - VLLM generation speed

4. **Policy Metrics**:
   - `policy_loss` - Policy gradient loss
   - `grad_norm` - Gradient norm (for stability)

### Log Files

Training logs are saved to:
- **Main log**: `results/$EXP_NAME/logs/`
- **Checkpoints**: `results/grpo_workplace_assistant/`
- **Penguin logs**: Detailed rollout information and tool execution traces

## Troubleshooting

### Common Issues

**1. Tool Execution Errors**

If you see errors like `got an unexpected keyword argument 'query'`, this is expected during exploration. The model is learning which arguments each tool accepts.

**2. Out of Memory**

Reduce batch sizes:
```bash
++grpo.num_prompts_per_step=16 \
++grpo.num_generations_per_prompt=4
```

**3. VLLM Generation Timeout**

Increase GPU memory utilization:
```bash
++policy.generation.vllm_cfg.gpu_memory_utilization=0.85
```

**4. Dataset Not Found**

Ensure you've run the download script and `ng_prepare_data` successfully:
```bash
ls -lh resources_servers/workplace_assistant/data/train.jsonl
ls -lh resources_servers/workplace_assistant/data/validation.jsonl
```

### Performance Optimization

**Single Node (8 GPUs)**:
- Use `tensor_parallel_size=2` (default in config)
- Set `num_prompts_per_step=32` for balanced throughput

**Multi-Node (64 GPUs)**:
- Use default config with `num_prompts_per_step=64`
- Ensure high-speed interconnect (InfiniBand/NVLink)

## Dataset Information

### Category Distribution

The dataset contains 5 task categories:
- **Email Management**: Searching, sending, replying to emails
- **Calendar Operations**: Event scheduling, search, updates
- **CRM Activities**: Customer data management and queries
- **Analytics**: Data visualization and reporting
- **Project Management**: Task tracking and project coordination

### Data Format

Each sample contains:
```json
{
  "id": 0,
  "responses_create_params": {
    "prompt": "...",
    "tools": [...]
  },
  "ground_truth": [...],
  "category": "...",
  "environment_name": "workplace_assistant"
}
```

## License

- **Code**: Apache 2.0
- **Dataset**: Apache 2.0
- **Model**: Qwen3 model license

## References

- **Dataset**: https://huggingface.co/datasets/nvidia/Nemotron-RL-agent-workplace_assistant
- **Model**: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- **NeMo-Gym Documentation**: See main repo README
- **Penguin Framework**: See `RL/3rdparty/Penguin-workspace/Penguin/README.md`

