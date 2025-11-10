# NeMo RL Integration Quick Reference

Fast reference for integrating NeMo Gym with NeMo RL. See {doc}`nemo-rl/index` for complete guide.

---

## Quick Start (3 Commands)

```bash
# 1. Transform Gym rollouts
python transform_gym_to_grpo.py --input rollouts.jsonl --output train.jsonl

# 2. Create config (see templates below)
nano grpo_config.yaml

# 3. Train
cd nemo-rl && uv run python examples/run_grpo_math.py --config grpo_config.yaml
```

---

## Algorithm Selection

```{list-table}
:header-rows: 1
:widths: 20 40 40

* - Algorithm
  - Use When
  - Data Requirement
* - **GRPO**
  - Task has verifiable outputs
  - Diverse prompts (1+ rollout each)
* - **SFT**
  - High-quality demonstrations
  - High reward rollouts (≥0.95)
* - **DPO**
  - Varied quality rollouts
  - 2+ rollouts per prompt with quality gap
```

---

## Data Transformations

### GRPO

```bash
python transform_gym_to_grpo.py \
    --input rollouts.jsonl \
    --output grpo_train.jsonl \
    --reward_threshold 0.0
```

**Output format**: `{"input": "...", "output": "...", "ground_truth": "..."}`

### SFT

```bash
# OpenAI format (with tool calling)
python transform_gym_to_sft.py \
    --input rollouts.jsonl \
    --output sft_train.jsonl \
    --reward_threshold 0.95 \
    --format openai

# Simple format (text only)
python transform_gym_to_sft.py \
    --input rollouts.jsonl \
    --output sft_train.jsonl \
    --reward_threshold 0.95 \
    --format simple
```

**Output format**: 
- OpenAI: `{"messages": [...], "tools": [...]}`
- Simple: `{"input": "...", "output": "..."}`

### DPO

```bash
# Same-prompt pairing (preferred)
python transform_gym_to_dpo.py \
    --input rollouts.jsonl \
    --output dpo_train.jsonl \
    --min_gap 0.2 \
    --strategy same_prompt

# Stratified pairing
python transform_gym_to_dpo.py \
    --input rollouts.jsonl \
    --output dpo_train.jsonl \
    --min_gap 0.2 \
    --strategy stratified
```

**Output format**: `{"prompt": "...", "chosen": "...", "rejected": "..."}`

---

## Configuration Templates

### GRPO Config

```yaml
defaults:
  - grpo_math_1B

data:
  dataset_name: ResponseDataset
  train_data_path: "grpo_train.jsonl"
  input_key: "input"
  output_key: "output"

policy:
  model_name: "Qwen/Qwen2.5-1.5B"
  max_total_sequence_length: 1024

checkpointing:
  checkpoint_dir: "results/gym_grpo"

logger:
  wandb_enabled: true
```

### SFT Config

```yaml
defaults:
  - sft

data:
  dataset_name: openai_format  # or ResponseDataset for simple
  train_data_path: "sft_train.jsonl"
  use_preserving_dataset: true

policy:
  model_name: "Qwen/Qwen2.5-1.5B"

checkpointing:
  checkpoint_dir: "results/gym_sft"
```

### DPO Config

```yaml
defaults:
  - dpo

data:
  dataset_name: BinaryPreferenceDataset
  train_data_path: "dpo_train.jsonl"
  prompt_key: "prompt"
  chosen_key: "chosen"
  rejected_key: "rejected"

policy:
  model_name: "Qwen/Qwen2.5-1.5B"

checkpointing:
  checkpoint_dir: "results/gym_dpo"
```

---

## Training Launch

```bash
# Single GPU
uv run python examples/run_{grpo_math|sft|dpo}.py --config config.yaml

# Multi-GPU (8 GPUs)
uv run python examples/run_{grpo_math|sft|dpo}.py \
    --config config.yaml \
    cluster.gpus_per_node=8

# Multi-node (Slurm)
COMMAND="uv run python examples/run_{grpo_math|sft|dpo}.py --config config.yaml cluster.num_nodes=2" \
CONTAINER=YOUR_CONTAINER \
sbatch --nodes=2 --gres=gpu:8 ray.sub
```

---

## Common Issues & Fixes

### "KeyError: 'input'"
```bash
# Check transformation output
head -n 1 transformed.jsonl | python -m json.tool

# Verify field names match config
```

### "No pairs created" (DPO)
```bash
# Check reward distribution
jq '.reward' rollouts.jsonl | sort | uniq -c

# Lower min_gap or adjust thresholds
--min_gap 0.1 --high_threshold 0.6 --low_threshold 0.5
```

### "CUDA out of memory"
```yaml
# Reduce batch sizes
policy:
  train_micro_batch_size: 2
  
# Enable checkpointing
policy:
  dtensor_cfg:
    activation_checkpointing: true
```

### "GRPO rewards all 0.0"
```python
# Verify ground_truth in data
data = json.loads(open('train.jsonl').readline())
print("Has ground_truth:", "ground_truth" in data)

# Check environment config
env:
  math:
    math_verify_impl: "hf_math_verify"
```

---

## Data Validation

```python
import json

# Load and check
data = [json.loads(line) for line in open('train.jsonl')]
print(f"Total: {len(data)}")

# Check fields
required = ["input", "output"]  # Adjust for format
missing = [i for i, d in enumerate(data) if not all(k in d for k in required)]
print(f"Missing fields: {len(missing)} examples")

# Check distribution (if applicable)
if "reward" in data[0]:
    rewards = [d["reward"] for d in data]
    print(f"Rewards: [{min(rewards):.2f}, {max(rewards):.2f}], mean={sum(rewards)/len(rewards):.2f}")
```

---

## Field Mapping Reference

```{list-table}
:header-rows: 1
:widths: 40 60

* - NeMo Gym Field
  - NeMo RL Equivalent
* - `responses_create_params.input`
  - `input` or `prompt` or `messages[user]`
* - `output`
  - `output` or `completion` or `messages[assistant]`
* - `reward`
  - Used for filtering/pairing (not directly in training data)
```

---

## Transformation Scripts Location

Find all scripts in the {doc}`nemo-rl` guide:
- **transform_gym_to_grpo.py** - Lines 100-200
- **transform_gym_to_sft.py** - Lines 400-600
- **transform_gym_to_dpo.py** - Lines 800-1000

Copy from documentation and save as `.py` files.

---

## Resource Requirements

```{list-table}
:header-rows: 1
:widths: 25 25 25 25

* - Model Size
  - Min GPUs
  - VRAM/GPU
  - Typical Config
* - 1-3B
  - 1
  - 24GB
  - A100/H100
* - 7-8B
  - 1-2
  - 40-80GB
  - A100 80GB
* - 32B+
  - 4-8
  - 80GB
  - Multi-node
```

---

## Next Steps

1. **Transform data** with appropriate script
2. **Validate output** format and distribution
3. **Create config** from template
4. **Test on 1 GPU** to verify data loading
5. **Scale up** to target hardware

**Need help?** See {doc}`nemo-rl` for:
- Complete transformation scripts
- Detailed troubleshooting
- Configuration explanations
- Advanced patterns

