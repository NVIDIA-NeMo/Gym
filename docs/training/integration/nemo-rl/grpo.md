(training-integration-nemo-rl-grpo)=

# GRPO Training

Train models with Group Relative Policy Optimization using NeMo Gym rollouts and automatic verification.

```{contents}
:local:
:depth: 2
```

---

### When to Use GRPO

**Group Relative Policy Optimization (GRPO)** is ideal when:
- You have tasks with verifiable outcomes (math, code, instruction following)
- Rewards are computed from automatic verification
- You want to maximize task success rate

**Example use cases**: Math problem solving, code generation, multi-turn tool use

### Step 1: Transform Data for GRPO

GRPO expects simple `input`/`output` pairs with ground truth for verification.

```python
"""
Transform NeMo Gym rollouts to NeMo RL ResponseDataset format for GRPO.

Usage:
    python transform_gym_to_grpo.py \
        --input rollouts.jsonl \
        --output grpo_train.jsonl \
        --reward_threshold 0.0
"""

import json
import argparse
from pathlib import Path


def extract_text_from_output(output_list):
    """Extract final text response from NeMo Gym output format."""
    for item in reversed(output_list):
        if item.get("type") == "message":
            # Handle both list and string content
            content = item.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                return content[0].get("text", "")
            elif isinstance(content, str):
                return content
    return ""


def extract_input_text(input_list):
    """Extract user query from input messages."""
    for msg in reversed(input_list):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def extract_ground_truth(responses_create_params):
    """
    Extract ground truth answer if available in the task metadata.
    
    Many resource servers include the ground truth in the input for verification.
    This is used by NeMo RL's math environment to score generations.
    """
    # Check for ground truth in various common locations
    if "ground_truth" in responses_create_params:
        return responses_create_params["ground_truth"]
    
    # Some servers embed it in metadata
    metadata = responses_create_params.get("metadata", {})
    if "ground_truth" in metadata:
        return metadata["ground_truth"]
    
    if "answer" in metadata:
        return metadata["answer"]
    
    return None


def transform_rollout(rollout, include_ground_truth=True):
    """Transform single NeMo Gym rollout to NeMo RL GRPO format."""
    # Extract input
    input_messages = rollout["responses_create_params"]["input"]
    input_text = extract_input_text(input_messages)
    
    # Extract output (agent's final response)
    output_text = extract_text_from_output(rollout["output"])
    
    # Build transformed data
    transformed = {
        "input": input_text,
        "output": output_text,
    }
    
    # Include ground truth if available and requested
    if include_ground_truth:
        ground_truth = extract_ground_truth(rollout["responses_create_params"])
        if ground_truth:
            transformed["ground_truth"] = ground_truth
    
    return transformed


def main():
    parser = argparse.ArgumentParser(
        description="Transform NeMo Gym rollouts to NeMo RL GRPO format"
    )
    parser.add_argument("--input", required=True, help="Input rollouts.jsonl path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--reward_threshold",
        type=float,
        default=0.0,
        help="Minimum reward to include (default: 0.0 = all rollouts)",
    )
    parser.add_argument(
        "--no_ground_truth",
        action="store_true",
        help="Exclude ground truth from output",
    )
    
    args = parser.parse_args()
    
    # Load rollouts
    rollouts = []
    with open(args.input, "r") as f:
        for line in f:
            rollouts.append(json.loads(line))
    
    print(f"Loaded {len(rollouts)} rollouts from {args.input}")
    
    # Filter by reward threshold
    filtered_rollouts = [
        r for r in rollouts if r.get("reward", 0.0) >= args.reward_threshold
    ]
    
    print(f"After filtering (reward >= {args.reward_threshold}): {len(filtered_rollouts)} rollouts")
    
    # Transform
    transformed = []
    for rollout in filtered_rollouts:
        try:
            transformed_data = transform_rollout(
                rollout, include_ground_truth=not args.no_ground_truth
            )
            transformed.append(transformed_data)
        except Exception as e:
            print(f"Warning: Failed to transform rollout: {e}")
            continue
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for item in transformed:
            f.write(json.dumps(item) + "\n")
    
    print(f"✅ Wrote {len(transformed)} transformed rollouts to {args.output}")
    
    # Show example
    if transformed:
        print("\n--- Example Transformed Rollout ---")
        print(json.dumps(transformed[0], indent=2))


if __name__ == "__main__":
    main()
```

**Run transformation**:

```bash
# Transform all rollouts (use for GRPO training)
python transform_gym_to_grpo.py \
    --input rollouts.jsonl \
    --output grpo_data/train.jsonl \
    --reward_threshold 0.0

# Create validation split (optional)
python transform_gym_to_grpo.py \
    --input validation_rollouts.jsonl \
    --output grpo_data/val.jsonl \
    --reward_threshold 0.0
```

### Step 2: Configure NeMo RL for GRPO

Create a training configuration `grpo_gym_data.yaml`:

```yaml
# Extend NeMo RL's base GRPO config
defaults:
  - grpo_math_1B  # Or grpo_math_8B, grpo_math_1B_megatron, etc.

# Override data settings to use your transformed data
data:
  dataset_name: ResponseDataset
  train_data_path: "grpo_data/train.jsonl"
  val_data_path: "grpo_data/val.jsonl"
  input_key: "input"       # Matches transformed format
  output_key: "output"     # Matches transformed format
  max_input_seq_length: 1024
  shuffle: true

# Configure environment for your task
env:
  math:
    num_workers: 8
    math_verify_impl: "hf_math_verify"  # Or your custom verification

# Optional: Adjust model and training
policy:
  model_name: "Qwen/Qwen2.5-1.5B"  # Or your preferred model
  max_total_sequence_length: 1024
  train_global_batch_size: 512
  train_micro_batch_size: 4

grpo:
  num_prompts_per_step: 32
  num_generations_per_prompt: 16
  val_period: 10

checkpointing:
  enabled: true
  checkpoint_dir: "results/gym_grpo"
  save_period: 10

logger:
  wandb_enabled: true
  wandb:
    project: "nemo-gym-grpo"
    name: "gym-data-run-1"
```

### Step 3: Launch GRPO Training

```bash
cd /path/to/nemo-rl

# Single GPU
uv run python examples/run_grpo_math.py \
    --config /path/to/grpo_gym_data.yaml

# Multi-GPU (8 GPUs)
uv run python examples/run_grpo_math.py \
    --config /path/to/grpo_gym_data.yaml \
    cluster.gpus_per_node=8

# Multi-node (2 nodes × 8 GPUs)
COMMAND="uv run python examples/run_grpo_math.py --config /path/to/grpo_gym_data.yaml cluster.num_nodes=2" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=2 \
    --account=YOUR_ACCOUNT \
    --partition=YOUR_PARTITION \
    --gres=gpu:8 \
    ray.sub
```

:::{tip}
If your config file is in the current directory, you can use `--config grpo_gym_data.yaml`. For configs in other locations, provide the full path: `--config /path/to/grpo_gym_data.yaml`.
:::


### Understanding GRPO Training Flow

```
1. Load transformed data (input/output pairs)
2. Sample prompts from dataset
3. Generate multiple responses per prompt using policy model
4. Score responses using environment (e.g., math verification)
5. Compute advantages using group-relative rewards
6. Update policy with PPO-style clipped objective
7. Repeat for N steps
```

:::{important}
**How GRPO uses your rollouts**: Your transformed NeMo Gym data provides the **prompts** (input queries) and **ground truth** (for verification). During training, GRPO generates **fresh responses** on-policy rather than reusing your collected agent outputs. This means:

- ✅ **Prompts**: Sampled from your transformed rollouts
- ✅ **Ground truth**: Used for environment verification/scoring  
- ❌ **Agent responses**: Generated fresh each training step (not from your rollouts)

This on-policy generation is a key feature of GRPO—the policy learns from its own current behavior.
:::

---

