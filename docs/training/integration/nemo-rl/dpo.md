(training-integration-nemo-rl-dpo)=

# DPO Training

Train models with Direct Preference Optimization using preference pairs from NeMo Gym rollouts.

```{contents}
:local:
:depth: 2
```

---

### When to Use DPO

**Direct Preference Optimization (DPO)** is ideal when:
- You have pairs of good and bad responses
- Rollouts have varied quality (not all high/low)
- You want to teach quality distinctions

**Example use cases**: Response quality tuning, style preferences, safety alignment

### Step 1: Create Preference Pairs

DPO needs pairs of chosen (high reward) and rejected (low reward) responses:

```python
"""
Transform NeMo Gym rollouts to NeMo RL DPO format.

Usage:
    python transform_gym_to_dpo.py \
        --input rollouts.jsonl \
        --output dpo_train.jsonl \
        --min_gap 0.2 \
        --strategy same_prompt
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict


def extract_prompt(input_messages: List[Dict]) -> str:
    """Extract user prompt from input messages."""
    for msg in reversed(input_messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def extract_response_text(output_list: List[Dict]) -> str:
    """Extract final response text from output."""
    for item in reversed(output_list):
        if item.get("type") == "message":
            content = item.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                return content[0].get("text", "")
            elif isinstance(content, str):
                return content
    return ""


def create_pairs_same_prompt(rollouts_by_prompt: Dict[str, List[Dict]], min_gap: float) -> List[Dict]:
    """
    Create DPO pairs from rollouts with the same prompt.
    
    Args:
        rollouts_by_prompt: Dict mapping prompt -> list of rollouts
        min_gap: Minimum reward difference between chosen and rejected
    
    Returns:
        List of DPO pairs
    """
    pairs = []
    
    for prompt, rollouts in rollouts_by_prompt.items():
        if len(rollouts) < 2:
            continue
        
        # Sort by reward
        rollouts.sort(key=lambda r: r.get("reward", 0.0), reverse=True)
        
        # Pair best with worst
        best = rollouts[0]
        worst = rollouts[-1]
        
        reward_gap = best.get("reward", 0.0) - worst.get("reward", 0.0)
        
        if reward_gap >= min_gap:
            pair = {
                "prompt": prompt,
                "chosen": extract_response_text(best["output"]),
                "rejected": extract_response_text(worst["output"]),
                "reward_gap": reward_gap,
                "chosen_reward": best.get("reward", 0.0),
                "rejected_reward": worst.get("reward", 0.0),
            }
            pairs.append(pair)
    
    return pairs


def create_pairs_stratified(rollouts: List[Dict], min_gap: float, high_threshold: float = 0.7, low_threshold: float = 0.4) -> List[Dict]:
    """
    Create DPO pairs using stratified sampling (high vs low quality).
    
    Args:
        rollouts: All rollouts
        min_gap: Minimum reward difference
        high_threshold: Minimum reward for "chosen" examples
        low_threshold: Maximum reward for "rejected" examples
    
    Returns:
        List of DPO pairs
    """
    high_quality = [r for r in rollouts if r.get("reward", 0.0) >= high_threshold]
    low_quality = [r for r in rollouts if r.get("reward", 0.0) <= low_threshold]
    
    pairs = []
    
    # Pair each high-quality with low-quality responses
    for chosen_rollout in high_quality:
        for rejected_rollout in low_quality:
            reward_gap = chosen_rollout.get("reward", 0.0) - rejected_rollout.get("reward", 0.0)
            
            if reward_gap >= min_gap:
                chosen_prompt = extract_prompt(chosen_rollout["responses_create_params"]["input"])
                rejected_prompt = extract_prompt(rejected_rollout["responses_create_params"]["input"])
                
                # Ideally same prompt, but not required for stratified
                pair = {
                    "prompt": chosen_prompt,  # Use chosen's prompt
                    "chosen": extract_response_text(chosen_rollout["output"]),
                    "rejected": extract_response_text(rejected_rollout["output"]),
                    "reward_gap": reward_gap,
                    "chosen_reward": chosen_rollout.get("reward", 0.0),
                    "rejected_reward": rejected_rollout.get("reward", 0.0),
                }
                pairs.append(pair)
    
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Transform NeMo Gym rollouts to NeMo RL DPO format"
    )
    parser.add_argument("--input", required=True, help="Input rollouts.jsonl")
    parser.add_argument("--output", required=True, help="Output DPO pairs JSONL")
    parser.add_argument(
        "--min_gap",
        type=float,
        default=0.2,
        help="Minimum reward gap between chosen and rejected (default: 0.2)",
    )
    parser.add_argument(
        "--strategy",
        choices=["same_prompt", "stratified"],
        default="same_prompt",
        help="Pairing strategy: 'same_prompt' or 'stratified'",
    )
    parser.add_argument(
        "--high_threshold",
        type=float,
        default=0.7,
        help="High quality threshold for stratified (default: 0.7)",
    )
    parser.add_argument(
        "--low_threshold",
        type=float,
        default=0.4,
        help="Low quality threshold for stratified (default: 0.4)",
    )
    
    args = parser.parse_args()
    
    # Load rollouts
    rollouts = []
    with open(args.input, "r") as f:
        for line in f:
            rollouts.append(json.loads(line))
    
    print(f"Loaded {len(rollouts)} rollouts from {args.input}")
    
    # Create pairs based on strategy
    if args.strategy == "same_prompt":
        # Group by prompt
        by_prompt = defaultdict(list)
        for r in rollouts:
            prompt = extract_prompt(r["responses_create_params"]["input"])
            by_prompt[prompt].append(r)
        
        print(f"Found {len(by_prompt)} unique prompts")
        pairs = create_pairs_same_prompt(by_prompt, args.min_gap)
    
    else:  # stratified
        pairs = create_pairs_stratified(
            rollouts,
            args.min_gap,
            args.high_threshold,
            args.low_threshold,
        )
    
    print(f"Created {len(pairs)} DPO pairs")
    
    if not pairs:
        print("❌ No pairs created. Try:")
        print(f"   - Lower --min_gap (current: {args.min_gap})")
        print(f"   - Adjust thresholds (high: {args.high_threshold}, low: {args.low_threshold})")
        print(f"   - Check reward distribution in rollouts")
        return
    
    # Show statistics
    if pairs:
        avg_gap = sum(p["reward_gap"] for p in pairs) / len(pairs)
        avg_chosen = sum(p["chosen_reward"] for p in pairs) / len(pairs)
        avg_rejected = sum(p["rejected_reward"] for p in pairs) / len(pairs)
        
        print(f"Average reward gap: {avg_gap:.3f}")
        print(f"Average chosen reward: {avg_chosen:.3f}")
        print(f"Average rejected reward: {avg_rejected:.3f}")
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    
    print(f"✅ Wrote {len(pairs)} pairs to {args.output}")
    
    # Show example
    if pairs:
        print("\n--- Example DPO Pair ---")
        print(json.dumps(pairs[0], indent=2))


if __name__ == "__main__":
    main()
```

**Run transformation**:

```bash
# Same-prompt pairing (preferred)
python transform_gym_to_dpo.py \
    --input rollouts.jsonl \
    --output dpo_data/train.jsonl \
    --min_gap 0.2 \
    --strategy same_prompt

# Stratified pairing (if you don't have multiple rollouts per prompt)
python transform_gym_to_dpo.py \
    --input rollouts.jsonl \
    --output dpo_data/train.jsonl \
    --min_gap 0.2 \
    --strategy stratified \
    --high_threshold 0.7 \
    --low_threshold 0.4
```

### Step 2: Configure NeMo RL for DPO

Create `dpo_gym_data.yaml` (configuration extends NeMo RL's base DPO config with Gym-specific data paths):

```yaml
# Extend NeMo RL's base DPO config
defaults:
  - dpo

# Configure for preference pairs
data:
  dataset_name: BinaryPreferenceDataset
  train_data_path: "dpo_data/train.jsonl"
  val_data_path: "dpo_data/val.jsonl"
  prompt_key: "prompt"
  chosen_key: "chosen"
  rejected_key: "rejected"

# Model configuration
policy:
  model_name: "Qwen/Qwen2.5-1.5B"
  max_total_sequence_length: 1024
  train_global_batch_size: 128
  train_micro_batch_size: 2

dpo:
  reference_policy_kl_penalty: 0.05
  preference_loss_weight: 1.0
  sft_loss_weight: 0.0
  max_num_epochs: 1
  val_period: 25

checkpointing:
  enabled: true
  checkpoint_dir: "results/gym_dpo"
  metric_name: "val:validation-default_loss"
  higher_is_better: false

logger:
  wandb_enabled: false  # Set to true after running 'wandb login'
  wandb:
    project: "nemo-gym-dpo"
    name: "gym-dpo-run-1"
```

### Step 3: Launch DPO Training

```bash
cd /path/to/nemo-rl

# Single GPU
uv run python examples/run_dpo.py \
    --config dpo_gym_data.yaml

# Multi-GPU (8 GPUs)
uv run python examples/run_dpo.py \
    --config dpo_gym_data.yaml \
    cluster.gpus_per_node=8

# Multi-node
COMMAND="uv run python examples/run_dpo.py --config dpo_gym_data.yaml cluster.num_nodes=2" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch --nodes=2 --gres=gpu:8 ray.sub
```

---

