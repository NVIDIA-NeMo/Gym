(training-integration-nemo-rl-dpo)=

# DPO Training

Detailed guide for **Direct Preference Optimization** using paired NeMo Gym rollouts.

**Best for**: Quality tuning, preference learning, style improvements

---

## Overview

DPO (Direct Preference Optimization) learns from preference pairs (chosen vs. rejected):

```{list-table}
:header-rows: 1
:widths: 30 70

* - Characteristic
  - Description
* - **Training Mode**
  - Preference learning (learns from comparisons)
* - **Data Requirements**
  - Multiple rollouts per prompt for pairing
* - **Learning Objective**
  - Distinguish good responses from bad ones
* - **Best Use Cases**
  - Quality tuning, style preferences, safety improvements
```

---

## Data Requirements

For DPO training, you need:

**Rollout Characteristics**:
- Multiple rollouts per prompt (at least 2)
- Quality variation (mix of high and low rewards)
- Same task, different responses

**Collection Strategy**:

```yaml
# Optimize for diversity and multiple samples
num_repeats: 2  # Or more for better pairing options
responses_create_params:
  temperature: 0.7  # Higher for variation
```

Refer to {doc}`../../rollout-collection/optimize-for-training/index` for detailed collection guidance.

---

## Data Transformation

Transform NeMo Gym rollouts to DPO preference pairs:

::::{tab-set}

:::{tab-item} Required Format

DPO expects `prompt`, `chosen`, and `rejected` fields:

```json
{
  "prompt": "What is 47 × 23?",
  "chosen": "Let me calculate: 47 * 23 = 1,081",
  "rejected": "I think it's around 1,000"
}
```
:::

:::{tab-item} Transformation Script

```python
import json
from itertools import combinations

def create_dpo_pairs(rollouts_for_prompt):
    """Create preference pairs from multiple rollouts."""
    pairs = []
    
    # Sort by reward
    sorted_rollouts = sorted(
        rollouts_for_prompt, 
        key=lambda r: r.get("reward", 0.0),
        reverse=True
    )
    
    # Create pairs: higher reward = chosen
    for high_r, low_r in combinations(sorted_rollouts, 2):
        high_reward = high_r.get("reward", 0.0)
        low_reward = low_r.get("reward", 0.0)
        
        # Only pair if quality difference is significant
        if high_reward - low_reward < 0.1:
            continue
        
        # Extract prompt (same for both)
        prompt = high_r["responses_create_params"]["input"][-1]["content"]
        
        # Extract outputs
        def get_output(rollout):
            parts = []
            for item in rollout["output"]:
                if item["type"] == "message":
                    parts.append(item["content"][0]["text"])
            return " ".join(parts)
        
        pairs.append({
            "prompt": prompt,
            "chosen": get_output(high_r),
            "rejected": get_output(low_r)
        })
    
    return pairs

# Group rollouts by prompt
from collections import defaultdict

with open("rollouts.jsonl") as f:
    all_rollouts = [json.loads(line) for line in f]

grouped = defaultdict(list)
for r in all_rollouts:
    prompt = r["responses_create_params"]["input"][-1]["content"]
    grouped[prompt].append(r)

# Create pairs
all_pairs = []
for prompt, rollouts in grouped.items():
    if len(rollouts) >= 2:
        all_pairs.extend(create_dpo_pairs(rollouts))

print(f"Created {len(all_pairs)} preference pairs from {len(all_rollouts)} rollouts")

# Save for NeMo RL
with open("dpo_train.jsonl", "w") as f:
    for pair in all_pairs:
        f.write(json.dumps(pair) + "\n")
```
:::

::::

---

## Training Configuration

Example NeMo RL configuration for DPO:

```yaml
data:
  dataset_name: BinaryPreferenceDataset
  train_data_path: "dpo_train.jsonl"
  prompt_key: "prompt"
  chosen_key: "chosen"
  rejected_key: "rejected"

model:
  pretrained_model: "Qwen/Qwen2.5-1.5B-Instruct"

training:
  algorithm: "dpo"
  num_epochs: 3
  train_micro_batch_size: 4
  train_global_batch_size: 256
```

Refer to [NeMo RL's DPO guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/dpo.md) for complete configuration options.

---

## Related Resources

**NeMo Gym**:
- {doc}`../index` - Integration overview
- {doc}`../../rollout-collection/optimize-for-training/index` - Optimize collection for DPO
- {doc}`../../data-quality/index` - Create high-quality preference pairs

**NeMo RL**:
- [DPO Guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/dpo.md) - Official NeMo RL documentation
- [Examples](https://github.com/NVIDIA-NeMo/RL/tree/main/examples) - Reference implementations

