(training-integration-nemo-rl-sft)=

# SFT Training

Detailed guide for **Supervised Fine-Tuning** using high-quality NeMo Gym rollouts.

**Best for**: High-reward demonstrations, imitation learning

---

## Overview

SFT (Supervised Fine-Tuning) trains models to imitate high-quality demonstrations:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Characteristic
  - Description
* - **Training Mode**
  - Supervised learning (standard next-token prediction)
* - **Data Selection**
  - Filters for high-reward rollouts (typically ≥ 0.8)
* - **Learning Objective**
  - Imitate successful behavior patterns
* - **Best Use Cases**
  - Expert demonstrations, high-quality tool use, consistent behavior
```

---

## Data Requirements

For SFT training, you need:

**Rollout Characteristics**:
- High-reward examples only (≥ 0.8 or ≥ 0.9)
- Consistent, correct behavior
- Representative of desired output quality

**Collection Strategy**:

```yaml
# Optimize for quality and consistency
responses_create_params:
  temperature: 0.2  # Low temperature for consistency
# Then filter for high rewards during transformation
```

Refer to {doc}`../../rollout-collection/optimize-for-training/index` for detailed collection guidance.

---

## Data Transformation

Transform NeMo Gym rollouts to SFT format:

::::{tab-set}

:::{tab-item} Required Format

SFT expects OpenAI message format:

```json
{
  "messages": [
    {"role": "user", "content": "What is 47 × 23?"},
    {"role": "assistant", "content": "Let me calculate: 47 * 23 = 1,081"}
  ]
}
```
:::

:::{tab-item} Transformation Script

```python
import json

def transform_for_sft(rollout, min_reward=0.8):
    """Convert high-reward rollouts to message format."""
    # Filter by reward
    if rollout.get("reward", 0.0) < min_reward:
        return None
    
    # Extract messages
    messages = []
    
    # Add input messages
    for msg in rollout["responses_create_params"]["input"]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Flatten output to assistant message
    output_parts = []
    for item in rollout["output"]:
        if item["type"] == "message":
            output_parts.append(item["content"][0]["text"])
    
    if output_parts:
        messages.append({
            "role": "assistant",
            "content": " ".join(output_parts)
        })
    
    return {"messages": messages}

# Transform and filter
with open("rollouts.jsonl") as f:
    rollouts = [json.loads(line) for line in f]

transformed = [
    t for r in rollouts 
    if (t := transform_for_sft(r, min_reward=0.8)) is not None
]

print(f"Filtered {len(rollouts)} → {len(transformed)} examples")

# Save for NeMo RL
with open("sft_train.jsonl", "w") as f:
    for item in transformed:
        f.write(json.dumps(item) + "\n")
```
:::

::::

---

## Training Configuration

Example NeMo RL configuration for SFT:

```yaml
data:
  dataset_name: OpenAIChatDataset
  train_data_path: "sft_train.jsonl"

model:
  pretrained_model: "Qwen/Qwen2.5-1.5B-Instruct"

training:
  algorithm: "sft"
  num_epochs: 3
  train_micro_batch_size: 8
  train_global_batch_size: 256
```

Refer to [NeMo RL's SFT guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/sft.md) for complete configuration options.

---

## Related Resources

**NeMo Gym**:
- {doc}`../index` - Integration overview
- {doc}`../../rollout-collection/optimize-for-training/index` - Optimize collection for SFT
- {doc}`../../data-quality/index` - Filter and curate high-quality data

**NeMo RL**:
- [SFT Guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/sft.md) - Official NeMo RL documentation
- [Examples](https://github.com/NVIDIA-NeMo/RL/tree/main/examples) - Reference implementations
