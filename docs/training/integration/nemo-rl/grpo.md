(training-integration-nemo-rl-grpo)=

# GRPO Training

Detailed guide for training with **Group Relative Policy Optimization** using NeMo Gym rollouts.

**Best for**: Math, code, and tasks with automatic verification

---

## Overview

GRPO (Group Relative Policy Optimization) is an online reinforcement learning algorithm that excels at tasks with verifiable outcomes:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Characteristic
  - Description
* - **Training Mode**
  - Online RL (generates rollouts during training)
* - **Reward Signal**
  - Uses automatic verification scores
* - **Data Efficiency**
  - Learns from both successful and failed attempts
* - **Best Use Cases**
  - Math, code execution, tool use with clear success criteria
```

---

## Data Requirements

For GRPO training, you need:

**Rollout Characteristics**:
- Diverse prompts (many different tasks)
- Clear reward signals (from verification)
- Balance of success and failure examples

**Collection Strategy**:

```yaml
# Optimize for diversity
num_samples_in_parallel: 50  # Many different prompts
responses_create_params:
  temperature: 0.5  # Moderate exploration
```

Refer to {doc}`../../rollout-collection/optimize-for-training/index` for detailed collection guidance.

---

## Data Transformation

Transform NeMo Gym rollouts to GRPO format:

::::{tab-set}

:::{tab-item} Required Format

GRPO expects `input`, `output`, and optionally `reward` fields:

```json
{
  "input": "What is 47 × 23?",
  "output": "Let me calculate: 47 * 23 = 1,081",
  "reward": 1.0
}
```
:::

:::{tab-item} Transformation Script

```python
import json

def transform_for_grpo(rollout):
    """Extract input/output from NeMo Gym rollout."""
    # Extract input text from last user message
    input_messages = rollout["responses_create_params"]["input"]
    input_text = next(
        (msg["content"] for msg in reversed(input_messages) 
         if msg["role"] == "user"),
        ""
    )
    
    # Flatten output to string
    output_parts = []
    for item in rollout["output"]:
        if item["type"] == "message":
            output_parts.append(item["content"][0]["text"])
    
    return {
        "input": input_text,
        "output": " ".join(output_parts),
        "reward": rollout.get("reward", 0.0)
    }

# Transform all rollouts
with open("rollouts.jsonl") as f:
    rollouts = [json.loads(line) for line in f]

transformed = [transform_for_grpo(r) for r in rollouts]

# Save for NeMo RL
with open("grpo_train.jsonl", "w") as f:
    for item in transformed:
        f.write(json.dumps(item) + "\n")
```
:::

::::

---

## Training Configuration

Example NeMo RL configuration for GRPO:

```yaml
data:
  dataset_name: ResponseDataset
  train_data_path: "grpo_train.jsonl"
  input_key: "input"
  output_key: "output"
  reward_key: "reward"  # Optional, can recompute during training

model:
  pretrained_model: "Qwen/Qwen2.5-1.5B-Instruct"

training:
  algorithm: "grpo"
  num_epochs: 3
  train_micro_batch_size: 4
  train_global_batch_size: 512
```

Refer to [NeMo RL's GRPO guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/grpo.md) for complete configuration options.

---

## Related Resources

**NeMo Gym**:
- {doc}`../index` - Integration overview
- {doc}`../../rollout-collection/optimize-for-training/index` - Optimize collection for GRPO
- {doc}`../../verification/index` - Configure verification for reward signals

**NeMo RL**:
- [GRPO Guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/grpo.md) - Official NeMo RL documentation
- [Examples](https://github.com/NVIDIA-NeMo/RL/tree/main/examples) - Reference implementations

