(training-integration-nemo-rl)=

# NeMo RL Integration

Use NeMo Gym rollouts to train models with NeMo RL's GRPO, SFT, and DPO algorithms.

NeMo RL is NVIDIA's post-training framework for reinforcement learning. It supports multiple training backends (DTensor, Megatron Core), high-performance generation (vLLM), and scales from single-GPU experiments to multi-node deployments.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item}
**What You'll Learn**

- Transform rollout data for NeMo RL
- Choose the right training algorithm
- Configure data paths and parameters
- Troubleshoot integration issues
:::

:::{grid-item}
**Integration Flow**

```{mermaid}
graph LR
    A[NeMo Gym<br/>Rollouts] --> B[Transform<br/>Format]
    B --> C[NeMo RL<br/>Training]
    C --> D[Trained<br/>Model]
```
:::

::::

:::{tip}
**Quick compatibility**: NeMo Gym's rollout format (OpenAI-style messages) works directly with NeMo RL's dataset loaders. Transform scripts handle field mapping and filtering.
:::

---

## Prerequisites

### From NeMo Gym

```{list-table}
:header-rows: 1
:widths: 30 70

* - Prerequisite
  - Description
* - **Collected rollouts**
  - JSONL file with `responses_create_params`, `output`, and `reward` fields
* - **Verified quality**
  - Rollouts scored by resource server verification ({doc}`../../verification/index`)
* - **Understood metrics**
  - Know what your reward signal means (binary, continuous, multi-metric)
```

### From NeMo RL

```{list-table}
:header-rows: 1
:widths: 30 70

* - Setup Item
  - Details
* - **Installation**
  - NeMo RL cloned and environment set up ([NeMo RL Quick Start](https://github.com/NVIDIA-NeMo/RL#quick-start))
* - **GPU access**
  - At least 1 GPU with ≥24GB VRAM for 1-3B models
* - **Environment variables**
  - `HF_HOME`, `WANDB_API_KEY` (optional), `HF_DATASETS_CACHE` (optional)
* - **HuggingFace login**
  - `huggingface-cli login` if using gated models (Llama, etc.)
```

---

## Data Format Reference

### NeMo Gym Rollout Structure

After rollout collection, you have JSONL where each line contains:

```{list-table}
:header-rows: 1
:widths: 25 75

* - Field
  - Description
* - `responses_create_params`
  - Original task: input messages and available tools
* - `output`
  - Complete agent response: tool calls, tool results, and final message
* - `reward`
  - Verification score from resource server (typically 0.0-1.0)
* - Additional metadata
  - Task-specific fields like `accuracy`, `test_results`, etc.
```

::::{dropdown} **Example: Math Task Rollout**
:open:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "user", "content": "What is 47 × 23?"}
    ],
    "tools": [{"type": "function", "name": "calculator"}]
  },
  "output": [
    {
      "type": "function_call",
      "name": "calculator",
      "arguments": "{\"expression\": \"47 * 23\"}"
    },
    {
      "type": "function_call_output",
      "output": "1081"
    },
    {
      "type": "message",
      "content": [{"text": "The answer is 1,081"}]
    }
  ],
  "reward": 1.0,
  "accuracy": 1.0
}
```

This captures everything: the task, the agent's tool usage, and the verification score.
::::

### NeMo RL Dataset Formats

NeMo RL expects different formats depending on the training algorithm:

::::{tab-set}

:::{tab-item} GRPO Format

**ResponseDataset** - Input/output pairs with optional rewards:

```json
{
  "input": "What is 47 × 23?",
  "output": "Let me calculate: 47 * 23 = 1,081",
  "reward": 1.0
}
```

**Transformation**: Extract text from `responses_create_params.input` and flatten `output` to a single string.
:::

:::{tab-item} SFT Format

**OpenAI Messages Format** - Standard chat format:

```json
{
  "messages": [
    {"role": "user", "content": "What is 47 × 23?"},
    {"role": "assistant", "content": "Let me calculate: 47 * 23 = 1,081"}
  ]
}
```

**Transformation**: Convert NeMo Gym's structured output to flattened message history. Filter for high-reward rollouts (≥ 0.8).
:::

:::{tab-item} DPO Format

**BinaryPreferenceDataset** - Preference pairs:

```json
{
  "prompt": "What is 47 × 23?",
  "chosen": "Let me calculate: 47 * 23 = 1,081",
  "rejected": "I think it's around 1,000"
}
```

**Transformation**: Collect multiple rollouts per prompt, then pair high-reward (chosen) with low-reward (rejected) responses.
:::

::::

---

## Getting Started

### Choose Your Training Algorithm

```{list-table}
:header-rows: 1
:widths: 25 40 35

* - Algorithm
  - Best For
  - Data Requirements
* - **GRPO**
  - Math, code, verifiable tasks
  - Diverse prompts with reward signals
* - **SFT**
  - Imitation learning, high-quality demos
  - Filtered high-reward rollouts (≥ 0.8)
* - **DPO**
  - Quality tuning, preference learning
  - Multiple rollouts per prompt for pairing
```

:::{dropdown} **Detailed Algorithm Selection Guide**

**Choose GRPO when**:
- Tasks have automatic verification (math, code execution)
- You want to maximize task success rate
- Rollouts have reward signals from verification
- Example: Training on math problems with calculator tools

**Choose SFT when**:
- You have high-quality demonstrations
- Most rollouts score ≥ 0.8 reward
- You want to imitate successful behavior
- Example: Training on expert-level code completions

**Choose DPO when**:
- You have multiple responses per prompt
- Want to teach quality distinctions
- Have paired examples (good vs. bad)
- Example: Improving response style and safety

:::

### Integration Steps

::::{tab-set}

:::{tab-item} 1. Collect Data

Use NeMo Gym to collect rollouts with verification:

```bash
ng_collect_rollouts \
  +agent_name=your_agent \
  +input_jsonl_fpath=tasks.jsonl \
  +output_jsonl_fpath=rollouts.jsonl
```

Refer to {doc}`../../rollout-collection/index` for collection strategies.
:::

:::{tab-item} 2. Transform Format

Write a Python script to transform rollouts:

```python
import json

# Example: Simple transformation for GRPO
def transform_for_grpo(rollout):
    # Extract input text
    input_text = rollout["responses_create_params"]["input"][-1]["content"]
    
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
with open("nemo_rl_data.jsonl", "w") as f:
    for item in transformed:
        f.write(json.dumps(item) + "\n")
```

:::

:::{tab-item} 3. Configure NeMo RL

Create a NeMo RL config YAML:

```yaml
data:
  dataset_name: ResponseDataset  # or OpenAIChatDataset for SFT
  train_data_path: "nemo_rl_data.jsonl"
  input_key: "input"
  output_key: "output"

model:
  pretrained_model: "Qwen/Qwen2.5-1.5B-Instruct"

training:
  algorithm: "grpo"  # or "sft", "dpo"
  num_epochs: 3
  batch_size: 4
```

Refer to [NeMo RL documentation](https://github.com/NVIDIA-NeMo/RL) for complete configuration options.
:::

:::{tab-item} 4. Launch Training

Run NeMo RL training:

```bash
# Navigate to NeMo RL repository
cd /path/to/nemo-rl

# Launch training
uv run python examples/run_grpo_math.py \
  --config your_config.yaml
```

Monitor training metrics via Weights & Biases or TensorBoard.
:::

::::

:::{seealso}
**For production workflows**: See {doc}`advanced` for multi-task training, reward shaping, and optimization techniques.
:::

---

## Additional Resources

### Algorithm-Specific Guides

::::{grid} 1 1 2 3
:gutter: 3

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` GRPO Training
:link: grpo
:link-type: doc

Detailed guide for **Group Relative Policy Optimization** with automatic verification and reward-weighted learning.
+++
{bdg-secondary}`online-rl` {bdg-secondary}`verification`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` SFT Training
:link: sft
:link-type: doc

Detailed guide for **Supervised Fine-Tuning** using high-quality demonstration data.
+++
{bdg-secondary}`supervised` {bdg-secondary}`demonstrations`
:::

:::{grid-item-card} {octicon}`git-compare;1.5em;sd-mr-1` DPO Training
:link: dpo
:link-type: doc

Detailed guide for **Direct Preference Optimization** using preference pairs.
+++
{bdg-secondary}`preference-learning` {bdg-secondary}`pairwise`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Quick Reference
:link: quick-reference
:link-type: doc

Fast lookup for transformations, configs, and commands across all algorithms.
+++
{bdg-info}`Cheat Sheet`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc

Common issues and solutions for data, training, and environment problems.
+++
{bdg-secondary}`debugging` {bdg-secondary}`solutions`
:::

:::{grid-item-card} {octicon}`telescope;1.5em;sd-mr-1` Advanced Topics
:link: advanced
:link-type: doc

Multi-task training, reward shaping, performance optimization, and production deployment.
+++
{bdg-secondary}`production` {bdg-secondary}`optimization`
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

grpo
sft
dpo
quick-reference
troubleshooting
advanced
```
