(training-integration-nemo-rl)=

# NeMo RL Integration

Use NeMo Gym rollouts to train models with NeMo RL's GRPO, SFT, and DPO algorithms.

NeMo RL is NVIDIA's post-training framework for reinforcement learning. It supports multiple training backends (DTensor, Megatron Core), high-performance generation (vLLM), and scales from single-GPU experiments to multi-node deployments.

```{contents}
:local:
:depth: 2
```

---

## Overview

### What You'll Learn

- Transform NeMo Gym rollouts to NeMo RL formats
- Configure NeMo RL to consume your data
- Launch GRPO, SFT, and DPO training runs
- Handle multi-turn conversations and tool calling
- Troubleshoot common integration issues

### Integration Architecture

```
NeMo Gym → rollouts.jsonl → Transformation → NeMo RL Training
  ↓           ↓                  ↓               ↓
Tasks      Verification      Format          GRPO/SFT/DPO
```

**Key compatibility**: NeMo Gym outputs JSONL with conversation-style data that maps directly to NeMo RL's expected formats with minimal transformation.

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

### NeMo Gym Output Format

After rollout collection, you have JSONL with this structure:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "user", "content": "What is 47 × 23?"}
    ],
    "tools": [{"type": "function", "name": "calculator",}]
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

**Key fields**:
- `responses_create_params.input` - Original prompt/conversation
- `output` - Agent's complete response (tool calls + text)
- `reward` - Primary training signal (0.0-1.0 or binary)
- Additional metrics - Task-specific scores (accuracy, etc.)

### NeMo RL Expected Formats

#### For GRPO/SFT (ResponseDataset)

```json
{
  "input": "What is 47 × 23?",
  "output": "First I'll calculate: 47 * 23 = 1,081. The answer is 1,081."
}
```

#### For SFT (OpenAI Format with Tool Calling)

```json
{
  "messages": [
    {"role": "user", "content": "What is 47 × 23?"},
    {"role": "assistant", "tool_calls": [{"name": "calculator", "arguments": {"expression": "47 * 23"}}]},
    {"role": "tool", "content": "1081"},
    {"role": "assistant", "content": "The answer is 1,081"}
  ],
  "tools": [{"name": "calculator", "description": "...", "parameters": {}}]
}
```

#### For DPO (BinaryPreferenceDataset)

```json
{
  "prompt": "What is 47 × 23?",
  "chosen": "First I'll calculate: 47 * 23 = 1,081. The answer is 1,081.",
  "rejected": "I think it's around 1,000."
}
```

---

## Training Guides

Choose the training algorithm that matches your data and objectives:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` GRPO Training
:link: grpo
:link-type: doc

Train with **Group Relative Policy Optimization** using automatic verification and reward-weighted learning.

**Best for**: Math, code, tasks with verifiable outcomes
+++
{bdg-secondary}`online-rl` {bdg-secondary}`verification`
:::

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` SFT Training
:link: sft
:link-type: doc

Train with **Supervised Fine-Tuning** using high-quality demonstration data.

**Best for**: High-reward rollouts, imitation learning
+++
{bdg-secondary}`supervised` {bdg-secondary}`demonstrations`
:::

:::{grid-item-card} {octicon}`git-compare;1.5em;sd-mr-1` DPO Training
:link: dpo
:link-type: doc

Train with **Direct Preference Optimization** using preference pairs.

**Best for**: Quality tuning, style preferences, safety
+++
{bdg-secondary}`preference-learning` {bdg-secondary}`pairwise`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc

Common issues and solutions for data, training, and environment problems.

**Quick fixes** for integration challenges
+++
{bdg-secondary}`debugging` {bdg-secondary}`solutions`
:::

::::

---

## Advanced Topics

For production deployments and advanced use cases:

- **{doc}`advanced`** - Multi-turn conversations, custom reward shaping, combining resource servers, performance optimization

---

## Quick Selection Guide

```{list-table}
:header-rows: 1
:widths: 30 35 35

* - Your Situation
  - Recommended Algorithm
  - Guide
* - Math/code tasks with automatic verification
  - GRPO
  - {doc}`grpo`
* - High-quality rollouts (reward ≥ 0.9)
  - SFT
  - {ref}`training-rollout-sampling-sft`
* - Mixed quality rollouts (varied rewards)
  - DPO
  - {ref}`training-rollout-sampling-dpo`
* - Multiple rollouts per prompt
  - DPO (create pairs)
  - {ref}`training-rollout-sampling-dpo`
* - Need to maximize task success rate
  - GRPO
  - {doc}`grpo`
* - Want to teach quality distinctions
  - DPO
  - {ref}`training-rollout-sampling-dpo`
```

---

## Integration Workflow

```
1. Collect Rollouts (NeMo Gym)
   ↓
2. Transform Data Format
   - GRPO: input/output pairs
   - SFT: filter high-quality demonstrations
   - DPO: create preference pairs
   ↓
3. Configure NeMo RL
   - Dataset paths and keys
   - Model and training settings
   ↓
4. Launch Training
   - Single/multi-GPU
   - Monitor metrics
   ↓
5. Evaluate & Iterate
   - Test trained model
   - Collect more data if needed
```

```{toctree}
:hidden:
:maxdepth: 1

grpo
sft
dpo
troubleshooting
advanced
```
