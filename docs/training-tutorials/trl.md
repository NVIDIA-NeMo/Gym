(training-trl)=
# TRL Training

```{note}
This page is a stub. Content is being developed.
```

Train models with NeMo Gym using [Hugging Face TRL](https://huggingface.co/docs/trl), the Transformer Reinforcement Learning library.

:::{card}

**Goal**: Fine-tune models using TRL's RL algorithms with NeMo Gym verifiers.

^^^

**In this tutorial, you will**:

1. Integrate NeMo Gym verifiers as TRL reward functions
2. Configure PPO/DPO training with NeMo Gym environments
3. Train a model on single-step tasks

:::

---

## Before You Begin

- ✅ **Hardware**: GPU recommended (single or multi-GPU)
- ✅ **Software**: Python 3.10+, PyTorch
- ✅ **Familiarity**: HuggingFace ecosystem, basic RL concepts

```bash
pip install trl transformers nemo-gym
```

## Why TRL + NeMo Gym?

**TRL** provides production-ready RL training for LLMs:
- PPO, DPO, ORPO, and other algorithms out of the box
- Seamless HuggingFace Hub integration (models, datasets, tokenizers)
- Active community and extensive documentation

**NeMo Gym** complements TRL by providing:
- Diverse verifiers for task-specific reward computation
- Ready-to-use training environments
- Standardized task interfaces for math, code, tool calling, and more

## Getting Started

<!-- TODO: Add notebook link or quick start command -->

```python
from trl import PPOTrainer, PPOConfig
from nemo_gym import ResourceServer

# TODO: Add integration code
```

## Integration Pattern

### NeMo Gym as Reward Function

Use NeMo Gym verifiers to compute rewards for TRL training:

<!-- TODO: Document how to wrap NeMo Gym verifiers as TRL reward functions -->

```python
# Example: Connecting NeMo Gym verifier to TRL
# TODO: Add integration code
```

### Environment Configuration

<!-- TODO: Document environment setup for TRL -->

## Supported Algorithms

| Algorithm | Status | Best For |
|-----------|--------|----------|
| PPO | TBD | Online RL training |
| DPO | TBD | Preference optimization |
| ORPO | TBD | Odds ratio preference optimization |

## Complete Example

<!-- TODO: Add end-to-end working example -->

## Troubleshooting

<!-- TODO: Add common issues and solutions -->

---

## What's Next?

After completing this tutorial, explore these options:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Use Other Training Environments
:link: https://github.com/NVIDIA-NeMo/Gym#-available-resource-servers

Browse available resource servers on GitHub to find other training environments.
+++
{bdg-secondary}`github` {bdg-secondary}`resource-servers`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Unsloth Training
:link: ../tutorials/unsloth-training
:link-type: doc

Fast, memory-efficient fine-tuning on a single GPU.
+++
{bdg-secondary}`unsloth` {bdg-secondary}`efficient`
:::

::::
