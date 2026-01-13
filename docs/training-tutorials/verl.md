(training-verl)=
# VeRL Training

```{note}
This page is a stub. Content is being developed.
```

Train models with NeMo Gym using [VeRL](https://github.com/volcengine/verl), a distributed reinforcement learning framework designed for large-scale LLM training.

:::{card}

**Goal**: Train models using VeRL's distributed RL algorithms with NeMo Gym verifiers.

^^^

**In this tutorial, you will**:

1. Integrate NeMo Gym verifiers as VeRL reward functions
2. Configure distributed PPO/GRPO training with NeMo Gym environments
3. Scale training from single-node to multi-node clusters
4. Monitor and optimize training performance

:::

---

## Before You Begin

Make sure you have these prerequisites ready:

- ✅ **Hardware**: Multi-GPU setup (VeRL excels at distributed training)
  - Single-node: 1+ NVIDIA GPUs (24GB+ VRAM each)
  - Multi-node: 2+ nodes with 8× GPUs each recommended
- ✅ **Software**: Python 3.10+, PyTorch 2.0+
- ✅ **Familiarity**: Basic RL concepts, distributed training fundamentals

```bash
pip install verl nemo-gym
```

**Optional accounts**:

- **Weights & Biases (W&B)**: For experiment tracking ([sign up](https://wandb.ai/signup))
- **HuggingFace**: For downloading models ([create token](https://huggingface.co/settings/tokens))

**Total time estimate**: ~2-4 hours (single-node) or ~4-8 hours (multi-node)

---

## Why VeRL + NeMo Gym?

**VeRL** is designed for large-scale RL training on LLMs:

- **Hybrid parallelism**: Flexible data, tensor, and pipeline parallelism strategies
- **Memory efficiency**: Optimized for large models with techniques like activation checkpointing
- **Research-friendly**: Clean APIs for algorithm experimentation and extension
- **Production-ready**: Battle-tested on billion-parameter models

**NeMo Gym** complements VeRL by providing:

- Diverse verifiers for task-specific reward computation
- Ready-to-use training environments (math, code, tool calling, reasoning)
- Standardized task interfaces across modalities

---

## 1. Integration Pattern

**Estimated time**: ~30 minutes

### NeMo Gym as Reward Function

Use NeMo Gym verifiers to compute rewards for VeRL training:

```python
from nemo_gym import ResourceServer

# Initialize resource server with your chosen environment
resource_server = ResourceServer(
    environment="your_environment",
    # TODO: Add configuration
)

# VeRL reward function wrapper
def nemo_gym_reward_fn(responses, prompts):
    """Compute rewards using NeMo Gym verifier."""
    rewards = []
    for response, prompt in zip(responses, prompts):
        result = resource_server.verify(response, prompt)
        rewards.append(result.reward)
    return rewards
```

### VeRL Configuration

<!-- TODO: Document VeRL config for NeMo Gym integration -->

```yaml
# verl_config.yaml
# TODO: Add VeRL configuration example
```

**✅ Success Check**: Resource server responds to verification requests.

---

## 2. Single-Node Training

**Estimated time**: ~1-2 hours

Start with single-node training to validate your setup:

### Launch Training

```bash
# TODO: Add single-node training command
```

### Monitor Progress

<!-- TODO: Document monitoring with W&B or logs -->

**✅ Success Check**: Training loss decreases, rewards increase over steps.

---

## 3. Multi-Node Training

**Estimated time**: ~3-6 hours

Scale to multi-node for production training:

### Cluster Configuration

<!-- TODO: Document multi-node cluster setup -->

### Launch Distributed Training

```bash
# TODO: Add multi-node training command with SLURM or Ray
```

### Parallelism Strategies

VeRL supports flexible parallelism configurations:

| Strategy | Use Case | Memory Savings |
|----------|----------|----------------|
| Data Parallel | Large batch sizes | Low |
| Tensor Parallel | Large model width | Medium |
| Pipeline Parallel | Large model depth | High |
| FSDP | Memory-constrained | Very High |

**✅ Success Check**: All nodes participate in training, gradient sync completes.

---

## Supported Algorithms

| Algorithm | Status | Best For |
|-----------|--------|----------|
| PPO | TBD | General online RL training |
| GRPO | TBD | Group-based policy optimization |
| ReMax | TBD | Reward maximization |
| REINFORCE | TBD | Policy gradient baseline |

---

## Supported Models

VeRL + NeMo Gym works with HuggingFace-compatible models:

| Model | Size | Tested |
|-------|------|--------|
| Llama 3.x | 8B-70B | TBD |
| Nemotron | 8B-70B | TBD |
| Qwen 2.x | 7B-72B | TBD |
| Mistral | 7B-22B | TBD |

---

## Complete Example

<!-- TODO: Add end-to-end working example with downloadable config -->

```python
# Full VeRL + NeMo Gym training script
# TODO: Add complete example
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM errors | Model too large for GPU | Enable FSDP or tensor parallelism |
| Slow training | Inefficient parallelism | Tune DP/TP/PP ratio for your cluster |
| Reward always zero | Verifier misconfiguration | Check resource server connectivity |
| Gradient NaN | Learning rate too high | Reduce LR, add gradient clipping |

<!-- TODO: Add detailed troubleshooting for common VeRL + NeMo Gym issues -->

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

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` NeMo RL with GRPO
:link: ../tutorials/nemo-rl-grpo/index
:link-type: doc

Multi-step tool calling with GRPO using NeMo RL.
+++
{bdg-secondary}`rl` {bdg-secondary}`grpo`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Unsloth Training
:link: ../tutorials/unsloth-training
:link-type: doc

Fast, memory-efficient fine-tuning on a single GPU.
+++
{bdg-secondary}`unsloth` {bdg-secondary}`efficient`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build a Custom Environment
:link: ../tutorials/creating-resource-server
:link-type: doc

Create your own resource server with custom tools.
+++
{bdg-secondary}`tutorial` {bdg-secondary}`custom-tools`
:::

::::
