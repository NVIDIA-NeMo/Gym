(training-nemo-rl-grpo-index)=

# RL Training with NeMo RL using GRPO

This tutorial trains NVIDIA [Nemotron Nano 9B v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) to improve its **{term}`multi-step <Multi-step>` {term}`tool-calling <Tool Use / Function Calling>`** capability using the **{term}`GRPO (Group Relative Policy Optimization) <GRPO (Group Relative Policy Optimization)>`** algorithm on the **Workplace Assistant** environment.

Workplace Assistant is a realistic office simulation (calendar, email, project management, etc.) with complex multi-step tasks, providing a strong data distribution for training enterprise-ready tool-using assistants.

:::{card}

**Goal**: Train a model for multi-step tool calling using GRPO on the Workplace Assistant environment.

^^^

**In this tutorial, you will**:

1. Set up NeMo RL and NeMo Gym for {term}`reinforcement learning <RL (Reinforcement Learning)>` training
2. Understand the Workplace Assistant environment and its multi-step tool calling tasks
3. Configure and run GRPO training on Nemotron Nano v2 9B
4. Monitor training progress via Weights & Biases (W&B)

:::

> **TL;DR:** Want to jump straight to running commands? Skip to {doc}`Setup <setup>`.

---

## Before You Begin

Make sure you have these prerequisites ready:

- ✅ **Hardware**: 1+ nodes with 8× NVIDIA GPUs (80GB+ each, such as H100 or A100)
  - Single-node testing: 1 node with 8 GPUs
  - Multi-node production: 8+ nodes with 8 GPUs each recommended
  - RAM: 64 GB+ per node
- ✅ **Storage**: 100 GB+ free disk space on a shared filesystem
- ✅ **Software**: Linux, Python 3.12+, Git, Slurm for multi-node training
- ✅ **Familiarity**: Python, LLM fine-tuning, basic RL concepts (in-depth RLVR/GRPO knowledge not required)

:::{note}
NeMo Gym does not require GPUs. GPUs are only necessary for GRPO training with NeMo RL.
:::

**Optional accounts**:

- **Weights & Biases (W&B)**: For experiment tracking ([sign up](https://wandb.ai/signup), [get API key](https://wandb.ai/authorize)). Training proceeds without W&B if not configured.
- **HuggingFace**: For downloading models ([create token](https://huggingface.co/settings/tokens)). Recommended to avoid rate limits.

**Total time estimate**: ~3-5 hours (including environment setup, data preparation, and training)

---

## Tutorial Steps

Follow these steps sequentially to complete the tutorial:

::::{grid} 1
:gutter: 2

:::{grid-item-card} 1. About the Workplace Assistant Training Environment
:link: training-nemo-rl-grpo-about-workplace-assistant
:link-type: ref

Understand the dataset you will train on and its multi-step tool calling tasks.
+++
{bdg-secondary}`background`
:::

:::{grid-item-card} 2. Gym Configuration
:link: training-nemo-rl-grpo-gym-configuration
:link-type: ref

Understand the Gym configuration component in the NeMo RL training config file.
+++
{bdg-secondary}`configuration`
:::

:::{grid-item-card} 3. NeMo RL Configuration
:link: training-nemo-rl-grpo-nemo-rl-configuration
:link-type: ref

Understand the GRPO and NeMo RL configuration components in the training config file.
+++
{bdg-secondary}`configuration`
:::

:::{grid-item-card} 4. Setup
:link: training-nemo-rl-grpo-setup
:link-type: ref

Clone repositories, install dependencies, and prepare the training data.
+++
{bdg-primary}`prerequisite`
:::

:::{grid-item-card} 5. Single Node Training
:link: training-nemo-rl-grpo-single-node-training
:link-type: ref

Perform a single node GRPO training run with success criteria.
+++
{bdg-primary}`training`
:::

:::{grid-item-card} 6. Multi-Node Training
:link: training-nemo-rl-grpo-multi-node-training
:link-type: ref

Scale to multi-node GRPO training for production.
+++
{bdg-primary}`training`
:::

::::

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

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build a Custom Training Environment
:link: ../creating-resource-server
:link-type: doc

Create your own resource server with custom tools and verification logic.
+++
{bdg-secondary}`tutorial` {bdg-secondary}`custom-tools`
:::

::::

```{toctree}
:caption: NeMo RL GRPO
:hidden:
:maxdepth: 1

about-workplace-assistant.md
gym-configuration.md
nemo-rl-configuration.md
setup.md
single-node-training.md
multi-node-training.md
🟡 Multi-Verifier Training <multi-verifier-training.md>
```