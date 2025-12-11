(training-nemo-rl-grpo-index)=

# RL Training with NeMo RL using GRPO

This tutorial trains NVIDIA [Nemotron Nano 9B v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) to improve its **{term}`multi-step <Multi-step>` {term}`tool-calling <Tool Use / Function Calling>`** capability using **{term}`GRPO (Group Relative Policy Optimization) <GRPO (Group Relative Policy Optimization)>`** algorithm on the **Workplace Assistant** environment. Workplace Assistant is a realistic office simulation (calendar, email, project management, etc.) with complex multi-step tasks, providing a strong data distribution for training enterprise-ready tool-using assistants.

**Total time estimate:** ~3-5 hours (including environment setup, data preparation, and training)

> **TL;DR:** Want to jump straight to running commands? Skip to {doc}`Setup Instructions <setup>`.

---

## Objectives

In this tutorial, you will:

1. Set up NeMo RL and NeMo Gym for {term}`reinforcement learning <RL (Reinforcement Learning)>` training
2. Understand the Workplace Assistant environment and its multi-step tool calling capability
3. Configure and run GRPO training on Nemotron Nano v2 9B using this environment in Gym
4. Monitor training progress via Weights & Biases (W&B)

---

## Prerequisites

::::{tab-set}

:::{tab-item} Required Knowledge

- You should be comfortable with Python, LLM fine-tuning, and basic reinforcement learning concepts such as policy optimization, rewards, and rollouts. While in-depth knowledge of Reinforcement Learning with Verifiable Rewards (RLVR) and the GRPO algorithm is not required, a high-level understanding is helpful.
- Some basic familiarity with Slurm is useful, but you can follow along using the example commands provided below.

:::

:::{tab-item} Hardware Requirements

**Minimum** 1 node of 8× NVIDIA GPUs with 80GB or more memory each (such as H100 or A100) is required.

NeMo Gym does not require GPUs. GPUs are only necessary for GRPO training with NeMo RL.

- **GPU**: Multi-GPU setup required for RL training
  - **Single-node testing**: 1 node with 8 GPUs (e.g., 8x A100 or H100 GPUs)
  - **Multi-node training**: 8+ nodes with 8 GPUs each recommended for production training
- **CPU**: Modern x86_64 processor
- **RAM**: 64 GB+ recommended per node
- **Storage**: 100 GB+ free disk space for:
  - NeMo RL repository and dependencies
  - Model checkpoints and training artifacts
  - Dataset storage (DAPO 17K and prepared data)

:::

:::{tab-item} Required Accounts & Tokens

- **Weights & Biases (W&B) API Key** (optional): For experiment tracking and visualization; Training metrics logging
  - [Create account](https://wandb.ai/signup)
  - Find your API key at [wandb.ai/authorize](https://wandb.ai/authorize)
  - If not provided, training will proceed without W&B logging
- **HuggingFace Token** (optional): For downloading models and datasets
  - [Create account](https://huggingface.co/join)
  - Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  - Recommended to avoid rate limits when downloading models and datasets
  - Ensure you have accepted the model license for Qwen 3 4B Instruct

:::

:::{tab-item} Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+ or equivalent)
- **Python**: 3.12 or higher
- **NeMo RL Container**: Pre-built container image with NeMo RL dependencies
  - Path example: `/path/to/nemo-rl/container` (update with your actual container path)
- **Slurm**: For multi-node training on GPU clusters
- **Git**: For cloning repositories
- **UV Package Manager**: Python package manager (installed during setup)

:::

:::{tab-item} Filesystem Access

- **Shared Filesystem**: Required for multi-node training
  - Example: `/shared/filesystem` mounted and accessible from all compute nodes
  - Used for storing code, data, checkpoints, and results

:::

::::

---

## RL Training Workflow

This tutorial will guide you through the entire RL training workflow using the following steps:

::::{grid} 1
:gutter: 1

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` About the Workplace Assistant Training Environment
:link: training-nemo-rl-grpo-about-workplace-assistant
:link-type: ref

Understand the dataset you will train on and the capabilities it corresponds to.
+++
{bdg-primary}`prerequisite` {bdg-secondary}`dataset`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` NeMo Gym Configuration for RL Training
:link: training-nemo-rl-grpo-gym-configuration
:link-type: ref

Understand the Gym configuration component in the NeMo RL training config file.
+++
{bdg-secondary}`configuration`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` NeMo RL Configuration for RL Training
:link: training-nemo-rl-grpo-nemo-rl-configuration
:link-type: ref

Understand the GRPO and NeMo RL configuration components in the NeMo RL training config file.
+++
{bdg-secondary}`configuration`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Setup
:link: training-nemo-rl-grpo-setup
:link-type: ref

Necessary NeMo RL and NeMo Gym setup instructions.
+++
{bdg-primary}`prerequisite`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Single Node Training
:link: training-nemo-rl-grpo-single-node-training
:link-type: ref

Perform a single node GRPO training run with success criteria.
+++
{bdg-primary}`training` {bdg-secondary}`single-node`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Multi Node Training
:link: training-nemo-rl-grpo-multi-node-training
:link-type: ref

Perform a multi node GRPO training run.
+++
{bdg-primary}`training` {bdg-secondary}`multi-node`
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
```
