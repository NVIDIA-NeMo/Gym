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

### Required Knowledge

- You should be comfortable with Python, LLM fine-tuning, and basic reinforcement learning concepts such as policy optimization, rewards, and rollouts. While in-depth knowledge of Reinforcement Learning with Verifiable Rewards (RLVR) and the GRPO algorithm is not required, a high-level understanding is helpful.
- Some basic familiarity with Slurm is useful, but you can follow along using the example commands provided below.

### Hardware Requirements

**Minimum** 1 node of 8× NVIDIA GPUs with 80GB or more memory each (e.g., H100, A100) is required.

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

### Required Accounts & Tokens
- **Weights & Biases (W&B) API Key** (optional): For experiment tracking and visualization; Training metrics logging
  - [Create account](https://wandb.ai/signup)
  - Find your API key at [wandb.ai/authorize](https://wandb.ai/authorize)
  - If not provided, training will proceed without W&B logging
- **HuggingFace Token** (optional): For downloading models and datasets
  - [Create account](https://huggingface.co/join)
  - Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
  - Recommended to avoid rate limits when downloading models and datasets
  - Ensure you have accepted the model license for Qwen 3 4B Instruct

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+ or equivalent)
- **Python**: 3.12 or higher
- **NeMo RL Container**: Pre-built container image with NeMo RL dependencies
  - Path example: `/path/to/nemo-rl/container` (update with your actual container path)
- **Slurm**: For multi-node training on GPU clusters
- **Git**: For cloning repositories
- **UV Package Manager**: Python package manager (installed during setup)

### Filesystem Access

- **Shared Filesystem**: Required for multi-node training
  - Example: `/shared/filesystem` mounted and accessible from all compute nodes
  - Used for storing code, data, checkpoints, and results


## Integration Components

Gym integration requires implementing the following components in your training framework:

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Generation Backend
:link: generation-backend-and-openai-compatible-http-server
:link-type: doc

OpenAI-compatible HTTP server requirements and existing implementations across RL frameworks.
+++
{bdg-primary}`prerequisite`
:::

::::

## Integration Workflow

The typical integration workflow follows this sequence:

```{list-table}
:header-rows: 1
:widths: 10 30 60

* - Step
  - Component
  - Description
* - 1
  - Generation backend
  - Expose your generation engine (vLLM, SGLang) as an OpenAI-compatible HTTP server
* - 2
  - On-policy corrections
  - Implement token ID fixes to prevent re-tokenization and re-templating issues
* - 3
  - Gym integration
  - Connect Gym to your training loop using the rollout orchestration APIs
* - 4
  - Validation
  - Verify integration using the success criteria benchmarks
```

```{toctree}
:caption: NeMo RL GRPO
:hidden:
:maxdepth: 1

about-workplace-assistant.md
setup.md
gym-configuration.md
nemo-rl-configuration.md
single-node-training.md
multi-node-training.md
```
