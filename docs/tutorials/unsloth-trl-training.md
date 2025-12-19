(training-unsloth-trl)=

# RL Training with Unsloth and TRL

This tutorial demonstrates how to use [Unsloth](https://github.com/unslothai/unsloth) and [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl) to fine-tune models for single-step tasks with NeMo Gym verifiers and datasets.

**Unsloth** is a fast, memory-efficient library for fine-tuning large language models. It provides optimized implementations that significantly reduce memory usage and training time, making it possible to fine-tune larger models on consumer hardware.

**TRL** is a library from HuggingFace for post-training models using techniques like SFT, GRPO, and DPO. It is built on top of [Transformers](https://github.com/huggingface/transformers) and supports a variety of model architectures and modalities. 


Both Unsloth and TRL can be used with NeMo Gym single-step verifiers including math tasks, structured outputs, instruction following, reasoning gym, and more. 

:::{card}

**Goal**: Fine-tune a model for single-step tasks using Unsloth and TRL with NeMo Gym verifiers.

^^^

**In this tutorial, you will**:

1. Set up Unsloth for efficient fine-tuning
2. Use NeMo Gym for tasks and verification
3. Train a model using GRPO on a single gpu 
4. Evaluate trained model performance 

:::

## Getting Started

Follow this interactive notebook to train your first model with Unsloth or TRL and NeMo Gym:

:::{button-link} https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/nemo_gym_sudoku.ipynb
:color: primary
:class: sd-rounded-pill

Unsloth GRPO notebook
:::

> **Note:** This notebook supports **single-step tasks** including math, structured outputs, instruction following, reasoning gym, and more. For multi-step tool calling scenarios, see the {doc}`GRPO with NeMo RL <nemo-rl-grpo/index>` tutorial. A complete multi-step and multi-turn integration with Unsloth and TRL is under construction! 

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

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Multi-Step Tool Calling
:link: nemo-rl-grpo/index
:link-type: doc

Scale to multi-step scenarios with GRPO and NeMo RL.
+++
{bdg-secondary}`rl` {bdg-secondary}`grpo` {bdg-secondary}`multi-step`
:::

::::
