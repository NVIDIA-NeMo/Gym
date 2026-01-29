(training-unsloth)=

# RL Training with Unsloth

This tutorial demonstrates how to use [Unsloth](https://github.com/unslothai/unsloth) to fine-tune models for single-step tasks with NeMo Gym verifiers and datasets.

**Unsloth** is a fast, memory-efficient library for fine-tuning large language models. It provides optimized implementations that significantly reduce memory usage and training time, making it possible to fine-tune larger models on consumer hardware.

Unsloth can be used with NeMo Gym single-step verifiers including math tasks, structured outputs, instruction following, reasoning gym, and more. 

:::{card}

**Goal**: Fine-tune a model for single-step tasks using Unsloth with NeMo Gym verifiers.

^^^

**In this tutorial, you will**:

1. Set up Unsloth for efficient fine-tuning
2. Use NeMo Gym for tasks and verification
3. Train a model using GRPO on a single GPU
4. Evaluate trained model performance 

:::

## Getting Started

Follow this interactive notebook to train your first model with Unsloth and NeMo Gym:

:::{button-link} https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/nemo_gym_sudoku.ipynb
:color: primary
:class: sd-rounded-pill

Unsloth GRPO notebook
:::

Check out [Unsloth's documentation](https://docs.unsloth.ai/models/nemotron-3#reinforcement-learning--nemo-gym) for more details.

> **Note:** This notebook supports **single-step tasks** including math, structured outputs, instruction following, reasoning gym, and more. For multi-step tool calling scenarios, see the {doc}`GRPO with NeMo RL <nemo-rl-grpo/index>` tutorial.

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
