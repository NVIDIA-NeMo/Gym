(training-unsloth)=

# RL Training with Unsloth

This tutorial demonstrates how to use [Unsloth](https://github.com/unslothai/unsloth) to fine-tune models with NeMo Gym environments.

**Unsloth** is a fast, memory-efficient library for fine-tuning large language models. It provides optimized implementations that significantly reduce memory usage and training time, making it possible to fine-tune larger models on consumer hardware.

## Prerequisites

- A Google account (for Colab) or a local GPU with 16GB+ VRAM
- Familiarity with NeMo Gym concepts ({doc}`/get-started/index`)

---

## Getting Started

Follow these interactive notebooks to train models with Unsloth and NeMo Gym:

:::{button-link} https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/NeMo-Gym-Sudoku.ipynb
:color: primary
:class: sd-rounded-pill

Sudoku
:::

:::{button-link} https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/NeMo-Gym-Multi-Environment.ipynb
:color: secondary
:class: sd-rounded-pill

Multi-Environment Training
:::

Check out [Unsloth's documentation](https://docs.unsloth.ai/models/nemotron-3#reinforcement-learning--nemo-gym) for more details.

