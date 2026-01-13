(gs-first-training-run)=
# First Training Run

```{warning}
This article was generated and has not been reviewed. Content may change.
```

This tutorial walks you through your first complete RL training run with NeMo Gym—from collected rollouts to a measurably improved model.

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em;` **Time**
30-60 minutes
:::

:::{grid-item-card} {octicon}`bookmark;1em;` **Prerequisites**

- Completed {doc}`rollout-collection`
- Single GPU with 16GB+ VRAM
- Google account (for Colab)

:::

::::

---

## Training Configuration

| Component | Value |
|-----------|-------|
| **Model** | Qwen-2.5 3B |
| **Compute** | Single GPU (T4/L4/A10 or better) |
| **Framework** | [Unsloth](https://github.com/unslothai/unsloth) |
| **Algorithm** | GRPO |

:::{tip}
**Why Unsloth?** Unsloth provides 2-5× faster training with 70% less memory, making it ideal for single-GPU experimentation. For production multi-node training, see the {doc}`/tutorials/nemo-rl-grpo/index` tutorial.
:::

---

## Interactive Notebook (Recommended)

The fastest path to your first trained model is our interactive Colab notebook. It handles all setup and runs end-to-end in about 30 minutes on a free T4 GPU.

:::{button-link} https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/nemo_gym_sudoku.ipynb
:color: primary
:class: sd-rounded-pill

Open in Google Colab
:::

The notebook demonstrates:

1. **Environment**: Sudoku puzzle solving with NeMo Gym verification
2. **Training**: GRPO with Unsloth's optimized implementation
3. **Evaluation**: Before/after comparison showing measurable improvement

---

## What You'll Learn

By completing this tutorial, you'll understand the core RL training loop with NeMo Gym:

```{mermaid}
flowchart LR
    A[Training Data] --> B[Model Generates Response]
    B --> C[Gym Verifies Response]
    C --> D[Compute Reward]
    D --> E[Update Model via GRPO]
    E --> B
```

**Key concepts covered**:

- **Reward signal**: How NeMo Gym's `verify()` function provides training signal
- **GRPO algorithm**: Group Relative Policy Optimization for efficient RL
- **LoRA fine-tuning**: Memory-efficient adaptation of large models

---

## Local Training (Advanced)

If you prefer to run training locally instead of using Colab, install Unsloth and follow the [NeMo Gym integration guide](https://docs.unsloth.ai/models/nemotron-3#reinforcement-learning--nemo-gym) in the Unsloth documentation.

```bash
# Install Unsloth (requires CUDA)
pip install unsloth
```

:::{note}
Local training requires an NVIDIA GPU with CUDA support. The Colab notebook is recommended for first-time users.
:::

---

## What's Next?

After completing your first training run:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Scale to Production
:link: /tutorials/nemo-rl-grpo/index
:link-type: doc

Multi-node GRPO training with NeMo RL for production workloads.
+++
{bdg-secondary}`multi-node` {bdg-secondary}`nemo-rl`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Try Different Environments
:link: https://github.com/NVIDIA-NeMo/Gym#-available-resource-servers

Browse available resource servers for math, code, tool-use, and more.
+++
{bdg-secondary}`environments`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build Custom Environments
:link: /tutorials/creating-resource-server
:link-type: doc

Create your own training environment with custom tools and verification.
+++
{bdg-secondary}`custom`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Offline Training
:link: /tutorials/offline-training-w-rollouts
:link-type: doc

Use collected rollouts for SFT or DPO training.
+++
{bdg-secondary}`sft` {bdg-secondary}`dpo`
:::

::::
