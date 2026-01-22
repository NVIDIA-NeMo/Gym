(gs-first-training-run)=
# First Training Run

This tutorial guides you through your first RL training run using NeMo Gym for task verification and Unsloth for efficient fine-tuning. By the end, you'll have a model that demonstrably improves at solving Sudoku puzzles.

:::{card}

**Goal**: Train your first RL model using NeMo Gym verification and Unsloth.

^^^

**In this tutorial, you will**:

1. Run a Colab notebook for Sudoku RL training
2. Understand the training loop and metrics
3. Evaluate before/after model performance

:::

:::{tip}
**No prior NeMo Gym setup required.** The Colab notebook is self-contained. For local training, see [Local Training](#local-training-advanced) below.
:::

---

## Training Configuration

| Component | Value |
|-----------|-------|
| **Model** | Qwen-2.5 3B |
| **Task** | Sudoku puzzle solving |
| **Compute** | Single GPU (Colab T4 or local GPU with 16GB+ VRAM) |
| **Framework** | [Unsloth](https://github.com/unslothai/unsloth) |
| **Algorithm** | GRPO (Group Relative Policy Optimization) |

:::{tip}
**Why Unsloth?** Unsloth provides optimized memory usage, making it possible to train on free Colab T4 GPUs. For production multi-node training, see {doc}`/tutorials/nemo-rl-grpo/index`.
:::

---

## Interactive Notebook (Recommended)

The Colab notebook is self-contained—it installs dependencies, downloads the model, and runs training in ~30 minutes on a free T4 GPU.

:::{button-link} https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/nemo_gym_sudoku.ipynb
:color: primary
:class: sd-rounded-pill

Open in Google Colab
:::

**What the notebook does**:

1. Installs Unsloth and NeMo Gym dependencies
2. Loads Qwen-2.5 3B with LoRA adapters
3. Trains on Sudoku puzzles using GRPO with automatic verification
4. Shows before/after accuracy comparison

**Expected outcome**: After training, model accuracy on Sudoku puzzles should noticeably improve. Exact results vary by random seed and runtime conditions.

---

## What You'll Learn

By completing this tutorial, you'll understand the core RL training loop:

```{mermaid}
flowchart LR
    A[Sudoku Puzzle] --> B[Model Generates Solution]
    B --> C[Verifier Checks Solution]
    C --> D[Reward: 1.0 or 0.0]
    D --> E[GRPO Updates Model]
    E --> B
```

**How verification works**: NeMo Gym's `reasoning_gym` integration automatically verifies solutions:

```python
# Simplified verification flow (handled by the notebook)
score_fn = reasoning_gym.get_score_answer_fn("sudoku")
reward = score_fn(answer=model_answer, entry=puzzle_entry)
# reward = 1.0 if correct, 0.0 if incorrect
```

**Key concepts**:

- **Reward signal**: Binary verification (correct/incorrect) drives model improvement
- **GRPO**: Groups responses and updates based on relative performance within the group
- **LoRA**: Trains adapter weights instead of full model, significantly reducing memory requirements

---

(local-training-advanced)=
## Local Training (Advanced)

Run training locally with an NVIDIA GPU (16GB+ VRAM recommended).

### Setup

```bash
# Create environment and install dependencies
pip install unsloth reasoning_gym

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Run Training

Download and run the notebook locally, or follow the [Unsloth NeMo Gym integration guide](https://docs.unsloth.ai/models/nemotron-3#reinforcement-learning--nemo-gym) for custom configurations.

```bash
# Download the notebook
wget https://raw.githubusercontent.com/unslothai/notebooks/main/nb/nemo_gym_sudoku.ipynb

# Run with Jupyter
jupyter notebook nemo_gym_sudoku.ipynb
```

:::{warning}
**Memory requirements**: Qwen-2.5 3B with LoRA requires ~10GB VRAM. If you encounter OOM errors, reduce `per_device_train_batch_size` in the notebook.
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
