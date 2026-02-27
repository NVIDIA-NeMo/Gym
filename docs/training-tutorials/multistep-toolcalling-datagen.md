(multistep-toolcalling-datagen)=

# Generating Multi-Step Tool-Calling Data

This tutorial walks through generating synthetic training data for multi-step tool-calling agents using [NVIDIA Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner). The generated data targets the **Workplace Assistant** environment and can be used for rollout collection and RLVR training with NeMo Gym.

:::{card}

**Goal**: Generate high-quality synthetic user queries and agent trajectories for multi-step tool-calling RL training.

^^^

**In this tutorial, you will**:

1. Load tool schemas for the Workplace Assistant environment (27 tools across 6 databases)
2. Use Data Designer to generate realistic multi-step user queries
3. Simulate agent trajectories (step-by-step tool-call solutions)
4. Apply dual-level LLM judge filtering to ensure data quality
5. Export training data in NeMo Gym JSONL format

:::

## Prerequisites

- **NVIDIA API Key** from [build.nvidia.com](https://build.nvidia.com) (or your own LLM endpoint)
- **Python 3.11+**
- Install tutorial dependencies:

  ```bash
  cd examples/multistep_toolcalling_datagen/
  uv pip install -r requirements.txt
  ```

## Running the Notebook

The tutorial is provided as a Jupyter notebook at `examples/multistep_toolcalling_datagen/multistep-toolcalling.ipynb`.

> **Important:** Run the notebook from the `examples/multistep_toolcalling_datagen/` directory so that relative imports for `tools/` and `utils/` resolve correctly.

:::{button-link} https://github.com/NVIDIA-NeMo/Gym/blob/main/examples/multistep_toolcalling_datagen/multistep-toolcalling.ipynb
:color: primary
:class: sd-rounded-pill

View Notebook on GitHub
:::

---

## What's Next?

After generating your training data, explore these options:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` Collect Rollouts
:link: ../get-started/rollout-collection
:link-type: doc

Use the generated JSONL with the Workplace Assistant resource server to collect rollouts.
+++
{bdg-secondary}`rollouts` {bdg-secondary}`workplace-assistant`
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` GRPO with NeMo RL
:link: nemo-rl-grpo/index
:link-type: doc

Train multi-step tool-calling agents with GRPO and NeMo RL.
+++
{bdg-secondary}`rl` {bdg-secondary}`grpo` {bdg-secondary}`multi-step`
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Offline Training with Rollouts
:link: offline-training-w-rollouts
:link-type: doc

Transform rollouts into SFT and DPO training data.
+++
{bdg-secondary}`sft` {bdg-secondary}`dpo`
:::

::::
