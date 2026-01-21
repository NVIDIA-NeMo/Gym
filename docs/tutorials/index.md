---
orphan: true
---

(tutorials-index)=

# NeMo Gym Tutorials

Hands-on tutorials to build and customize your training environments.

:::{tip}
**New to NeMo Gym?** Begin with the {doc}`Get Started <../get-started/index>` section for a guided tutorial from installation through your first verified agent. Return here afterward to learn about advanced topics like additional rollout collection methods and training data generation. You can find the project repository on [GitHub](https://github.com/NVIDIA-NeMo/Gym).
:::
---

## Building Custom Components

Create custom resource servers and implement tool-based agent interactions.

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Creating a Resource Server
:link: creating-resource-server
:link-type: doc
Implement or integrate existing tools and define task verification logic.
+++
{bdg-primary}`beginner` {bdg-secondary}`30 min` {bdg-secondary}`custom-environments` {bdg-secondary}`tools`
:::

::::

---

## Rollout Collection and Training Data

Implement rollout generation and training data preparation for RL, SFT, and DPO.

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Offline Training with Rollouts
:link: offline-training-w-rollouts
:link-type: doc
Transform rollouts into training data for {term}`supervised fine-tuning (SFT) <SFT (Supervised Fine-Tuning)>` and {term}`direct preference optimization (DPO) <DPO (Direct Preference Optimization)>`.
+++
{bdg-secondary}`sft` {bdg-secondary}`dpo`
:::

::::

---

## RL Training

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` GRPO with NeMo RL
:link: training-nemo-rl-grpo-index
:link-type: ref
Learn how to set up NeMo Gym and NeMo RL training environments, run tests, prepare data, and launch single-node and multi-node training runs.
+++
{bdg-primary}`training` {bdg-secondary}`rl` {bdg-secondary}`grpo` {bdg-secondary}`multi-step`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Unsloth
:link: training-unsloth
:link-type: ref
Fast, memory-efficient fine-tuning for single-step tasks: math, structured outputs, instruction following, reasoning gym and more.
+++
{bdg-primary}`training` {bdg-secondary}`unsloth` {bdg-secondary}`single-step`
:::

::::
