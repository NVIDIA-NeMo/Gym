(tutorials-index)=

# NeMo Gym Tutorials

Hands-on learning experiences that guide you through building, training, and deploying AI agents with NeMo Gym.

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
Build custom resource servers with tools, verification logic, and business logic for your AI agents.
+++
{bdg-primary}`beginner` {bdg-secondary}`30 min`
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

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` RL Training with NeMo RL
:link: rl-training-with-nemo-rl
:link-type: doc
Train a model with NeMo RL. Learn how to set up NeMo Gym and NeMo RL training environments, run tests, prepare data, and launch single-node and multi-node training runs.
+++
{bdg-secondary}`rl` {bdg-secondary}`training`
:::

::::
