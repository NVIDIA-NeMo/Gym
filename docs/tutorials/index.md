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


---

## Training framework integration

Implement NeMo Gym integration into a new training framework. This is only for expert users that cannot use existing training framework integrations with Gym and need to implement their own.

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Generation backend and OpenAI-compatible HTTP server
:link: generation-backend-and-http-server
:link-type: doc
OpenAI compatible HTTP server pre-requisites for Gym integration into an RL framework.
+++
{bdg-secondary}`training` {bdg-secondary}`infra`
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` OpenAI-compatible HTTP server On-Policy correction
:link: rl-training-with-nemo-rl
:link-type: doc
Helpful On-Policy fixes for OpenAI-compatible HTTP server implementations in multi step and multi turn scenarios.
+++
{bdg-secondary}`training` {bdg-secondary}`infra`
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Gym integration footprint and form factor
:link: rl-training-with-nemo-rl
:link-type: doc
Implementation details like footprint and form factor for the actual Gym + RL integration.
+++
{bdg-secondary}`training` {bdg-secondary}`infra`
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Gym + RL framework integration success criteria
:link: rl-training-with-nemo-rl
:link-type: doc
What success criteria can be used to validate Gym + RL framework integration correctness.
+++
{bdg-secondary}`training` {bdg-secondary}`infra`
:::

::::
