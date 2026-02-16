---
description: "NeMo Gym is an open-source library for building reinforcement learning (RL) training environments for large language models (LLMs)"
categories:
  - documentation
  - home
tags:
  - reinforcement-learning
  - llm-training
  - rollout-collection
  - agent-environments
  - rl-environments
personas:
  - Data Scientists
  - Machine Learning Engineers
  - RL Researchers
difficulty: beginner
content_type: index
---

(gym-home)=

# NeMo Gym Documentation

[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) is a library for building reinforcement learning (RL) training environments for large language models (LLMs). NeMo Gym provides infrastructure to develop environments, scale rollout collection, and integrate seamlessly with your preferred training framework.

A training environment consists of three server components: **Agents** orchestrate the rollout lifecycle—calling models, executing tool calls through resources, and coordinating verification. **Models** provide stateless text generation using LLM inference endpoints. **Resources** define tasks, tool implementations, and verification logic.

````{div} sd-d-flex-row
```{button-ref} gs-quickstart
:ref-type: ref
:color: primary
:class: sd-rounded-pill sd-mr-3

Quickstart
```

```{button-ref} environment-tutorials/index
:ref-type: doc
:color: secondary
:class: sd-rounded-pill

Explore Tutorials
```
````

---

## Introduction to NeMo Gym

Understand NeMo Gym's purpose and core components before diving into tutorials.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` About NeMo Gym
:link: about/index
:link-type: doc
Motivation and benefits of NeMo Gym.
+++
{bdg-secondary}`motivation` {bdg-secondary}`benefits`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Concepts
:link: about/concepts/index
:link-type: doc
Core components, configuration, verification and RL terminology.
+++
{bdg-secondary}`environments` {bdg-secondary}`agents` {bdg-secondary}`models` {bdg-secondary}`resources`
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Ecosystem
:link: about/ecosystem
:link-type: doc
Understand how NeMo Gym fits within the RL environment ecosystem.
+++
{bdg-secondary}`ecosystem` {bdg-secondary}`integrations`
:::

::::

## Get Started

Install and run NeMo Gym to start collecting rollouts.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quickstart
:link: get-started/index
:link-type: doc
Install, start servers, and collect your first rollouts in one page.
+++
{bdg-primary}`start here` {bdg-secondary}`5 min`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Detailed Setup Guide
:link: get-started/detailed-setup
:link-type: doc
Step-by-step installation with requirements, configuration, and troubleshooting.
+++
{bdg-secondary}`15 min` {bdg-secondary}`environment` {bdg-secondary}`configuration`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Rollout Collection
:link: get-started/rollout-collection
:link-type: doc
Generate batches of scored interactions and view them with the rollout viewer.
+++
{bdg-secondary}`10 min` {bdg-secondary}`rollouts` {bdg-secondary}`training-data`
:::

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` First Training Run
:link: get-started/first-training-run
:link-type: doc
Train a Sudoku-solving model with GRPO using a self-contained Colab notebook.
+++
{bdg-secondary}`30 min` {bdg-secondary}`training` {bdg-secondary}`colab`
:::

::::

## Environment Configuration

Configure and customize environment components and prepare datasets.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` Model Server
:link: model-server/index
:link-type: doc
Configure LLM inference backends including vLLM.
+++
{bdg-secondary}`inference` {bdg-secondary}`vllm`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Data
:link: data/index
:link-type: doc
Prepare and validate training datasets.
+++
{bdg-secondary}`datasets` {bdg-secondary}`jsonl`
:::

::::

## Environment Tutorials

Learn how to build custom training environments for various RL scenarios.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`plus-circle;1.5em;sd-mr-1` Creating Environments
:link: environment-tutorials/creating-training-environment
:link-type: doc
Build a complete training environment from scratch.
+++
{bdg-primary}`beginner` {bdg-secondary}`foundational`
:::

::::

```{button-ref} environment-tutorials/index
:ref-type: doc
:color: secondary
:class: sd-rounded-pill

View all environment tutorials →
```

## Training Tutorials

Train models using NeMo Gym with your preferred RL framework.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` RL (GRPO)
:link: training-tutorials/index
:link-type: doc
Hands-on tutorials with NeMo RL, TRL, Unsloth, and more.
+++
{bdg-secondary}`grpo`
:::

:::{grid-item-card} {octicon}`file;1.5em;sd-mr-1` SFT & DPO
:link: offline-training-w-rollouts
:link-type: ref
Transform rollouts into SFT and DPO format.
+++
{bdg-secondary}`sft` {bdg-secondary}`dpo`
:::

::::

```{button-ref} training-tutorials/index
:ref-type: doc
:color: secondary
:class: sd-rounded-pill

View all training tutorials →
```

## Infrastructure

Deploy NeMo Gym and plan cluster resources for training.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Deployment Topology
:link: infrastructure/deployment-topology
:link-type: doc
Production deployment patterns and configurations.
+++
{bdg-secondary}`deployment` {bdg-secondary}`topology`
:::

::::

## Contribute

Contribute to NeMo Gym development.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Contribute Environments
:link: contribute/environments/index
:link-type: doc
Contribute new environments or integrate existing benchmarks.
+++
{bdg-primary}`environments`
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Integrate RL Frameworks
:link: contribute/rl-framework-integration/index
:link-type: doc
Implement NeMo Gym integration into a new training framework.
+++
{bdg-primary}`training-integration`
:::

::::

---

```{toctree}
:hidden:
Home <self>
```

```{toctree}
:caption: About
:hidden:
:maxdepth: 2

Overview <about/index.md>
Concepts <about/concepts/index>
Ecosystem <about/ecosystem>
```

```{toctree}
:caption: Get Started
:hidden:
:maxdepth: 1

Quickstart <get-started/index>
Detailed Setup Guide <get-started/detailed-setup.md>
Rollout Collection <get-started/rollout-collection.md>
🟡 First Training Run <get-started/first-training-run.md>
```

```{toctree}
:caption: Model Server
:hidden:
:maxdepth: 1

Overview <model-server/index>
vLLM <model-server/vllm>
```

```{toctree}
:caption: Data
:hidden:
:maxdepth: 1

Overview <data/index>
Prepare and Validate <data/prepare-validate>
Download from Hugging Face <data/download-huggingface>
```

```{toctree}
:caption: Environment Tutorials
:hidden:
:maxdepth: 1

Overview <environment-tutorials/index>
🟡 Creating Training Environment <environment-tutorials/creating-training-environment>
Multi-Environment Training <environment-tutorials/multi-environment-training>
```

```{toctree}
:caption: Training Tutorials
:hidden:
:maxdepth: 1

Overview <training-tutorials/index>
NeMo RL <training-tutorials/nemo-rl-grpo/index.md>
TRL <training-tutorials/trl>
Unsloth <training-tutorials/unsloth-training>
Offline Training (SFT/DPO) <training-tutorials/offline-training-w-rollouts>
```

```{toctree}
:caption: Model Recipes
:hidden:
:maxdepth: 1

🟡 Overview <model-recipes/index>
🟡 Nemotron 3 Nano <model-recipes/nemotron-nano>
```

```{toctree}
:caption: Infrastructure
:hidden:
:maxdepth: 1

Overview <infrastructure/index>
Deployment Topology <infrastructure/deployment-topology>
Engineering Notes <infrastructure/engineering-notes/index>
```

```{toctree}
:caption: Reference
:hidden:
:maxdepth: 1

Configuration <reference/configuration>
reference/cli-commands.md
apidocs/index.rst
FAQ <reference/faq>
```

```{toctree}
:caption: Troubleshooting
:hidden:
:maxdepth: 1

troubleshooting/configuration.md
```

```{toctree}
:caption: Contribute
:hidden:
:maxdepth: 1

Overview <contribute/index>
Development Setup <contribute/development-setup>
Environments <contribute/environments/index>
Integrate RL Frameworks <contribute/rl-framework-integration/index>
```
