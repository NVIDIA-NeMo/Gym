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

```{button-ref} tutorials/index
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
{bdg-secondary}`agents` {bdg-secondary}`models` {bdg-secondary}`resources`
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Ecosystem
:link: about/ecosystem
:link-type: doc
Understand how NeMo Gym fits within the NVIDIA NeMo Framework.
+++
{bdg-secondary}`nemo-framework`
:::

::::

## Get Started

Install and run NeMo Gym to start collecting rollouts.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quickstart
:link: get-started/index
:link-type: doc
Run a training environment and start collecting rollouts in under 5 minutes.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Detailed Setup Guide
:link: get-started/detailed-setup
:link-type: doc
Detailed walkthrough of running your first training environment.
+++
{bdg-secondary}`environment` {bdg-secondary}`configuration`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Rollout Collection
:link: get-started/rollout-collection
:link-type: doc
Collect and view rollouts
+++
{bdg-secondary}`rollouts` {bdg-secondary}`training-data`
:::

::::

## Tutorials

Hands-on tutorials to build and customize your training environments.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build a Resource Server
:link: tutorials/creating-resource-server
:link-type: doc
Implement or integrate existing tools and define task verification logic.
+++
{bdg-secondary}`custom-environments` {bdg-secondary}`tools`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Offline Training (SFT, DPO)
:link: tutorials/offline-training-w-rollouts
:link-type: doc
Train with SFT or DPO using collected rollouts.
+++
{bdg-secondary}`sft` {bdg-secondary}`dpo`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` RL Training with NeMo RL
:link: tutorials/rl-training-with-nemo-rl
:link-type: doc
Train with GRPO using NeMo RL and NeMo Gym.
+++
{bdg-secondary}`grpo` {bdg-secondary}`nemo-rl`
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
```

```{toctree}
:caption: Tutorials
:hidden:
:maxdepth: 1

tutorials/index.md
tutorials/creating-resource-server
tutorials/offline-training-w-rollouts
tutorials/rl-training-with-nemo-rl
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
