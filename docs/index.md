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

NeMo Gym is a framework for building reinforcement learning (RL) training environments large language models (LLMs). Gym provides training environment development scaffolding and training environment patterns such as multi-step, multi-turn, and user modeling scenarios.

At the core of NeMo Gym are three server concepts: **Responses API Model servers** are model endpoints, **Resources servers** contain tool implementations and verification logic, and **Response API Agent servers** orchestrate the interaction between models and resources.

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

## Introduction to Gym

Learn about NeMo Gym, how it works at a high level, and the key concepts.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` About Gym
:link: about/index
:link-type: doc
Overview of NeMo Gym and its capabilities for building RL training environments.
+++
{bdg-secondary}`target-users` {bdg-secondary}`how-it-works`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Concepts
:link: about/concepts/index
:link-type: doc
Explore the core components: Agents, Models, and Resources.
+++
{bdg-secondary}`agents` {bdg-secondary}`models` {bdg-secondary}`resources`
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Ecosystem
:link: about/ecosystem
:link-type: doc
Understand how NeMo Gym integrates with NeMo RL and other training frameworks.
+++
{bdg-secondary}`nemo-rl` {bdg-secondary}`integrations`
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
+++
{bdg-primary}`beginner`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Setup and Installation
:link: get-started/setup-installation
:link-type: doc
Complete guide with requirements, model provider options, and troubleshooting.
+++
{bdg-secondary}`environment` {bdg-secondary}`configuration`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Rollout Collection
:link: get-started/rollout-collection
:link-type: doc
Batch processing, parameter tuning, and the rollout viewer tool.
+++
{bdg-secondary}`training-data` {bdg-secondary}`scale`
:::

::::

## Tutorials

Hands-on tutorials to build and customize your training environments.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build a Resource Server
:link: tutorials/creating-resource-server
:link-type: doc
Implement custom tools and define task verification logic.
+++
{bdg-secondary}`custom-environments` {bdg-secondary}`tools`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Offline Training
:link: tutorials/offline-training-w-rollouts
:link-type: doc
Train models offline using collected rollouts.
+++
{bdg-secondary}`training` {bdg-secondary}`datasets`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` RL Training with NeMo RL
:link: tutorials/rl-training-with-nemo-rl
:link-type: doc
Integrate NeMo Gym with NeMo RL for end-to-end RL training.
+++
{bdg-secondary}`nemo-rl` {bdg-secondary}`integration`
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

about/index.md
Concepts <about/concepts/index>
Ecosystem <about/ecosystem>
```

```{toctree}
:caption: Get Started
:hidden:
:maxdepth: 1

Overview <get-started/index>
get-started/setup-installation.md
get-started/rollout-collection.md
```

```{toctree}
:caption: Tutorials
:hidden:
:maxdepth: 1

tutorials/index.md
tutorials/creating-resource-server
tutorials/offline-training-w-rollouts
tutorials/rl-training-with-nemo-rl
how-to-faq.md
```

```{toctree}
:caption: Reference
:hidden:
:maxdepth: 1

reference/cli-commands.md
```

```{toctree}
:caption: Development
:hidden:

apidocs/index.rst
```
