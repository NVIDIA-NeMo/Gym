(gym-home)=

# NeMo Gym Documentation

Welcome to the NeMo Gym documentation.

## Introduction to Gym

Learn about the Gym, how it works at a high-level, and the key features.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` About NeMo Gym
:link: about-overview
:link-type: ref
Overview of NeMo Gym and its approach to scalable rollout collection.
+++
{bdg-secondary}`target-users` {bdg-secondary}`core-components`
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Key Features
:link: about-features
:link-type: ref
Comprehensive catalog of NeMo Gym capabilities and design principles.
+++
{bdg-secondary}`features` {bdg-secondary}`capabilities`
:::

:::{grid-item-card} {octicon}`stack;1.5em;sd-mr-1` Architecture
:link: about-architecture
:link-type: ref
How NeMo Gym components work together and interact.
+++
{bdg-secondary}`system-design` {bdg-secondary}`deployment`
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Ecosystem
:link: about-ecosystem
:link-type: ref
NeMo Gym's place in the NVIDIA NeMo Framework and ecosystem.
+++
{bdg-secondary}`nemo-framework` {bdg-secondary}`positioning`
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Concepts
:link: about-concepts
:link-type: ref
Core concepts behind models, resources, agents, and verification.
+++
{bdg-secondary}`mental-models` {bdg-secondary}`abstractions`
:::

::::

## Get Started

New to NeMo Gym? Follow our guided tutorial path to build your first agent.

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` 1. Setup and Installation
:link: get-started/setup-installation
:link-type: doc
Get NeMo Gym installed and servers running with your first successful agent interaction.
+++
{bdg-secondary}`environment` {bdg-secondary}`first-run`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` 2. Your First Agent
:link: get-started/first-agent
:link-type: doc
Understand how your weather agent works and learn to interact with it.
+++
{bdg-secondary}`workflow` {bdg-secondary}`tools`
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` 3. Verifying Agent Results
:link: get-started/verifying-agent-results
:link-type: doc
Understand how NeMo Gym evaluates agent performance and what verification means for training.
+++
{bdg-secondary}`rewards` {bdg-secondary}`scoring`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` 4. Collecting Rollouts
:link: get-started/collecting-rollouts
:link-type: doc
Generate your first batch of rollouts and understand how they become training data.
+++
{bdg-secondary}`training-data` {bdg-secondary}`scale`
:::

::::

---

## Setup & Deployment

Configure NeMo Gym for your project. Manage environments, secure secrets, and deploy in development, testing, or production scenarios.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration
:link: setup-deployment/configuration/index
:link-type: doc
Master the three-tier configuration system for managing dev/test/prod environments and secrets.
+++
{bdg-secondary}`how-to` {bdg-secondary}`config` {bdg-secondary}`secrets`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Deployment
:link: setup-deployment/deployment/index
:link-type: doc
Deploy NeMo Gym locally, on remote servers, or in containerized infrastructure.
+++
{bdg-secondary}`how-to` {bdg-secondary}`deployment` {bdg-secondary}`docker`
:::

:::{grid-item-card} {octicon}`pulse;1.5em;sd-mr-1` Operations
:link: setup-deployment/operations/index
:link-type: doc
Monitor, test, and debug your NeMo Gym deployment for reliable operation.
+++
{bdg-secondary}`how-to` {bdg-secondary}`monitoring` {bdg-secondary}`debugging`
:::

::::

---

## Training

Scale up training data generation and integrate with RL frameworks. Master rollout collection, data quality, and framework integration for production workflows.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Rollout Collection
:link: training/rollout-collection/index
:link-type: doc
Generate training rollouts at scale with optimized sampling and parallelization.
+++
{bdg-secondary}`data-generation` {bdg-secondary}`throughput`
:::

:::{grid-item-card} {octicon}`trophy;1.5em;sd-mr-1` Verification
:link: training/verification/index
:link-type: doc
Design reward signals and verification strategies that drive effective training.
+++
{bdg-secondary}`rewards` {bdg-secondary}`scoring` {bdg-secondary}`verification`
:::

:::{grid-item-card} {octicon}`filter;1.5em;sd-mr-1` Data Quality
:link: training/data-quality/index
:link-type: doc
Filter, curate, and balance rollouts to ensure high-quality training datasets.
+++
{bdg-secondary}`filtering` {bdg-secondary}`curation`
:::

:::{grid-item-card} {octicon}`package-dependencies;1.5em;sd-mr-1` Datasets
:link: training/datasets/index
:link-type: doc
Organize, validate, and prepare datasets for RL training frameworks.
+++
{bdg-secondary}`formats` {bdg-secondary}`sft-dpo`
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Integration
:link: training/integration/index
:link-type: doc
Connect to NeMo-RL, VeRL, OpenRLHF, TRL, and custom frameworks.
+++
{bdg-secondary}`rl-frameworks` {bdg-secondary}`integration`
:::

::::

---

## Models

Configure model serving methods for generating agent responses and tool-calling decisions.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` vLLM
:link: models/vllm/index
:link-type: doc
Self-hosted open models with high-throughput inference.
+++
{bdg-secondary}`self-hosted` {bdg-secondary}`high-performance`
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` NVIDIA NIM
:link: models/nvidia-nim/index
:link-type: doc
Enterprise-grade model deployment with NVIDIA NIM microservices.
+++
{bdg-secondary}`enterprise` {bdg-secondary}`production`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` OpenAI
:link: models/openai/index
:link-type: doc
Use OpenAI models through the Responses API format.
+++
{bdg-secondary}`managed` {bdg-secondary}`cloud`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Azure OpenAI
:link: models/azure-openai/index
:link-type: doc
Access Azure OpenAI endpoints for LLM inference.
+++
{bdg-secondary}`azure` {bdg-secondary}`enterprise`
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` OpenRouter
:link: models/openrouter/index
:link-type: doc
Unified access to multiple model providers.
+++
{bdg-secondary}`multi-provider` {bdg-secondary}`flexible`
:::

::::

---

## Resources

Quick reference materials to support your work with NeMo Gym.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Glossary
:link: resources/glossary
:link-type: doc
Essential terminology for agent training and RL workflows.
+++
{bdg-secondary}`reference` {bdg-secondary}`terminology`
:::

::::

---

```{toctree}
:caption: About
:hidden:
:maxdepth: 2

Overview <about/index>
Ecosystem <about/ecosystem>
Architecture <about/architecture>
Key Features <about/features>
Concepts <about/concepts/index>
Release Notes <about/release-notes/index>
```

```{toctree}
:caption: Get Started
:hidden:
:maxdepth: 1

get-started/index
get-started/setup-installation
get-started/first-agent
get-started/verifying-agent-results
get-started/collecting-rollouts
```

```{toctree}
:caption: Setup & Deployment
:hidden:
:maxdepth: 2

Overview <setup-deployment/index>
Configuration <setup-deployment/configuration/index>
Deployment <setup-deployment/deployment/index>
Operations <setup-deployment/operations/index>
```

```{toctree}
:caption: Tutorials
:hidden:
:maxdepth: 1

Overview <tutorials/index>
tutorials/offline-training-w-rollouts
tutorials/separate-policy-and-judge-models
```

```{toctree}
:caption: Training
:hidden:
:maxdepth: 2

Overview <training/index>
Resource Servers <training/resource-servers/index>
Rollout Collection <training/rollout-collection/index>
Verification <training/verification/index>
Data Quality <training/data-quality/index>
Datasets <training/datasets/index>
```

```{toctree}
:caption: Models
:hidden:
:maxdepth: 2

Overview <models/index>
vLLM <models/vllm/index>
NVIDIA NIM <models/nvidia-nim/index>
OpenAI <models/openai/index>
Azure OpenAI <models/azure-openai/index>
OpenRouter <models/openrouter/index>
```

```{toctree}
:caption: Development
:hidden:

apidocs/index.rst
README
```

```{toctree}
:caption: Resources
:hidden:
:maxdepth: 1

Overview <resources/index>
resources/glossary
```
