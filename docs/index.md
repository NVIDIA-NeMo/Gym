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
:caption: Tutorials
:hidden:
:maxdepth: 1

Overview <tutorials/index>
tutorials/offline-training-w-rollouts
tutorials/separate-policy-and-judge-models
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

