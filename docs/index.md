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

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Concepts
:link: about-concepts
:link-type: ref
Core concepts behind models, resources, agents, and verification.
+++
{bdg-secondary}`mental-models` {bdg-secondary}`abstractions`
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

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Key Features
:link: about-features
:link-type: ref
Comprehensive catalog of NeMo Gym capabilities and design principles.
+++
{bdg-secondary}`features` {bdg-secondary}`capabilities`
:::

::::

## Tutorial Highlights

Start with these tutorials to learn NeMo Gym fundamentals and test and train a sample agent.

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Setup and Installation
:link: tutorials/02-setup
:link-type: doc
Get NeMo Gym installed and servers running.
+++
{bdg-primary}`beginner` {bdg-secondary}`installation`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Your First Agent
:link: tutorials/03-your-first-agent
:link-type: doc
Understand how your weather agent works and learn to interact with it.
+++
{bdg-primary}`beginner` {bdg-secondary}`hands-on`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Rollout Collection
:link: tutorials/05-rollout-collection
:link-type: doc
Master rollout collection for training data and evaluation.
+++
{bdg-secondary}`data-generation` {bdg-secondary}`training`
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
Concepts <about/concepts/index>
Architecture <about/architecture>
Ecosystem <about/ecosystem>
<!-- Key Features <about/features> -->
Release Notes <about/release-notes/index>
```

```{toctree}
:caption: Tutorials
:hidden:
:maxdepth: 1

Overview <tutorials/index>
tutorials/02-setup
tutorials/03-your-first-agent
tutorials/04-verifying-results
tutorials/05-rollout-collection
tutorials/07-sft-dpo-rollout-collection
tutorials/09-configuration-guide
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

