(building-environments-index)=

# Building RL Environments in NeMo Gym

A comprehensive guide to understanding and building RL environments for LLM training using NVIDIA NeMo Gym. This tutorial walks through four progressively complex environments, from a simple single-tool server to a real-world workplace assistant.

:::{card}

**Goal**: Understand NeMo Gym's architecture and learn to build custom RL environments (Resource Servers) with tools, session state, and reward functions.

^^^

**In this guide, you will**:

1. Learn NeMo Gym's three-tier microservices architecture
2. Build a simple single-tool environment
3. Build a multi-step tool calling environment
4. Build a stateful environment with session management
5. Study a real-world environment (Workplace Assistant)
6. Configure YAML files and training data
7. Understand the server infrastructure and RL training loop

:::

> **TL;DR:** Want to create a resource server quickly? See the {doc}`Creating a Resource Server <../creating-resource-server>` quick-start tutorial. This guide provides the in-depth conceptual understanding.

---

## Before You Begin

Make sure you have these prerequisites:

- Basic Python and FastAPI knowledge
- Familiarity with REST APIs and HTTP servers
- Basic RL concepts (rewards, episodes, policies)
- NeMo Gym installed ({doc}`detailed setup <../get-started/detailed-setup>`)

---

## Guide Sections

Follow these sections sequentially to build your understanding:

::::{grid} 1
:gutter: 2

:::{grid-item-card} 1. Architecture and Core Concepts
:link: building-environments-architecture
:link-type: ref

NeMo Gym's three-tier architecture, the Resource Server lifecycle, and key base classes.
+++
{bdg-secondary}`concepts`
:::

:::{grid-item-card} 2. Single-Step Environment
:link: building-environments-single-step
:link-type: ref

Build a simple stateless environment with one tool endpoint and a basic reward function.
+++
{bdg-primary}`beginner` {bdg-secondary}`example`
:::

:::{grid-item-card} 3. Multi-Step Environment
:link: building-environments-multi-step
:link-type: ref

Build an environment requiring multiple sequential tool calls with ground-truth verification.
+++
{bdg-primary}`intermediate` {bdg-secondary}`example`
:::

:::{grid-item-card} 4. Stateful Environment
:link: building-environments-stateful
:link-type: ref

Add per-episode session state using `SESSION_ID_KEY` and the session middleware.
+++
{bdg-primary}`intermediate` {bdg-secondary}`example`
:::

:::{grid-item-card} 5. Real-World Environment (Workplace Assistant)
:link: building-environments-real-world
:link-type: ref

Study a production environment with dynamic tool routing and state-based verification.
+++
{bdg-primary}`advanced` {bdg-secondary}`example`
:::

:::{grid-item-card} 6. Configuration and Training Data
:link: building-environments-configuration
:link-type: ref

YAML configuration files, JSONL training data format, and tool definitions.
+++
{bdg-secondary}`reference`
:::

:::{grid-item-card} 7. Infrastructure, Training, and Next Steps
:link: building-environments-infrastructure
:link-type: ref

Server infrastructure, the RL training loop, directory structure, and a step-by-step checklist.
+++
{bdg-secondary}`reference`
:::

::::

---

## What's Next?

After completing this guide, explore these options:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Create a Resource Server
:link: ../creating-resource-server
:link-type: doc

Follow the quick-start tutorial to scaffold and run your own resource server.
+++
{bdg-secondary}`tutorial` {bdg-secondary}`quick-start`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: training-nemo-rl-grpo-index
:link-type: ref

Train a model on the Workplace Assistant environment using GRPO.
+++
{bdg-secondary}`tutorial` {bdg-secondary}`training`
:::

::::

```{toctree}
:caption: Building Environments
:hidden:
:maxdepth: 1

architecture-and-concepts.md
single-step-environment.md
multi-step-environment.md
stateful-environment.md
real-world-environment.md
configuration-and-data.md
infrastructure-and-training.md
```
