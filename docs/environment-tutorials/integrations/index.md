(environment-integrations-index)=

# Environment Integrations

Run environments from external RL frameworks in NeMo Gym.

---

## Overview

NeMo Gym integrates with popular environment libraries, allowing you to use their pre-built tasks for training with NeMo RL. Each integration wraps the external framework's environments to expose them via the NeMo Gym API.

| Integration | Environments | Focus Areas |
|-------------|--------------|-------------|
| **Reasoning Gym** | 100+ tasks | Algebra, arithmetic, logic, graph theory, games |
| **Aviary** | Multi-step agents | Scientific tasks, biological sequences, literature search |
| **Verifiers** | 600+ tasks | Math, reasoning, agent environments |

---

## Integrations

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Reasoning Gym
:link: reasoning-gym
:link-type: doc

100+ procedurally generated reasoning tasks across multiple domains.

+++
{bdg-secondary}`integration` {bdg-secondary}`15-20 min`
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Aviary
:link: aviary
:link-type: doc

Custom language agent environments for scientific and reasoning tasks.

+++
{bdg-secondary}`integration` {bdg-secondary}`10-15 min`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Verifiers
:link: verifiers
:link-type: doc

600+ environments from Prime Intellect's Environments Hub.

+++
{bdg-secondary}`integration` {bdg-secondary}`20 min`
:::

::::

---

## Common Pattern

All integrations follow the same workflow:

```{mermaid}
flowchart LR
    A[Install Dependencies] --> B[Configure Model Server]
    B --> C[Launch NeMo Gym Servers]
    C --> D[Collect Rollouts]
    D --> E[Train with NeMo RL]
```

1. **Install** — Add the external library and any environment-specific packages
2. **Configure** — Set up `env.yaml` with model server connection details
3. **Launch** — Start NeMo Gym with the integration's config
4. **Collect** — Run `ng_collect_rollouts` to generate training data
5. **Train** — Use collected rollouts with NeMo RL or another framework

---

## Next Steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build Custom Environments
:link: /environment-tutorials/creating-training-environment
:link-type: doc
Create your own training environment from scratch.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: training-nemo-rl-grpo-index
:link-type: ref
Start training models on your environment.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

reasoning-gym
aviary
verifiers
```
