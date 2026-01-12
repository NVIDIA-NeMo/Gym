---
orphan: true
---

(about-concepts)=
# Understanding Concepts for {{product_name}}

NeMo Gym concepts explain the mental model behind building reliable agent systems: how services collaborate, how teams capture interaction data, and how verification signals drive learning. Use this page as a compass to decide which explanation to read next.

::::{tip}
Need a refresher on reinforcement learning language? Refer to the {doc}`key-terminology` before diving in.
::::

---

## Concept Highlights

Each explainer below covers one foundational idea and links to deeper material.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Core Components
:link: core-components
:link-type: ref
Understand the three server components that make up a training environment.
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Architecture
:link: architecture
:link-type: ref
How components interact during startup and rollout collection.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration System
:link: configuration-concepts
:link-type: ref
Understand how servers are configured and connected.
:::

:::{grid-item-card} {octicon}`check-circle;1.5em;sd-mr-1` Task Verification
:link: task-verification
:link-type: ref
Understand the importance of verification and common implementation patterns.
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Key Terminology
:link: key-terminology
:link-type: ref
Essential vocabulary for agent training, RL workflows, and NeMo Gym.
:::

::::

---

```{toctree}
:hidden:
:maxdepth: 1

Core Components <core-components>
Architecture <architecture>
Configuration System <configuration>
Task Verification <task-verification>
Key Terminology <key-terminology>
```
