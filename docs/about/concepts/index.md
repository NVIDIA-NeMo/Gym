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
Understand how Models, Resources, and Agents remain decoupled yet coordinated as independent HTTP services, including which endpoints each component exposes.
:::

:::{grid-item-card} {octicon}`check-circle;1.5em;sd-mr-1` Task Verification
:link: task-verification
:link-type: ref
Explore how resource servers score agent outputs with `verify()` implementations that transform correctness, quality, and efficiency checks into reward signals.
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Key Terminology
:link: key-terminology
:link-type: ref
Essential vocabulary for agent training, RL workflows, and NeMo Gym. This glossary defines terms you'll encounter throughout the tutorials and documentation.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration System
:link: configuration-concepts
:link-type: ref
Understand the three-level config pattern and why server IDs and implementations are independent choices.
:::

::::

---

```{toctree}
:hidden:
:maxdepth: 1

Core Components <core-components>
Configuration System <configuration>
Task Verification <task-verification>
Key Terminology <key-terminology>
```
