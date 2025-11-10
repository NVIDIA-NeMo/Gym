---
orphan: true
---

(about-concepts)=
# Understanding Concepts for {{product_name}}

NeMo Gym concepts explain the mental model behind building reliable agent systems: how services collaborate, how teams capture interaction data, and how verification signals drive learning. Use this page as a compass to decide which explanation to read next.

::::{tip}
Need a refresher on reinforcement learning language? Refer to the {doc}`../../resources/glossary` before diving in.
::::

---

## How to Navigate This Section

- Read these explanations when your team needs shared vocabulary for configuring Agents, Models, and Resources together.
- Pair each concept page with its related tutorials when you are ready to practice tasks such as assembling interaction datasets or scoring agent behavior.
- Return here whenever you add a new teammate so that they can orient and choose the depth that fits their role.

---

## Concept Highlights

Each explainer below covers one foundational idea and links to deeper material.

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Core Abstractions
:link: concepts-core-abstractions
:link-type: ref
Understand how Agents, Models, and Resources remain decoupled yet coordinated as independent HTTP services, including which endpoints each abstraction exposes.
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` Rollout Collection Fundamentals
:link: concepts-rc-fundamentals
:link-type: ref
Learn why complete interaction transcripts matter for reinforcement learning, how they enable evaluation, and how collection orchestrators stream results to JSONL datasets.
:::

:::{grid-item-card} {octicon}`check-circle;1.5em;sd-mr-1` Verifying Agent Results
:link: concepts-verifying-results
:link-type: ref
Explore how resource servers score agent outputs with `verify()` implementations that transform correctness, quality, and efficiency checks into reward signals.
:::

::::

---

## Continue Learning

- Reinforce the architecture concepts by stepping through {doc}`../../tutorials/03-your-first-agent` before exploring large-scale collection.
- Apply the verification patterns in practice with the {doc}`../../tutorials/04-verifying-results` tutorial, then move on to high-volume data generation in {doc}`../../tutorials/05-rollout-collection`.
- Catalog the services you plan to deploy by referencing {doc}`../features` once the core concepts are familiar.

---

```{toctree}
:hidden:
:maxdepth: 1

Core Abstractions <core-abstractions>
Rollout Collection Fundamentals <rollout-collection-fundamentals>
Verifying Agent Results <verifying-agent-results>
```
