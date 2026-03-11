(env-real-world-environment)=

# Real-World Environment (Workplace Assistant)

The Workplace Assistant environment simulates an office with email, calendar, analytics, project management, and CRM toolkits. It uses dynamic routing, per-session state, and state-based verification to grade outcomes.

:::{button-ref} stateful-environment
:color: secondary
:outline:
:ref-type: doc

< Previous: Stateful Environment
:::

---

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Generating Training Data
:link: real-world-data-generation
:link-type: doc
Generate synthetic multi-step tool-calling data using NeMo Data Designer.
+++
{bdg-primary}`start here`
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Resources Server Implementation
:link: real-world-implementation
:link-type: doc
Dynamic routing, per-session state, state-based verification, and rollout transcripts.
+++
{bdg-primary}`advanced`
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

Generating Training Data <real-world-data-generation>
Resources Server Implementation <real-world-implementation>
```

---

:::{button-ref} index
:color: secondary
:outline:
:ref-type: doc

< Back to Building Environments
:::
