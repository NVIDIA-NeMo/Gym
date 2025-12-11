(integrate-reference)=

# Integration Reference

Technical reference material for custom training framework integration. These pages provide ecosystem context, implementation checklists, and validation benchmarks.

---

## Ecosystem Context

Understand the generation backend landscape before implementing.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Generation Backends
:link: generation-backends
:link-type: doc

OpenAI-compatible HTTP server implementations across RL frameworks (NeMo RL, VeRL, TRL, Slime, OpenPIPE ART).
+++
{bdg-secondary}`landscape`
{bdg-secondary}`vLLM`
{bdg-secondary}`SGLang`
:::

::::

---

## Implementation Reference

Component checklist and validation benchmarks for integration correctness.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Integration Footprint
:link: integration-footprint
:link-type: doc

5-component checklist with NeMo RL code pointers and test references.
+++
{bdg-secondary}`checklist`
{bdg-secondary}`NeMo RL`
:::

:::{grid-item-card} {octicon}`verified;1.5em;sd-mr-1` Success Criteria
:link: success-criteria
:link-type: doc

Validation benchmarks: DAPO17k (85% AIME24), Workplace Assistant.
+++
{bdg-secondary}`benchmarks`
{bdg-secondary}`validation`
:::

::::

---

## Related Concepts

- {doc}`/about/concepts/on-policy-token-alignment` — Why token alignment matters for multi-turn training
- {doc}`/about/concepts/training-integration-architecture` — Architecture deep-dive

---

```{toctree}
:maxdepth: 1
:hidden:

generation-backends
integration-footprint
success-criteria
```
