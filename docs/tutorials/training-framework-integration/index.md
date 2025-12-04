# Training framework integration

Implement NeMo Gym integration into a new training framework. This is only for expert users that cannot use existing training framework integrations with Gym and need to implement their own.

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Generation backend and OpenAI-compatible HTTP server
:link: generation-backend-and-openai-compatible-http-server
:link-type: doc
OpenAI compatible HTTP server pre-requisites for Gym integration into an RL framework.
+++
{bdg-secondary}`training` {bdg-secondary}`infra`
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` OpenAI-compatible HTTP server On-Policy correction
:link: openai-compatible-http-server-on-policy-correction
:link-type: doc
Helpful On-Policy fixes for OpenAI-compatible HTTP server implementations in multi step and multi turn scenarios.
+++
{bdg-secondary}`training` {bdg-secondary}`infra`
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Gym integration footprint and form factor
:link: gym-integration-footprint-and-form-factor
:link-type: doc
Implementation details like footprint and form factor for the actual Gym + RL integration.
+++
{bdg-secondary}`training` {bdg-secondary}`infra`
:::

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` Gym + RL framework integration success criteria
:link: gym-rl-framework-integration-success-criteria
:link-type: doc
What success criteria can be used to validate Gym + RL framework integration correctness.
+++
{bdg-secondary}`training` {bdg-secondary}`infra`
:::

::::

```{toctree}
:caption: Training framework integration
:hidden:
:maxdepth: 1

generation-backend-and-openai-compatible-http-server
openai-compatible-http-server-on-policy-correction
gym-integration-footprint-and-form-factor
gym-rl-framework-integration-success-criteria
```
