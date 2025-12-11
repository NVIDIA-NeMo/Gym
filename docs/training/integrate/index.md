(training-integrate-overview)=

# Integrate External Training Frameworks

Connect NeMo Gym to external training frameworks using OpenAI-compatible HTTP endpoints. These guides are for partners integrating existing frameworks (such as VeRL or TRL) with NeMo Gym as an alternative to NeMo RL.

:::{tip}
**Using NeMo RL?** Skip these guides — NeMo RL handles Gym integration automatically. Refer to {doc}`/tutorials/integrate-training-frameworks/train-with-nemo-rl` instead.
:::

---

## How-To Guides

Work through these sequentially when integrating a new framework, or jump to the guide you need.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Expose an OpenAI-Compatible Endpoint
:link: expose-openai-endpoint
:link-type: doc
Configure your generation backend to serve an HTTP endpoint that Gym can connect to.
+++
{bdg-secondary}`vllm`
{bdg-secondary}`http`
{bdg-secondary}`20 min`
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Connect Gym to Your Training Loop
:link: connect-gym-to-training
:link-type: doc
Integrate Gym's rollout collection into your framework's training loop.
+++
{bdg-secondary}`RunHelper`
{bdg-secondary}`rollouts`
{bdg-secondary}`30 min`
:::

:::{grid-item-card} {octicon}`check-circle;1.5em;sd-mr-1` Validate Your Integration
:link: validate-integration
:link-type: doc
Verify your integration works correctly end-to-end.
+++
{bdg-secondary}`testing`
{bdg-secondary}`validation`
{bdg-secondary}`15 min`
:::

::::

## Reference

Technical reference for framework integration implementers.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Generation Backends
:link: reference/generation-backends
:link-type: doc
OpenAI-compatible HTTP servers across training frameworks.
+++
{bdg-secondary}`landscape`
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Integration Footprint
:link: reference/integration-footprint
:link-type: doc
Component checklist with NeMo RL code pointers.
+++
{bdg-secondary}`checklist`
:::

:::{grid-item-card} {octicon}`verified;1.5em;sd-mr-1` Success Criteria
:link: reference/success-criteria
:link-type: doc
Validation benchmarks for integration correctness.
+++
{bdg-secondary}`validation`
:::

::::

## Related

- {doc}`/about/concepts/on-policy-token-alignment` — Why token alignment matters (re-tokenization, off-policyness)
- {doc}`/about/concepts/training-integration-architecture` — Architecture deep-dive
- {doc}`/training/rollout-collection/process-multi-turn-rollouts` — Token alignment how-to

---

```{toctree}
:hidden:
:maxdepth: 1

expose-openai-endpoint
connect-gym-to-training
validate-integration
reference/index
```
