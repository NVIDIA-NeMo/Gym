(models-openrouter)=

# OpenRouter - Multi-Provider API Gateway

Unified access to multiple LLM providers through a single API for flexible model selection and cost optimization in NeMo Gym.

---

## What is OpenRouter?

OpenRouter provides a unified API gateway to access models from multiple providers including OpenAI, Anthropic, Google, and others. NeMo Gym integrates with OpenRouter to enable flexible model selection and cost optimization across providers.

::::{tab-set}

:::{tab-item} When to use
:selected:

- Want to experiment with models from different providers
- Need cost optimization across multiple providers
- Require provider fallback for reliability
- Prefer unified billing for multiple services
- Evaluating which model/provider works best for your use case

:::

:::{tab-item} Why OpenRouter

- **Unified API**: Single integration for multiple model providers
- **Cost optimization**: Compare pricing and switch between providers
- **Provider redundancy**: Automatic fallback if a provider is down
- **Simplified billing**: One bill for all providers
- **Easy experimentation**: Test different models without changing code

:::

::::

:::{seealso}
**Not sure which model serving method to use?** See the [Models Overview](../index.md#choosing-a-model-serving-method) for a comparison.
:::

---

## Quick Example

Once configured, using OpenRouter with NeMo Gym is straightforward:

```bash
# Start NeMo Gym with OpenRouter configuration
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"

ng_run "+config_paths=[${config_paths}]"

# Collect rollouts
ng_collect_rollouts \
  +agent_name=simple_weather_simple_agent \
  +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
  +output_jsonl_fpath=results/openrouter_rollouts.jsonl \
  +limit=100 \
  +concurrency=10
```

:::{tip}
**OpenRouter uses OpenAI-compatible endpoints**, so you configure it using the `openai_model` adapter with the OpenRouter URL.
:::

:::{seealso}
For a complete walkthrough of rollout collection, see [Collecting Rollouts](../../get-started/collecting-rollouts.md).
:::

---

## Topics

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quick Start
:link: quick-start
:link-type: doc

Get OpenRouter connected to NeMo Gym in under 3 minutes.
+++
{bdg-secondary}`tutorial` {bdg-secondary}`setup`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Configuration Reference
:link: configuration
:link-type: doc

Complete guide to OpenRouter configuration and available models.
+++
{bdg-secondary}`reference` {bdg-secondary}`config`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc

Common issues and solutions for OpenRouter integration.
+++
{bdg-secondary}`how-to` {bdg-secondary}`debugging`
:::

::::

```{toctree}
:caption: Models
:hidden:
:maxdepth: 2

quick-start
configuration
troubleshooting
```

