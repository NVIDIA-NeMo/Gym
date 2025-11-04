(models-openai)=

# OpenAI - Managed API Service

Rapid prototyping and experimentation with OpenAI models through managed API service for agent training in NeMo Gym.

---

## What is OpenAI?

OpenAI provides state-of-the-art language models through a managed API service. NeMo Gym integrates with OpenAI's endpoints using the Responses API format, enabling quick experimentation without infrastructure management.

::::{tab-set}

:::{tab-item} When to use
:selected:

- Quick prototyping and proof-of-concept work
- Experimentation with different agent approaches
- Small to medium-scale rollout collection
- Development and testing before production deployment

:::

:::{tab-item} Why OpenAI

- **Rapid prototyping**: Get started in minutes without infrastructure setup
- **Latest models**: Access to GPT-4, GPT-4 Turbo, and other cutting-edge models
- **Managed service**: High availability with no server maintenance
- **Pay-per-use**: Cost-effective for experimentation and small-scale projects
- **Easy migration**: Simple switch to self-hosted options when ready to scale

:::

::::

:::{seealso}
**Not sure which model serving method to use?** See the [Models Overview](../index.md#choosing-a-model-serving-method) for a comparison.
:::

---

## Quick Example

Once configured, using OpenAI with NeMo Gym is straightforward:

```bash
# Start NeMo Gym with OpenAI configuration
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"

ng_run "+config_paths=[${config_paths}]"

# Collect rollouts
ng_collect_rollouts \
  +agent_name=simple_weather_simple_agent \
  +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
  +output_jsonl_fpath=results/openai_rollouts.jsonl \
  +limit=100 \
  +concurrency=10
```

:::{tip}
**Start small**: OpenAI charges per token, so begin with lower concurrency (`+concurrency=10`) and small rollout counts while testing.
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

Get OpenAI connected to NeMo Gym in under 3 minutes.
+++
{bdg-secondary}`tutorial` {bdg-secondary}`setup`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Configuration Reference
:link: configuration
:link-type: doc

Complete guide to all OpenAI configuration options and models.
+++
{bdg-secondary}`reference` {bdg-secondary}`config`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc

Common issues and solutions for OpenAI integration.
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

