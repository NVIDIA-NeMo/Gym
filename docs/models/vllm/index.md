(models-vllm)=

# vLLM - Self-Hosted Open Models

vLLM enables high-throughput inference with self-hosted open-source models for large-scale agent training in NeMo Gym.

---

## What is vLLM?

vLLM is a fast and memory-efficient inference engine for large language models. NeMo Gym's vLLM adapter lets you deploy open-source models (Llama, Qwen, Mistral, etc.) on your own infrastructure with optimized throughput for generating training rollouts.

::::{tab-set}

:::{tab-item} When to use
:selected:

- Training with open-source models (Llama, Qwen, Mistral)
- Large-scale rollout collection requiring high throughput
- Cost-sensitive projects with predictable infrastructure costs
- Research requiring full model control and reproducibility

:::

:::{tab-item} Why vLLM

**High throughput**: Optimized batch processing generates thousands of rollouts concurrently  
**Full control**: Deploy any open-source model on your infrastructure  
**Cost-effective**: No per-token API costs for large-scale training workloads  
**Training-ready**: Built-in support for token IDs and log probabilities required by RL frameworks  
**Reasoning support**: Automatic parsing and extraction of reasoning tokens using `<think>` tags

:::

::::

:::{seealso}
**Not sure which model serving method to use?** See the [Models Overview](../index.md#choosing-a-model-serving-method) for a comparison.
:::

---

## Quick Example

Once configured, collecting rollouts with vLLM is straightforward:

```bash
# Start NeMo Gym with vLLM configuration
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"

ng_run "+config_paths=[${config_paths}]"

# Collect rollouts
ng_collect_rollouts \
  +agent_name=simple_weather_simple_agent \
  +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
  +output_jsonl_fpath=results/vllm_rollouts.jsonl \
  +limit=100 \
  +concurrency=50
```

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

Get vLLM running with NeMo Gym in under 5 minutes.
+++
{bdg-secondary}`tutorial` {bdg-secondary}`setup`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Configuration Reference
:link: configuration
:link-type: doc

Complete guide to all vLLM adapter configuration options.
+++
{bdg-secondary}`reference` {bdg-secondary}`config`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Optimization
:link: optimization
:link-type: doc

Load balancing, performance tuning, and production deployment.
+++
{bdg-secondary}`how-to` {bdg-secondary}`production`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc

Common issues and solutions for vLLM integration.
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
optimization
troubleshooting
```
