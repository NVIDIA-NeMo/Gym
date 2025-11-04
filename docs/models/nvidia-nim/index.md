(models-nvidia-nim)=

# NVIDIA NIM - Enterprise Model Deployment

Enterprise-grade model serving with production reliability, NVIDIA optimization, and support for large-scale agent training in NeMo Gym.

---

## What is NVIDIA NIM?

NVIDIA NIM (NVIDIA Inference Microservices) delivers production-ready AI model inference with enterprise features. NeMo Gym's NIM integration enables reliable, scalable model serving for training workflows requiring enterprise-grade reliability and support.

::::{tab-set}

:::{tab-item} When to use
:selected:

- Production deployments requiring SLAs and support
- Enterprise environments with compliance requirements
- Teams wanting optimized performance without infrastructure expertise
- Organizations already using NVIDIA infrastructure

:::

:::{tab-item} Why NVIDIA NIM

- **Enterprise reliability**: Production SLAs, support, and uptime guarantees
- **Optimized performance**: NVIDIA-tuned inference engines for maximum GPU utilization
- **Enterprise security**: Compliance certifications, audit logging, and secure deployment
- **Scalable infrastructure**: Kubernetes-native deployment with auto-scaling
- **Simplified operations**: Managed microservices reduce operational overhead

:::

::::

:::{seealso}
**Not sure which model serving method to use?** See the [Models Overview](../index.md#choosing-a-model-serving-method) for a comparison.
:::

---

## Quick Example

Once configured, using NVIDIA NIM with NeMo Gym is straightforward:

```bash
# Start NeMo Gym with NVIDIA NIM configuration
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"

ng_run "+config_paths=[${config_paths}]"

# Collect rollouts
ng_collect_rollouts \
  +agent_name=simple_weather_simple_agent \
  +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
  +output_jsonl_fpath=results/nim_rollouts.jsonl \
  +limit=100 \
  +concurrency=50
```

:::{tip}
**NVIDIA NIM uses OpenAI-compatible endpoints**, so you configure it using the `openai_model` adapter with your NIM endpoint URL.
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

Get NVIDIA NIM connected to NeMo Gym in under 5 minutes.
+++
{bdg-secondary}`tutorial` {bdg-secondary}`setup`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Configuration Reference
:link: configuration
:link-type: doc

Complete guide to all NVIDIA NIM configuration options.
+++
{bdg-secondary}`reference` {bdg-secondary}`config`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc

Common issues and solutions for NVIDIA NIM integration.
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

