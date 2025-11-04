(models-overview)=

# Models

NeMo Gym supports multiple model serving methods, enabling you to choose the inference solution that best fits your deployment, performance, and cost requirements. Whether you need self-hosted open models, enterprise-grade deployment, or managed API services, NeMo Gym provides consistent interfaces across all options.

This section covers how to configure and use each model serving method with NeMo Gym for generating agent rollouts.

---

## Available Model Serving Methods

Choose the model serving approach that best matches your infrastructure and requirements:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` vLLM
:link: vllm/index
:link-type: doc

Self-hosted open models with high-throughput inference using vLLM.
+++
{bdg-secondary}`self-hosted` {bdg-secondary}`open-models` {bdg-secondary}`high-performance`
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` NVIDIA NIM
:link: nvidia-nim
:link-type: doc

Enterprise-grade model deployment with NVIDIA NIM microservices.
+++
{bdg-secondary}`enterprise` {bdg-secondary}`production` {bdg-secondary}`scalable`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` OpenAI Responses API
:link: openai
:link-type: doc

Use OpenAI models through the Responses API format.
+++
{bdg-secondary}`managed` {bdg-secondary}`cloud` {bdg-secondary}`easy-setup`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Azure OpenAI Responses API
:link: azure-openai
:link-type: doc

Access Azure OpenAI endpoints for LLM inference.
+++
{bdg-secondary}`azure` {bdg-secondary}`enterprise` {bdg-secondary}`compliance`
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` OpenRouter API
:link: openrouter
:link-type: doc

Unified access to multiple model providers through OpenRouter.
+++
{bdg-secondary}`multi-provider` {bdg-secondary}`flexible` {bdg-secondary}`unified-api`
:::

::::

---

## Choosing a Model Serving Method

Consider these factors when selecting your model serving approach:

```{list-table}
:header-rows: 1
:widths: 25 35 40

* - Method
  - Best For
  - Key Considerations
* - **vLLM**
  - High-throughput inference with open models, full control over deployment
  - Requires infrastructure setup and maintenance, best performance for batch workloads
* - **NVIDIA NIM**
  - Enterprise production deployments, compliance requirements, support needs
  - Enterprise features, production-grade reliability, NVIDIA support
* - **OpenAI**
  - Rapid prototyping, experimentation, minimal infrastructure
  - Pay-per-use pricing, no infrastructure management, latest OpenAI models
* - **Azure OpenAI**
  - Azure-native deployments, enterprise compliance, regional requirements
  - Azure integration, compliance features, regional availability
* - **OpenRouter**
  - Access to multiple model providers through single API, cost optimization
  - Unified interface, flexible provider selection, comparison testing
```

---

## Common Configuration Pattern

All model servers in NeMo Gym follow a consistent configuration pattern using YAML files and environment variables:

```yaml
policy_model:
  responses_api_models:
    <model_type>:
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
```

Environment variables are defined in `env.yaml`:

```yaml
policy_base_url: <your_endpoint_url>
policy_api_key: <your_api_key>
policy_model_name: <model_identifier>
```

Each model type has specific configuration options and usage patterns—refer to the individual guides for details.

---

## Next Steps

Select the model serving method that fits your needs and follow the configuration guide:

:::{button-ref} vllm/index
:color: primary
:outline:
:ref-type: doc

Start with vLLM for Self-Hosted Models →
:::

:::{tip}
**Not sure which to choose?** Start with OpenAI for quick experimentation, then move to vLLM or NVIDIA NIM for production workloads.
:::

