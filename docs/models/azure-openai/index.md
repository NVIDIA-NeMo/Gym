(models-azure-openai)=

# Azure OpenAI - Enterprise Azure Integration

Enterprise-grade OpenAI models through Microsoft Azure with compliance, regional deployment, and Azure ecosystem integration for NeMo Gym.

---

## What is Azure OpenAI?

Azure OpenAI Service provides OpenAI models through Microsoft Azure infrastructure, enabling enterprise deployments with Azure's security, compliance, and regional availability. NeMo Gym integrates using the `azure_openai_model` adapter, which handles Azure-specific authentication and endpoint formatting.

::::{tab-set}

:::{tab-item} When to use
:selected:

- Already using Azure infrastructure
- Need compliance certifications (HIPAA, SOC 2, etc.)
- Regional data residency requirements
- Microsoft enterprise support needed
- Azure ecosystem integration (Key Vault, Monitor, etc.)

:::

:::{tab-item} Why Azure OpenAI

- **Azure integration**: Native integration with Azure services and infrastructure
- **Enterprise compliance**: SOC 2, HIPAA, ISO 27001, and regional data residency
- **Microsoft support**: Enterprise SLAs and Microsoft support channels
- **Regional deployment**: Deploy in specific Azure regions for data residency
- **Familiar pricing**: Azure billing and cost management

:::

::::

:::{button-ref} models-comparison-table
:color: secondary
:outline:
:ref-type: ref

← Compare Options
:::

---

## Quick Example

Once configured, using Azure OpenAI with NeMo Gym is straightforward:

```bash
# Start NeMo Gym with Azure OpenAI configuration
config_paths="responses_api_models/azure_openai_model/configs/azure_openai_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml"

ng_run "+config_paths=[${config_paths}]" \
    +policy_model.responses_api_models.azure_openai_model.default_query.api-version=2024-10-21

# Collect rollouts
ng_collect_rollouts \
  +agent_name=simple_weather_simple_agent \
  +input_jsonl_fpath=resources_servers/simple_weather/data/example.jsonl \
  +output_jsonl_fpath=results/azure_rollouts.jsonl \
  +limit=100 \
  +concurrency=10
```

:::{seealso}
For a complete walkthrough of rollout collection, see [Collecting Rollouts](../../get-started/collecting-rollouts.md).
:::

---

## Topics

Connect to Azure OpenAI and configure it for enterprise deployments.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quick Start
:link: quick-start
:link-type: doc

Get Azure OpenAI connected to NeMo Gym in under 5 minutes.
+++
{bdg-secondary}`tutorial` {bdg-secondary}`setup`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Configuration Reference
:link: configuration
:link-type: doc

Complete guide to Azure OpenAI configuration and deployments.
+++
{bdg-secondary}`reference` {bdg-secondary}`config`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc

Common issues and solutions for Azure OpenAI integration.
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

