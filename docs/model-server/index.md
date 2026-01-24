(model-server-index)=
# Model Server

Model servers provide stateless LLM inference via OpenAI-compatible endpoints. They implement `SimpleResponsesAPIModel` and expose two endpoints:

- **`/v1/chat/completions`** — Standard Chat Completions API
- **`/v1/responses`** — Responses API with tool calling support

## Choosing a Backend

| Backend | Use Case | Function Calling | Latency |
|---------|----------|------------------|---------|
| [vLLM](vllm) | Self-hosted models, custom fine-tunes | ✅ Via chat template | Low |
| [OpenAI](openai) | Quick prototyping, GPT models | ✅ Native | Medium |
| [Azure OpenAI](azure-openai) | Enterprise deployments | ✅ Native | Medium |
| [Responses-Native](responses-native) | Models with native Responses API | ✅ Native | Low |

## Backend Guides

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` vLLM
:link: vllm
:link-type: doc
Self-hosted inference with vLLM for maximum control.
+++
{bdg-secondary}`self-hosted` {bdg-secondary}`open-source`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` OpenAI
:link: openai
:link-type: doc
Connect to OpenAI's API for GPT models.
+++
{bdg-secondary}`cloud` {bdg-secondary}`api`
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Azure OpenAI
:link: azure-openai
:link-type: doc
Enterprise deployments with Azure.
+++
{bdg-secondary}`azure` {bdg-secondary}`enterprise`
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` Responses-Native Models
:link: responses-native
:link-type: doc
Models with native Responses API support.
+++
{bdg-secondary}`responses-api` {bdg-secondary}`native`
:::

::::

## Configuration Example

Model servers are configured in YAML:

```yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_base_url: ${policy_base_url}
      openai_api_key: ${policy_api_key}
      openai_model: ${policy_model_name}
```

See {doc}`/reference/configuration` for complete configuration reference.

