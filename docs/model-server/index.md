(model-server-index)=
# Model Server

Model servers provide stateless LLM inference via OpenAI-compatible endpoints. They implement `SimpleResponsesAPIModel` and expose two endpoints:

- **`/v1/chat/completions`** — Standard Chat Completions API
- **`/v1/responses`** — Responses API with tool calling support

## Choosing a Backend

| Backend | Use Case | Function Calling | Latency |
|---------|----------|------------------|---------|
| [vLLM](vllm) | Self-hosted models, custom fine-tunes | ✅ Via chat template | Low |

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

