(model-server-index)=
# Model Server

Model servers provide stateless LLM inference via OpenAI-compatible endpoints. They implement `ResponsesAPIModel` and expose two endpoints:

- **`/v1/responses`** — [OpenAI Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create)
  - This is the default input/output schema for all NeMo Gym rollouts.
- **`/v1/chat/completions`** — [OpenAI Chat Completions API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)


## Backend Guides

Guides for OpenAI and Azure OpenAI Responses API models and more are coming soon!

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
