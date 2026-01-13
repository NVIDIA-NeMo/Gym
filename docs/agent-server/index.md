(agent-server-index)=
# Agent Server

Agent servers orchestrate the rollout lifecycle—calling models, executing tool calls through resources servers, and coordinating verification. They implement `SimpleResponsesAPIAgent` and expose two endpoints:

- **`/v1/responses`** — Responses API passthrough for direct model interaction
- **`/run`** — Execute a complete rollout with tool calling and verification

## How Agents Work

1. **Receive task** — Agent receives input via `/run` endpoint
2. **Call model** — Agent sends prompt to model server
3. **Execute tools** — Model's tool calls are routed to resources server
4. **Iterate** — Process continues until model stops or max steps reached
5. **Verify** — Resources server evaluates the rollout and returns reward

## Built-in Agents

NeMo Gym includes the **Simple Agent** (`responses_api_agents/simple_agent/`) which handles:
- Multi-step tool calling loops
- Conversation history management
- Configurable step limits

## Configuration

Agent servers reference both a model server and resources server:

```yaml
my_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_resources
      model_server:
        type: responses_api_models
        name: policy_model
      max_steps: 10  # Maximum tool-calling iterations
```

## Integrating External Agents

You can integrate agents from external frameworks:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` OpenAI Agents SDK
:link: integrate-agents/openai-agents-sdk
:link-type: doc
Integrate agents built with OpenAI's Agents SDK.
+++
{bdg-secondary}`openai` {bdg-secondary}`agents-sdk`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` NeMo Agent Toolkit
:link: integrate-agents/nemo-agent-toolkit
:link-type: doc
Use agents from NeMo Agent Toolkit.
+++
{bdg-secondary}`nemo` {bdg-secondary}`agent-toolkit`
:::

::::

## Custom Agent Implementation

To create a custom agent, extend `SimpleResponsesAPIAgent`:

```python
class MyAgent(SimpleResponsesAPIAgent):
    async def responses(self, body) -> NeMoGymResponse:
        # Custom response handling
        pass

    async def run(self, body) -> BaseVerifyResponse:
        # Custom rollout orchestration
        pass
```