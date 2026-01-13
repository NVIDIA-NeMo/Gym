(agent-server-integrate-index)=
# Integrate Existing Agents

```{note}
This section is being developed.
```

Learn how to integrate agents from external frameworks into NeMo Gym, enabling you to use existing agent implementations for RL training.

## Why Integrate External Agents?

- **Leverage existing implementations** — Use battle-tested agent code
- **Specialized capabilities** — Some frameworks excel at specific agent patterns
- **Team familiarity** — Keep using frameworks your team knows

## Integration Requirements

To integrate an external agent with NeMo Gym, it must:

1. **Accept Responses API input** — Process `NeMoGymResponseCreateParamsNonStreaming`
2. **Call tools via HTTP** — Route tool calls to resources server endpoints
3. **Return verification results** — Produce `BaseVerifyResponse` with reward

## Available Integrations

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` OpenAI Agents SDK
:link: openai-agents-sdk
:link-type: doc
Integrate OpenAI's Agents SDK for structured agent workflows.
+++
{bdg-secondary}`openai` {bdg-secondary}`agents-sdk`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` NeMo Agent Toolkit
:link: nemo-agent-toolkit
:link-type: doc
Use NeMo Agent Toolkit's enterprise agents.
+++
{bdg-secondary}`nemo` {bdg-secondary}`agent-toolkit`
:::

::::

## Integration Pattern

The general pattern for integrating external agents:

```python
class ExternalAgentWrapper(SimpleResponsesAPIAgent):
    def model_post_init(self, context):
        # Initialize external agent
        self.external_agent = ExternalAgent(...)

    async def run(self, body: BaseRunRequest) -> BaseVerifyResponse:
        # Convert NeMo Gym format to external agent format
        external_input = convert_input(body)

        # Run external agent
        result = await self.external_agent.run(external_input)

        # Verify via resources server
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json={"responses_create_params": body.responses_create_params, "response": result}
        )

        return BaseVerifyResponse.model_validate(await verify_response.json())
```

```{toctree}
:hidden:
:maxdepth: 1

openai-agents-sdk
nemo-agent-toolkit
```
