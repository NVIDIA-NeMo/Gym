(agent-server-index)=
# Agent Server

Agent servers orchestrate the rollout lifecycle—calling models, executing tool calls through resources servers, and coordinating verification. They implement `SimpleResponsesAPIAgent` and expose two endpoints:

- **`/v1/responses`** — Responses API passthrough for direct model interaction
- **`/run`** — Execute a complete rollout with tool calling and verification

---

## Rollout Lifecycle

The agent orchestrates a complete rollout through these steps:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           Agent Server (/run)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Receive task ──► 2. Initialize session ──► 3. Call model            │
│         │                    │                       │                  │
│         │                    ▼                       ▼                  │
│         │            Resources Server        Model Server               │
│         │            /seed_session           /v1/responses              │
│         │                                          │                    │
│         │                                          ▼                    │
│         │            4. Execute tool calls ◄── Tool calls?              │
│         │                    │                     │                    │
│         │                    ▼                     │ No                 │
│         │            Resources Server              │                    │
│         │            /{tool_name}                  ▼                    │
│         │                    │              5. Verify rollout           │
│         │                    │                     │                    │
│         │                    ▼                     ▼                    │
│         │               Loop back to        Resources Server            │
│         │               step 3 (until       /verify                     │
│         │               max_steps or              │                     │
│         │               no tool calls)            ▼                     │
│         │                                   Return reward               │
│         └─────────────────────────────────────────────────────────────► │
└─────────────────────────────────────────────────────────────────────────┘
```

### Lifecycle Steps

1. **Receive task** — Agent receives input via `/run` endpoint with `responses_create_params`
2. **Initialize session** — Call `/seed_session` on resources server to initialize state
3. **Call model** — Send prompt to model server via `/v1/responses`
4. **Execute tools** — Route model's tool calls to appropriate resources server endpoints
5. **Iterate** — Repeat steps 3-4 until model stops calling tools or `max_steps` reached
6. **Verify** — Call `/verify` on resources server to compute reward

---

## Built-in Agents

NeMo Gym includes the **Simple Agent** (`responses_api_agents/simple_agent/`) which handles:

- Multi-step tool calling loops
- Conversation history management
- Configurable step limits
- Automatic session management

---

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
      max_steps: 10
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `entrypoint` | `str` | Required | Python file containing the agent class |
| `resources_server` | `ref` | Required | Reference to resources server |
| `model_server` | `ref` | Required | Reference to model server |
| `max_steps` | `int` | `null` | Maximum tool-calling iterations (no limit if null) |

:::{note}
The agent automatically ends the rollout when the model outputs a message without tool calls. This behavior is built into `SimpleAgent` and is not configurable.
:::

---

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

---

## Custom Agent Implementation

To create a custom agent, extend `SimpleResponsesAPIAgent`:

```python
from fastapi import Body, Request, Response

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


class MyCustomAgent(SimpleResponsesAPIAgent):
    """Custom agent with specialized behavior."""

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Override for custom model interaction logic."""
        # Your custom implementation
        pass

    async def run(
        self,
        request: Request,
        body: BaseRunRequest,
    ) -> BaseVerifyResponse:
        """Override for custom rollout orchestration."""
        # Your custom implementation
        pass
```

See `responses_api_agents/simple_agent/app.py` for a complete implementation.

---

## API Endpoints

| Endpoint | Description | Returns |
|----------|-------------|---------|
| `/v1/responses` | Direct model passthrough | `NeMoGymResponse` |
| `/run` | Complete rollout with verification | `BaseVerifyResponse` |

## Source Code

The agent base classes are defined in `nemo_gym/base_responses_api_agent.py`. See `responses_api_agents/simple_agent/` for the reference implementation.