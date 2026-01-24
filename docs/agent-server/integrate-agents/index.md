(agent-server-integrate-index)=
# Integrate Existing Agents

Integrate agents from external frameworks by implementing `SimpleResponsesAPIAgent`. Your agent gains access to NeMo Gym's RL training infrastructure: rollout orchestration, tool calling, and verification rewards.

:::{admonition} Prerequisites
:class: note

Understand the [NeMo Gym server architecture](/about/architecture):

- **Agent servers** — orchestrate rollouts (this guide)
- **Model servers** — LLM inference at `/v1/responses`
- **Resources servers** — environment state and verification
- **Head server** — configuration discovery
:::

---

## Why `SimpleResponsesAPIAgent`?

| Feature | What you get |
|---------|-------------|
| **Service discovery** | `ServerClient` resolves server names to host/port |
| **Session management** | Middleware assigns session IDs, propagates cookies |
| **Standard endpoints** | `/v1/responses` and `/run` pre-wired |
| **Config injection** | Your config class parsed and injected at startup |

For a minimal HTTP wrapper without these features, see the [Complete Example](../index.md#complete-example) section.

---

## Interface Overview

Subclass `SimpleResponsesAPIAgent` and implement two abstract methods:

| Method | Purpose | Returns |
|--------|---------|---------|
| `responses()` | Multi-step tool-calling loop | `NeMoGymResponse` |
| `run()` | Complete rollout: seed → responses → verify | `BaseVerifyResponse` with `reward` |

:::{dropdown} Base class definition
:icon: code

```python
from abc import abstractmethod
from fastapi import Body, FastAPI

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer


class BaseResponsesAPIAgentConfig(BaseRunServerInstanceConfig):
    pass


class BaseResponsesAPIAgent(BaseServer):
    config: BaseResponsesAPIAgentConfig


class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        return app

    @abstractmethod
    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        pass

    @abstractmethod
    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        pass
```

**Source**: `nemo_gym/base_responses_api_agent.py`
:::

---

## What You Implement

:::::{tab-set}

::::{tab-item} responses()
The tool-calling loop: call model, execute tools, repeat until done.

```python
async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
    outputs = []
    while True:
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body.model_copy(update={"input": body.input + outputs}),
        )
        # Parse response, execute tool calls, append to outputs
        # Break when model returns message without tool calls
    return model_response
```
::::

::::{tab-item} run()
Complete rollout: seed session → call responses → verify for reward.

```python
async def run(self, body: BaseRunRequest) -> BaseVerifyResponse:
    # 1. Initialize session
    await self.server_client.post(
        server_name=self.config.resources_server.name,
        url_path="/seed_session",
        json=body.model_dump(),
    )

    # 2. Execute tool loop
    response = await self.server_client.post(
        server_name=self.config.name,
        url_path="/v1/responses",
        json=body.responses_create_params,
    )

    # 3. Verify and return reward
    verify_response = await self.server_client.post(
        server_name=self.config.resources_server.name,
        url_path="/verify",
        json={"responses_create_params": body.responses_create_params, "response": response},
    )
    return BaseVerifyResponse.model_validate(await verify_response.json())
```
::::

:::::

---

## Complete Example

Create `my_agent/app.py`:

```python
from fastapi import Body, Request
from pydantic import ConfigDict

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status


class MyAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef


class MyAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class MyAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class MyAgent(SimpleResponsesAPIAgent):
    config: MyAgentConfig

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
        )
        await raise_for_status(model_response)
        return NeMoGymResponse.model_validate(await get_response_json(model_response))

    async def run(self, request: Request, body: MyAgentRunRequest) -> MyAgentVerifyResponse:
        cookies = request.cookies

        # 1. Seed session
        seed_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_response)
        cookies = seed_response.cookies

        # 2. Run responses
        response = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(response)
        cookies = response.cookies

        # 3. Verify
        verify_request = MyAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": await get_response_json(response)}
        )
        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        return MyAgentVerifyResponse.model_validate(await get_response_json(verify_response))


class MyAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


if __name__ == "__main__":
    MyAgent.run_webserver()
```

---

## Configuration

Create `my_agent/configs/my_agent.yaml`:

```yaml
my_agent:
  responses_api_agents:
    my_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_resources
      model_server:
        type: responses_api_models
        name: policy_model
```

| Field | Type | Description |
|-------|------|-------------|
| `entrypoint` | `str` | Python file containing agent class |
| `resources_server` | `ref` | Resources server for tools and verification |
| `model_server` | `ref` | Model server for LLM inference |

---

## Test Your Integration

```bash
# Start servers
nemo-gym run --config resources_servers/math_with_judge/configs/math_with_judge.yaml \
             --config my_agent/configs/my_agent.yaml

# Test /run endpoint
curl -X POST http://localhost:<agent_port>/run \
  -H "Content-Type: application/json" \
  -d '{"responses_create_params": {"input": "What is 2+2?"}}'
```

The response includes `reward` from the resources server's `/verify` endpoint.

---

## Error Handling

`ServerClient` retries requests 3 times with 0.5s delays. For custom retry logic, use [tenacity](https://tenacity.readthedocs.io/):

```python
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=5))
async def _seed_session(self, task_idx: int):
    response = await self.server_client.post(
        server_name=self.config.resources_server.name,
        url_path="/seed_session",
        json={"task_idx": task_idx},
    )
    response.raise_for_status()
    return response
```

---

## Available Integrations

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`workflow;1.5em;sd-mr-1` OpenAI Agents SDK
:link: openai-agents-sdk
:link-type: doc
Adapt OpenAI Agents SDK agents to NeMo Gym.
+++
{bdg-secondary}`openai` {bdg-secondary}`agents-sdk`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` NeMo Agent Toolkit
:link: nemo-agent-toolkit
:link-type: doc
Integrate NeMo Agent Toolkit agents.
+++
{bdg-secondary}`nemo` {bdg-secondary}`agent-toolkit`
:::

::::

---

## Next Steps

- [Agent Server overview](../index.md) — Full tool-calling loop implementation
- `responses_api_agents/simple_agent/app.py` — Production reference
- {doc}`/resources-server/index` — Verification and tool endpoints

```{toctree}
:hidden:
:maxdepth: 1

OpenAI Agents SDK <openai-agents-sdk>
NeMo Agent Toolkit <nemo-agent-toolkit>
```
