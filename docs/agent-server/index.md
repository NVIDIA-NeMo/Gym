(agent-server-index)=
# Agent Server

Agent servers orchestrate the **rollout** lifecycle—a rollout is one complete episode of agent-environment interaction where the model generates actions, tools execute them, and a reward is computed.

Agent servers coordinate three components:

- **[Model Server](../model-server/index)** — Generates model responses
- **[Resources Server](../resources-server/index)** — Executes tools and computes rewards

They expose two HTTP endpoints:

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/responses` | Execute a multi-step tool-calling loop |
| `POST /run` | Complete rollout: seed → tool loop → verification |

---

## Prerequisites

Before using agent servers:

1. **Install NeMo Gym** — See [Installation](../get-started/detailed-setup)
2. **Set up a model server** — See [Model Server](../model-server/index)
3. **Set up a resources server** — See [Resources Server](../resources-server/index)

---

## Architecture

Agent servers extend `SimpleResponsesAPIAgent`, which registers endpoints via FastAPI:

```python
class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, SimpleServer):
    config: BaseResponsesAPIAgentConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        return app
```

Subclasses implement two abstract methods:

- `responses()` — Handle `/v1/responses` endpoint
- `run()` — Handle `/run` endpoint

---

## Rollout Lifecycle

The `/run` endpoint executes a complete rollout through three server interactions:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           Agent Server (/run)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. /seed_session ──► 2. /v1/responses ──► 3. /verify                   │
│         │                    │                   │                      │
│         ▼                    ▼                   ▼                      │
│  Resources Server     Agent Server        Resources Server              │
│  (init state)         (tool loop)         (compute reward)              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Seed Session

The agent initializes environment state by calling `/seed_session` on the resources server:

```python
seed_session_response = await self.server_client.post(
    server_name=self.config.resources_server.name,
    url_path="/seed_session",
    json=body.model_dump(),
    cookies=cookies,
)
await raise_for_status(seed_session_response)
cookies = seed_session_response.cookies
```

### Step 2: Tool-Calling Loop

The `/v1/responses` method iterates until the model stops calling tools:

```python
while True:
    step += 1
    
    # 1. Call model server
    model_response = await self.server_client.post(
        server_name=self.config.model_server.name,
        url_path="/v1/responses",
        json=new_body,
        cookies=model_server_cookies,
    )
    await raise_for_status(model_response)
    
    # 2. Parse model output
    output = model_response.output
    all_fn_calls = [o for o in output if o.type == "function_call"]
    all_output_messages = [o for o in output if o.type == "message"]
    
    # 3. Check termination conditions
    if not all_fn_calls and all_output_messages:
        break  # Model finished (no more tool calls)
    
    # 4. Execute each tool call on resources server
    for fn_call in all_fn_calls:
        tool_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path=f"/{fn_call.name}",
            json=json.loads(fn_call.arguments),
        )
        # Tool errors are valid returns (e.g., invalid arguments)
    
    # 5. Check step limit
    if self.config.max_steps and step >= self.config.max_steps:
        break
```

### Termination Conditions

| Condition | Behavior |
|-----------|----------|
| Model returns message without tool calls | Loop exits normally |
| `max_steps` limit reached | Loop exits with partial rollout |
| Model returns `incomplete_details.reason == "max_output_tokens"` | Loop exits early |
| Model server returns error | `raise_for_status` raises exception |

### Step 3: Verification

After the tool loop completes, the agent calls `/verify` to compute the reward:

```python
verify_request = SimpleAgentVerifyRequest.model_validate(
    body.model_dump() | {"response": await get_response_json(response)}
)
verify_response = await self.server_client.post(
    server_name=self.config.resources_server.name,
    url_path="/verify",
    json=verify_request.model_dump(),
    cookies=cookies,
)
await raise_for_status(verify_response)
return SimpleAgentVerifyResponse.model_validate(await get_response_json(verify_response))
```

The reward is a float computed by the resources server based on task completion.

---

## Choosing an Agent

NeMo Gym includes three agent implementations:

| Agent | Use Case | Key Feature |
|-------|----------|-------------|
| **Simple Agent** | General-purpose tool calling | Direct tool routing via `/{tool_name}` |
| **Aviary Agent** | Environment-based tasks | State management with `/step`, `/close` |
| **Mini SWE Agent** | SWE-bench evaluation | Ray-distributed, container-based |

### Simple Agent

**Location**: `responses_api_agents/simple_agent/`

Best for standard tool-calling tasks where each tool is an HTTP endpoint.

```python
class SimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None  # No limit if None
```

### Aviary Agent

**Location**: `responses_api_agents/aviary_agent/`

Best for tasks with explicit environment state (games, simulations).

```python
class AviaryAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int | None = None
    max_total_sequence_length: int | None = None  # Token limit
    done_if_no_tool_calls: bool = True
    collapse_old_env_states: bool = False  # Compress history
```

The Aviary agent uses `/step` and `/close` endpoints instead of direct tool routing:

- `/step` — Execute action, return observation
- `/close` — Clean up environment resources

### Mini SWE Agent

**Location**: `responses_api_agents/mini_swe_agent/`

Best for SWE-bench code editing evaluation at scale.

```python
class MiniSWEAgentConfig(BaseResponsesAPIAgentConfig):
    model_server: ModelServerRef
    resources_server: ResourcesServerRef
    env: Literal["docker", "singularity"]  # Container runtime
    concurrency: int  # Parallel rollouts
    step_timeout: int = 600   # 10 min per step
    eval_timeout: int = 1800  # 30 min per evaluation
    step_limit: int = 250     # Max steps
```

:::{note}
Mini SWE Agent does not implement `/v1/responses` (raises `NotImplementedError`). Use `/run` only.
:::

---

## Configuration

Agent servers are configured in YAML under `responses_api_agents`:

```yaml
# configs/my_agent.yaml
my_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      max_steps: 10
```

### Configuration Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `entrypoint` | `str` | Yes | Python file containing the agent class |
| `resources_server` | `ResourcesServerRef` | Yes | Reference to resources server |
| `model_server` | `ModelServerRef` | Yes | Reference to model server |
| `max_steps` | `int` | No | Maximum iterations (unlimited if omitted) |

### Server References

Agent servers reference other servers using typed refs:

```python
class ModelServerRef(BaseModel):
    type: Literal["responses_api_models"]
    name: str

class ResourcesServerRef(BaseModel):
    type: Literal["resources_servers"]
    name: str
```

---

## Complete Example

A minimal working agent that delegates to the built-in Simple Agent logic:

```python
# my_echo_agent.py
"""Minimal agent that echoes tool responses."""
import json
from typing import List

from fastapi import Body, Request, Response

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import get_response_json, raise_for_status


class EchoAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = 5


class EchoAgent(SimpleResponsesAPIAgent):
    config: EchoAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        """Execute tool-calling loop until model stops or max_steps reached."""
        # Normalize string input to message list
        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        outputs: List = []
        step = 0

        while step < (self.config.max_steps or 100):
            step += 1

            # Call model server
            model_resp = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path="/v1/responses",
                json=body.model_copy(update={"input": body.input + outputs}),
            )
            await raise_for_status(model_resp)
            model_data = NeMoGymResponse.model_validate(await get_response_json(model_resp))

            # Collect outputs
            outputs.extend(model_data.output)

            # Check for function calls
            fn_calls: List[NeMoGymResponseFunctionToolCall] = [
                o for o in model_data.output if o.type == "function_call"
            ]
            messages: List[NeMoGymResponseOutputMessage] = [
                o for o in model_data.output if o.type == "message"
            ]

            # Exit if no function calls
            if not fn_calls and messages:
                break

            # Execute each tool call
            for fn_call in fn_calls:
                tool_resp = await self.server_client.post(
                    server_name=self.config.resources_server.name,
                    url_path=f"/{fn_call.name}",
                    json=json.loads(fn_call.arguments),
                )
                tool_output = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=fn_call.call_id,
                    output=(await tool_resp.content.read()).decode(),
                )
                outputs.append(tool_output)

        model_data.output = outputs
        return model_data

    async def run(self, request: Request, body: BaseRunRequest) -> BaseVerifyResponse:
        """Execute complete rollout: seed -> responses -> verify."""
        cookies = request.cookies

        # 1. Seed session
        seed_resp = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_resp)
        cookies = seed_resp.cookies

        # 2. Run tool-calling loop
        resp = await self.server_client.post(
            server_name=self.config.name,
            url_path="/v1/responses",
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(resp)
        cookies = resp.cookies

        # 3. Verify and compute reward
        verify_resp = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=body.model_dump() | {"response": await get_response_json(resp)},
            cookies=cookies,
        )
        await raise_for_status(verify_resp)

        return BaseVerifyResponse.model_validate(await get_response_json(verify_resp))


if __name__ == "__main__":
    EchoAgent.run_webserver()
```

Run the agent:

```bash
python my_echo_agent.py --config configs/my_agent.yaml
```

---

## API Reference

### Request Types

**BaseRunRequest** — Input to `/run` endpoint

```python
class BaseRunRequest(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
```

### Response Types

**BaseVerifyResponse** — Output from `/run` endpoint

```python
class BaseVerifyResponse(BaseVerifyRequest):
    reward: float  # Task completion score
```

### Example Response

```text
{
  "responses_create_params": {"input": [...]},
  "response": {"output": [...], "model": "..."},
  "reward": 1.0
}
```

---

## Error Handling

| Error | Cause | Behavior |
|-------|-------|----------|
| Model server 500 | Model inference failed | `raise_for_status` raises, rollout fails |
| Resources server 500 | Tool execution failed | Tool returns error string, loop continues |
| Network timeout | Server unreachable | `aiohttp` raises `ClientError` |
| Invalid tool arguments | Model output malformed | `json.loads` raises `JSONDecodeError` |

The Simple Agent treats tool errors as valid returns—the error message is passed back to the model as `function_call_output`. Only model server errors abort the rollout.

---

## Integrating External Agents

Integrate agents from external frameworks by implementing the `SimpleResponsesAPIAgent` interface:

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

