(resources-server-index)=
# Resources Server

Resources servers define RL training environments—tools, tasks, and verification logic. During training, the agent server orchestrates **rollouts** (sequences of model interactions with tools) and calls your resources server to evaluate performance.

**Key responsibilities**:
- **Tools**: FastAPI endpoints the model calls during rollouts (e.g., `/get_weather`)
- **Verification**: Evaluate model output after rollout completes, return reward (0.0–1.0)
- **Session state**: Optional per-rollout initialization (e.g., reset game state)

:::{tip}
A **rollout** is one complete interaction sequence: the model receives a prompt, makes tool calls, and produces output. Your resources server provides the tools and judges the result.
:::

---

## Request Flow

```text
Agent Server           Resources Server           Model Server
     │                        │                         │
     │──── /seed_session ────▶│                         │
     │◀─── session ready ─────│                         │
     │                        │                         │
     │──────────────── /v1/responses ─────────────────▶│
     │                        │◀── /get_weather ────────│
     │                        │─── weather data ───────▶│
     │◀─────────────── response ──────────────────────│
     │                        │                         │
     │──── /verify ──────────▶│                         │
     │◀─── reward: 1.0 ───────│                         │
```

---

## Core Endpoints

Every resources server exposes two required endpoints:

| Endpoint | Purpose | When Called |
|----------|---------|-------------|
| `/seed_session` | Initialize session state | Before each rollout |
| `/verify` | Evaluate rollout, compute reward | After rollout completes |

Additional tool endpoints (e.g., `/get_weather`) are defined by your implementation.

---

## Quick Start

Create a resources server in 3 steps:

**1. Initialize** from repository root:

```bash
ng_init_resources_server +entrypoint=resources_servers/my_tool
```

**2. Implement** `resources_servers/my_tool/app.py` (see example below)

**3. Run** with a model:

```bash
ng_run "+config_paths=[resources_servers/my_tool/configs/my_tool.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

---

## Minimal Working Example

A complete, runnable resources server (`resources_servers/my_tool/app.py`):

```python
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class GetWeatherRequest(BaseModel):
    city: str


class GetWeatherResponse(BaseModel):
    city: str
    weather_description: str


class WeatherResourcesServer(SimpleResourcesServer):
    config: BaseResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/get_weather")(self.get_weather)
        return app

    async def get_weather(self, body: GetWeatherRequest) -> GetWeatherResponse:
        return GetWeatherResponse(
            city=body.city,
            weather_description=f"The weather in {body.city} is cold."
        )

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # Custom verification logic here
        reward = 1.0  # Return 0.0-1.0 based on task success
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    WeatherResourcesServer.run_webserver()
```

**Config** (`resources_servers/my_tool/configs/my_tool.yaml`):

```yaml
my_tool_resources_server:
  resources_servers:
    my_tool:
      entrypoint: app.py
      domain: agent
      host: 0.0.0.0
      port: 8080
```

Based on: `resources_servers/example_single_tool_call/app.py`

---

## Base Classes

Import from `nemo_gym.base_resources_server`:

### Class Hierarchy

```text
BaseServer (server_utils.py)
├── SimpleServer (session middleware, FastAPI setup)
│   └── SimpleResourcesServer ← Use this
└── BaseResourcesServer (config management)
```

**`SimpleResourcesServer`** combines both parents—use it for all implementations.

### Request/Response Types

| Type | Purpose | Key Fields |
|------|---------|------------|
| `BaseVerifyRequest` | Input to `/verify` | `responses_create_params`, `response` |
| `BaseVerifyResponse` | Output from `/verify` | `reward` (float, 0.0–1.0) |
| `BaseSeedSessionRequest` | Input to `/seed_session` | — |
| `BaseSeedSessionResponse` | Output from `/seed_session` | — |

### Required Methods

| Method | Required | Purpose |
|--------|----------|---------|
| `setup_webserver()` | Override | Register tool endpoints on FastAPI app |
| `verify()` | **Abstract** | Evaluate rollout, return reward |
| `seed_session()` | Optional | Initialize per-rollout state (default: no-op) |

---

## Configuration

Resources servers require a `domain` field in the config YAML:

```yaml
my_weather_server:
  resources_servers:
    my_weather_tool:
      entrypoint: app.py
      domain: agent  # Required
```

### Domain Values

| Domain | Use Case |
|--------|----------|
| `math` | Mathematical problem-solving |
| `coding` | Code generation, programming |
| `agent` | Tool calling, agent workflows |
| `knowledge` | Question answering |
| `instruction_following` | Instruction benchmarks |
| `long_context` | Long context handling |
| `safety` | Safety and alignment |
| `games` | Game-playing scenarios |
| `translation` | Translation tasks |
| `e2e` | End-to-end workflows |
| `other` | General purpose |

See {py:class}`~nemo_gym.config_types.Domain` for the enum definition.

---

## How-To Guides

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Integrate Python Tools
:link: integrate-python-tools
:link-type: doc
Wrap existing Python functions as tools.
+++
{bdg-secondary}`python` {bdg-secondary}`tools`
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Integrate APIs
:link: integrate-apis
:link-type: doc
Connect external REST/GraphQL APIs.
+++
{bdg-secondary}`api` {bdg-secondary}`integration`
:::

:::{grid-item-card} {octicon}`container;1.5em;sd-mr-1` Containerize
:link: containerize
:link-type: doc
Package for Docker deployment.
+++
{bdg-secondary}`docker` {bdg-secondary}`deployment`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Profile Performance
:link: profile
:link-type: doc
Measure and optimize throughput.
+++
{bdg-secondary}`performance` {bdg-secondary}`profiling`
:::

::::

---

## Error Handling

The `SimpleServer` base class wraps all endpoint handlers with exception middleware. If your tool or verify method raises:

- Exceptions are caught and logged with full stack trace
- `repr(e)` is returned to the caller with HTTP 500
- The server remains running

For custom error responses, raise `fastapi.HTTPException` explicitly.

---

## Session Isolation

Each rollout gets a unique session ID via cookie-based middleware. Session state is:
- **Isolated per-rollout**: Different rollouts don't share state
- **Persisted within rollout**: Multiple tool calls in one rollout share state

Access the session in endpoints via `request.session` (standard Starlette sessions).

---

## Next Steps

- {doc}`/tutorials/creating-resource-server` — Step-by-step tutorial from scratch
- {doc}`/reference/configuration` — Complete configuration reference

## Source Code

Base classes: `nemo_gym/base_resources_server.py` (74 lines), `nemo_gym/server_utils.py` (SimpleServer)