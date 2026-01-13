(resources-server-index)=
# Resources Server

```{warning}
This article has not been reviewed by a developer SME. Content may change.
```

Resources servers define the training environment—tasks, tools, and verification logic. They implement `SimpleResourcesServer` and expose two core endpoints:

- **`/seed_session`** — Initialize session state before a rollout
- **`/verify`** — Evaluate the completed rollout and compute reward

---

## Core Concepts

### Tools

Tools are FastAPI endpoints that models can call. Each tool has a request/response schema defined with Pydantic:

```python
app.post("/get_weather")(self.get_weather)

async def get_weather(self, body: GetWeatherRequest) -> GetWeatherResponse:
    return GetWeatherResponse(city=body.city, weather_description="Sunny")
```

### Verification

The `verify()` method evaluates rollout performance and returns a reward signal:

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    # Evaluate the model's tool usage and outputs
    reward = 1.0 if correct else 0.0
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

### Session Management

Sessions isolate state between rollouts. The `seed_session()` method initializes per-rollout state:

```python
async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
    # Initialize state for this rollout
    return BaseSeedSessionResponse()
```

---

## Base Classes

Resources servers inherit from base classes in `nemo_gym/base_resources_server.py`:

### Request/Response Types

| Type | Description | Key Fields |
|------|-------------|------------|
| `BaseRunRequest` | Input for tool execution | `responses_create_params` |
| `BaseVerifyRequest` | Input for verification | `responses_create_params`, `response` |
| `BaseVerifyResponse` | Output from verification | `reward` (float) |
| `BaseSeedSessionRequest` | Input for session initialization | — |
| `BaseSeedSessionResponse` | Output from session initialization | — |

### Class Hierarchy

```text
BaseServer
    └── BaseResourcesServer
            └── SimpleResourcesServer  ← Use this for most implementations
```

- **`BaseResourcesServer`**: Abstract base with config management
- **`SimpleResourcesServer`**: Adds FastAPI setup, session middleware, and default endpoints

---

## Minimal Working Example

```python
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class MyConfig(BaseResourcesServerConfig):
    domain: str = "math"


class MyResourcesServer(SimpleResourcesServer):
    config: MyConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # Register custom tools
        app.post("/my_tool")(self.my_tool)
        return app

    async def my_tool(self, body: MyToolRequest) -> MyToolResponse:
        return MyToolResponse(result="computed")

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # Your verification logic
        reward = 1.0 if self.is_correct(body.response) else 0.0
        return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

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

## Configuration

Resources servers require a `domain` field:

```yaml
my_resource:
  resources_servers:
    my_implementation:
      entrypoint: app.py
      domain: agent  # Required: math, coding, agent, knowledge, etc.
```

See {doc}`/reference/configuration` for all domain values.

---

## Creating a New Resources Server

For a complete tutorial on building a resources server from scratch, see {doc}`/tutorials/creating-resource-server`.

## Source Code

The base classes are defined in `nemo_gym/base_resources_server.py` (73 lines).