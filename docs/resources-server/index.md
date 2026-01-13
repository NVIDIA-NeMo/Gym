(resources-server-index)=
# Resources Server

```{warning}
This article was generated and has not been reviewed. Content may change.
```

Resources servers define the training environment—tasks, tools, and verification logic. They implement `SimpleResourcesServer` and expose two core endpoints:

- **`/seed_session`** — Initialize session state before a rollout
- **`/verify`** — Evaluate the completed rollout and compute reward

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

## Creating a New Resources Server

For a complete tutorial on building a resources server from scratch, see {doc}`/tutorials/creating-resource-server`.

