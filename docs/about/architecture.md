(about-architecture)=
# NeMo Gym Architecture

This page explains how NeMo Gym components work together. Understanding the
architecture helps platform engineers run the system and helps developers
integrate servers and workflows correctly.

:::{tip}
For the product's place in the broader NVIDIA ecosystem, refer to
{ref}`about-ecosystem`.
:::

---

## High-Level Architecture Diagram

NeMo Gym coordinates three kinds of servers (model, resources, agent) under a
shared configuration and provides utilities for high-throughput data
collection.

![High-level overview of NeMo Gym servers and data flow.](../../resources/rl_verifiers_system_design.png)

---

## Component Layers

NeMo Gym organizes into three layers that work together to deliver RL training
data and evaluation signals.

### Functional Servers

Functional servers expose HTTP endpoints that your workflows call directly.

::::{tab-set}

:::{tab-item} Responses API Agent Server
Runs the minimal agent loop and returns metrics.

**Endpoints**: `POST /v1/responses`, `POST /run`

```py
class SimpleResponsesAPIAgent(BaseResponsesAPIAgent, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/v1/responses")(self.responses)
        app.post("/run")(self.run)
        return app
```
:::

:::{tab-item} Responses API Model Server
Wraps model endpoints with OpenAI-compatible routes.

**Endpoints**: `POST /v1/chat/completions`, `POST /v1/responses`

```py
class SimpleResponsesAPIModel(BaseResponsesAPIModel, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/v1/chat/completions")(self.chat_completions)
        app.post("/v1/responses")(self.responses)
        return app
```
:::

:::{tab-item} Resources Server
Provides tools and verification to compute rewards.

**Endpoints**: `POST /seed_session`, `POST /verify`

```py
class SimpleResourcesServer(BaseResourcesServer, SimpleServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/seed_session")(self.seed_session)
        app.post("/verify")(self.verify)
        return app
```
:::

::::

---

### Runtime Services

Common runtime behavior is provided by shared utilities:

- Session handling per request
- Exception handling with structured responses
- Optional CPU-time profiling with a `/stats` endpoint
- ASGI app startup with configurable logging

The head server serves the resolved global configuration for other processes.

```py
class HeadServer(BaseServer):
    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.get("/global_config_dict_yaml")(self.global_config_dict_yaml)
        return app
```

---

### Configuration Model

NeMo Gym centralizes runtime settings in a single configuration dictionary.

Key points:

- Parsed with Hydra and merged with optional entries from `.env.yaml`
- Supports `config_paths` for composing multiple files
- Validates server references and fills default host/port
- Cached and reused across processes

Code:

```py
class GlobalConfigDictParserConfig(BaseModel):
    ...
```

```py
config_paths, extra_configs = self.load_extra_config_paths(config_paths)
# validation and default population
```

Schemas for servers and datasets:

::::{tab-set}

:::{tab-item} Base
Base configuration inherited by all server types.

```py
class BaseRunServerConfig(BaseServerConfig):
    entrypoint: str
    domain: Optional[Domain] = None
```
:::

:::{tab-item} Responses API Agent
```py
class ResponsesAPIAgentServerTypeConfig(...)
```
:::

:::{tab-item} Responses API Model
```py
class ResponsesAPIModelServerTypeConfig(...)
```
:::

:::{tab-item} Resources
```py
class ResourcesServerTypeConfig(...)
```
:::

::::

---

## Data Flow

The data collection helper drives high-throughput requests to the agent server
and writes outputs for training.

Flow:

1. Read rows from an input JSONL file; optionally repeat rows
2. Build a server client bound to the head server configuration
3. For each row, `POST /run` to the agent server; write the JSON result
4. Track numeric fields to compute averages

Concurrency uses an optional semaphore to cap the number of in-flight
requests.

::::{tab-set}

:::{tab-item} Configuration
Schema for rollout collection settings.

```py
class RolloutCollectionConfig(BaseModel):
    agent_name: str
    input_jsonl_fpath: str
    output_jsonl_fpath: str
```
:::

:::{tab-item} Concurrency Control
Optional semaphore to limit parallel requests.

```py
semaphore = nullcontext()
if config.num_samples_in_parallel:
    semaphore = Semaphore(config.num_samples_in_parallel)
```
:::

:::{tab-item} Request Handler
Async coroutine that posts to the agent server and writes results.

```py
with open(config.output_jsonl_fpath, "a") as f:
    async def _post_coroutine(row: dict) -> None:
        row["responses_create_params"] = row["responses_create_params"] | config.responses_create_params
        async with semaphore:
            response = await server_client.post(server_name=config.agent_name, url_path="/run", json=row)
            response.raise_for_status()
            result = await response.json()
            f.write(json.dumps(result) + "\n")
```
:::

::::

---

## Deployment Topologies

NeMo Gym runs locally or connects to an existing Ray cluster depending on
configuration.

```py
def initialize_ray() -> None:
    if ray_head_node_address is not None:
        ray.init(address=ray_head_node_address, ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True)
```
