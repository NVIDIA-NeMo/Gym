(resources-server-apis)=
# Integrate External APIs

Connect external REST APIs, GraphQL endpoints, or web services as tools in your resources server.

**Goal**: Add external API calls (REST, GraphQL) as tools in your resources server.

**Prerequisites**: Complete {doc}`/get-started/detailed-setup`

---

## Overview

External API integration allows models to call real-world services during rollouts (model inference + tool execution cycles). The pattern:

1. Add API credentials to `env.yaml` (in repo root, gitignored)
2. Create a config class extending `BaseResourcesServerConfig`
3. Define Pydantic request/response schemas
4. Register tool endpoints in `setup_webserver()`
5. Implement async tool methods with error handling

:::{warning}
Store API credentials in `env.yaml` (gitignored) or environment variables. Never commit secrets to version control.
:::

---

## Example: Google Search Integration

The `google_search` resources server demonstrates a complete API integration. It provides:

| Tool | Endpoint | Description |
|------|----------|-------------|
| `search` | `/search` | Query Google Programmable Search Engine |
| `browse` | `/browse` | Fetch and extract webpage content using [trafilatura](https://trafilatura.readthedocs.io/) |

### Configuration

Add credentials to `env.yaml`:

```yaml
google_search:
  resources_servers:
    google_search:
      google_api_key: <your_api_key>
      google_cx: <your_search_engine_id>
```

Get credentials from [Google Programmable Search Engine](https://developers.google.com/custom-search/v1/using_rest).

### Server Implementation

The server extends `SimpleResourcesServer` and registers tool endpoints:

```python
from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    SimpleResourcesServer,
)


class GoogleSearchResourcesServerConfig(BaseResourcesServerConfig):
    google_api_key: str
    google_cx: str


class GoogleSearchResourcesServer(SimpleResourcesServer):
    config: GoogleSearchResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/search")(self.search)
        app.post("/browse")(self.browse)
        return app
```

Reference: `resources_servers/google_search/app.py:33-35,90-100`

### Tool Definitions

Tools are defined in the dataset JSONL file (each row's `responses_create_params.tools` array) or in agent configuration. The tool `name` must match the endpoint path:

```python
tools = [
    {
        "type": "function",
        "name": "search",
        "description": "Search Google for a query and return up to 10 search results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The term to search for",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "browse",
        "description": "Returns the cleaned content of a webpage. Long pages are truncated.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the page to get content from",
                }
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]
```

:::{note}
Tool `name` must match the FastAPI endpoint path (without leading `/`).
:::

### Running the Example

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/google_search/configs/google_search.yaml"

ng_run "+config_paths=[$config_paths]"

ng_collect_rollouts +agent_name=simple_agent \
    +input_jsonl_fpath=resources_servers/google_search/data/example.jsonl \
    +output_jsonl_fpath=results/rollouts.jsonl \
    +limit=1
```

---

## Basic REST API Integration

Use the synchronous `requests` library for simple integrations:

```python
import requests
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class MyAPIConfig(BaseResourcesServerConfig):
    api_key: str
    base_url: str = "https://api.example.com"


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    results: list[str]


class MyAPIResourcesServer(SimpleResourcesServer):
    config: MyAPIConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/search")(self.search)
        return app

    async def search(self, body: SearchRequest) -> SearchResponse:
        response = requests.get(
            f"{self.config.base_url}/search",
            params={"q": body.query},
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        return SearchResponse(results=response.json()["results"])

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # Implement verification logic
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)
```

:::{note}
This example uses synchronous `requests` inside an `async def`. This blocks the event loop during the HTTP call, which is acceptable for low-to-moderate concurrency. For high-concurrency scenarios (thousands of concurrent requests), use `aiohttp`—NeMo Gym provides a global aiohttp client at `nemo_gym/server_utils.py:83-98`.
:::

---

## Authentication Patterns

:::::{tab-set}

::::{tab-item} Bearer Token
```python
class MyConfig(BaseResourcesServerConfig):
    api_key: str

# In your tool method:
headers = {"Authorization": f"Bearer {self.config.api_key}"}
response = requests.get(url, headers=headers)
```
::::

::::{tab-item} Query Parameter
```python
# Google APIs use query parameter authentication
params = {
    "key": self.config.google_api_key,
    "cx": self.config.google_cx,
    "q": body.query,
}
response = requests.get(url, params=params)
```

Reference: `resources_servers/google_search/app.py:103-107`
::::

:::::

---

## Error Handling

Handle API errors gracefully to prevent rollout failures:

```python
import requests
from loguru import logger


async def search(self, body: SearchRequest) -> SearchResponse:
    try:
        response = requests.get(
            f"{self.config.base_url}/search",
            params={"q": body.query},
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=30,
        )
        if response.status_code == 429:
            logger.warning("Rate limited, returning empty results")
            return SearchResponse(results=[])
        response.raise_for_status()
        return SearchResponse(results=response.json()["results"])
    except requests.RequestException as e:
        logger.error(f"API error: {e}")
        return SearchResponse(results=[])
```

---

## Environment Inheritance

Inherit from existing resources servers to reuse their tools:

```python
from resources_servers.google_search.app import (
    GoogleSearchResourcesServer,
    GoogleSearchVerifyRequest,
    GoogleSearchVerifyResponse,
)


class MyCustomServer(GoogleSearchResourcesServer):
    """Inherit search and browse tools, add custom verification."""

    async def verify(self, body: GoogleSearchVerifyRequest) -> GoogleSearchVerifyResponse:
        # Use inherited search/browse, add custom logic
        return GoogleSearchVerifyResponse(**body.model_dump(), reward=1.0, parsed_option="A")
```

Reference: `resources_servers/google_search/app.py:59-63`

---

## Related

- {doc}`./index` — Resources server overview
- {doc}`./integrate-python-tools` — Integrate Python libraries as tools
- {doc}`./containerize` — Package resources servers for deployment
- `resources_servers/google_search/` — Complete reference implementation
