(resources-server-apis)=
# Integrate Existing APIs

Connect external REST APIs, GraphQL endpoints, or other web services as tools in your resources server.

---

## Overview

External API integration allows models to access real-world data and services. The pattern involves:

1. Configure API credentials via `env.yaml`
2. Create an async HTTP client (use `aiohttp` for async operations)
3. Define request/response schemas that map to the API
4. Handle authentication, errors, and rate limiting

---

## Example: Google Search Integration

The `google_search` resources server demonstrates a complete API integration pattern. It provides two tools:

| Tool | Description |
|------|-------------|
| `search` | Query Google and return structured results |
| `browse` | Fetch and parse webpage content |

### Configuration

Set API credentials in `env.yaml`:

```yaml
google_search:
  resources_servers:
    google_search:
      google_api_key: <your_api_key>
      google_cx: <your_cx_engine>
```

Get credentials from [Google Programmable Search Engine](https://developers.google.com/custom-search/v1/using_rest).

### Tool Definitions

These tools can be inherited by other environments:

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
        "description": "Returns the cleaned content of a webpage. Truncated to 10,000 words.",
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

### Running the Example

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/google_search/configs/google_search.yaml"

ng_run "+config_paths=[$config_paths]"

ng_collect_rollouts +agent_name=simple_agent \
    +input_jsonl_fpath=data/Nemotron-RL-knowledge-web_search-mcqa.jsonl \
    +output_jsonl_fpath=results/rollouts.jsonl \
    +limit=1
```

---

## Basic REST API Integration

```python
import aiohttp
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
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

    def model_post_init(self, context):
        # Create reusable HTTP session for better performance
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.config.api_key}"}
            )
        return self._session

    async def search(self, body: SearchRequest) -> SearchResponse:
        session = await self._get_session()
        async with session.get(
            f"{self.config.base_url}/search",
            params={"q": body.query}
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return SearchResponse(results=data["results"])
```

---

## Authentication Patterns

:::::{tab-set}

::::{tab-item} API Key

```python
class MyResourcesServerConfig(BaseResourcesServerConfig):
    api_key: str

# In your tool method:
headers = {"Authorization": f"Bearer {self.config.api_key}"}
```

::::

::::{tab-item} Query Parameter

```python
# Some APIs use query parameter authentication
params = {
    "key": self.config.api_key,
    "cx": self.config.search_engine_id,
    "q": body.query,
}
```

::::

:::::

---

## Error Handling

Handle API errors gracefully to prevent rollout failures:

```python
import aiohttp
from loguru import logger


async def search(self, body: SearchRequest) -> SearchResponse:
    try:
        session = await self._get_session()
        async with session.get(
            f"{self.config.base_url}/search",
            params={"q": body.query},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 429:
                logger.warning("Rate limited, returning empty results")
                return SearchResponse(results=[])
            response.raise_for_status()
            data = await response.json()
            return SearchResponse(results=data["results"])
    except aiohttp.ClientError as e:
        logger.error(f"API error: {e}")
        return SearchResponse(results=[])
```

---

## Environment Inheritance

APIs integrated into one resources server can be reused by others. The `google_search` tools are available for inheritance:

```python
from resources_servers.google_search.app import GoogleSearchResourcesServer


class MyCustomServer(GoogleSearchResourcesServer):
    """Inherit search and browse tools, add custom verification."""

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # Custom verification using search results
        ...
```

---

## Source Code

For the complete implementation, see `resources_servers/google_search/` (README: 383 lines).
