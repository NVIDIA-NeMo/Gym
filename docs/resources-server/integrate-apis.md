(resources-server-apis)=
# Integrate Existing APIs

```{note}
This page is a stub. Content is being developed.
```

Connect external REST APIs, GraphQL endpoints, or other web services as tools in your resources server.

---

## Overview

To integrate external APIs:
1. Create an async HTTP client (use `aiohttp` for async operations)
2. Define request/response schemas that map to the API
3. Handle authentication and error cases

## Basic REST API Integration

```python
import aiohttp
from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    results: list[str]

class MyResourcesServer(SimpleResourcesServer):
    async def search(self, body: SearchRequest) -> SearchResponse:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.example.com/search",
                params={"q": body.query},
                headers={"Authorization": f"Bearer {self.config.api_key}"}
            ) as response:
                data = await response.json()
                return SearchResponse(results=data["results"])
```

## Authentication Patterns

### API Key Authentication

```python
class MyResourcesServerConfig(BaseResourcesServerConfig):
    api_key: str

# In your tool method:
headers = {"Authorization": f"Bearer {self.config.api_key}"}
```

### OAuth/Token Authentication

<!-- TODO: Document OAuth patterns -->

### Session-Based Authentication

<!-- TODO: Document session handling -->

## Error Handling

### HTTP Errors

<!-- TODO: Document HTTP error handling -->

### Timeout Handling

<!-- TODO: Document timeout configuration -->

### Retry Logic

<!-- TODO: Document retry patterns -->

## Rate Limiting

<!-- TODO: Document rate limit handling -->

## GraphQL Integration

<!-- TODO: Document GraphQL client patterns -->

## Caching Responses

<!-- TODO: Document caching strategies -->

## Examples

For complete examples, see:
- `resources_servers/google_search/` — Google Search API integration
