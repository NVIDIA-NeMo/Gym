(resources-server-python-tools)=
# Integrate Existing Python Tools

```{note}
This page is a stub. Content is being developed.
```

Learn how to wrap existing Python functions and libraries as tools in your resources server.

---

## Overview

NeMo Gym tools are FastAPI endpoints. To integrate existing Python code:
1. Define Pydantic schemas for input/output
2. Create an async endpoint method
3. Register the endpoint in `setup_webserver()`

## Basic Pattern

```python
from pydantic import BaseModel
from nemo_gym.base_resources_server import SimpleResourcesServer

class MyToolRequest(BaseModel):
    """Input schema — defines what the model sends."""
    query: str

class MyToolResponse(BaseModel):
    """Output schema — defines what the tool returns."""
    result: str

class MyResourcesServer(SimpleResourcesServer):
    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/my_tool")(self.my_tool)
        return app

    async def my_tool(self, body: MyToolRequest) -> MyToolResponse:
        # Your existing Python code here
        result = existing_function(body.query)
        return MyToolResponse(result=result)
```

## Wrapping Synchronous Code

For blocking operations, use `asyncio.to_thread`:

```python
import asyncio

async def my_tool(self, body: MyToolRequest) -> MyToolResponse:
    # Run blocking code in thread pool
    result = await asyncio.to_thread(blocking_function, body.query)
    return MyToolResponse(result=result)
```

## Complex Input/Output Types

### Nested Objects

<!-- TODO: Document nested Pydantic models -->

### Lists and Optionals

<!-- TODO: Document list and optional handling -->

### File Handling

<!-- TODO: Document file upload/download patterns -->

## Error Handling

<!-- TODO: Document error handling best practices -->

## Configuration via Server Config

Pass configuration to your tools via the server config:

```python
class MyResourcesServerConfig(BaseResourcesServerConfig):
    api_endpoint: str
    timeout: int = 30

class MyResourcesServer(SimpleResourcesServer):
    config: MyResourcesServerConfig

    async def my_tool(self, body: MyToolRequest) -> MyToolResponse:
        # Access config
        endpoint = self.config.api_endpoint
```

## Testing Tools

<!-- TODO: Document testing patterns -->

## Examples

For complete examples, see:
- `resources_servers/example_single_tool_call/` — Simple weather tool
- `resources_servers/google_search/` — External API integration
- `resources_servers/math_advanced_calculations/` — Complex computation
