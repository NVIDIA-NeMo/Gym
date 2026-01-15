(resources-server-python-tools)=
# Integrate Existing Python Tools

Wrap existing Python functions and libraries as tools in your resources server.

**Goal**: Wrap Python functions as FastAPI tool endpoints in your resources server.

**Prerequisites**: Familiar with {doc}`index` and basic FastAPI.

---

NeMo Gym tools are FastAPI endpoints. To integrate existing Python code:
1. Define Pydantic schemas for input/output
2. Create an async endpoint method
3. Register the endpoint in `setup_webserver()`

---

## Basic Pattern

Create a file at `resources_servers/<your_server>/app.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class MyToolResourcesServerConfig(BaseResourcesServerConfig):
    pass


class MyToolRequest(BaseModel):
    """Input schema — defines what the model sends."""
    query: str


class MyToolResponse(BaseModel):
    """Output schema — defines what the tool returns."""
    result: str


class MyToolResourcesServer(SimpleResourcesServer):
    config: MyToolResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/my_tool")(self.my_tool)
        return app

    async def my_tool(self, body: MyToolRequest) -> MyToolResponse:
        # Replace with your actual Python function
        result = body.query.upper()  # Example: uppercase the input
        return MyToolResponse(result=result)

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        """Evaluate rollout performance and return a reward (0.0 to 1.0).
        
        Called after a rollout completes. Use body.response to access
        the model's outputs and compute task-specific metrics.
        """
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


if __name__ == "__main__":
    MyToolResourcesServer.run_webserver()
```

---

## Wrapping Synchronous Code

For blocking operations, use `asyncio.get_running_loop()` with `run_in_executor`:

```python
import asyncio

async def my_tool(self, body: MyToolRequest) -> MyToolResponse:
    loop = asyncio.get_running_loop()
    # Run blocking code in thread pool
    result = await loop.run_in_executor(None, blocking_function, body.query)
    return MyToolResponse(result=result)
```

For simpler cases, `asyncio.to_thread` provides a cleaner API:

```python
import asyncio

async def my_tool(self, body: MyToolRequest) -> MyToolResponse:
    result = await asyncio.to_thread(blocking_function, body.query)
    return MyToolResponse(result=result)
```

---

## Complex Input/Output Types

### Nested Objects

Define nested Pydantic models for structured data:

```python
from pydantic import BaseModel
from typing import Optional


class ContactInfo(BaseModel):
    email: str
    phone: Optional[str] = None


class UserRequest(BaseModel):
    name: str
    contact: ContactInfo  # Nested model
```

### Lists and Optionals

Use standard Python type hints:

```python
from typing import List, Optional, Any
from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    items: List[str]                           # Required list
    weights: Optional[List[float]] = None      # Optional list
    metadata: Optional[dict[str, Any]] = None  # Optional dict


class AnalysisResponse(BaseModel):
    results: List[int]
    confidence_scores: List[float]
```

### Flexible Schemas

For tools that accept arbitrary fields, use `ConfigDict(extra="allow")`:

```python
from pydantic import BaseModel, ConfigDict


class FlexibleRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    # Any additional fields passed will be accepted
```

Access extra fields by using `body.model_dump(exclude_unset=True)`.

---

## Error Handling

### Return Errors in Response (Recommended)

Return error information to the model so it can self-correct during rollouts. This approach allows the model to retry with different parameters rather than failing the entire rollout:

```python
import asyncio

async def search(self, body: SearchRequest) -> SearchResponse:
    try:
        # Use run_in_executor for blocking requests
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(url, params=params, timeout=10)
        )
        response.raise_for_status()
        return SearchResponse(results=response.json())
    except Exception as e:
        return SearchResponse(results=f"Error: {str(e)}")
```

### HTTPException for Invalid State

Use `HTTPException` for session or state validation errors that indicate a client bug:

```python
from fastapi import HTTPException, Request
from nemo_gym.server_utils import SESSION_ID_KEY


async def my_tool(self, request: Request, body: MyToolRequest) -> MyToolResponse:
    session_id = request.session[SESSION_ID_KEY]
    
    if session_id not in self.session_data:
        raise HTTPException(
            status_code=400,
            detail="Session not initialized. Call seed_session first.",
        )
    
    # Process request...
```

---

## Configuration Through Server Config

Pass configuration to your tools through the server config:

```python
import asyncio
import requests
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class MyAPIResourcesServerConfig(BaseResourcesServerConfig):
    api_endpoint: str
    api_key: str
    timeout: int = 30


class APIRequest(BaseModel):
    query: str


class APIResponse(BaseModel):
    data: dict


class MyAPIResourcesServer(SimpleResourcesServer):
    config: MyAPIResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/call_api")(self.call_api)
        return app

    async def call_api(self, body: APIRequest) -> APIResponse:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(
                self.config.api_endpoint,
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                params={"q": body.query},
                timeout=self.config.timeout,
            )
        )
        return APIResponse(data=response.json())

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)
```

Configuration values come from the YAML config file when the server is launched.

---

## Session State

Store per-session data using the session ID from the request:

```python
from typing import Dict, Any
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY


class StatefulResourcesServerConfig(BaseResourcesServerConfig):
    pass


class MyToolRequest(BaseModel):
    query: str


class MyToolResponse(BaseModel):
    result: str
    history_length: int


class StatefulResourcesServer(SimpleResourcesServer):
    config: StatefulResourcesServerConfig
    session_data: Dict[str, Any] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/my_tool")(self.my_tool)
        return app

    async def seed_session(
        self, request: Request, body: BaseSeedSessionRequest
    ) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        # Initialize session state
        self.session_data[session_id] = {"history": [], "context": {}}
        return BaseSeedSessionResponse()

    async def my_tool(self, request: Request, body: MyToolRequest) -> MyToolResponse:
        session_id = request.session[SESSION_ID_KEY]
        session = self.session_data[session_id]
        
        # Access and update session state
        session["history"].append(body.query)
        result = body.query.upper()
        
        return MyToolResponse(result=result, history_length=len(session["history"]))

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)
```

---

## Testing Tools

### Basic Test Pattern

```python
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.your_server.app import (
    MyResourcesServer,
    MyResourcesServerConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = MyResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        MyResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
```

### Testing Tool Methods

```python
import pytest
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.your_server.app import (
    MyResourcesServer,
    MyResourcesServerConfig,
    MyToolRequest,
    MyToolResponse,
)


class TestApp:
    @pytest.fixture
    def server(self):
        config = MyResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        return MyResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )

    @pytest.mark.asyncio
    async def test_my_tool(self, server) -> None:
        request = MyToolRequest(query="test")
        response = await server.my_tool(request)
        
        assert isinstance(response, MyToolResponse)
        assert response.result is not None
```

### HTTP Integration Testing

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from nemo_gym.server_utils import ServerClient
from resources_servers.your_server.app import (
    MyResourcesServer,
    MyResourcesServerConfig,
)


class TestApp:
    @pytest.fixture
    def server(self):
        config = MyResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        return MyResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )

    @pytest.fixture
    def client(self, server):
        app = server.setup_webserver()
        return TestClient(app)

    def test_endpoint(self, client) -> None:
        response = client.post("/my_tool", json={"query": "test"})
        assert response.status_code == 200
```

---

## Examples

For complete implementations, see:
- `resources_servers/example_single_tool_call/` — Simple weather tool
- `resources_servers/google_search/` — External API integration with error handling
- `resources_servers/math_with_code/` — Synchronous code execution with session state
- `resources_servers/workplace_assistant/` — Dynamic routing to Python functions
- `resources_servers/example_multi_step/` — Multi-tool server with complex types
