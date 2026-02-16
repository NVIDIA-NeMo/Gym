(building-environments-stateful)=

# Stateful Environment

For environments that need to maintain state across multiple tool calls within an episode, NeMo Gym provides session management via middleware.

---

## What You'll Build

A counter environment where the agent must increment a counter and read its value. The session state (counter value) persists across tool calls within an episode using `SESSION_ID_KEY`.

### Episode Flow

```text
Goal (what the agent is learning)
  - Learn multi-step stateful tool usage: perform actions that change environment state and then read/verify the final state.

Inputs
  - seed input: initial_count (e.g., 3)
  - ground truth for grading: expected_count (e.g., 5)

Flow (state is stored per session_id inside the ResourcesServer)
  1) POST ResourcesServer /seed_session {"initial_count": 3}
     - stores session_id_to_counter[session_id] = 3
  2) POST ModelServer /v1/responses → function_call: increment_counter({"count": 2})
  3) POST ResourcesServer /increment_counter {"count": 2}
     - counter becomes 5 for this session_id
  4) POST ModelServer /v1/responses → function_call: get_counter_value({})
  5) POST ResourcesServer /get_counter_value {}
     - returns {"count": 5}
  6) POST ResourcesServer /verify {"expected_count": 5, ...}
     - reward = 1.0 iff stored counter == expected_count
```

---

## Implementation

**File (simplified excerpt, source-aligned): `resources_servers/example_session_state_mgmt/app.py`**

```python
# simplified
from typing import Dict

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY  # Critical import!


class StatefulCounterResourcesServerConfig(BaseResourcesServerConfig):
    pass


# Custom seed request to initialize state
class StatefulCounterSeedSessionRequest(BaseSeedSessionRequest):
    initial_count: int


class IncrementCounterRequest(BaseModel):
    count: int


class IncrementCounterResponse(BaseModel):
    success: bool


class GetCounterValueResponse(BaseModel):
    count: int


class StatefulCounterVerifyRequest(BaseVerifyRequest):
    expected_count: int


class StatefulCounterResourcesServer(SimpleResourcesServer):
    config: StatefulCounterResourcesServerConfig

    # Session state storage - maps session_id -> state
    session_id_to_counter: Dict[str, int] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/increment_counter")(self.increment_counter)
        app.post("/get_counter_value")(self.get_counter_value)
        return app

    # Initialize session state
    async def seed_session(
        self,
        request: Request,  # Must include Request to access session
        body: StatefulCounterSeedSessionRequest
    ) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]  # Get unique session ID
        self.session_id_to_counter.setdefault(session_id, body.initial_count)
        return BaseSeedSessionResponse()

    # Stateful tool - modifies session state
    async def increment_counter(
        self,
        request: Request,
        body: IncrementCounterRequest
    ) -> IncrementCounterResponse:
        session_id = request.session[SESSION_ID_KEY]
        counter = self.session_id_to_counter.setdefault(session_id, 0)
        counter += body.count
        self.session_id_to_counter[session_id] = counter
        return IncrementCounterResponse(success=True)

    # Read-only tool
    async def get_counter_value(self, request: Request) -> GetCounterValueResponse:
        session_id = request.session[SESSION_ID_KEY]
        counter = self.session_id_to_counter.setdefault(session_id, 0)
        return GetCounterValueResponse(count=counter)

    # Verify against expected final state
    async def verify(
        self,
        request: Request,
        body: StatefulCounterVerifyRequest
    ) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]

        reward = 0.0
        if session_id in self.session_id_to_counter:
            counter = self.session_id_to_counter[session_id]
            reward = float(body.expected_count == counter)

        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    StatefulCounterResourcesServer.run_webserver()
```

### Key Pattern

Use `SESSION_ID_KEY` from the request session middleware to maintain per-episode state. The session ID is automatically assigned by the framework's middleware (set up in `SimpleResourcesServer.setup_webserver()` via `self.setup_session_middleware(app)`).

**To access session state:**
1. Add `request: Request` as a parameter to any endpoint method
2. Read the session ID with `request.session[SESSION_ID_KEY]`
3. Store state in an instance-level dictionary keyed by session ID

:::{note}
The actual source file (`resources_servers/example_session_state_mgmt/app.py`) contains a local redefinition of `BaseVerifyResponse` that shadows the import. This simplified version removes that redundancy and uses the imported `BaseVerifyResponse` directly, which has the same structure (`reward: float` inheriting from `BaseVerifyRequest`).
:::

---

## Rollout Transcript

```text
[Episode start]

Agent → ResourcesServer: POST /seed_session {"initial_count": 3}
  (ResourcesServer stores session_id_to_counter[session_id] = 3)

User: "Increment the counter by 2, then tell me the current value."

Agent → ModelServer: POST /v1/responses (tools: increment_counter, get_counter_value)
Model calls tool:
  function_call: increment_counter({"count": 2})

Agent → ResourcesServer: POST /increment_counter {"count": 2}
ResourcesServer → Agent:
  {"success": true}
  (counter is now 5 for this session_id)

Agent → ModelServer: POST /v1/responses
Model calls tool:
  function_call: get_counter_value({})

Agent → ResourcesServer: POST /get_counter_value {}
ResourcesServer → Agent:
  {"count": 5}

[Episode end → grading]

Agent → ResourcesServer: POST /verify {"expected_count": 5, ...}
ResourcesServer:
  - reads counter for this session_id
  - reward = 1.0 if counter == expected_count else 0.0
```

---

> **Previous**: {ref}`Multi-Step Environment <building-environments-multi-step>` | **Next**: {ref}`Real-World Environment <building-environments-real-world>`
