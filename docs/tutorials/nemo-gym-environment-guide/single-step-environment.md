(building-environments-single-step)=

# Single-Step Environment

This is the simplest possible environment — a stateless tool that always succeeds.

---

## What You'll Build

A weather tool environment where the agent calls `get_weather` with a city name and receives a weather description. The reward is always 1.0 (the focus is on structure, not verification logic).

### Episode Flow

```text
Goal (what the agent is learning)
  - Learn single-step tool usage: call get_weather with the right arguments, then use the tool output.

Inputs
  - user question (e.g., "What's the weather in Boston?")
  - tool schema for get_weather(city: str)

Flow
  1) POST ResourcesServer /seed_session
     - no persistent environment state needed for this example
  2) POST ModelServer /v1/responses
     - model emits function_call: get_weather({"city": "Boston"})
  3) POST ResourcesServer /get_weather
     - returns {"city": "...", "weather_description": "..."}
  4) POST ModelServer /v1/responses
     - model produces final assistant message
  5) POST ResourcesServer /verify
     - returns reward (here: always 1.0)
```

---

## Implementation

**File (simplified excerpt, source-aligned): `resources_servers/example_single_tool_call/app.py`**

```python
# simplified
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


# Step 1: Define your configuration class
class SimpleWeatherResourcesServerConfig(BaseResourcesServerConfig):
    pass


# Step 2: Define request/response models for your tools
class GetWeatherRequest(BaseModel):
    city: str


class GetWeatherResponse(BaseModel):
    city: str
    weather_description: str


# Step 3: Implement the server
class SimpleWeatherResourcesServer(SimpleResourcesServer):
    config: SimpleWeatherResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        # Call parent to get base app with /seed_session and /verify
        app = super().setup_webserver()

        # Register your custom tool endpoints
        app.post("/get_weather")(self.get_weather)

        return app

    # Your tool implementation
    async def get_weather(self, body: GetWeatherRequest) -> GetWeatherResponse:
        return GetWeatherResponse(
            city=body.city,
            weather_description=f"The weather in {body.city} is cold."
        )

    # The reward function - REQUIRED
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # In this simple case, always return reward of 1.0
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


# Entry point
if __name__ == "__main__":
    SimpleWeatherResourcesServer.run_webserver()
```

### Key Points

- Inherit from `SimpleResourcesServer`
- Override `setup_webserver()` to add custom tool endpoints
- Implement `verify()` to compute the reward

---

## Rollout Transcript

A complete trace of one episode:

```text
[Episode start]

Agent → ResourcesServer: POST /seed_session
  (no custom state needed for this stateless example)

User: "What's the weather in Boston?"

Agent → ModelServer: POST /v1/responses (tool available: get_weather)
Model chooses to call the tool:
  function_call: get_weather({"city": "Boston"})

Agent → ResourcesServer: POST /get_weather {"city": "Boston"}
ResourcesServer → Agent:
  {"city": "Boston", "weather_description": "The weather in Boston is cold."}

Agent → ModelServer: POST /v1/responses (now includes tool output)
Model produces final assistant message:
  "The weather in Boston is cold."

[Episode end → grading]

Agent → ResourcesServer: POST /verify
ResourcesServer:
  - returns reward: 1.0   (this example verifier always succeeds)
```

---

> **Previous**: {ref}`Architecture and Core Concepts <building-environments-architecture>` | **Next**: {ref}`Multi-Step Environment <building-environments-multi-step>`
