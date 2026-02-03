# Building RL Environments and Resource Servers in NeMo Gym

A comprehensive guide to understanding and building RL environments for LLM training using NVIDIA NeMo Gym.

---

## Table of Contents

1. [Overview: What is NeMo Gym?](#overview-what-is-nemo-gym)
2. [Architecture: The Three-Tier System](#architecture-the-three-tier-system)
3. [Core Concepts](#core-concepts)
4. [Building Your First Environment](#building-your-first-environment)
   - [Example 1: Simple Single-Tool Environment](#example-1-simple-single-tool-environment-weather)
   - [Example 2: Multi-Step Tool Calling Environment](#example-2-multi-step-tool-calling-environment)
   - [Example 3: Stateful Environment with Session Management](#example-3-stateful-environment-with-session-management)
   - [Example 4: Real-World Environment (Workplace Assistant)](#example-4-real-world-environment-workplace-assistant)
5. [Configuration: YAML Files](#configuration-yaml-files)
6. [Training Data Format: JSONL](#training-data-format-jsonl)
7. [The Server Infrastructure](#the-server-infrastructure)
8. [The RL Training Loop](#the-rl-training-loop)
9. [Directory Structure for a New Environment](#directory-structure-for-a-new-environment)
10. [Summary](#summary-building-your-environment)

---

## Overview: What is NeMo Gym?

NeMo Gym is a framework for building RL (Reinforcement Learning) environments designed specifically for training Large Language Models (LLMs). It provides a microservices-based architecture where environments (called **Resource Servers**) interact with LLM agents through well-defined HTTP APIs.

**Key characteristics:**
- **Microservices architecture**: Each component runs as an independent HTTP server
- **OpenAI-compatible APIs**: Uses the Responses API format for tool calling
- **Scalable**: Components can be distributed across machines
- **Composable**: Mix and match environments, agents, and models

### A Note on Terminology: Environments vs Resource Servers

In NeMo Gym, **"Resource Server" and "Environment" are the same thing**. The codebase uses "Resource Server" to emphasize the microservices architecture, but functionally these serve the same role as environments in traditional RL frameworks.

| Traditional RL (OpenAI Gym) | NeMo Gym |
|----------------------------|----------|
| Environment (Python object) | Resource Server (HTTP microservice) |
| `env.step(action)` | `POST /{tool_endpoint}` |
| `env.reset()` | `POST /seed_session` |
| Reward from `step()` | Reward from `POST /verify` |

This naming is reflected in the codebase:
- The base class is `SimpleResourcesServer` (see `nemo_gym/base_resources_server.py:55-73`)
- All environments live in `resources_servers/` directory
- Config types use `ResourcesServerTypeConfig` (see `nemo_gym/config_types.py:449-454`)

Throughout this guide, "environment" and "resource server" are used interchangeably.

---

## Architecture: The Three-Tier System

NeMo Gym uses a three-tier microservices architecture:

```
┌─────────────────────────────────────────────────────────────┐
│              HeadServer (Port 11000)                        │
│   - Global configuration distribution                       │
│   - Server discovery and registry                           │
└─────────────────────────────────────────────────────────────┘
              ↓                    ↓                    ↓
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│ ResponsesAPIModel    │ │ ResponsesAPIAgent    │ │ ResourcesServer      │
│ (LLM inference)      │ │ (Orchestrator)       │ │ (Your Environment)   │
│                      │ │                      │ │                      │
│ /v1/chat/completions │ │ /v1/responses        │ │ /verify              │
│ /v1/responses        │ │ /run                 │ │ /seed_session        │
│                      │ │                      │ │ /{custom_tools}      │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
```

### Component Responsibilities

| Component | Purpose | Key Endpoints |
|-----------|---------|---------------|
| **HeadServer** | Central registry, config distribution | `/global_config_dict_yaml`, `/server_instances` |
| **ResponsesAPIModel** | LLM inference (OpenAI, vLLM, etc.) | `/v1/chat/completions`, `/v1/responses` |
| **ResponsesAPIAgent** | Orchestrates rollouts | `/v1/responses`, `/run` |
| **ResourcesServer** | Your RL environment | `/seed_session`, `/verify`, `/{tools}` |

**Key insight**: Unlike traditional RL environments (like OpenAI Gym), NeMo Gym environments are HTTP servers that can be distributed, scaled, and composed independently.

---

## Core Concepts

### 1. The Resource Server Lifecycle

Every RL episode in NeMo Gym follows this lifecycle:

```
1. seed_session() → Initialize environment state for a task. Called once at the start of each episode (triggered by new task or rollout step)
         ↓
2. [Tool Calls]   → Agent interacts with environment via custom endpoints. This happens repeatedly over the course of the rollout.
         ↓
3. verify()       → Evaluate final response, compute reward. Called once at the end of rollout step.
```

### 2. The Base Classes

The foundation of every Resource Server is `SimpleResourcesServer` from `nemo_gym/base_resources_server.py`:

```python
from abc import abstractmethod
from fastapi import FastAPI
from pydantic import BaseModel
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import BaseRunServerInstanceConfig, BaseServer, SimpleServer


class BaseRunRequest(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class BaseVerifyRequest(BaseRunRequest):
    response: NeMoGymResponse  # The model's full response


class BaseVerifyResponse(BaseVerifyRequest):
    reward: float  # The RL reward signal (typically 0.0-1.0)


class BaseSeedSessionRequest(BaseModel):
    pass  # Extend for custom initialization data


class BaseSeedSessionResponse(BaseModel):
    pass  # Extend for custom initialization responses


class SimpleResourcesServer(BaseResourcesServer, SimpleServer):
    config: BaseResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_session_middleware(app)
        app.post("/seed_session")(self.seed_session)
        app.post("/verify")(self.verify)
        return app

    async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        return BaseSeedSessionResponse()

    @abstractmethod
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        pass  # YOU MUST IMPLEMENT THIS
```

### 3. Key Abstractions

| Class | Purpose |
|-------|---------|
| `SimpleResourcesServer` | Base class for all environments |
| `BaseVerifyRequest` | Contains the model's response for evaluation |
| `BaseVerifyResponse` | Must include `reward: float` |
| `BaseSeedSessionRequest/Response` | Initialize episode state |
| `SESSION_ID_KEY` | Unique identifier for stateful sessions |

---

## Building Your First Environment

### Example 1: Simple Single-Tool Environment (Weather)

This is the simplest possible environment - a stateless tool that always succeeds.

**Visual: one agent rollout (one episode), end-to-end**

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

**File: `resources_servers/example_single_tool_call/app.py`**

```python
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

**Key points:**
- Inherit from `SimpleResourcesServer`
- Override `setup_webserver()` to add custom tool endpoints
- Implement `verify()` to compute the reward

**Illustrative rollout transcript**

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

### Example 2: Multi-Step Tool Calling Environment

This environment requires the agent to make multiple tool calls in sequence and extract specific values.

**Visual: one agent rollout (one episode), end-to-end**

```text
Goal (what the agent is learning)
  - Learn a multi-step tool workflow: call the right tool(s), carry values forward, and submit them in the required format.
  - It is *not* “learning ASCII math” as a capability. The ASCII-sum is just a deterministic placeholder tool so we can grade behavior reliably.

Inputs (from one JSONL row)
  - expected_synonyms:         ["Warm", "Blazing", ...]
  - expected_synonym_values:   [407, 711, ...]          # ground truth for grading
  - minefield_label/value:     ("Hot", 299)             # optional failure-mode tracking

What does synonym_value mean?
  - `synonym_value` is the numeric output returned by the tool `/get_synonym_value`.
  - In this example implementation, it’s computed as the sum of character code points for the synonym string (e.g., "Warm" → 407).

┌───────────────────────────── ResponsesAPIAgent (/run) ─────────────────────────────┐
│                                                                                    │
│  1) Initialize episode state                                                       │
│     POST ResourcesServer /seed_session                                              │
│                                                                                    │
│  2) Interaction loop (repeat up to max_steps)                                       │
│     POST ModelServer /v1/responses                                                  │
│       ├─ if output contains text: keep it in the conversation                       │
│       └─ if output contains function_call(name=TOOL, arguments=...):                │
│              POST ResourcesServer /{TOOL}                                           │
│                - /get_synonym_value(synonym="Warm")   → synonym_value=407           │
│                - /get_synonym_value(synonym="Blazing")→ synonym_value=711           │
│              append tool result back into the conversation                          │
│                                                                                    │
│     (agent eventually submits)                                                      │
│       function_call: extract_synonym_values(synonym_values=[407, 711, ...])         │
│                                                                                    │
│  3) Grade the rollout (reward)                                                     │
│     POST ResourcesServer /verify                                                    │
│       - parse the final extract_synonym_values(...) arguments from the response     │
│       - compare to expected_synonym_values                                          │
│       - reward = 1.0 if exact match else 0.0 (plus extra metrics like minefields)   │
└────────────────────────────────────────────────────────────────────────────────────┘
```

**Illustrative rollout transcript (what it “looks like” end-to-end)**

Below is a simplified, human-readable trace of one episode. In reality, the agent and model exchange OpenAI Responses-style messages, but the logic maps cleanly to this:

```text
[Episode start]

Agent → ResourcesServer: POST /seed_session
  (environment is initialized for this episode)

User: "For the synonyms ['Warm', 'Blazing'], look up each synonym_value and then submit the list."

Agent → ModelServer: POST /v1/responses (tools available: get_synonym_value, extract_synonym_values)
Model decides to call a tool:
  function_call: get_synonym_value({"synonym": "Warm"})

Agent → ResourcesServer: POST /get_synonym_value {"synonym": "Warm"}
ResourcesServer → Agent:
  {"synonym_value": 407}

Agent → ModelServer: POST /v1/responses (now includes tool output 407)
Model calls next tool:
  function_call: get_synonym_value({"synonym": "Blazing"})

Agent → ResourcesServer: POST /get_synonym_value {"synonym": "Blazing"}
ResourcesServer → Agent:
  {"synonym_value": 711}

Agent → ModelServer: POST /v1/responses (now includes tool output 711)
Model submits final answer via the “submit” tool:
  function_call: extract_synonym_values({"synonym_values": [407, 711]})

[Episode end → grading]

Agent → ResourcesServer: POST /verify (includes the full response trace + ground truth fields)
ResourcesServer:
  - parses the extract_synonym_values(...) arguments → actual=[407, 711]
  - compares to expected_synonym_values from the dataset row
  - returns reward: 1.0 if exact match else 0.0
```

**File: `resources_servers/example_multi_step/app.py`**

```python
import json
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class ExampleMultiStepResourcesServerConfig(BaseResourcesServerConfig):
    pass


# Custom request types with task-specific metadata
class ExampleMultiStepRunRequest(BaseRunRequest):
    id: int
    expected_synonym_values: List[int]
    expected_synonyms: List[str]
    minefield_label: str
    minefield_label_value: int


class ExampleMultiStepVerifyRequest(ExampleMultiStepRunRequest, BaseVerifyRequest):
    pass


# Extended verify response with detailed metrics
class ExampleMultiStepVerifyResponse(BaseVerifyResponse):
    parsed_synonym_values: List[int]
    accuracy: bool
    set_overlap: float
    original_term_minefield_hit: bool
    order_instruction_following_failure: bool


# Tool request/response models
class GetSynonymValueRequest(BaseModel):
    synonym: str


class GetSynonymValueResponse(BaseModel):
    synonym_value: int


class ExtractSynonymValuesRequest(BaseModel):
    synonym_values: List[int]


class ExtractSynonymValuesResponse(BaseModel):
    success: bool


class ExampleMultiStepResourcesServer(SimpleResourcesServer):
    config: ExampleMultiStepResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Register multiple tool endpoints
        app.post("/get_synonym_value")(self.get_synonym_value)
        app.post("/extract_synonym_values")(self.extract_synonym_values)

        return app

    # Tool 1: Get the numeric value for a synonym
    async def get_synonym_value(self, body: GetSynonymValueRequest) -> GetSynonymValueResponse:
        # Simple deterministic function: sum of ASCII values
        return GetSynonymValueResponse(synonym_value=sum(map(ord, body.synonym)))

    # Tool 2: Extract/submit the final answer
    async def extract_synonym_values(
        self, body: ExtractSynonymValuesRequest
    ) -> ExtractSynonymValuesResponse:
        return ExtractSynonymValuesResponse(success=True)

    # THE REWARD FUNCTION - This is where RL magic happens
    async def verify(
        self, body: ExampleMultiStepVerifyRequest
    ) -> ExampleMultiStepVerifyResponse:
        expected = body.expected_synonym_values # Pulls the ground truth

        # Parse the agent's final answer from its response
        actual = []
        for output in reversed(body.response.output):
            if output.type == "function_call" and output.name == "extract_synonym_values":
                actual = json.loads(output.arguments)["synonym_values"]
                break

        # Compute reward based on exact match
        accuracy = expected == actual
        set_overlap = len(set(actual) & set(expected)) / len(expected) if expected else 0.0

        return ExampleMultiStepVerifyResponse(
            **body.model_dump(),
            reward=float(accuracy),  # 1.0 if correct, 0.0 otherwise
            parsed_synonym_values=actual,
            accuracy=accuracy,
            set_overlap=set_overlap,
            original_term_minefield_hit=body.minefield_label in actual,
            order_instruction_following_failure=not accuracy and set_overlap == 1.0,
        )


if __name__ == "__main__":
    ExampleMultiStepResourcesServer.run_webserver()
```

**Key insight**: The `verify()` function parses the agent's tool calls from `body.response.output` and computes a reward by comparing against ground truth (`body.expected_synonym_values`).

---

### Example 3: Stateful Environment with Session Management

For environments that need to maintain state across multiple tool calls within an episode:

**Visual: one agent rollout (one episode), end-to-end**

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

**Illustrative rollout transcript**

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

**File: `resources_servers/example_session_state_mgmt/app.py`**

```python
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


class BaseVerifyResponse(BaseVerifyRequest):
    reward: float


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

**Key pattern**: Use `SESSION_ID_KEY` from request session middleware to maintain per-episode state. The session ID is automatically assigned by the framework's middleware.

---

### Example 4: Real-World Environment (Workplace Assistant)

A more advanced environment with multiple tools in a realistic office setting:

**Visual: one agent rollout (one episode), end-to-end**

```text
Goal (what the agent is learning)
  - Learn realistic multi-step tool calling workflows (search → decide → act) with persistent per-episode state.

Inputs
  - user instruction + tool schemas (email/calendar/analytics/...)
  - ground truth calls (or other grading metadata) for verify()

Flow (state is stored per session_id inside the ResourcesServer)
  1) POST ResourcesServer /seed_session
     - initializes toolkits + in-memory data for this session_id
  2) POST ModelServer /v1/responses
     - model emits one or more function_call tool invocations (e.g., email_search_emails, email_reply_email)
  3) POST ResourcesServer /{tool_name}
     - executes the tool against the session’s state and returns output/errors
     - agent appends tool outputs back into the conversation
  4) POST ResourcesServer /verify
     - parses the full trace and grades outcome (often state-based), returning reward ∈ [0, 1]
```


**Illustrative rollout transcript**

```text
[Episode start]

Agent → ResourcesServer: POST /seed_session
  (ResourcesServer initializes a fresh in-memory “workbench” for this session_id:
   email/calendar/analytics/project_management/crm toolkits + their data)

User: "Reply to Carlos's last email about 'Task Update' with 'Thanks, I'll follow up tomorrow.'"

Agent → ModelServer: POST /v1/responses (many tools available)
Model calls tools to reach the goal (one possible path):
  function_call: email_search_emails({"query": "carlos Task Update"})

Agent → ResourcesServer: POST /email_search_emails {"query": "carlos Task Update"}
ResourcesServer → Agent:
  {"emails": [...], "pagination": {...}}

Agent → ModelServer: POST /v1/responses (now includes search results)
Model calls:
  function_call: email_reply_email({"email_id": "00000057", "body": "Thanks, I'll follow up tomorrow."})

Agent → ResourcesServer: POST /email_reply_email {"email_id": "00000057", "body": "..."}
ResourcesServer → Agent:
  "Email replied successfully."

[Episode end → grading]

Agent → ResourcesServer: POST /verify (includes full trace + ground truth calls for this task)
ResourcesServer:
  - extracts predicted function calls from the trace
  - compares against ground truth using deterministic checks (Workplace Assistant uses state-based verification)
  - returns reward 1.0 or 0.0
```

**File: `resources_servers/workplace_assistant/app.py`**

```python
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.workplace_assistant.utils import get_tools, is_correct


class WorkbenchResourcesServerConfig(BaseResourcesServerConfig):
    pass


class WorkbenchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class WorkbenchResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class WorkbenchVerifyRequest(BaseVerifyRequest):
    ground_truth: list[Dict[str, str]] | str
    id: int
    category: str
    environment_name: str


class WorkbenchVerifyResponse(BaseVerifyResponse):
    pass


class WorkbenchResourcesServer(SimpleResourcesServer):
    config: WorkbenchResourcesServerConfig
    session_id_to_tool_env: Dict[str, Any] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # Dynamic routing: any path becomes a tool call
        app.post("/{path}")(self.route_to_python_function)
        return app

    async def seed_session(
        self,
        request: Request,
        body: BaseSeedSessionRequest
    ) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]

        # Initialize multiple toolkits for this session
        toolkits = [
            "email",
            "calendar",
            "analytics",
            "project_management",
            "customer_relationship_manager",
        ]
        self.session_id_to_tool_env[session_id] = get_tools(toolkits)
        return BaseSeedSessionResponse()

    # Generic tool router - dispatches to Python functions dynamically
    async def route_to_python_function(
        self,
        path: str,
        body: WorkbenchRequest,
        request: Request
    ) -> WorkbenchResponse:
        session_id = request.session[SESSION_ID_KEY]

        if session_id not in self.session_id_to_tool_env:
            raise HTTPException(
                status_code=400,
                detail="Session not initialized. Please call seed_session first.",
            )

        tool_env = self.session_id_to_tool_env[session_id]
        args = {k: v for k, v in body.model_dump(exclude_unset=True).items() if v is not None}

        try:
            function = tool_env["functions"][path]
            result = function(**args)
            return WorkbenchResponse(output=result)
        except Exception as e:
            # Return error to model so it can self-correct
            return WorkbenchResponse(output=f"Error executing tool '{path}': {str(e)}")

    async def verify(self, body: WorkbenchVerifyRequest) -> WorkbenchVerifyResponse:
        ground_truth = body.ground_truth
        response = body.response.output

        # Extract function calls from response
        predicted_function_calls = [
            message.model_dump()
            for message in response
            if message.type == "function_call"
        ]

        # Compute reward using custom evaluation function
        total_score = is_correct(predicted_function_calls, ground_truth, None) * 1.0
        return WorkbenchVerifyResponse(**body.model_dump(), reward=total_score)


if __name__ == "__main__":
    WorkbenchResourcesServer.run_webserver()
```

**Key pattern**: Dynamic routing with `/{path}` allows the environment to expose an arbitrary number of tools without hardcoding each endpoint.

**Verification: trajectory matching vs state matching (and what Workplace Assistant uses)**

There are two common ways to grade tool-using agents:

1. **Trajectory matching (sequence matching)**: compare the *exact* tool call sequence (names + arguments, sometimes order) against a reference trajectory.
   - **Pros**: simple to implement; easy to debug.
   - **Cons**: brittle—penalizes alternative correct paths (different searches, different ordering, equivalent updates).

2. **State matching (outcome matching)**: execute the agent’s predicted calls in a fresh sandbox, execute the ground truth calls in another fresh sandbox, then compare the **final environment state**.
   - **Pros**: rewards correct outcomes even when the path differs; better reflects “did the work get done?”
   - **Cons**: requires you to define what “state” is (tables, files, DB rows, etc.) and how to compare it (case sensitivity, ordering, floating-point tolerance).

**Workplace Assistant uses state matching.** Its `verify()` extracts the predicted function calls, then calls `is_correct(...)`, which:
- Replays predicted calls and ground truth calls separately (fresh tool env each time)
- Compares the final DataFrames for email/calendar/analytics/project management/CRM (mostly case-insensitive)

This choice makes sense because workplace tasks often have **multiple valid tool sequences** that reach the same correct final state.


---

## Configuration: YAML Files

Every environment needs a configuration file that defines:
1. The Resource Server itself
2. The Agent that uses the Resource Server
3. Training/validation datasets

**File: `resources_servers/example_multi_step/configs/example_multi_step.yaml`**

```yaml
# Define the Resource Server
example_multi_step_resources_server:
  resources_servers:
    example_multi_step:
      entrypoint: app.py                    # Entry point script
      domain: instruction_following         # Domain classification
      verified: false                       # Verification status
      description: Multi-step tool calling  # Human-readable description

# Define the Agent that uses this Resource Server
example_multi_step_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: example_multi_step_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
        - name: train
          type: train
          jsonl_fpath: resources_servers/example_multi_step/data/train.jsonl
          gitlab_identifier:
            dataset_name: example_multi_step
            version: 0.0.1
            artifact_fpath: train.jsonl
          license: Apache 2.0
        - name: validation
          type: validation
          jsonl_fpath: resources_servers/example_multi_step/data/validation.jsonl
          gitlab_identifier:
            dataset_name: example_multi_step
            version: 0.0.1
            artifact_fpath: validation.jsonl
          license: Apache 2.0
        - name: example
          type: example
          jsonl_fpath: resources_servers/example_multi_step/data/example.jsonl
```

### Available Domains

From `nemo_gym/config_types.py`:

```python
class Domain(str, Enum):
    MATH = "math"
    CODING = "coding"
    AGENT = "agent"
    KNOWLEDGE = "knowledge"
    INSTRUCTION_FOLLOWING = "instruction_following"
    LONG_CONTEXT = "long_context"
    SAFETY = "safety"
    GAMES = "games"
    TRANSLATION = "translation"
    E2E = "e2e"
    OTHER = "other"
```

### Server References

To link servers together in configuration:

```yaml
# Reference a model server
model_server:
  type: responses_api_models
  name: policy_model

# Reference a resource server
resources_server:
  type: resources_servers
  name: my_resources_server

# Reference an agent server
agent_server:
  type: responses_api_agents
  name: my_agent
```

---

## Training Data Format: JSONL

Training data is stored in JSONL (JSON Lines) format where each line is a complete JSON object representing one training sample.

### Structure

```json
{
  "id": 0,
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "You are an extraction agent..."},
      {"role": "user", "content": "What factors contribute to high temperatures?"}
    ],
    "tools": [
      {
        "type": "function",
        "name": "get_synonym_value",
        "description": "Get the synonym value for a synonym.",
        "parameters": {
          "properties": {"synonym": {"type": "string"}},
          "type": "object",
          "required": ["synonym"],
          "additionalProperties": false
        },
        "strict": true
      },
      {
        "type": "function",
        "name": "extract_synonym_values",
        "description": "Extract the synonym values you retrieved.",
        "parameters": {
          "properties": {
            "synonym_values": {
              "items": {"type": "integer"},
              "type": "array"
            }
          },
          "type": "object",
          "required": ["synonym_values"],
          "additionalProperties": false
        },
        "strict": true
      }
    ],
    "parallel_tool_calls": false
  },
  "expected_synonyms": ["Blazing", "Warm"],
  "expected_synonym_values": [711, 407],
  "minefield_label": "Hot",
  "minefield_label_value": 299
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `id` | Unique sample identifier |
| `responses_create_params` | OpenAI Responses API-compatible input |
| `responses_create_params.input` | Conversation messages (system, user, assistant) |
| `responses_create_params.tools` | Available tools/functions for the agent |
| `expected_*` | Ground truth fields for reward computation in `verify()` |

### Tool Definition Format

Each tool follows the OpenAI function calling schema:

```json
{
  "type": "function",
  "name": "tool_name",
  "description": "What this tool does",
  "parameters": {
    "type": "object",
    "properties": {
      "param1": {"type": "string", "description": "..."},
      "param2": {"type": "integer", "description": "..."}
    },
    "required": ["param1"],
    "additionalProperties": false
  },
  "strict": true
}
```

---

## The Server Infrastructure

### HeadServer: Central Configuration

The HeadServer runs on port 11000 and serves as the central registry:

```python
class HeadServer(BaseServer):
    config: BaseServerConfig
    _server_instances: List[dict] = []

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.get("/global_config_dict_yaml")(self.global_config_dict_yaml)
        app.get("/server_instances")(self.get_server_instances)
        return app

    async def global_config_dict_yaml(self) -> str:
        return OmegaConf.to_yaml(get_global_config_dict())
```

### ServerClient: Inter-Service Communication

```python
class ServerClient(BaseModel):
    head_server_config: BaseServerConfig
    global_config_dict: DictConfig

    async def post(
        self,
        server_name: str,
        url_path: str,
        **kwargs
    ) -> ClientResponse:
        """Make HTTP POST to another server by name"""
        server_config_dict = get_first_server_config_dict(
            self.global_config_dict, server_name
        )
        base_url = f"http://{server_config_dict.host}:{server_config_dict.port}"
        return await request(method="POST", url=f"{base_url}{url_path}", **kwargs)

    async def get(
        self,
        server_name: str,
        url_path: str,
        **kwargs
    ) -> ClientResponse:
        """Make HTTP GET to another server by name"""
        return await self.request(
            server_name=server_name,
            url_path=url_path,
            method="GET",
            **kwargs,
        )

    def poll_for_status(self, server_name: str) -> ServerStatus:
        """Check if a server is running"""
        # Returns: "success", "connection_error", "timeout", or "unknown_error"
```

### SimpleServer: Base Infrastructure

```python
SESSION_ID_KEY = "session_id"

class SimpleServer(BaseServer):
    server_client: ServerClient

    @abstractmethod
    def setup_webserver(self) -> FastAPI:
        pass

    def setup_session_middleware(self, app: FastAPI) -> None:
        """Automatically assigns unique session IDs to each request"""
        @app.middleware("http")
        async def add_session_id(request: Request, call_next):
            if SESSION_ID_KEY not in request.session:
                request.session[SESSION_ID_KEY] = str(uuid4())
            return await call_next(request)

        session_middleware_key = self.get_session_middleware_key()
        app.add_middleware(
            SessionMiddleware,
            secret_key=session_middleware_key,
            session_cookie=session_middleware_key
        )

    def setup_exception_middleware(self, app: FastAPI) -> None:
        """Catches exceptions and returns formatted error responses"""
        @app.middleware("http")
        async def exception_handling_middleware(request: Request, call_next):
            try:
                return await call_next(request)
            except Exception as e:
                return JSONResponse(content=repr(e), status_code=500)

    @classmethod
    def run_webserver(cls) -> FastAPI:
        """Start the server with uvicorn"""
        server_config = cls.load_config_from_global_config()
        server_client = ServerClient.load_from_global_config()
        server = cls(config=server_config, server_client=server_client)

        app = server.setup_webserver()
        server.setup_exception_middleware(app)

        uvicorn.run(
            app,
            host=server.config.host,
            port=server.config.port,
            timeout_graceful_shutdown=0.5
        )
        return app
```

---

## The RL Training Loop

Here's how a single rollout works end-to-end:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                               │
│    Agent calls POST /seed_session on ResourcesServer            │
│    → Environment initializes state for the task                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. GET INITIAL PROMPT                                           │
│    Agent loads prompt from training data (JSONL)                │
│    → System message + user query + tool definitions             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. INTERACTION LOOP                                             │
│                                                                 │
│    a. Agent calls POST /v1/responses on ModelServer             │
│       → LLM generates text OR function_call                     │
│                                                                 │
│    b. If function_call:                                         │
│       Agent calls POST /{tool_name} on ResourcesServer          │
│       → Environment executes tool, returns result               │
│       → Agent appends tool result to conversation               │
│                                                                 │
│    c. If termination condition met (max turns, done flag):      │
│       → Break loop                                              │
│                                                                 │
│    d. Else: Continue to step (a)                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. VERIFICATION                                                 │
│    Agent calls POST /verify on ResourcesServer                  │
│    → Environment computes reward by comparing response          │
│      to ground truth                                            │
│    → Returns reward: float for RL gradient computation          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. TRAINING                                                     │
│    Training pipeline uses (response, reward) pairs              │
│    → Compute policy gradients                                   │
│    → Update model weights                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Batch Rollout Collection

For efficiency, NeMo Gym supports batch rollout collection:

```python
class RolloutCollectionConfig(BaseNeMoGymCLIConfig):
    """Batch rollout collection from JSONL dataset"""
    agent_name: str
    input_jsonl_fpath: str
    output_jsonl_fpath: str
    limit: Optional[int] = None
    concurrency: int = 1
    dry_run: bool = False
    global_aiohttp_connector_limit: int = 100 * 1024
```

CLI command: `ng_collect_rollouts`

---

## Directory Structure for a New Environment

```
resources_servers/my_environment/
├── app.py                      # Main server implementation
├── client.py                   # Optional: client for testing
├── utils.py                    # Optional: helper functions
├── configs/
│   └── my_environment.yaml     # Configuration
├── data/
│   ├── train.jsonl             # Training data
│   ├── validation.jsonl        # Validation data
│   └── example.jsonl           # Example data for testing
├── tests/
│   └── test_app.py             # Unit tests
└── requirements.txt            # Dependencies (optional)
```

### Minimal app.py Template

```python
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class MyEnvironmentConfig(BaseResourcesServerConfig):
    pass


class MyToolRequest(BaseModel):
    # Define your tool parameters
    param1: str
    param2: int


class MyToolResponse(BaseModel):
    # Define your tool response
    result: str


class MyEnvironmentServer(SimpleResourcesServer):
    config: MyEnvironmentConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/my_tool")(self.my_tool)
        return app

    async def my_tool(self, body: MyToolRequest) -> MyToolResponse:
        # Implement your tool logic
        return MyToolResponse(result=f"Processed {body.param1}")

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # Implement your reward function
        reward = 1.0  # Compute based on body.response
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    MyEnvironmentServer.run_webserver()
```

---

## Summary: Building Your Environment

### Step-by-Step Checklist

1. **Create the directory structure**
   ```bash
   mkdir -p resources_servers/my_env/{configs,data,tests}
   touch resources_servers/my_env/app.py
   touch resources_servers/my_env/configs/my_env.yaml
   ```

2. **Define your request/response models** with Pydantic
   - Tool input parameters
   - Tool output format
   - Extended verify request with ground truth fields
   - Extended verify response with metrics

3. **Implement your Resource Server class**
   - Inherit from `SimpleResourcesServer`
   - Override `setup_webserver()` to add tool endpoints
   - Implement `verify()` with your reward function
   - Optionally override `seed_session()` for stateful environments

4. **Write your configuration YAML**
   - Define resource server with domain and entrypoint
   - Define agent with server references
   - Specify datasets with paths and licenses

5. **Create training data in JSONL format**
   - System prompts with task instructions
   - Tool definitions
   - Ground truth for reward computation

6. **Test your environment**
   ```bash
   python resources_servers/my_env/app.py
   # Or via CLI:
   ng_run +config=resources_servers/my_env/configs/my_env.yaml
   ```

7. **Generate initial data (start small, then scale)**
   - Create a tiny `data/example.jsonl` first (5–20 rows) so you can debug the full loop quickly.
   - Then expand to `data/train.jsonl` / `data/validation.jsonl` once verification is stable.
   - Common approaches:
     - Hand-written seeds (highest quality, lowest scale)
     - Synthetic Data Generation (using LLM) for tasks with automatic checks + spot review
     - Programmatic task generators (best for templated problems)

8. **Collect rollouts (turn tasks into scored trajectories)**
   - Once servers are running, use `ng_collect_rollouts` to produce rollout traces you can inspect and/or use for offline training.
   - The official walkthrough is in `docs/get-started/rollout-collection.md`. A typical command looks like:

   ```bash
   ng_collect_rollouts +agent_name=my_env_simple_agent \
       +input_jsonl_fpath=resources_servers/my_env/data/example.jsonl \
       +output_jsonl_fpath=results/my_env_rollouts.jsonl \
       +limit=100 \
       +num_repeats=4 \
       +num_samples_in_parallel=10
   ```

9. **Train (Gym provides environments; training happens in an RL framework)**
   - NeMo Gym’s job is to run the environment, collect rollouts, and compute rewards.
   - For GRPO training with NeMo RL, see `docs/tutorials/nemo-rl-grpo/index.md` (end-to-end training on Workplace Assistant).
   - If you’re doing offline training from collected rollouts, see `docs/tutorials/offline-training-w-rollouts.md`.

### Key Differentiators from Traditional RL

| Traditional RL | NeMo Gym |
|----------------------------|----------|
| Python function calls | HTTP microservices |
| Single process | Distributed architecture |
| `step()`, `reset()` | `seed_session()`, tool endpoints, `verify()` |
| Observation/action spaces | OpenAI Responses API format |
| In-memory state | Session-based state with `SESSION_ID_KEY` |

The key differentiator is that NeMo Gym environments are HTTP microservices, enabling:
- **Distributed training** across multiple machines
- **Horizontal scaling** of environment instances
- **Composition** of multiple environments in complex training scenarios
- **Language-agnostic** tool implementations (anything that speaks HTTP)

---