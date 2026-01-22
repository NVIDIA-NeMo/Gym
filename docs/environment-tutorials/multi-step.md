(env-multi-step)=
# Multi-Step Environments

Build training environments where models make sequential tool calls, using results from previous calls to inform the next action.

:::{card}

**Goal**: Build a multi-step environment where models make sequential tool calls.

^^^

**In this tutorial, you will**:

1. Understand how multi-step tool-calling loops work
2. Define tools that models can call sequentially
3. Manage state across tool calls
4. Verify multi-step rollouts

:::

:::{button-ref} llm-as-judge
:color: secondary
:outline:
:ref-type: doc

← Previous: LLM-as-a-Judge Verification
:::

---

## Key Concepts

Before starting, understand these NeMo Gym terms:

| Term | Definition |
|------|------------|
| **Rollout** | A complete task execution: prompt → tool calls → verification → reward |
| **Resources Server** | A FastAPI server that provides tools the model can call |
| **Agent** | Orchestrates the loop: calls the model, routes tool calls to the resources server, collects results |
| **Session** | Isolated state for one rollout, identified by a unique session ID |

### How It Works

```text
┌─────────────────────────────────────────────────────────────┐
│                         Agent                               │
│  (orchestrates the loop)                                    │
└─────────────────────────────────────────────────────────────┘
         │                                    │
         │ 1. Send prompt                     │ 3. Route tool calls
         ▼                                    ▼
┌─────────────────┐                  ┌─────────────────┐
│  Model Server   │                  │Resources Server │
│  (LLM inference)│                  │  (your tools)   │
└─────────────────┘                  └─────────────────┘
         │                                    │
         │ 2. Return tool calls               │ 4. Return results
         └────────────────────────────────────┘
```

The agent repeats steps 1-4 until the model stops calling tools or hits `max_steps`.

---

## Multi-Step vs Single-Step

In single-step environments, the model makes one tool call and receives a reward. Multi-step environments allow **sequential tool calls**, where each result informs the next action.

```text
Model → Tool Call 1 → Result 1 → Tool Call 2 → Result 2 → ... → Final Answer
```

Example workflow from `example_multi_step`:

1. Model calls `get_synonym_value("Blazing")` → returns `711`
2. Model calls `get_synonym_value("Warm")` → returns `407`
3. Model calls `extract_synonym_values([711, 407])` → submits final answer

The agent loop exits when:

| Condition | Behavior |
|-----------|----------|
| Model returns message without tool calls | Loop exits normally |
| `max_steps` limit reached | Loop exits with partial rollout |
| Model response reports `max_output_tokens` | Loop exits early |

---

## Quick Start

Run the built-in multi-step example.

### 1. Start Servers

```bash
export OPENAI_API_KEY="your-api-key"  # pragma: allowlist secret

config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_multi_step/configs/example_multi_step.yaml"

ng_run "+config_paths=[$config_paths]"
```

**Expected output:**

```text
INFO:     Starting servers...
INFO:     example_multi_step_resources_server running on http://0.0.0.0:8001
INFO:     example_multi_step_simple_agent running on http://0.0.0.0:8002
INFO:     Press Ctrl+C to stop
```

### 2. Collect Rollouts

In another terminal:

```bash
ng_collect_rollouts \
    +agent_name=example_multi_step_simple_agent \
    +input_jsonl_fpath=resources_servers/example_multi_step/data/example.jsonl \
    +output_jsonl_fpath=data/multi_step_rollouts.jsonl
```

**Expected output:**

```text
Processing 5 samples...
Sample 1/5: reward=1.0 (3 tool calls)
Sample 2/5: reward=1.0 (3 tool calls)
...
Complete. Results saved to data/multi_step_rollouts.jsonl
```

### 3. Inspect Rollout Output

Each line in the output JSONL contains the full tool-calling sequence:

```json
{
  "response": {
    "output": [
      {"type": "function_call", "name": "get_synonym_value", "arguments": "{\"synonym\": \"Blazing\"}"},
      {"type": "function_call_output", "output": "{\"synonym_value\": 711}"},
      {"type": "function_call", "name": "get_synonym_value", "arguments": "{\"synonym\": \"Warm\"}"},
      {"type": "function_call_output", "output": "{\"synonym_value\": 407}"},
      {"type": "function_call", "name": "extract_synonym_values", "arguments": "{\"synonym_values\": [711, 407]}"},
      {"type": "function_call_output", "output": "{\"success\": true}"},
      {"type": "message", "content": "[711, 407]"}
    ]
  },
  "reward": 1.0
}
```

---

## Build a Multi-Step Environment

### Core Pattern

A multi-step resources server needs:

1. **Tool endpoints** — Register with `app.post("/tool_name")`
2. **Request/Response schemas** — Pydantic models for each tool
3. **Verify method** — Score the rollout by examining tool calls

:::{dropdown} Complete Example
:icon: code
:open:

```python
# app.py
import json
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


# Request/Response schemas
class GetSynonymValueRequest(BaseModel):
    synonym: str


class GetSynonymValueResponse(BaseModel):
    synonym_value: int


class ExtractSynonymValuesRequest(BaseModel):
    synonym_values: List[int]


class ExtractSynonymValuesResponse(BaseModel):
    success: bool


# Verify request with task-specific fields
class MultiStepVerifyRequest(BaseVerifyRequest):
    expected_synonym_values: List[int]


class MultiStepVerifyResponse(BaseVerifyResponse):
    parsed_synonym_values: List[int]
    accuracy: bool


# Config (can be empty if no custom settings needed)
class MultiStepConfig(BaseResourcesServerConfig):
    pass


class MultiStepResourcesServer(SimpleResourcesServer):
    config: MultiStepConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Register tools as POST endpoints
        # The endpoint name must match the tool name in your data
        app.post("/get_synonym_value")(self.get_synonym_value)
        app.post("/extract_synonym_values")(self.extract_synonym_values)

        return app

    async def get_synonym_value(
        self, body: GetSynonymValueRequest
    ) -> GetSynonymValueResponse:
        """Tool 1: Get numeric value for a synonym."""
        value = sum(map(ord, body.synonym))
        return GetSynonymValueResponse(synonym_value=value)

    async def extract_synonym_values(
        self, body: ExtractSynonymValuesRequest
    ) -> ExtractSynonymValuesResponse:
        """Tool 2: Submit the collected values."""
        return ExtractSynonymValuesResponse(success=True)

    async def verify(
        self, body: MultiStepVerifyRequest
    ) -> MultiStepVerifyResponse:
        """Score the rollout by comparing submitted vs expected values."""
        expected = body.expected_synonym_values

        # Extract values from the final extract_synonym_values call
        actual = []
        for output in reversed(body.response.output):
            if (output.type == "function_call"
                and output.name == "extract_synonym_values"):
                actual = json.loads(output.arguments)["synonym_values"]
                break

        accuracy = expected == actual
        return MultiStepVerifyResponse(
            **body.model_dump(),
            reward=float(accuracy),
            parsed_synonym_values=actual,
            accuracy=accuracy,
        )


if __name__ == "__main__":
    MultiStepResourcesServer.run_webserver()
```

:::

Reference: `resources_servers/example_multi_step/app.py`

### Error Handling

Tool responses are **passed verbatim** to the model—design your schema to include error information:

```python
from typing import Optional
from pydantic import BaseModel

class ToolResponse(BaseModel):
    result: Optional[str] = None
    error: Optional[str] = None

async def my_tool(self, body: ToolRequest) -> ToolResponse:
    try:
        result = do_something(body.value)
        return ToolResponse(result=result)
    except ValueError as e:
        return ToolResponse(error=str(e))  # Model sees error, may retry
```

**Behavior**: Unhandled exceptions return the HTTP error to the model; the rollout continues.

---

## State Management

For tools that track **state across calls** (counters, accumulated results, game state), use **session IDs** to isolate each rollout.

### Pattern: Session State Dictionary

Use a dictionary keyed by session ID to store per-rollout state:

```python
from nemo_gym.server_utils import SESSION_ID_KEY

class StatefulServer(SimpleResourcesServer):
    session_id_to_state: Dict[str, Any] = Field(default_factory=dict)

    async def seed_session(self, request: Request, body: SeedRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_state.setdefault(session_id, body.initial_value)
        return BaseSeedSessionResponse()

    async def my_tool(self, request: Request, body: ToolRequest) -> ToolResponse:
        session_id = request.session[SESSION_ID_KEY]
        # Read/modify self.session_id_to_state[session_id]
        ...
```

:::{dropdown} Complete Stateful Counter Example
:icon: code

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
from nemo_gym.server_utils import SESSION_ID_KEY


class StatefulCounterResourcesServerConfig(BaseResourcesServerConfig):
    pass


class IncrementCounterRequest(BaseModel):
    count: int


class IncrementCounterResponse(BaseModel):
    success: bool


class GetCounterValueResponse(BaseModel):
    count: int


class StatefulCounterVerifyRequest(BaseVerifyRequest):
    expected_count: int


class StatefulCounterSeedSessionRequest(BaseSeedSessionRequest):
    initial_count: int


class StatefulCounterResourcesServer(SimpleResourcesServer):
    config: StatefulCounterResourcesServerConfig
    session_id_to_counter: Dict[str, int] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/increment_counter")(self.increment_counter)
        app.post("/get_counter_value")(self.get_counter_value)
        return app

    async def seed_session(
        self, request: Request, body: StatefulCounterSeedSessionRequest
    ) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_counter.setdefault(session_id, body.initial_count)
        return BaseSeedSessionResponse()

    async def increment_counter(
        self, request: Request, body: IncrementCounterRequest
    ) -> IncrementCounterResponse:
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_counter[session_id] += body.count
        return IncrementCounterResponse(success=True)

    async def get_counter_value(self, request: Request) -> GetCounterValueResponse:
        session_id = request.session[SESSION_ID_KEY]
        return GetCounterValueResponse(count=self.session_id_to_counter.get(session_id, 0))

    async def verify(
        self, request: Request, body: StatefulCounterVerifyRequest
    ) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        actual = self.session_id_to_counter.get(session_id, 0)
        reward = float(body.expected_count == actual)
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    StatefulCounterResourcesServer.run_webserver()
```

:::

Reference: `resources_servers/example_session_state_mgmt/app.py`

Key points:

- Define a custom `SeedSessionRequest` with initialization fields (e.g., `initial_count`)
- Override `seed_session` to initialize per-session state using `setdefault`
- Access `request.session[SESSION_ID_KEY]` in any method to get the session ID
- State is isolated between concurrent rollouts

---

## Data Format

### Required Fields

Every task record must include `responses_create_params` with:

- **`input`** — Messages (system prompt + user query)
- **`tools`** — Tool definitions (JSON Schema format)
- **Task-specific fields** — Ground truth for verification (e.g., `expected_synonym_values`)

:::{dropdown} Full Data Format Example
:icon: file-code

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "You are an extraction agent..."},
      {"role": "user", "content": "Get synonym values for: Blazing, Warm"}
    ],
    "tools": [
      {
        "type": "function",
        "name": "get_synonym_value",
        "description": "Get the numeric value for a synonym.",
        "parameters": {
          "type": "object",
          "properties": {
            "synonym": {"type": "string", "description": "The synonym"}
          },
          "required": ["synonym"],
          "additionalProperties": false
        },
        "strict": true
      },
      {
        "type": "function",
        "name": "extract_synonym_values",
        "description": "Submit the collected synonym values.",
        "parameters": {
          "type": "object",
          "properties": {
            "synonym_values": {
              "type": "array",
              "items": {"type": "integer"},
              "description": "The values to submit"
            }
          },
          "required": ["synonym_values"],
          "additionalProperties": false
        },
        "strict": true
      }
    ],
    "parallel_tool_calls": false
  },
  "expected_synonym_values": [711, 407]
}
```

:::

### Tool Definition Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"function"` | Always `"function"` |
| `name` | string | Must match endpoint: `app.post("/{name}")` |
| `description` | string | Describes what the tool does |
| `parameters` | object | JSON Schema for arguments |
| `strict` | bool | Enable strict schema validation |

Reference: `resources_servers/example_multi_step/data/example.jsonl`

---

## Configuration

### Agent Configuration

```yaml
# configs/my_multi_step.yaml
my_resources_server:
  resources_servers:
    my_multi_step:
      entrypoint: app.py

my_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      max_steps: 10  # Recommended: set a limit
```

### Agent Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_steps` | int | `None` | Max tool-calling iterations. Set this in production. |

:::{warning}
**Production recommendation**: Always set `max_steps` to prevent infinite loops. A typical value is 10-50 depending on task complexity.
:::

Reference: `responses_api_agents/simple_agent/app.py:43-46`

---

## Verification Strategies

:::::{tab-set}

::::{tab-item} Exact Match

Verify the final tool call contains exactly the expected values:

```python
import json

async def verify(self, body: VerifyRequest) -> VerifyResponse:
    expected = body.expected_values

    # Find the final submission tool call
    actual = []
    for output in reversed(body.response.output):
        if output.type == "function_call" and output.name == "submit":
            actual = json.loads(output.arguments)["values"]
            break

    correct = expected == actual
    return VerifyResponse(**body.model_dump(), reward=float(correct))
```

::::

::::{tab-item} Partial Credit

Award partial credit based on overlap:

```python
import json

async def verify(self, body: VerifyRequest) -> VerifyResponse:
    expected = set(body.expected_values)

    # Extract submitted values
    actual = set()
    for output in body.response.output:
        if output.type == "function_call" and output.name == "submit":
            actual = set(json.loads(output.arguments)["values"])

    if not expected:
        reward = 0.0
    else:
        reward = len(actual & expected) / len(expected)

    return VerifyResponse(**body.model_dump(), reward=reward)
```

::::

::::{tab-item} State-Based

Verify based on final environment state:

```python
async def verify(
    self, request: Request, body: VerifyRequest
) -> VerifyResponse:
    session_id = request.session[SESSION_ID_KEY]

    # Check final state matches expected
    final_count = self.session_id_to_counter.get(session_id, 0)
    reward = float(body.expected_count == final_count)

    return VerifyResponse(**body.model_dump(), reward=reward)
```

Reference: `resources_servers/example_session_state_mgmt/app.py:91-99`

::::

:::::

---

## Examples

| Example | Description | Location |
|---------|-------------|----------|
| **Basic multi-step** | Synonym extraction with 2 tools | `resources_servers/example_multi_step/` |
| **Stateful counter** | Session state management | `resources_servers/example_session_state_mgmt/` |
| **Workplace assistant** | 26 tools, 690 tasks | `resources_servers/workplace_assistant/` |

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| `Session not initialized` | Tool called before `seed_session` | Ensure agent calls `/seed_session` first |
| `max_steps reached` with low reward | Model making extra tool calls | Improve prompt clarity or increase `max_steps` |
| `KeyError` in session state | Session ID not found | Verify `seed_session` initializes the key |
| Tool returns error string | Unhandled exception | Add try/except, return error in response schema |

---

## Next Steps

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`comment-discussion;1.5em;sd-mr-1` Multi-Turn Environments
:link: multi-turn
:link-type: doc
Add conversation history for dialogue training.
:::

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` LLM-as-a-Judge
:link: llm-as-judge
:link-type: doc
Use LLMs for flexible verification.
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Train with NeMo RL
:link: training-nemo-rl-grpo-index
:link-type: ref
Start training on your environment.
:::

::::
