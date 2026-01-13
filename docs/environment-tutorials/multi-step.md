(env-multi-step)=
# Multi-Step Environments

```{warning}
This article was generated and has not been reviewed. Content may change.
```

Build training environments with sequential tool calling and intermediate states.

::::{grid} 2
:gutter: 3

:::{grid-item-card} {octicon}`clock;1em;` **Time**
30-45 minutes
:::

:::{grid-item-card} {octicon}`bookmark;1em;` **Prerequisites**

- Completed {doc}`/get-started/detailed-setup`
- Completed {doc}`/tutorials/creating-resource-server`

:::

::::

---

## What is Multi-Step?

Multi-step environments allow models to make multiple tool calls in sequence, using results from previous calls to inform subsequent actions.

```text
Model → Tool Call 1 → Result 1 → Tool Call 2 → Result 2 → ... → Final Answer
```

Example: An agent extracting synonyms might:

1. Call `get_synonym_value("happy")` → returns 532
2. Call `get_synonym_value("joyful")` → returns 645
3. Call `extract_synonym_values([532, 645])` → submits final answer

---

## When to Use Multi-Step

| Pattern | Description | Example |
|---------|-------------|---------|
| **Single-step** | One model response | Math calculation, Q&A |
| **Multi-step** | Sequential tool calls | Data extraction, workflows |
| **Multi-turn** | Back-and-forth dialogue | Customer support |

Use multi-step when:

- Tasks require multiple tool invocations
- Later steps depend on earlier results
- Complex workflows need orchestration
- Information must be gathered incrementally

:::{note}
Multi-step differs from multi-turn: multi-step involves tool-calling loops within a single turn, while multi-turn involves conversation history across user/assistant exchanges.
:::

---

## Quick Start

Run the built-in multi-step example:

### 1. Start Servers

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_multi_step/configs/example_multi_step.yaml"

ng_run "+config_paths=[$config_paths]"
```

### 2. Collect Rollouts

```bash
ng_collect_rollouts \
    +agent_name=example_multi_step_simple_agent \
    +input_jsonl_fpath=resources_servers/example_multi_step/data/example.jsonl \
    +output_jsonl_fpath=data/multi_step_rollouts.jsonl
```

---

## Implementation Pattern

### Defining Tools

Register tools with FastAPI endpoints and define request/response schemas:

```python
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class GetValueRequest(BaseModel):
    item: str


class GetValueResponse(BaseModel):
    value: int


class MultiStepResourcesServer(SimpleResourcesServer):
    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        
        # Register tools as POST endpoints
        app.post("/get_value")(self.get_value)
        app.post("/submit_values")(self.submit_values)
        
        return app

    async def get_value(self, body: GetValueRequest) -> GetValueResponse:
        # Tool logic here
        return GetValueResponse(value=sum(map(ord, body.item)))

    async def submit_values(self, body: SubmitRequest) -> SubmitResponse:
        return SubmitResponse(success=True)
```

### State Management

For tools that need to track state across calls, use session IDs:

```python
from typing import Dict
from fastapi import Request
from pydantic import Field

from nemo_gym.base_resources_server import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY


class StatefulResourcesServer(SimpleResourcesServer):
    # Per-session state storage
    session_state: Dict[str, dict] = Field(default_factory=dict)

    async def seed_session(
        self, request: Request, body: BaseSeedSessionRequest
    ) -> BaseSeedSessionResponse:
        """Initialize state when a new session starts."""
        session_id = request.session[SESSION_ID_KEY]
        self.session_state[session_id] = {"results": []}
        return BaseSeedSessionResponse()

    async def get_value(self, request: Request, body: GetValueRequest) -> GetValueResponse:
        """Tools can read/write session state."""
        session_id = request.session[SESSION_ID_KEY]
        state = self.session_state.get(session_id, {})
        
        value = compute_value(body.item)
        state.setdefault("results", []).append(value)
        
        return GetValueResponse(value=value)
```

### Tool Dependencies

Design tools that build on each other:

```python
# Tool 1: Search returns IDs
async def search(self, body: SearchRequest) -> SearchResponse:
    results = perform_search(body.query)
    return SearchResponse(item_ids=[r.id for r in results])

# Tool 2: Get details using IDs from search
async def get_details(self, body: GetDetailsRequest) -> GetDetailsResponse:
    item = fetch_item(body.item_id)  # Uses ID from search
    return GetDetailsResponse(details=item.details)

# Tool 3: Submit using details
async def submit(self, body: SubmitRequest) -> SubmitResponse:
    return SubmitResponse(success=True)
```

---

## Verification Strategies

:::::{tab-set}

::::{tab-item} Final State Verification

Verify the final tool call contains the correct answer:

```python
import json


async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    expected = body.expected_values
    
    # Extract values from the final relevant tool call
    actual = []
    for output in reversed(body.response.output):
        if output.type == "function_call" and output.name == "submit_values":
            actual = json.loads(output.arguments)["values"]
            break
    
    correct = expected == actual
    return BaseVerifyResponse(**body.model_dump(), reward=float(correct))
```

::::

::::{tab-item} Partial Credit Verification

Award partial credit based on progress:

```python
async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
    expected = set(body.expected_values)
    
    # Extract submitted values
    actual = set(extract_submitted_values(body.response))
    
    # Partial credit for overlap
    if not expected:
        reward = 0.0
    else:
        overlap = len(actual & expected) / len(expected)
        reward = overlap
    
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

::::

:::::

---

## Data Format

### Required Fields

```json
{
  "responses_create_params": {
    "input": [
      {"role": "user", "content": "Your task prompt here"}
    ]
  },
  "expected_values": [532, 645, 789]
}
```

### Example Record

From `example_multi_step`:

```json
{
  "id": 1,
  "responses_create_params": {
    "input": [
      {
        "role": "user", 
        "content": "Get the synonym values for: happy, joyful, glad"
      }
    ]
  },
  "expected_synonym_values": [532, 645, 416],
  "expected_synonyms": ["happy", "joyful", "glad"]
}
```

---

## Configuration Reference

### Agent Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_steps` | int | None | Maximum tool-calling iterations before stopping |
| `done_if_no_tool_calls` | bool | true | End rollout if model stops calling tools |

Example configuration:

```yaml
my_multi_step_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_multi_step_resources
      model_server:
        type: responses_api_models
        name: policy_model
      max_steps: 10
```

---

## Examples

| Example | Description | Location |
|---------|-------------|----------|
| **Basic multi-step** | Synonym value extraction | `resources_servers/example_multi_step/` |
| **Session state** | Stateful counter across calls | `resources_servers/example_session_state_mgmt/` |
| **Workplace assistant** | 26 tools, 690 tasks | `resources_servers/workplace_assistant/` |

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
