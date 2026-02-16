(building-environments-multi-step)=

# Multi-Step Environment

This environment requires the agent to make multiple tool calls in sequence and extract specific values. The reward function compares the agent's submitted values against ground truth.

---

## What You'll Build

An environment with two tools: `get_synonym_value` (lookup) and `extract_synonym_values` (submit). The agent must look up values for each synonym, then submit the complete list. Reward is 1.0 for exact match, 0.0 otherwise.

### Episode Flow

```text
Goal (what the agent is learning)
  - Learn a multi-step tool workflow: call the right tool(s), carry values forward, and submit them in the required format.
  - It is *not* "learning ASCII math" as a capability. The ASCII-sum is just a deterministic placeholder tool so we can grade behavior reliably.

Inputs (from one JSONL row)
  - expected_synonyms:         ["Warm", "Blazing", ...]
  - expected_synonym_values:   [407, 711, ...]          # ground truth for grading
  - minefield_label/value:     ("Hot", 299)             # optional failure-mode tracking

What does synonym_value mean?
  - `synonym_value` is the numeric output returned by the tool `/get_synonym_value`.
  - In this example implementation, it's computed as the sum of character code points for the synonym string (e.g., "Warm" → 407).

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

---

## Implementation

**File (simplified excerpt, source-aligned): `resources_servers/example_multi_step/app.py`**

```python
# simplified
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
        # Simple deterministic function: sum of character code points
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
        set_overlap = len(set(actual) & set(expected)) / len(expected)

        return ExampleMultiStepVerifyResponse(
            **body.model_dump(),
            reward=float(accuracy),  # 1.0 if correct, 0.0 otherwise
            parsed_synonym_values=actual,
            accuracy=accuracy,
            set_overlap=set_overlap,
            original_term_minefield_hit=body.minefield_label in actual or body.minefield_label_value in actual,
            order_instruction_following_failure=not accuracy and set_overlap == 1.0,
        )


if __name__ == "__main__":
    ExampleMultiStepResourcesServer.run_webserver()
```

### Key Insight

The `verify()` function parses the agent's tool calls from `body.response.output` and computes a reward by comparing against ground truth (`body.expected_synonym_values`). The ground truth fields come from the JSONL dataset row and are passed through the `ExampleMultiStepVerifyRequest`.

---

## Rollout Transcript

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
Model submits final answer via the "submit" tool:
  function_call: extract_synonym_values({"synonym_values": [407, 711]})

[Episode end → grading]

Agent → ResourcesServer: POST /verify (includes the full response trace + ground truth fields)
ResourcesServer:
  - parses the extract_synonym_values(...) arguments → actual=[407, 711]
  - compares to expected_synonym_values from the dataset row
  - returns reward: 1.0 if exact match else 0.0
```

---

> **Previous**: {ref}`Single-Step Environment <building-environments-single-step>` | **Next**: {ref}`Stateful Environment <building-environments-stateful>`
