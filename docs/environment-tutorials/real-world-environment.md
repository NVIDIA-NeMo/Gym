(env-real-world-environment)=

# Real-World Environment (Workplace Assistant)

This tutorial focuses on the **Resources Server** implementation for a production-scale environment with multiple tools, dynamic routing, and state-based verification. The full workflow — task data preparation, agent/model configuration, rollout collection, and training — follows the same steps as the {doc}`main guide <index>`. What changes here is the scale and complexity of the Resources Server.

:::{button-ref} stateful-environment
:color: secondary
:outline:
:ref-type: doc

< Previous: Stateful Environment
:::

---

## What You'll Build (Conceptually)

The Workplace Assistant environment simulates an office with email, calendar, analytics, project management, and CRM toolkits. It uses dynamic routing (`/{path}`) to dispatch tool calls to Python functions, and state-based verification to grade outcomes.

### Episode Flow

```text
Goal (what the agent is learning)
  - Learn realistic multi-step tool calling workflows (search -> decide -> act) with persistent per-episode state.

Inputs
  - user instruction + tool schemas (company_directory + email/calendar/analytics/...)
  - ground truth calls (or other grading metadata) for verify()

Flow (state is stored per session_id inside the ResourcesServer)
  1) POST ResourcesServer /seed_session
     - initializes toolkits + in-memory data for this session_id
  2) POST ModelServer /v1/responses
     - model emits one or more function_call tool invocations (e.g., email_search_emails, email_reply_email)
  3) POST ResourcesServer /{tool_name}
     - executes the tool against the session's state and returns output/errors
     - agent appends tool outputs back into the conversation
  4) POST ResourcesServer /verify
     - extracts predicted function calls from the response and grades the replayed outcome, returning reward in [0, 1]
```

---

## Implementation

This Resources Server introduces three patterns not seen in the earlier tutorials:

- **Dynamic routing** --- a single `/{path}` catch-all endpoint dispatches to any tool function, so you don't need to register each tool individually.
- **Per-session toolkit initialization** --- `seed_session()` creates an independent set of toolkits and data for each episode, so concurrent rollouts don't interfere.
- **State-based verification** --- `verify()` extracts the agent's function calls, replays them in a fresh environment alongside the ground truth, and compares the resulting state rather than the exact call sequence.

**File (simplified from [`resources_servers/workplace_assistant/app.py`](https://github.com/NVIDIA-NeMo/Gym/tree/main/resources_servers/workplace_assistant/app.py)):**

```python
# simplified
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

### Key Pattern

Dynamic routing with `/{path}` allows the environment to expose an arbitrary number of tools without hardcoding each endpoint. The `route_to_python_function` method dispatches incoming requests to Python functions in the per-session `tool_env["functions"]` dictionary.

:::{warning}
The `/{path}` catch-all route **must be registered after** `super().setup_webserver()`. The parent method registers `/seed_session` and `/verify` --- if your catch-all is registered first, it will intercept those requests and break the server lifecycle.
:::

:::{admonition} What does `get_tools()` return?
:class: note

`get_tools(toolkits)` initializes a dictionary containing:
- `"functions"`: A mapping of tool names (e.g. `"email_search_emails"`) to Python callables
- Per-toolkit in-memory data (DataFrames for emails, calendar events, analytics, etc.)

Each session gets its own independent copy of this state, so tool calls in one episode cannot affect another.
:::

:::{admonition} What does `is_correct()` do?
:class: note

`is_correct(predicted_calls, ground_truth, env)` performs **state-based verification**:
1. Replays the predicted tool calls against a fresh environment
2. Replays the ground-truth calls against another fresh environment
3. Compares five specific mutable DataFrames: `email._emails`, `calendar._calendar_events`, `analytics._plots_data`, `project_management._project_tasks`, and `customer_relationship_manager._crm_data` (mostly case-insensitive)
4. Returns `1.0` if all five match, `0.0` otherwise

Note that read-only state (e.g. `company_directory`) is not compared --- only mutable state that tools can modify. Tool execution errors during replay are caught and skipped rather than treated as immediate failures.

This is more flexible than trajectory matching because it rewards correct outcomes regardless of the specific tool call sequence.
:::

---

## Rollout Transcript

```text
[Episode start]

Agent -> ResourcesServer: POST /seed_session
  (ResourcesServer initializes a fresh in-memory "workbench" for this session_id:
   company_directory + email/calendar/analytics/project_management/crm toolkits + their data)

User: "Reply to Carlos's last email about 'Task Update' with 'Thanks, I'll follow up tomorrow.'"

Agent -> ModelServer: POST /v1/responses (many tools available)
Model calls tools to reach the goal (one possible path):
  function_call: email_search_emails({"query": "carlos Task Update"})

Agent -> ResourcesServer: POST /email_search_emails {"query": "carlos Task Update"}
ResourcesServer -> Agent:
  {"output": {"emails": [...], "pagination": {...}}}

Agent -> ModelServer: POST /v1/responses (now includes search results)
Model calls:
  function_call: email_reply_email({"email_id": "00000057", "body": "Thanks, I'll follow up tomorrow."})

Agent -> ResourcesServer: POST /email_reply_email {"email_id": "00000057", "body": "..."}
ResourcesServer -> Agent:
  {"output": "Email replied successfully."}

[Episode end -> grading]

Agent -> ResourcesServer: POST /verify (includes response + ground truth calls for this task)
ResourcesServer:
  - extracts predicted function calls from the response (ignores text output)
  - replays predicted and ground-truth calls, compares final state
  - returns reward 1.0 or 0.0
```

---

## Verification: Trajectory Matching vs State Matching

There are two common ways to grade tool-using agents:

### 1. Trajectory Matching (Sequence Matching)

Compare the *exact* tool call sequence (names + arguments, sometimes order) against a reference trajectory.

- **Pros**: Simple to implement; easy to debug.
- **Cons**: Brittle --- penalizes alternative correct paths (different searches, different ordering, equivalent updates).

### 2. State Matching (Outcome Matching)

Execute the agent's predicted calls in a fresh sandbox, execute the ground truth calls in another fresh sandbox, then compare the **final environment state**.

- **Pros**: Rewards correct outcomes even when the path differs; better reflects "did the work get done?"
- **Cons**: Requires you to define what "state" is (tables, files, DB rows, etc.) and how to compare it (case sensitivity, ordering, floating-point tolerance).

### What Workplace Assistant Uses

**Workplace Assistant uses state matching.** Its `verify()` extracts only the `function_call` items from the response (text output is ignored for scoring), then calls `is_correct(...)`, which:
- Replays predicted calls and ground truth calls separately (fresh tool env each time)
- Compares five mutable DataFrames (email, calendar, analytics plots, project management tasks, CRM data) mostly case-insensitive

This choice makes sense because workplace tasks often have **multiple valid tool sequences** that reach the same correct final state.

:::{tip}
For a deeper dive into verification strategies, reward shaping, and common pitfalls, see {ref}`task-verification`.
:::

---

:::{button-ref} index
:color: secondary
:outline:
:ref-type: doc

< Back to Building Environments
:::
