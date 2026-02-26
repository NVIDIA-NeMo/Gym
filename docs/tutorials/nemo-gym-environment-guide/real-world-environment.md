(building-environments-real-world)=

# Real-World Environment (Workplace Assistant)

A production environment with multiple tools in a realistic office setting, demonstrating dynamic tool routing and state-based verification.

---

## What You'll Build (Conceptually)

The Workplace Assistant environment simulates an office with email, calendar, analytics, project management, and CRM toolkits. It uses dynamic routing (`/{path}`) to dispatch tool calls to Python functions, and state-based verification to grade outcomes.

### Episode Flow

```text
Goal (what the agent is learning)
  - Learn realistic multi-step tool calling workflows (search → decide → act) with persistent per-episode state.

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
     - parses the full trace and grades outcome (often state-based), returning reward ∈ [0, 1]
```

---

## Implementation

**File (simplified excerpt, source-aligned): `resources_servers/workplace_assistant/app.py`**

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

---

## Rollout Transcript

```text
[Episode start]

Agent → ResourcesServer: POST /seed_session
  (ResourcesServer initializes a fresh in-memory "workbench" for this session_id:
   company_directory + email/calendar/analytics/project_management/crm toolkits + their data)

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

---

## Verification: Trajectory Matching vs State Matching

There are two common ways to grade tool-using agents:

### 1. Trajectory Matching (Sequence Matching)

Compare the *exact* tool call sequence (names + arguments, sometimes order) against a reference trajectory.

- **Pros**: Simple to implement; easy to debug.
- **Cons**: Brittle — penalizes alternative correct paths (different searches, different ordering, equivalent updates).

### 2. State Matching (Outcome Matching)

Execute the agent's predicted calls in a fresh sandbox, execute the ground truth calls in another fresh sandbox, then compare the **final environment state**.

- **Pros**: Rewards correct outcomes even when the path differs; better reflects "did the work get done?"
- **Cons**: Requires you to define what "state" is (tables, files, DB rows, etc.) and how to compare it (case sensitivity, ordering, floating-point tolerance).

### What Workplace Assistant Uses

**Workplace Assistant uses state matching.** Its `verify()` extracts the predicted function calls, then calls `is_correct(...)`, which:
- Replays predicted calls and ground truth calls separately (fresh tool env each time)
- Compares the final DataFrames for email/calendar/analytics/project management/CRM (mostly case-insensitive)

This choice makes sense because workplace tasks often have **multiple valid tool sequences** that reach the same correct final state.

---

> **Previous**: {ref}`Stateful Environment <building-environments-stateful>` | **Next**: {ref}`Configuration and Training Data <building-environments-configuration>`
