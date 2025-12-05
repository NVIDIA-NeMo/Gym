(tutorial-counter-game-server)=

# Build a Counter Game Server

In {doc}`Build a Data Extraction Server <multi-step-interactions>`, tools were stateless. Now you'll build a counter game where tools modify shared state that persists across calls within a session.

:::{card}

**Goal**: Build a stateful counter server where agents must reach a target number.

^^^

**In this tutorial, you will**:

1. Implement session state that persists across tool calls
2. Create `increment`, `decrement`, and `get_count` tools
3. Verify based on whether the agent reaches the target
4. Walk away with a pattern for stateful agent interactions

:::

:::{button-ref} /tutorials/resource-servers/index
:color: secondary
:outline:
:ref-type: doc

← Resource Server Patterns
:::

---

## Before You Begin

Make sure you have:

- ✅ Completed {doc}`multi-step-interactions`
- ✅ Understanding of session management concepts
- ✅ NeMo Gym installed and working

**What you'll build**: A counter game where the agent starts at 0 and must reach a target number (e.g., 5) using increment and decrement operations.

:::{tip}
**Reference implementation**: `resources_servers/example_stateful_counter/`
:::

---

## 1. Understand What We're Building

The counter game has persistent state:

1. User says: "Reach the number 5"
2. Agent calls `get_count()` → returns `0`
3. Agent calls `increment()` → state becomes `1`, returns `1`
4. Agent calls `increment()` → state becomes `2`, returns `2`
5. ... continues until reaching 5
6. Agent says: "I've reached 5"
7. Verification checks: Is the counter actually at 5?

```{mermaid}
sequenceDiagram
    participant Agent
    participant Server
    Note over Server: state = 0
    Agent->>Server: get_count()
    Server-->>Agent: 0
    Agent->>Server: increment()
    Note over Server: state = 1
    Server-->>Agent: 1
    Agent->>Server: increment()
    Note over Server: state = 2
    Server-->>Agent: 2
    Note over Agent,Server: ... continues ...
    Agent->>Server: Done! Counter is at 5
    Server-->>Agent: ✅ reward=1.0 (state == target)
```

**✅ Success Check**: You understand that state persists between tool calls.

---

## 2. Implement Session State

<!-- SME: Extract and adapt from example_stateful_counter/app.py -->

```python
# app.py
from nemo_gym.base_resources_server import SimpleResourcesServer, BaseResourcesServerConfig
from typing import Dict, Any

class CounterServerConfig(BaseResourcesServerConfig):
    pass

class CounterServer(SimpleResourcesServer):
    
    def __init__(self, config: CounterServerConfig):
        super().__init__(config)
        # Session state: keyed by session_id
        self.sessions: Dict[str, int] = {}
    
    def _get_session_id(self, responses_create_params: dict) -> str:
        """Extract or generate session ID from request."""
        # In practice, this comes from the request context
        # For simplicity, use a hash of the input
        return str(hash(str(responses_create_params.get("input", []))))
    
    def _get_counter(self, session_id: str) -> int:
        """Get counter for session, initializing if needed."""
        if session_id not in self.sessions:
            self.sessions[session_id] = 0
        return self.sessions[session_id]
    
    def _set_counter(self, session_id: str, value: int):
        """Set counter for session."""
        self.sessions[session_id] = value
```

**✅ Success Check**: Each session has isolated state.

---

## 3. Create Stateful Tools

```python
# Add to app.py

def get_tools(self):
    return [
        {
            "type": "function",
            "function": {
                "name": "get_count",
                "description": "Get the current counter value",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "increment",
                "description": "Increase the counter by 1",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "decrement",
                "description": "Decrease the counter by 1",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

def call_tool(self, tool_name: str, tool_args: dict, session_id: str = None) -> str:
    if session_id is None:
        session_id = "default"
    
    if tool_name == "get_count":
        count = self._get_counter(session_id)
        return f"Current count: {count}"
    
    elif tool_name == "increment":
        count = self._get_counter(session_id) + 1
        self._set_counter(session_id, count)
        return f"Incremented. Current count: {count}"
    
    elif tool_name == "decrement":
        count = self._get_counter(session_id) - 1
        self._set_counter(session_id, count)
        return f"Decremented. Current count: {count}"
    
    return "Unknown tool"
```

**✅ Success Check**: Tools modify and return the current state.

---

## 4. Verify Final State

```python
# Add to app.py

def verify(self, responses_create_params: dict, output: list) -> float:
    """Verify the agent reached the target number."""
    
    # Extract target from user message
    user_message = responses_create_params["input"][0]["content"]
    
    # Parse target number (e.g., "Reach the number 5")
    import re
    match = re.search(r'(\d+)', user_message)
    if not match:
        return 0.0
    target = int(match.group(1))
    
    # Get session ID and current counter value
    session_id = self._get_session_id(responses_create_params)
    final_count = self._get_counter(session_id)
    
    # Clean up session
    if session_id in self.sessions:
        del self.sessions[session_id]
    
    # Reward based on reaching target
    if final_count == target:
        return 1.0
    else:
        # Partial credit based on how close
        distance = abs(target - final_count)
        return max(0, 1.0 - (distance * 0.1))
```

**✅ Success Check**: Returns 1.0 only if final state equals target.

---

## 5. Create Test Data

Create `data/example.jsonl`:

```json
{"responses_create_params": {"input": [{"role": "user", "content": "Use the counter tools to reach the number 5"}], "model": "gpt-4"}}
{"responses_create_params": {"input": [{"role": "user", "content": "Start from 0 and get to 3 using increment"}], "model": "gpt-4"}}
{"responses_create_params": {"input": [{"role": "user", "content": "Reach -2 by decrementing the counter"}], "model": "gpt-4"}}
```

---

## 6. Run and Test

```bash
ng_collect_rollouts +agent_name=counter_simple_agent \
    +input_jsonl_fpath=resources_servers/counter/data/example.jsonl \
    +output_jsonl_fpath=results/counter_rollouts.jsonl \
    +limit=3
```

**✅ Success Check**: Reward=1.0 means agent successfully reached target number.

---

## Adapt for Your Use Case

This pattern works for any stateful interaction:

- **Shopping cart**: `add_item()`, `remove_item()`, `checkout()` → verify final cart
- **Game state**: `move()`, `attack()`, `heal()` → verify win condition
- **Form filling**: `set_field()`, `validate()`, `submit()` → verify all fields complete

---

## Learn More

- {doc}`/training/verification/index` — State-based verification patterns
- {doc}`/about/concepts/core-abstractions` — How Gym manages session state

---

## What's Next?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`law;1.5em;sd-mr-1` Math Verifier
:link: math-verifier-server
:link-type: doc

Use an LLM to verify open-ended math answers.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Tester
:link: code-tester-server
:link-type: doc

Execute and verify generated code.
:::

::::
