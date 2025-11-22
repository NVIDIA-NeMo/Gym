# Stateful Counter Resource Server

A reference implementation demonstrating **session state management** for multi-step agent interactions in NeMo Gym. This server shows how to maintain state across multiple tool calls within a single trajectory.

## Overview

The Stateful Counter is a simple but complete example of a stateful resource server. It maintains a counter that agents can increment and read across multiple tool calls, with each session maintaining its own independent counter state.

**Use this as a reference for:**
- Understanding session-based state management
- Building resource servers that track agent actions
- Implementing multi-step tasks requiring state persistence
- Learning state management patterns (in-memory, database, external services)

## What is Multi-Step State Management?

In multi-step agent interactions, the agent makes multiple tool calls to complete a task. State management allows the resource server to remember what happened in previous steps.

**Example Task**: *"Add 4 then add 3 then get the count"*

```
Initial State: counter = 0

Step 1: Agent calls increment_counter(count=4)
        → Server updates: counter = 4
        → Returns: {"success": true}

Step 2: Agent calls increment_counter(count=3)
        → Server updates: counter = 7  (remembers previous state!)
        → Returns: {"success": true}

Step 3: Agent calls get_counter_value()
        → Server reads: counter = 7
        → Returns: {"count": 7}

Verification: Expected count == 7? ✓ Reward = 1.0
```

Without state management, the server wouldn't remember the counter value between calls.

---

## Quick Start

### Prerequisites

```bash
# Ensure NeMo Gym is installed and virtual environment is activated
source .venv/bin/activate

# Verify installation
python -c "import nemo_gym; print('NeMo Gym installed')"
```

### Generate Example Dataset

```bash
# From the nemo-gym root directory
python resources_servers/example_stateful_counter/create_examples.py
```

This creates `data/example.jsonl` with sample tasks like:
- "add 1 then add 2 then get the count" (initial_count=3, expected=6)
- "add 4 then add 5 then get the count" (initial_count=6, expected=15)

### Run the Server

**Terminal 1: Start servers**
```bash
config_paths="resources_servers/example_stateful_counter/configs/stateful_counter.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

**Terminal 2: Test with client**
```bash
source .venv/bin/activate
python resources_servers/example_stateful_counter/client.py
```

### Collect Rollouts

```bash
# Collect verified rollouts for training
ng_collect_rollouts \
    +agent_name=stateful_counter_simple_agent \
    +input_jsonl_fpath=resources_servers/example_stateful_counter/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/example_stateful_counter/data/example_rollouts.jsonl

# View results
cat resources_servers/example_stateful_counter/data/example_rollouts.jsonl | python -m json.tool
```

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│ SimpleAgent (Orchestrator)                              │
│ - Manages conversation loop                             │
│ - Routes tool calls to resource server                  │
│ - Maintains session cookies                             │
└────────────┬────────────────────────────────────────────┘
             │
             │ (with session cookie)
             ▼
┌─────────────────────────────────────────────────────────┐
│ StatefulCounterResourcesServer                          │
│                                                         │
│ Endpoints:                                              │
│ - /seed_session     → Initialize counter                │
│ - /increment_counter → Add to counter                   │
│ - /get_counter_value → Read counter value               │
│ - /verify           → Check final count                 │
│                                                         │
│ State Storage:                                          │
│ - session_id_to_counter: Dict[str, int]                │
│   (in-memory, keyed by session ID)                      │
└─────────────────────────────────────────────────────────┘
```

### Session Lifecycle

1. **Request arrives** → Session middleware checks for session cookie
2. **No cookie?** → Generate unique session ID, return as cookie
3. **Has cookie?** → Extract session ID from cookie
4. **Tool execution** → Access state using `request.session[SESSION_ID_KEY]`
5. **Return response** → Cookie automatically included in response
6. **Next tool call** → Client sends cookie, state is maintained

---

## Code Walkthrough

### State Storage

The server maintains a dictionary mapping session IDs to counter values:

```python
from typing import Dict
from pydantic import Field

class StatefulCounterResourcesServer(SimpleResourcesServer):
    # In-memory state storage
    session_id_to_counter: Dict[str, int] = Field(default_factory=dict)
    #                      ^session_id  ^counter value
```

**Key Point**: Each session gets its own counter. Multiple concurrent sessions don't interfere with each other.

### Seed Session: Initializing State

The `seed_session` endpoint is called by the agent before any tool calls to set up initial state:

```python
from nemo_gym.server_utils import SESSION_ID_KEY

class StatefulCounterSeedSessionRequest(BaseSeedSessionRequest):
    initial_count: int  # Custom parameter for this resource server

async def seed_session(self, request: Request, 
                      body: StatefulCounterSeedSessionRequest):
    # Get the session ID assigned by middleware
    session_id = request.session[SESSION_ID_KEY]
    
    # Initialize the counter for this session
    self.session_id_to_counter[session_id] = body.initial_count
    
    return BaseSeedSessionResponse()
```

**When to use `seed_session`:**
- Setting initial environment parameters
- Loading task-specific data
- Creating temporary resources (files, connections)
- Randomizing initial conditions

### Tool Implementation: Modifying State

The `increment_counter` tool modifies the session's counter:

```python
async def increment_counter(self, request: Request, 
                           body: IncrementCounterRequest):
    # 1. Get session ID from the request
    session_id = request.session[SESSION_ID_KEY]
    
    # 2. Get current counter (default to 0 if not set)
    counter = self.session_id_to_counter.setdefault(session_id, 0)
    
    # 3. Update counter
    counter += body.count
    
    # 4. Save updated state
    self.session_id_to_counter[session_id] = counter
    
    return IncrementCounterResponse(success=True)
```

**Pattern**: Get session ID → Load state → Modify → Save

### Tool Implementation: Reading State

The `get_counter_value` tool reads the current counter:

```python
async def get_counter_value(self, request: Request):
    session_id = request.session[SESSION_ID_KEY]
    
    # Retrieve counter (default to 0 if session doesn't exist)
    counter = self.session_id_to_counter.setdefault(session_id, 0)
    
    return GetCounterValueResponse(count=counter)
```

### Verification: Using State

Verification checks if the agent reached the expected count:

```python
class StatefulCounterVerifyRequest(BaseVerifyRequest):
    expected_count: int  # What the counter should be

async def verify(self, request: Request, 
                body: StatefulCounterVerifyRequest):
    session_id = request.session[SESSION_ID_KEY]
    
    reward = 0.0
    if session_id in self.session_id_to_counter:
        counter = self.session_id_to_counter[session_id]
        # Binary reward: 1.0 if correct, 0.0 otherwise
        reward = float(body.expected_count == counter)
    
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

---

## Testing

### Unit Tests

The test suite (`tests/test_app.py`) demonstrates:
- Session isolation (different cookies = different sessions)
- State persistence within a session
- Default initialization

**Key test pattern:**

```python
# Session 1: Start new session
response1 = client.post("/get_counter_value")
cookies1 = response1.cookies
assert response1.json() == {"count": 0}

# Session 1: Increment
client.post("/increment_counter", json={"count": 2}, cookies=cookies1)
assert client.post("/get_counter_value", cookies=cookies1).json() == {"count": 2}

# Session 2: Independent session (no cookies shared)
response2 = client.post("/increment_counter", json={"count": 4})
cookies2 = response2.cookies
assert client.post("/get_counter_value", cookies=cookies2).json() == {"count": 4}

# Session 1: Still at 2 (not affected by session 2)
assert client.post("/get_counter_value", cookies=cookies1).json() == {"count": 2}
```

### Running Tests

```bash
# Run unit tests
pytest resources_servers/example_stateful_counter/tests/ -v

# Run with coverage
pytest resources_servers/example_stateful_counter/tests/ --cov=resources_servers.example_stateful_counter
```

---

## API Reference

### Endpoints

#### POST `/seed_session`

Initialize the counter for a new session.

**Request:**
```json
{
  "initial_count": 0
}
```

**Response:**
```json
{}
```

**Headers:** Sets session cookie

---

#### POST `/increment_counter`

Add a value to the session's counter.

**Request:**
```json
{
  "count": 5
}
```

**Response:**
```json
{
  "success": true
}
```

**Side effect:** Updates counter in session state

---

#### POST `/get_counter_value`

Get the current counter value.

**Request:** Empty body `{}`

**Response:**
```json
{
  "count": 7
}
```

---

#### POST `/verify`

Verify if the final counter matches expected value.

**Request:**
```json
{
  "responses_create_params": { ... },
  "response": { ... },
  "expected_count": 10
}
```

**Response:**
```json
{
  "responses_create_params": { ... },
  "response": { ... },
  "expected_count": 10,
  "reward": 1.0
}
```

**Reward:**
- `1.0` if `counter == expected_count`
- `0.0` otherwise

---

## Extending This Example

### Add Database Persistence

Replace in-memory storage with a database:

```python
import aioredis

class StatefulCounterResourcesServer(SimpleResourcesServer):
    redis: aioredis.Redis
    
    async def increment_counter(self, request: Request, body: IncrementCounterRequest):
        session_id = request.session[SESSION_ID_KEY]
        
        # Load from Redis
        counter = await self.redis.get(f"counter:{session_id}") or 0
        counter = int(counter)
        
        # Update
        counter += body.count
        
        # Save to Redis
        await self.redis.set(f"counter:{session_id}", counter)
        
        return IncrementCounterResponse(success=True)
```

### Add More Operations

```python
class DecrementCounterRequest(BaseModel):
    count: int

async def decrement_counter(self, request: Request, 
                           body: DecrementCounterRequest):
    session_id = request.session[SESSION_ID_KEY]
    counter = self.session_id_to_counter.setdefault(session_id, 0)
    counter -= body.count
    self.session_id_to_counter[session_id] = counter
    return DecrementCounterResponse(success=True)
```

### Track Operation History

```python
class StatefulCounterResourcesServer(SimpleResourcesServer):
    session_id_to_counter: Dict[str, int] = Field(default_factory=dict)
    session_id_to_history: Dict[str, List[str]] = Field(default_factory=dict)
    
    async def increment_counter(self, request: Request, body: IncrementCounterRequest):
        session_id = request.session[SESSION_ID_KEY]
        
        # Update counter
        counter = self.session_id_to_counter.setdefault(session_id, 0)
        counter += body.count
        self.session_id_to_counter[session_id] = counter
        
        # Track history
        history = self.session_id_to_history.setdefault(session_id, [])
        history.append(f"increment({body.count}) -> {counter}")
        
        return IncrementCounterResponse(success=True)
```

---

## State Management Patterns Comparison

### In-Memory (Current Implementation)

**Pros:**
- Simple to implement
- Fast access (no I/O)
- No external dependencies

**Cons:**
- Lost on server restart
- Not shared across multiple server replicas
- Memory usage grows unbounded

**Best for:** Development, testing, single-instance deployments

### Redis-Backed

```python
session_counter = await redis.get(f"session:{session_id}:counter")
```

**Pros:**
- Persists across restarts
- Shared across multiple server instances
- Built-in expiration (TTL)

**Cons:**
- Requires Redis setup
- Network latency
- Additional infrastructure

**Best for:** Production multi-instance deployments

### Database-Backed (PostgreSQL/MySQL)

```python
counter = await db.fetchval(
    "SELECT counter FROM sessions WHERE session_id = $1", 
    session_id
)
```

**Pros:**
- Durable persistence
- Rich querying capabilities
- Transaction support

**Cons:**
- Slower than in-memory/Redis
- Requires database setup
- More complex queries

**Best for:** Complex state with relationships, audit trails

---

## Common Patterns

### Pattern 1: Session Cleanup

```python
def setup_webserver(self) -> FastAPI:
    app = super().setup_webserver()
    
    @app.post("/cleanup_session")
    async def cleanup_session(request: Request):
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_counter.pop(session_id, None)
        return {"status": "cleaned"}
    
    return app
```

### Pattern 2: State Validation

```python
async def increment_counter(self, request: Request, body: IncrementCounterRequest):
    session_id = request.session[SESSION_ID_KEY]
    counter = self.session_id_to_counter.setdefault(session_id, 0)
    
    # Validate state
    if counter + body.count > 1000:
        raise ValueError("Counter would exceed maximum")
    
    counter += body.count
    self.session_id_to_counter[session_id] = counter
    return IncrementCounterResponse(success=True)
```

### Pattern 3: State Snapshots

```python
async def get_state_snapshot(self, request: Request):
    session_id = request.session[SESSION_ID_KEY]
    return StateSnapshot(
        counter=self.session_id_to_counter.get(session_id, 0),
        operations_count=self.operation_counts.get(session_id, 0),
    )
```

---

## Troubleshooting

### State Not Persisting Between Calls

**Problem:** Counter resets to 0 after each call

**Causes:**
1. Not passing cookies between requests
2. Session middleware not set up
3. Using different session IDs

**Solution:**
```python
# ✅ Correct: Pass cookies
response1 = client.post("/increment_counter", json={"count": 2})
cookies = response1.cookies
response2 = client.post("/get_counter_value", cookies=cookies)

# ❌ Wrong: No cookies
response1 = client.post("/increment_counter", json={"count": 2})
response2 = client.post("/get_counter_value")  # New session!
```

### State Shared Between Sessions

**Problem:** Different clients see the same counter

**Cause:** Using global state instead of session-keyed state

**Solution:**
```python
# ✅ Correct: Session-specific state
session_id = request.session[SESSION_ID_KEY]
counter = self.session_id_to_counter[session_id]

# ❌ Wrong: Global state
counter = self.global_counter
```

### Memory Leaks

**Problem:** Server memory grows indefinitely

**Cause:** Sessions never cleaned up

**Solution:** Implement cleanup or use external storage with TTL:
```python
# Redis with expiration
await redis.setex(f"counter:{session_id}", 3600, counter)  # 1 hour TTL
```

---

## Related Documentation

- **[Multi-Step Patterns Tutorial](../../../docs/tutorials/development/multi-step-patterns.md)**: Comprehensive guide to multi-step interactions
- **[Core Abstractions](../../../docs/about/concepts/core-abstractions.md)**: Understanding resources, models, and agents
- **[Key Terminology](../../../docs/about/concepts/key-terminology.md)**: Glossary of terms

---

## Configuration

The server is configured via `configs/stateful_counter.yaml`:

```yaml
stateful_counter:
  resources_servers:
    example_stateful_counter:
      entrypoint: app.py
      host: 0.0.0.0
      port: 12002

stateful_counter_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        name: stateful_counter
      model_server:
        name: policy_model
```

---

## Licensing Information

- **Code**: Apache 2.0
- **Data**: Apache 2.0

### Dependencies

- `nemo_gym`: Apache 2.0

---

## Contributing

Found a bug or have a feature idea? 
- **[Report Issues](https://github.com/NVIDIA-NeMo/Gym/issues)**
- **[Contributing Guide](../../../CONTRIBUTING.md)**

---

## Summary

This resource server demonstrates:

✅ **Session-based state management** using cookies  
✅ **Session isolation** (each session has independent state)  
✅ **State initialization** with `seed_session`  
✅ **State persistence** across multiple tool calls  
✅ **State-aware verification** for reward calculation  
✅ **Testing patterns** for stateful servers  

Use this as a template when building resource servers that need to maintain state across multi-step agent interactions.
