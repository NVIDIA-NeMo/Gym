(sessions-explained)=

# Sessions Explained

A deep dive into how NeMo Gym uses sessions to enable stateful multi-step interactions.

## What is a Session?

A **session** is a unique identifier that connects multiple requests into a single logical trajectory. Sessions enable resource servers to maintain state across tool calls, allowing agents to perform complex multi-step tasks.

**Key Concept**: Without sessions, each tool call would be isolated—the server would have no memory of previous calls. Sessions provide that memory.

---

## Session Lifecycle

### 1. Session Creation

```
┌─────────────────┐
│ Agent Request   │
│ (no cookie)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Session Middleware      │
│ - Generates UUID        │
│ - Sets in request obj   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Returns Response        │
│ + Session Cookie        │
└─────────────────────────┘
```

**Code:**
```python
# In SessionMiddleware
if SESSION_ID_KEY not in request.session:
    request.session[SESSION_ID_KEY] = str(uuid4())
    # e.g., "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```

### 2. Session Routing

```
┌─────────────────┐
│ Agent Request   │
│ (with cookie)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Session Middleware      │
│ - Extracts session_id   │
│ - Attaches to request   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Tool Handler            │
│ - Gets session_id       │
│ - Loads session state   │
│ - Executes tool         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│ Returns Response        │
│ (same cookie)           │
└─────────────────────────┘
```

**Code:**
```python
async def my_tool(self, request: Request, body: ToolRequest):
    # Session ID automatically available
    session_id = request.session[SESSION_ID_KEY]
    
    # Use it to load/save state
    state = self.session_state.get(session_id, {})
```

### 3. Session Persistence

```
Tool Call 1:
┌──────────────────────────┐
│ session_id: abc-123      │
│ counter: 0 → 5           │
└──────────────────────────┘

Tool Call 2 (same session):
┌──────────────────────────┐
│ session_id: abc-123      │
│ counter: 5 → 10          │  ← Remembered from call 1
└──────────────────────────┘

Tool Call 3 (different session):
┌──────────────────────────┐
│ session_id: xyz-789      │
│ counter: 0 → 3           │  ← Independent state
└──────────────────────────┘
```

### 4. Session Cleanup

Sessions in NeMo Gym are **ephemeral**—they exist for the duration of a trajectory and can be cleaned up after verification.

**Options:**
1. **Natural expiration**: In-memory state naturally disappears when server restarts
2. **Explicit cleanup**: Implement cleanup endpoints for long-running servers
3. **TTL expiration**: Use Redis/cache with automatic expiration
4. **Post-verification cleanup**: Clean up after verification completes

---

## How Sessions Work Under the Hood

### The Cookie Mechanism

NeMo Gym uses HTTP cookies to track sessions. The cookie name is unique per server instance:

```python
# Server generates unique cookie name
cookie_name = f"{self.__class__.__name__}___{self.config.name}"
# e.g., "StatefulCounterResourcesServer___stateful_counter"

# Middleware adds session cookie to response
app.add_middleware(
    SessionMiddleware, 
    secret_key=cookie_name,
    session_cookie=cookie_name
)
```

**Important**: The agent orchestrator automatically propagates cookies between model and resource servers:

```python
# In SimpleAgent.responses()
resources_server_cookies = request.cookies

for fn_call in all_fn_calls:
    api_response = await self.server_client.post(
        server_name=self.config.resources_server.name,
        url_path=f"/{fn_call.name}",
        json=fn_call.arguments,
        cookies=resources_server_cookies,  # Maintains session
    )
    resources_server_cookies = api_response.cookies
```

### The Session Middleware

The middleware runs on every request to ensure a session ID exists:

```python
@app.middleware("http")
async def add_session_id(request: Request, call_next):
    # Check if session has an ID
    if SESSION_ID_KEY not in request.session:
        # Generate new UUID for this session
        request.session[SESSION_ID_KEY] = str(uuid4())
    
    # Process request
    response = await call_next(request)
    return response
```

**Order matters**: This middleware must be registered **before** `SessionMiddleware` to ensure IDs are assigned before session data is written to cookies.

---

## Session Isolation

One of the most important properties of sessions is **isolation**—each session maintains completely independent state.

### Example: Multiple Concurrent Sessions

```
# Session A
Counter: [increment 5] → [increment 3] → [get] = 8

# Session B (concurrent with A)
Counter: [increment 10] → [get] = 10

# Session A (continues)
Counter: [get] = 8  ← Still 8, not affected by session B
```

**Implementation:**
```python
class StatefulServer(SimpleResourcesServer):
    # State keyed by session_id ensures isolation
    session_counters: Dict[str, int] = Field(default_factory=dict)
    
    async def increment(self, request: Request, body: IncrementRequest):
        session_id = request.session[SESSION_ID_KEY]
        
        # Each session has its own counter
        counter = self.session_counters.setdefault(session_id, 0)
        counter += body.count
        self.session_counters[session_id] = counter
        
        return IncrementResponse(success=True)
```

---

## Session vs. Conversation Turn

It's important to distinguish between sessions and conversation turns:

| Aspect | Session | Conversation Turn |
|--------|---------|------------------|
| **Duration** | Entire trajectory | Single user ↔ assistant exchange |
| **Scope** | Multiple tool calls | Single message exchange |
| **Purpose** | State management | Conversation history |
| **Example** | "Add 4, then add 3, then get count" | User: "Hello" → Assistant: "Hi there!" |

**Example: Multi-Step in Single Turn**

```
User: "Add 4 then add 3 then get the count"
↓
[Single conversation turn, but multiple tool calls in one session]
Agent: increment_counter(4) [session: abc-123]
Agent: increment_counter(3) [session: abc-123]
Agent: get_counter_value() [session: abc-123]
↓
Agent: "The count is 7"
```

**Example: Multi-Turn Conversation**

```
Turn 1:
User: "Add 4 to the counter"
Agent: increment_counter(4) [session: abc-123]
Agent: "Done, added 4"

Turn 2:
User: "Now add 3"
Agent: increment_counter(3) [session: abc-123]
Agent: "Done, added 3"

Turn 3:
User: "What's the count?"
Agent: get_counter_value() [session: abc-123]
Agent: "The count is 7"
```

In multi-turn scenarios, the **same session** persists across turns, allowing state to accumulate.

---

## Session State Storage Patterns

### Pattern 1: In-Memory Dictionary

**Best for**: Development, testing, single-server deployments

```python
class MyServer(SimpleResourcesServer):
    session_state: Dict[str, MyState] = Field(default_factory=dict)
    
    async def my_tool(self, request: Request, body: ToolRequest):
        session_id = request.session[SESSION_ID_KEY]
        state = self.session_state.setdefault(session_id, MyState())
        # ... use state ...
```

**Pros**: Simple, fast, no dependencies  
**Cons**: Lost on restart, not shared across instances

### Pattern 2: Redis Cache

**Best for**: Production, distributed systems

```python
import aioredis

class MyServer(SimpleResourcesServer):
    redis: aioredis.Redis
    
    async def my_tool(self, request: Request, body: ToolRequest):
        session_id = request.session[SESSION_ID_KEY]
        
        # Load from Redis
        state_json = await self.redis.get(f"session:{session_id}:state")
        state = MyState.parse_raw(state_json) if state_json else MyState()
        
        # ... use state ...
        
        # Save to Redis with TTL
        await self.redis.setex(
            f"session:{session_id}:state",
            3600,  # 1 hour expiration
            state.json()
        )
```

**Pros**: Persists across restarts, shared across instances, TTL support  
**Cons**: Requires Redis, network latency

### Pattern 3: Database

**Best for**: Complex state, audit requirements

```python
class MyServer(SimpleResourcesServer):
    async def my_tool(self, request: Request, body: ToolRequest):
        session_id = request.session[SESSION_ID_KEY]
        
        # Load from database
        state = await self.db.fetch_one(
            "SELECT state FROM sessions WHERE session_id = ?",
            session_id
        )
        
        # ... use state ...
        
        # Save to database
        await self.db.execute(
            "UPDATE sessions SET state = ? WHERE session_id = ?",
            state, session_id
        )
```

**Pros**: Durable, queryable, supports transactions  
**Cons**: Slower, requires database setup

---

## Common Session Patterns

### Pattern: Seeding Initial State

Use `seed_session` to initialize state before the agent starts:

```python
async def seed_session(self, request: Request, body: SeedRequest):
    session_id = request.session[SESSION_ID_KEY]
    
    # Initialize state for this trajectory
    self.session_state[session_id] = {
        "initial_value": body.initial_value,
        "created_at": datetime.now(),
        "operations": []
    }
    
    return BaseSeedSessionResponse()
```

### Pattern: Accumulating History

Track all operations within a session:

```python
async def my_tool(self, request: Request, body: ToolRequest):
    session_id = request.session[SESSION_ID_KEY]
    
    state = self.session_state.setdefault(session_id, {"history": []})
    
    # Record operation
    state["history"].append({
        "tool": "my_tool",
        "args": body.dict(),
        "timestamp": datetime.now()
    })
    
    return response
```

### Pattern: Session Validation

Ensure session is in valid state before executing tools:

```python
async def my_tool(self, request: Request, body: ToolRequest):
    session_id = request.session[SESSION_ID_KEY]
    
    if session_id not in self.session_state:
        raise HTTPException(
            status_code=400,
            detail="Session not initialized. Call seed_session first."
        )
    
    state = self.session_state[session_id]
    # ... continue ...
```

### Pattern: Cleanup

Clean up resources when session ends:

```python
async def cleanup_session(self, request: Request):
    session_id = request.session[SESSION_ID_KEY]
    
    if session_id in self.session_state:
        # Clean up any resources
        state = self.session_state[session_id]
        if "temp_files" in state:
            for f in state["temp_files"]:
                f.unlink()  # Delete temp files
        
        # Remove from state
        del self.session_state[session_id]
    
    return {"status": "cleaned"}
```

---

## Debugging Sessions

### Viewing Session State

Add debug endpoints to inspect session state:

```python
@app.get("/debug/session")
async def debug_session(request: Request):
    session_id = request.session[SESSION_ID_KEY]
    state = self.session_state.get(session_id, {})
    return {
        "session_id": session_id,
        "state": state,
        "exists": session_id in self.session_state
    }
```

### Logging Session Activity

```python
import logging

logger = logging.getLogger(__name__)

async def my_tool(self, request: Request, body: ToolRequest):
    session_id = request.session[SESSION_ID_KEY]
    logger.info(f"Session {session_id}: executing my_tool with {body}")
    # ... execute ...
    logger.info(f"Session {session_id}: completed my_tool")
```

### Testing Session Isolation

```python
def test_session_isolation():
    client = TestClient(app)
    
    # Session 1
    r1 = client.post("/increment", json={"count": 5})
    c1 = r1.cookies
    
    # Session 2
    r2 = client.post("/increment", json={"count": 10})
    c2 = r2.cookies
    
    # Verify isolation
    assert client.post("/get_value", cookies=c1).json()["value"] == 5
    assert client.post("/get_value", cookies=c2).json()["value"] == 10
```

---

## Best Practices

### ✅ Do

1. **Always use session IDs** for stateful operations
2. **Handle missing state gracefully** with `setdefault()` or `get()`
3. **Initialize state in `seed_session`** when needed
4. **Test session isolation** in your test suite
5. **Log session activity** for debugging
6. **Clean up resources** when sessions end (if using expensive resources)

### ❌ Don't

1. **Don't use global state** without session keys
2. **Don't assume sessions persist forever** (they're ephemeral)
3. **Don't share state across sessions** (breaks isolation)
4. **Don't store sensitive data** in cookies (use session IDs only)
5. **Don't forget to propagate cookies** in client code

---

## Related Documentation

- **{doc}`../../tutorials/development/multi-step-patterns`**: Complete guide to multi-step patterns
- **{doc}`core-abstractions`**: Understanding resources, models, and agents
- **{doc}`key-terminology`**: Glossary of terms

---

## Summary

**Sessions enable stateful multi-step interactions by:**

1. **Unique Identification**: Each trajectory gets a unique session ID
2. **State Isolation**: Sessions maintain independent state
3. **Cookie Propagation**: Session IDs travel via HTTP cookies
4. **Automatic Management**: Middleware handles session creation/routing
5. **Flexible Storage**: Support for in-memory, Redis, database, or custom storage

**Key Takeaway**: Sessions are the foundation of stateful resource servers. Understanding how they work is essential for building agents that maintain context across multiple tool calls.

