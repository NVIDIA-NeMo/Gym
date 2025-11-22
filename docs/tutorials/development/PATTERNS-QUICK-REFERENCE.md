---
orphan: true
---

# Multi-Step Patterns Quick Reference

A quick decision guide for choosing the right patterns for your resource server.

## When to Use Stateful vs Stateless

### Use STATELESS when:
- ✅ Each tool call is independent
- ✅ All information is in the request
- ✅ No dependencies between calls
- ✅ Examples: weather lookup, fact checking, simple calculations

**Example:** Simple Weather
```python
async def get_weather(self, body: GetWeatherRequest):
    # No state needed - just return data
    return GetWeatherResponse(city=body.city, weather="sunny")
```

### Use STATEFUL when:
- ✅ Tool calls build upon previous results
- ✅ Need to track agent actions
- ✅ Complex verification requires history
- ✅ Managing resources (files, connections)
- ✅ Examples: code environments, multi-step math, databases

**Example:** Stateful Counter
```python
session_state: Dict[str, int] = Field(default_factory=dict)

async def increment(self, request: Request, body: IncrementRequest):
    session_id = request.session[SESSION_ID_KEY]
    counter = self.session_state.setdefault(session_id, 0)
    counter += body.count
    self.session_state[session_id] = counter
    return IncrementResponse(success=True)
```

---

## State Storage Decision Tree

```
┌─────────────────────────────┐
│ Choose State Storage        │
└─────────────┬───────────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Development/Testing  │───→ In-Memory Dict
    │ - Quick iteration   │
    │ - Single server     │
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Production          │
    │ - Single instance   │───→ In-Memory + Monitoring
    │ - Low traffic       │
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Production          │
    │ - Multi-instance    │───→ Redis/Database
    │ - High availability │
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │ Complex             │
    │ - Large state       │───→ External Service
    │ - Special needs     │     (S3, Custom API)
    └─────────────────────┘
```

---

## Essential Code Patterns

### Pattern 1: Get Session ID
```python
from nemo_gym.server_utils import SESSION_ID_KEY

async def my_tool(self, request: Request, body: ToolRequest):
    session_id = request.session[SESSION_ID_KEY]
```

### Pattern 2: Initialize State
```python
async def seed_session(self, request: Request, body: SeedRequest):
    session_id = request.session[SESSION_ID_KEY]
    self.session_state[session_id] = body.initial_value
    return BaseSeedSessionResponse()
```

### Pattern 3: Load State with Default
```python
# Use setdefault for initialization
state = self.session_state.setdefault(session_id, default_value())

# Or use get with default
state = self.session_state.get(session_id, default_value())
```

### Pattern 4: Update State
```python
# Load
state = self.session_state[session_id]

# Modify
state.counter += 1

# Save (if using mutable objects)
self.session_state[session_id] = state
```

### Pattern 5: Verify with State
```python
async def verify(self, request: Request, body: VerifyRequest):
    session_id = request.session[SESSION_ID_KEY]
    
    if session_id not in self.session_state:
        return BaseVerifyResponse(**body.model_dump(), reward=0.0)
    
    state = self.session_state[session_id]
    reward = calculate_reward(state, body.expected)
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

---

## Common Mistakes

### ❌ Don't: Use Global State
```python
class BadServer:
    global_counter = 0  # Shared across all sessions!
    
    async def increment(self):
        self.global_counter += 1  # Wrong!
```

### ✅ Do: Use Session-Specific State
```python
class GoodServer:
    session_counters: Dict[str, int] = Field(default_factory=dict)
    
    async def increment(self, request: Request):
        session_id = request.session[SESSION_ID_KEY]
        counter = self.session_counters.setdefault(session_id, 0)
        self.session_counters[session_id] = counter + 1
```

---

### ❌ Don't: Assume State Exists
```python
counter = self.session_state[session_id]  # KeyError if not set!
```

### ✅ Do: Handle Missing State
```python
counter = self.session_state.setdefault(session_id, 0)
# or
counter = self.session_state.get(session_id, 0)
```

---

### ❌ Don't: Forget to Pass Cookies
```python
# Client-side error
response1 = client.post("/increment", json={"count": 2})
response2 = client.post("/get_value")  # New session! Counter reset
```

### ✅ Do: Maintain Session Cookies
```python
response1 = client.post("/increment", json={"count": 2})
cookies = response1.cookies
response2 = client.post("/get_value", cookies=cookies)  # Same session
```

---

## State Storage Comparison

| Pattern | Speed | Persistence | Multi-Instance | Complexity |
|---------|-------|-------------|----------------|------------|
| In-Memory | ⚡⚡⚡ Fast | ❌ Lost on restart | ❌ No | ⭐ Simple |
| Redis | ⚡⚡ Fast | ✅ Persists | ✅ Yes | ⭐⭐ Medium |
| Database | ⚡ Slower | ✅ Persists | ✅ Yes | ⭐⭐⭐ Complex |
| External Service | ⚡ Varies | ✅ Persists | ✅ Yes | ⭐⭐⭐ Complex |

---

## Testing Checklist

- [ ] Sessions are isolated (different cookies = different state)
- [ ] State persists across multiple tool calls
- [ ] Missing state handled gracefully
- [ ] Verification uses correct session state
- [ ] Cleanup (if needed) works correctly

---

## Next Steps

📖 **Full Tutorial**: {doc}`multi-step-patterns`  
📦 **Example Code**: `resources_servers/example_stateful_counter/`  
🧪 **Tests**: `resources_servers/example_stateful_counter/tests/`

---

## Summary

**Key Points:**
1. Use sessions (`SESSION_ID_KEY`) for all stateful operations
2. Always handle missing state with `setdefault()` or `get()`
3. Choose storage based on requirements (dev vs prod, single vs multi-instance)
4. Test session isolation and state persistence
5. Remember: Multi-step ≠ Multi-turn (tool calls vs conversations)

