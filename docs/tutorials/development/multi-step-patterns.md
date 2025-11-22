(multi-step-patterns)=

# Multi-Step Interactions

**Estimated Time**: 15-20 minutes

Multi-step interactions are fundamental to building capable AI agents. This tutorial explains how NeMo Gym handles scenarios where agents make multiple tool calls within a single task trajectory to accomplish complex goals.

:::{tip}
**What is Multi-Step?** A multi-step interaction is when an agent makes multiple sequential tool calls within a single task to reach a goal. For example: searching for information, then analyzing it, then formatting results. This is different from multi-turn conversations (user ↔ assistant exchanges).
:::

---

## What are Multi-Step Interactions?

In NeMo Gym, a **multi-step interaction** occurs when:

1. An agent receives a task (prompt + available tools)
2. The agent makes multiple tool calls sequentially
3. Each tool call's output informs the next decision
4. The agent continues until completing the task
5. The final result is verified for training

**Example Task**: *"Add 4 then add 3 then get the count"*

```
Step 1: Agent calls increment_counter(count=4)
        → Response: {"success": true}
Step 2: Agent calls increment_counter(count=3)  
        → Response: {"success": true}
Step 3: Agent calls get_counter_value()
        → Response: {"count": 7}
Step 4: Agent responds with final answer
        → Verification checks if count == 7
```

This entire sequence is one **trajectory** or **rollout**.

---

## The Session Model

NeMo Gym uses **sessions** to maintain state across multiple tool calls within a single trajectory. Understanding sessions is crucial for building stateful resource servers.

### Session Lifecycle

```
New Request
    │
    ▼
Session Cookie?
    │
    ├─[No]──> Create Session ID ──┐
    │                              │
    └─[Yes]─> Load Session ID ─────┤
                                   │
                                   ▼
                        Route to Resource Server
                                   │
                                   ▼
                          Execute Tool Call
                                   │
                                   ▼
                     Return Response + Cookie
                                   │
                                   ▼
                         More Tool Calls?
                                   │
                            ├─[Yes]─┘
                            │
                            └─[No]──> Verify and Score
```

### How Sessions Work

1. **Session Creation**: When a request arrives without a session cookie, NeMo Gym automatically generates a unique session ID
2. **Session Routing**: All subsequent requests with the same session cookie are routed to maintain state
3. **Session Storage**: Resource servers can store data keyed by session ID
4. **Session Cleanup**: Sessions are ephemeral and exist only for the duration of a trajectory

### Session Implementation

Sessions are implemented using cookies and FastAPI's `SessionMiddleware`:

```python
from fastapi import Request
from nemo_gym.server_utils import SESSION_ID_KEY

async def my_tool(self, request: Request, body: ToolRequest):
    # Get the session ID for this trajectory
    session_id = request.session[SESSION_ID_KEY]
    
    # Use session_id to maintain state
    state = self.session_state.get(session_id, {})
    # ... update state ...
    self.session_state[session_id] = state
```

The `SimpleServer` base class handles session middleware automatically:

```python
def setup_session_middleware(self, app: FastAPI) -> None:
    @app.middleware("http")
    async def add_session_id(request: Request, call_next):
        if SESSION_ID_KEY not in request.session:
            request.session[SESSION_ID_KEY] = str(uuid4())
        response = await call_next(request)
        return response
    
    app.add_middleware(SessionMiddleware, 
                      secret_key=self.get_session_middleware_key())
```

---

## Stateful vs. Stateless Patterns

### When to Use Stateless Resource Servers

**Stateless servers** don't maintain information between tool calls. Each request is independent.

**Use stateless when:**
- Each tool call is self-contained
- All necessary information is in the request
- No dependencies between tool calls
- Examples: weather lookup, fact checking, single calculations

**Example: Simple Weather**

```python
class SimpleWeatherResourcesServer(SimpleResourcesServer):
    async def get_weather(self, body: GetWeatherRequest) -> GetWeatherResponse:
        # No state needed - just return weather for requested city
        return GetWeatherResponse(
            city=body.city, 
            weather_description=f"The weather in {body.city} is cold."
        )
```

### When to Use Stateful Resource Servers

**Stateful servers** maintain information across multiple tool calls within a trajectory.

**Use stateful when:**
- Tool calls build upon previous results
- Need to track what the agent has done
- Complex verification requires historical context
- Managing resources (files, database connections, counters)
- Examples: code execution environments, multi-step math problems, database queries

**Example: Stateful Counter**

```python
class StatefulCounterResourcesServer(SimpleResourcesServer):
    session_id_to_counter: Dict[str, int] = Field(default_factory=dict)
    
    async def increment_counter(self, request: Request, body: IncrementCounterRequest):
        session_id = request.session[SESSION_ID_KEY]
        counter = self.session_id_to_counter.setdefault(session_id, 0)
        counter += body.count
        self.session_id_to_counter[session_id] = counter
        return IncrementCounterResponse(success=True)
    
    async def get_counter_value(self, request: Request):
        session_id = request.session[SESSION_ID_KEY]
        counter = self.session_id_to_counter.get(session_id, 0)
        return GetCounterValueResponse(count=counter)
```

---

## State Management Patterns

### Pattern 1: In-Memory State

**Best for**: Development, testing, single-server deployments

```python
class MyResourcesServer(SimpleResourcesServer):
    # State stored in memory, keyed by session_id
    session_state: Dict[str, Any] = Field(default_factory=dict)
    
    async def my_tool(self, request: Request, body: ToolRequest):
        session_id = request.session[SESSION_ID_KEY]
        state = self.session_state.setdefault(session_id, initial_state())
        # ... use and update state ...
        return response
```

**Pros:**
- Simple to implement
- Fast access
- No external dependencies

**Cons:**
- Lost on server restart
- Not shared across multiple server instances
- Memory usage grows with sessions

### Pattern 2: Database-Backed State

**Best for**: Production, distributed deployments, persistent state

```python
class MyResourcesServer(SimpleResourcesServer):
    db_connection: DatabaseClient
    
    async def my_tool(self, request: Request, body: ToolRequest):
        session_id = request.session[SESSION_ID_KEY]
        
        # Load state from database
        state = await self.db_connection.get(session_id) or initial_state()
        
        # Update state
        # ...
        
        # Save state back to database
        await self.db_connection.set(session_id, state)
        return response
```

**Pros:**
- Survives server restarts
- Works across multiple server instances
- Can persist for debugging

**Cons:**
- Requires database setup
- Slightly slower than in-memory
- More complex implementation

### Pattern 3: External Service State

**Best for**: Complex environments, specialized storage needs

```python
class MyResourcesServer(SimpleResourcesServer):
    async def my_tool(self, request: Request, body: ToolRequest):
        session_id = request.session[SESSION_ID_KEY]
        
        # Use external service (Redis, S3, custom API)
        state = await self.external_service.get(session_id)
        # ...
        await self.external_service.set(session_id, state)
        return response
```

---

## The seed_session Pattern

The `seed_session` endpoint allows you to initialize state at the beginning of a trajectory:

```python
class StatefulCounterSeedSessionRequest(BaseSeedSessionRequest):
    initial_count: int

class StatefulCounterResourcesServer(SimpleResourcesServer):
    async def seed_session(self, request: Request, 
                          body: StatefulCounterSeedSessionRequest):
        session_id = request.session[SESSION_ID_KEY]
        # Initialize state for this trajectory
        self.session_id_to_counter[session_id] = body.initial_count
        return BaseSeedSessionResponse()
```

**When to use `seed_session`:**
- Setting up initial environment state
- Loading task-specific configuration
- Initializing resources (temporary files, connections)
- Randomizing environment parameters

The agent orchestrator calls `seed_session` before the agent starts making tool calls:

```python
# In the agent's /run endpoint
seed_session_response = await self.server_client.post(
    server_name=self.config.resources_server.name,
    url_path="/seed_session",
    json=body.model_dump(),  # Contains seed parameters
    cookies=cookies,
)
```

---

## Agent Orchestration Loop

Understanding how agents orchestrate multi-step interactions helps when building resource servers.

Here's the simplified flow from `SimpleAgent`:

```python
async def responses(self, request: Request, response: Response, body: ...):
    new_outputs = []
    step = 0
    resources_server_cookies = request.cookies
    
    while True:
        step += 1
        
        # 1. Call model with current conversation history
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json={"input": body.input + new_outputs},
            cookies=model_server_cookies,
        )
        
        output = model_response.output
        new_outputs.extend(output)
        
        # 2. Check if model wants to call functions
        all_fn_calls = [o for o in output if o.type == "function_call"]
        if not all_fn_calls:
            break  # Model returned final answer
        
        # 3. Execute each function call
        for fn_call in all_fn_calls:
            api_response = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path=f"/{fn_call.name}",
                json=json.loads(fn_call.arguments),
                cookies=resources_server_cookies,  # Maintains session!
            )
            resources_server_cookies = api_response.cookies
            
            # Add tool response to conversation history
            tool_response = FunctionCallOutput(
                call_id=fn_call.call_id,
                output=await api_response.content.read(),
            )
            new_outputs.append(tool_response)
        
        # 4. Check max steps limit
        if self.config.max_steps and step >= self.config.max_steps:
            break
    
    return NeMoGymResponse(output=new_outputs)
```

**Key Points:**
1. Cookies are passed with every request to maintain session
2. Tool outputs are added to conversation history
3. Loop continues until model returns text (no more tool calls)
4. Optional `max_steps` prevents infinite loops

---

## Decision Tree: Do I Need State Management?

Use this decision tree to determine your state management needs:

```
┌─────────────────────────────────────┐
│ Does your task require multiple     │
│ tool calls to complete?             │
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────┐
        │ YES         │ NO → Use stateless (like simple_weather)
        │             │
        ▼             │
┌───────────────────┐ │
│ Do tool calls     │ │
│ depend on         │ │
│ previous results? │ │
└────────┬──────────┘ │
         │            │
  ┌──────┴──────┐     │
  │ YES         │ NO → Use stateless (parallel tool calls)
  │             │
  ▼             │
┌─────────────┐ │
│ Use stateful│ │
│ pattern     │ │
└──────┬──────┘ │
       │        │
       ▼        ▼
┌──────────────────────────────┐
│ Which state storage?         │
├──────────────────────────────┤
│ Development/Testing          │
│ → In-memory (Dict)          │
│                              │
│ Single instance production   │
│ → In-memory with monitoring │
│                              │
│ Multi-instance production    │
│ → Database (Redis/Postgres) │
│                              │
│ Complex state needs          │
│ → External service          │
└──────────────────────────────┘
```

---

## Hands-On Example: Stateful Counter

The `example_stateful_counter` resource server demonstrates stateful patterns. Let's walk through it.

### Setup and Running

```bash
# Terminal 1: Start servers
config_paths="resources_servers/example_stateful_counter/configs/stateful_counter.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"

# Terminal 2: Interact with the agent
source .venv/bin/activate
python resources_servers/example_stateful_counter/client.py
```

### Task: Multi-Step Counting

The agent receives a task like "add 4 then add 3 then get the count" with an initial counter value.

**Expected behavior:**
1. Start with `initial_count = 0`
2. Call `increment_counter(count=4)` → counter becomes 4
3. Call `increment_counter(count=3)` → counter becomes 7
4. Call `get_counter_value()` → returns 7
5. Verification checks if final count matches expected

### Code Walkthrough

**State Storage:**
```python
class StatefulCounterResourcesServer(SimpleResourcesServer):
    # In-memory dictionary keyed by session_id
    session_id_to_counter: Dict[str, int] = Field(default_factory=dict)
```

**Initializing State:**
```python
async def seed_session(self, request: Request, 
                      body: StatefulCounterSeedSessionRequest):
    session_id = request.session[SESSION_ID_KEY]
    # Set initial counter value for this session
    self.session_id_to_counter[session_id] = body.initial_count
    return BaseSeedSessionResponse()
```

**Modifying State:**
```python
async def increment_counter(self, request: Request, 
                           body: IncrementCounterRequest):
    session_id = request.session[SESSION_ID_KEY]
    # Get current counter (default to 0 if not set)
    counter = self.session_id_to_counter.setdefault(session_id, 0)
    
    # Update counter
    counter += body.count
    self.session_id_to_counter[session_id] = counter
    
    return IncrementCounterResponse(success=True)
```

**Reading State:**
```python
async def get_counter_value(self, request: Request):
    session_id = request.session[SESSION_ID_KEY]
    counter = self.session_id_to_counter.setdefault(session_id, 0)
    return GetCounterValueResponse(count=counter)
```

**Verifying with State:**
```python
async def verify(self, request: Request, 
                body: StatefulCounterVerifyRequest):
    session_id = request.session[SESSION_ID_KEY]
    
    reward = 0.0
    if session_id in self.session_id_to_counter:
        counter = self.session_id_to_counter[session_id]
        # Check if agent reached the expected count
        reward = float(body.expected_count == counter)
    
    return BaseVerifyResponse(**body.model_dump(), reward=reward)
```

### Testing

The test suite demonstrates session isolation:

```python
# Session 1: increment by 2, check value
response = client.post("/increment_counter", json={"count": 2}, 
                      cookies=session1_cookies)
response = client.post("/get_counter_value", cookies=session1_cookies)
assert response.json() == {"count": 2}

# Session 2: increment by 4 (independent of session 1)
response = client.post("/increment_counter", json={"count": 4}, 
                      cookies=session2_cookies)
response = client.post("/get_counter_value", cookies=session2_cookies)
assert response.json() == {"count": 4}

# Back to Session 1: still at 2
response = client.post("/get_counter_value", cookies=session1_cookies)
assert response.json() == {"count": 2}
```

---

## Best Practices

### 1. Always Use Session IDs

```python
# ✅ Good: Get session_id from request
async def my_tool(self, request: Request, body: ToolRequest):
    session_id = request.session[SESSION_ID_KEY]
    state = self.session_state[session_id]

# ❌ Bad: Using global state without session_id
class BadServer(SimpleResourcesServer):
    global_counter = 0  # Shared across all sessions!
    
    async def increment(self):
        self.global_counter += 1  # Wrong!
```

### 2. Handle Missing State Gracefully

```python
# ✅ Good: Use setdefault or get with default
counter = self.session_id_to_counter.setdefault(session_id, 0)
# or
counter = self.session_id_to_counter.get(session_id, 0)

# ❌ Bad: Assuming state exists
counter = self.session_id_to_counter[session_id]  # KeyError!
```

### 3. Clean Up Resources

For production systems, consider implementing cleanup:

```python
class MyResourcesServer(SimpleResourcesServer):
    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        
        # Add cleanup endpoint
        @app.post("/cleanup_session")
        async def cleanup_session(request: Request):
            session_id = request.session[SESSION_ID_KEY]
            # Clean up resources
            if session_id in self.session_state:
                self.cleanup_resources(self.session_state[session_id])
                del self.session_state[session_id]
            return {"status": "cleaned"}
        
        return app
```

### 4. Limit State Size

```python
# ✅ Good: Store minimal state
self.session_state[session_id] = {
    "counter": 5,
    "last_operation": "increment"
}

# ❌ Bad: Storing excessive data
self.session_state[session_id] = {
    "full_conversation_history": [...],  # Already in agent
    "all_intermediate_results": [...],   # Not needed
    "large_data_blob": "..." * 10000     # Use external storage
}
```

### 5. Use Type-Safe Models

```python
# ✅ Good: Pydantic models for requests and responses
class IncrementRequest(BaseModel):
    count: int

async def increment_counter(self, body: IncrementRequest):
    # body.count is validated as int
    pass

# ❌ Bad: Untyped dictionaries
async def increment_counter(self, body: dict):
    count = body["count"]  # No validation!
```

---

## Common Patterns

### Pattern: Accumulating Results

```python
class ResultAccumulator(SimpleResourcesServer):
    session_results: Dict[str, List[Any]] = Field(default_factory=dict)
    
    async def add_result(self, request: Request, body: ResultRequest):
        session_id = request.session[SESSION_ID_KEY]
        results = self.session_results.setdefault(session_id, [])
        results.append(body.result)
        return AddResultResponse(total_count=len(results))
    
    async def get_all_results(self, request: Request):
        session_id = request.session[SESSION_ID_KEY]
        results = self.session_results.get(session_id, [])
        return GetAllResultsResponse(results=results)
```

### Pattern: Conditional Tool Availability

```python
class ConditionalTools(SimpleResourcesServer):
    session_unlocked: Dict[str, Set[str]] = Field(default_factory=dict)
    
    async def unlock_tool(self, request: Request, body: UnlockRequest):
        session_id = request.session[SESSION_ID_KEY]
        unlocked = self.session_unlocked.setdefault(session_id, set())
        unlocked.add(body.tool_name)
        return UnlockResponse(success=True)
    
    async def advanced_tool(self, request: Request, body: ToolRequest):
        session_id = request.session[SESSION_ID_KEY]
        unlocked = self.session_unlocked.get(session_id, set())
        
        if "advanced_tool" not in unlocked:
            return ErrorResponse(error="Tool not unlocked yet")
        
        # Execute tool
        return ToolResponse(result="...")
```

### Pattern: Resource Management

```python
class FileManager(SimpleResourcesServer):
    session_files: Dict[str, Path] = Field(default_factory=dict)
    
    async def seed_session(self, request: Request, 
                          body: BaseSeedSessionRequest):
        session_id = request.session[SESSION_ID_KEY]
        # Create temporary directory for this session
        temp_dir = Path(f"/tmp/session_{session_id}")
        temp_dir.mkdir(exist_ok=True)
        self.session_files[session_id] = temp_dir
        return BaseSeedSessionResponse()
    
    async def write_file(self, request: Request, body: WriteFileRequest):
        session_id = request.session[SESSION_ID_KEY]
        session_dir = self.session_files[session_id]
        file_path = session_dir / body.filename
        file_path.write_text(body.content)
        return WriteFileResponse(path=str(file_path))
```

---

## Next Steps

Now that you understand multi-step interactions, explore:

- **Database Integration** (coming soon): Scale stateful servers with databases
- **Complex Environments**: See the `workplace_assistant` resource server for multi-step in action
- **Custom Resource Servers** (coming soon): Build your own stateful environment

---

## Reference Implementation

The complete `example_stateful_counter` implementation is available at:
- **Code**: `resources_servers/example_stateful_counter/app.py`
- **Tests**: `resources_servers/example_stateful_counter/tests/test_app.py`
- **Example Data**: `resources_servers/example_stateful_counter/data/`
- **README**: `resources_servers/example_stateful_counter/README.md`

---

## Summary

**Key Takeaways:**

1. **Multi-step** = multiple sequential tool calls within one trajectory
2. **Sessions** maintain state across tool calls using cookies
3. **Stateless** for independent tool calls, **stateful** for dependent operations
4. **State storage** options: in-memory (dev), database (production), external (complex)
5. **`seed_session`** initializes environment before agent starts
6. Always use `request.session[SESSION_ID_KEY]` to access session-specific state

**Remember:** Multi-step is about multiple tool calls in one task. This is separate from multi-turn conversations (multiple user-assistant exchanges).

