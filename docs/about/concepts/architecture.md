(architecture)=

# Architecture

This section describes how NeMo Gym components interact during startup and execution. For an overview of the three server types (Model, Resources, Agent), see {ref}`core-components`.

## Control Plane: Server Startup

When you run `ng_run`, the system starts up in four phases:

```{mermaid}
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333', 'primaryColor': '#e3f2fd', 'secondaryColor': '#f5f5f5'}}}%%
flowchart LR
    A[Parse CLI] --> B[Load Configs]
    B --> C[Init Ray]
    C --> D[Start Servers]
```

### Phase 1: Parse CLI

The `ng_run` command uses Hydra to parse command-line arguments. Users specify configuration files via `+config_paths`:

```bash
ng_run "+config_paths=[resources_servers/math/configs/math.yaml, responses_api_models/openai_model/configs/openai_model.yaml]"
```

### Phase 2: Load and Merge Configs

Configuration is loaded from multiple sources in order of priority (later sources override earlier):

1. YAML files specified in `config_paths`
2. Local `env.yaml` file (for sensitive values like API keys)
3. Command-line arguments (highest priority)

**Port allocation**: Users can explicitly specify `host` and `port` in their config. If not provided, the framework automatically allocates ports from available system ports, tracking used ports to prevent conflicts.

### Phase 3: Initialize Ray

The system initializes a Ray cluster for distributed coordination. If `ray_head_node_address` is specified in the config, it connects to an existing cluster; otherwise, it starts a new one.

### Phase 4: Start Servers

Servers are started in two stages:

```{mermaid}
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
flowchart TB
    Main["Main Process<br/>ng_run"]
    Head["Head Server<br/>Port 11000"]
    Model["Model Server<br/>Own venv + port"]
    Agent["Agent Server<br/>Own venv + port"]
    Resources["Resources Server<br/>Own venv + port"]
    
    Main --> Head
    Main -->|spawn| Model
    Main -->|spawn| Agent
    Main -->|spawn| Resources
```

1. **Head Server**: Started as a background thread in the main process. Provides endpoints for config discovery (`/global_config_dict_yaml`) and server instance listing (`/server_instances`).

2. **Server Subprocesses**: Each configured server is spawned as an independent OS process:
   - Each server has its own Python virtual environment in order to isolate dependencies.
   - Each runs uvicorn with a FastAPI application listening on `http://{host}:{port}`.
   - The global config is passed via environment variable `NEMO_GYM_CONFIG_DICT`.
   - The specific server identity is passed via `NEMO_GYM_CONFIG_PATH`.
   - Server URLs are registered in the global config, allowing other servers to discover and call them.

3. **Health Check**: The main process polls each server's HTTP endpoint until all return 200, then reports "All servers ready!"

## Running State

Once all servers are healthy, the system enters steady state:

- The main process sleeps and periodically polls subprocess health
- Each server process runs its own uvicorn event loop, handling requests asynchronously
- Servers communicate with each other only via HTTP (no shared memory)
- Session state is maintained via cookies for multi-step rollouts

## Shutdown

When the user presses Ctrl+C (or the process receives SIGINT):

1. SIGINT is forwarded to all server subprocesses
2. Main process waits for subprocesses to terminate (with timeout)
3. Head server thread is stopped
4. Process exits cleanly

---

## HTTP Request Flow: Example

During a single rollout, servers communicate via HTTP. This example shows a math problem with tool use:

```{mermaid}
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
sequenceDiagram
    autonumber
    participant Agent as Agent
    participant Model as Model
    participant Resources as Resources
    
    Note over Agent,Resources: Initialize
    Agent->>Resources: POST /seed_session
    Resources-->>Agent: OK
    
    Note over Agent,Resources: First generation
    Agent->>Model: POST /v1/responses
    Model-->>Agent: function_call: calculate
    
    Note over Agent,Resources: Tool execution
    Agent->>Resources: POST /calculate
    Resources-->>Agent: "4"
    
    Note over Agent,Resources: Second generation
    Agent->>Model: POST /v1/responses
    Model-->>Agent: message: "The answer is 4"
    
    Note over Agent,Resources: Verification
    Agent->>Resources: POST /verify
    Resources-->>Agent: reward: 1.0
```

**Key Design Points:**

- **HTTP-only communication**: All servers communicate via HTTP, enabling language-agnostic implementations and deployment flexibility
- **Stateless model servers**: Model servers perform single-call generation without memory; the agent maintains conversation state
- **Session state in resources**: Resources servers use session cookies to maintain per-rollout state across multiple tool calls
- **OpenAI API compatibility**: Model servers expose `/v1/responses` endpoints compatible with the OpenAI Responses API
- **uvicorn + FastAPI**: All servers use uvicorn as the ASGI server with FastAPI for HTTP routing and request handling

---

## Data Plane: Rollout Collection

When you run `ng_collect_rollouts`, the system collects training data by executing rollouts in parallel:

```{mermaid}
%%{init: {'theme': 'default', 'themeVariables': { 'lineColor': '#5c6bc0', 'primaryTextColor': '#333'}}}%%
sequenceDiagram
    autonumber
    participant Client as Client
    participant Head as Head Server
    participant Agent as Agent
    participant Model as Model
    participant Resources as Resources
    
    Client->>Head: GET /global_config_dict_yaml
    Head-->>Client: Server addresses
    
    Client->>Agent: POST /run
    
    loop Each prompt in parallel
        Agent->>Resources: Seed session
        Resources-->>Agent: OK
        
        loop Generation loop
            Agent->>Model: Generate response
            Model-->>Agent: Output
            
            opt Tool calls
                Agent->>Resources: Execute tool
                Resources-->>Agent: Result
            end
        end
        
        Agent->>Resources: Verify and score
        Resources-->>Agent: Reward
    end
    
    Agent-->>Client: Results with rewards
```

The client first queries the Head Server to discover server addresses from the global config, then reads input JSONL and dispatches prompts to the Agent. Completed rollouts are written to output JSONL.

**Concurrency behavior differs by use case:**
- **Standalone rollout collection** (`ng_collect_rollouts`): A semaphore gates concurrency via `num_samples_in_parallel` to control load.
- **Training framework integration** (e.g., NeMo RL): All requests are sent without gating; the training framework manages concurrency externally.
