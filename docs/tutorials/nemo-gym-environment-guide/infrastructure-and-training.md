(building-environments-infrastructure)=

# Infrastructure, Training, and Next Steps

This section covers the server infrastructure that connects everything together, the RL training loop, and a step-by-step checklist for building your own environment.

---

## The Server Infrastructure

### HeadServer: Central Configuration

The HeadServer defaults to port 11000 and serves as the central registry:

```python
# simplified
class HeadServer(BaseServer):
    config: BaseServerConfig
    _server_instances: List[dict] = []

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.get("/global_config_dict_yaml")(self.global_config_dict_yaml)
        app.get("/server_instances")(self.get_server_instances)
        return app

    async def global_config_dict_yaml(self) -> str:
        return OmegaConf.to_yaml(get_global_config_dict())
```

### ServerClient: Inter-Service Communication

```python
# simplified
class ServerClient(BaseModel):
    head_server_config: BaseServerConfig
    global_config_dict: DictConfig

    async def request(
        self, server_name: str, url_path: str, method: str, **kwargs
    ) -> ClientResponse:
        server_config_dict = get_first_server_config_dict(self.global_config_dict, server_name)
        base_url = self._build_server_base_url(server_config_dict)

        if "json" in kwargs:
            json_obj = kwargs["json"]
            if isinstance(json_obj, BaseModel):
                kwargs["json"] = json_obj.model_dump(exclude_unset=True)

        return await request(method=method, url=f"{base_url}{url_path}", _internal=True, **kwargs)

    async def post(self, server_name: str, url_path: str, **kwargs) -> ClientResponse:
        return await self.request(server_name=server_name, url_path=url_path, method="POST", **kwargs)

    async def get(
        self,
        server_name: str,
        url_path: str,
        **kwargs
    ) -> ClientResponse:
        """Make HTTP GET to another server by name"""
        return await self.request(
            server_name=server_name,
            url_path=url_path,
            method="GET",
            **kwargs,
        )

    def poll_for_status(self, server_name: str) -> ServerStatus:
        """Check if a server is running"""
        # Returns: "success", "connection_error", "timeout", or "unknown_error"
```

### SimpleServer: Base Infrastructure

```python
# simplified
SESSION_ID_KEY = "session_id"

class SimpleServer(BaseServer):
    server_client: ServerClient

    @abstractmethod
    def setup_webserver(self) -> FastAPI:
        pass

    def setup_session_middleware(self, app: FastAPI) -> None:
        """Automatically assigns unique session IDs to each request"""
        @app.middleware("http")
        async def add_session_id(request: Request, call_next):
            if SESSION_ID_KEY not in request.session:
                request.session[SESSION_ID_KEY] = str(uuid4())
            return await call_next(request)

        session_middleware_key = self.get_session_middleware_key()
        app.add_middleware(
            SessionMiddleware,
            secret_key=session_middleware_key,
            session_cookie=session_middleware_key
        )

    def setup_exception_middleware(self, app: FastAPI) -> None:
        """Catches exceptions and returns formatted error responses"""
        @app.middleware("http")
        async def exception_handling_middleware(request: Request, call_next):
            try:
                return await call_next(request)
            except Exception as e:
                return JSONResponse(content=repr(e), status_code=500)

    @classmethod
    def run_webserver(cls) -> FastAPI:
        global_config_dict = get_global_config_dict()
        initialize_ray()

        server_config = cls.load_config_from_global_config()
        server_client = ServerClient(
            head_server_config=ServerClient.load_head_server_config(),
            global_config_dict=global_config_dict,
        )
        server = cls(config=server_config, server_client=server_client)

        app = server.setup_webserver()
        server.setup_exception_middleware(app)

        uvicorn.run(
            app,
            host=server.config.host,
            port=server.config.port,
            timeout_graceful_shutdown=0.5,
        )
        return app
```

---

## The RL Training Loop

Here's how a single rollout works end-to-end:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                               │
│    Agent calls POST /seed_session on ResourcesServer            │
│    → Environment initializes state for the task                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. GET INITIAL PROMPT                                           │
│    Agent loads prompt from training data (JSONL)                │
│    → System message + user query + tool definitions             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. INTERACTION LOOP                                             │
│                                                                 │
│    a. Agent calls POST /v1/responses on ModelServer             │
│       → LLM generates text OR function_call                     │
│                                                                 │
│    b. If function_call:                                         │
│       Agent calls POST /{tool_name} on ResourcesServer          │
│       → Environment executes tool, returns result               │
│       → Agent appends tool result to conversation               │
│                                                                 │
│    c. If termination condition met (max turns, done flag):      │
│       → Break loop                                              │
│                                                                 │
│    d. Else: Continue to step (a)                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. VERIFICATION                                                 │
│    Agent calls POST /verify on ResourcesServer                  │
│    → Environment computes reward by comparing response          │
│      to ground truth                                            │
│    → Returns reward: float for RL gradient computation          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. TRAINING                                                     │
│    Training pipeline uses (response, reward) pairs              │
│    → Compute policy gradients                                   │
│    → Update model weights                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Batch Rollout Collection

For efficiency, NeMo Gym supports batch rollout collection:

```python
# simplified
class RolloutCollectionConfig(BaseNeMoGymCLIConfig):
    """Perform a batch of rollout collection."""
    agent_name: Optional[str] = None
    input_jsonl_fpath: str
    output_jsonl_fpath: str
    limit: Optional[int] = None
    num_repeats: Optional[int] = None
    num_samples_in_parallel: Optional[int] = None
    responses_create_params: Dict[str, Any] = Field(default_factory=dict)
```

CLI command: `ng_collect_rollouts`

---

## Directory Structure for a New Environment

```
resources_servers/my_environment/
├── app.py                      # Main server implementation
├── client.py                   # Optional: client for testing
├── utils.py                    # Optional: helper functions
├── configs/
│   └── my_environment.yaml     # Configuration
├── data/
│   ├── train.jsonl             # Training data
│   ├── validation.jsonl        # Validation data
│   └── example.jsonl           # Example data for testing
├── tests/
│   └── test_app.py             # Unit tests
└── requirements.txt            # Dependencies (optional)
```

### Minimal app.py Template

```python
# simplified
from fastapi import FastAPI
from pydantic import BaseModel

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


class MyEnvironmentConfig(BaseResourcesServerConfig):
    pass


class MyToolRequest(BaseModel):
    # Define your tool parameters
    param1: str
    param2: int


class MyToolResponse(BaseModel):
    # Define your tool response
    result: str


class MyEnvironmentServer(SimpleResourcesServer):
    config: MyEnvironmentConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/my_tool")(self.my_tool)
        return app

    async def my_tool(self, body: MyToolRequest) -> MyToolResponse:
        # Implement your tool logic
        return MyToolResponse(result=f"Processed {body.param1}")

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # Implement your reward function
        reward = 1.0  # Compute based on body.response
        return BaseVerifyResponse(**body.model_dump(), reward=reward)


if __name__ == "__main__":
    MyEnvironmentServer.run_webserver()
```

---

## Summary: Building Your Environment

### Step-by-Step Checklist

1. **Create the directory structure**
   ```bash
   # simplified
   mkdir -p resources_servers/my_env/{configs,data,tests}
   touch resources_servers/my_env/app.py
   touch resources_servers/my_env/configs/my_env.yaml
   ```

2. **Define your request/response models** with Pydantic
   - Tool input parameters
   - Tool output format
   - Extended verify request with ground truth fields
   - Extended verify response with metrics

3. **Implement your Resource Server class**
   - Inherit from `SimpleResourcesServer`
   - Override `setup_webserver()` to add tool endpoints
   - Implement `verify()` with your reward function
   - Optionally override `seed_session()` for stateful environments

4. **Write your configuration YAML**
   - Define resource server with domain and entrypoint
   - Define agent with server references
   - Specify datasets with paths and licenses

5. **Create training data in JSONL format**
   - System prompts with task instructions
   - Tool definitions
   - Ground truth for reward computation

6. **Test your environment**
   ```bash
   # simplified
   python resources_servers/my_env/app.py
   # Or via CLI:
   ng_run +config=resources_servers/my_env/configs/my_env.yaml
   ```

7. **Generate initial data (start small, then scale)**
   - Create a tiny `data/example.jsonl` first (5-20 rows) so you can debug the full loop quickly.
   - Then expand to `data/train.jsonl` / `data/validation.jsonl` once verification is stable.
   - Common approaches:
     - Hand-written seeds (highest quality, lowest scale)
     - Synthetic Data Generation (using LLM) for tasks with automatic checks + spot review
     - Programmatic task generators (best for templated problems)

8. **Collect rollouts (turn tasks into scored trajectories)**
   - Once servers are running, use `ng_collect_rollouts` to produce rollout traces you can inspect and/or use for offline training.
   - The official walkthrough is in `docs/get-started/rollout-collection.md`. A typical command looks like:

   ```bash
   # simplified
   ng_collect_rollouts +agent_name=my_env_simple_agent \
       +input_jsonl_fpath=resources_servers/my_env/data/example.jsonl \
       +output_jsonl_fpath=results/my_env_rollouts.jsonl \
       +limit=100 \
       +num_repeats=4 \
       +num_samples_in_parallel=10
   ```

9. **Train (Gym provides environments; training happens in an RL framework)**
   - NeMo Gym's job is to run the environment, collect rollouts, and compute rewards.
   - For GRPO training with NeMo RL, see `docs/tutorials/nemo-rl-grpo/index.md` (end-to-end training on Workplace Assistant).
   - If you're doing offline training from collected rollouts, see `docs/tutorials/offline-training-w-rollouts.md`.

### Key Differentiators from Traditional RL

| Traditional RL | NeMo Gym |
|----------------------------|----------|
| Python function calls | HTTP microservices |
| Single process | Distributed architecture |
| `step()`, `reset()` | `seed_session()`, tool endpoints, `verify()` |
| Observation/action spaces | OpenAI Responses API format |
| In-memory state | Session-based state with `SESSION_ID_KEY` |

The key differentiator is that NeMo Gym environments are HTTP microservices, enabling:
- **Distributed training** across multiple machines
- **Horizontal scaling** of environment instances
- **Composition** of multiple environments in complex training scenarios
- **Language-agnostic** tool implementations (anything that speaks HTTP)

---

> **Previous**: {ref}`Configuration and Training Data <building-environments-configuration>` | {ref}`Back to Guide Overview <building-environments-index>`
