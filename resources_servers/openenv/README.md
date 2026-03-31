# OpenEnv Resource Server

Use any [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment as a NeMo-Gym resource server. Point the adapter at an environment class via YAML config and it handles the rest -- session management, tool endpoints, reward accumulation, and verification.

## Included Environments

| Environment | Type | Description |
|-------------|------|-------------|
| **Echo** | MCP | Echoes messages back with length-based rewards. Tools: `echo_message`, `echo_with_length` |
| **Coding** | Non-MCP | Execute Python code, get stdout/stderr. Tool: `step` (code string) |
| **Maze** | Non-MCP | Navigate an 8x8 maze to find the exit. Tool: `step` (direction: 0-3) |

## Quick Start

All commands below should be run from the **NeMo-Gym repository root** (the directory containing the top-level `pyproject.toml`).

### 1. Setup

```bash
# Set up the project venv
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra dev
```

### 2. Configure API Credentials

Create an `env.yaml` file at the repo root (auto-loaded by NeMo-Gym, already gitignored):

```yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: <your-api-key>
policy_model_name: <your-model-name>
```

### 3. Start Servers and Collect Rollouts

Running an environment is a **two-step process**:

1. **Start the servers** with `ng_run` — this launches the resource server, agent server, and model server. Wait until you see `All 3 / 3 servers ready!`.
2. **Collect rollouts** with `ng_collect_rollouts` in a **separate terminal** — this sends the JSONL prompts through the agent, which calls the LLM and environment in a loop, and writes trajectories with rewards to an output file.

#### Echo

```bash
# Terminal 1: Start servers
source .venv/bin/activate
ng_run "+config_paths=[resources_servers/openenv/configs/openenv_echo.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"

# Terminal 2: Collect rollouts (after "All 3 / 3 servers ready!")
source .venv/bin/activate
ng_collect_rollouts \
  +agent_name=openenv_echo_simple_agent \
  +input_jsonl_fpath=resources_servers/openenv/data/echo/example.jsonl \
  +output_jsonl_fpath=output_echo.jsonl \
  +num_samples_in_parallel=5
```

#### Coding

```bash
# Terminal 1: Start servers
source .venv/bin/activate
ng_run "+config_paths=[resources_servers/openenv/configs/openenv_coding.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"

# Terminal 2: Collect rollouts
source .venv/bin/activate
ng_collect_rollouts \
  +agent_name=openenv_coding_simple_agent \
  +input_jsonl_fpath=resources_servers/openenv/data/coding/example.jsonl \
  +output_jsonl_fpath=output_coding.jsonl \
  +num_samples_in_parallel=5
```

#### Maze

```bash
# Terminal 1: Start servers
source .venv/bin/activate
ng_run "+config_paths=[resources_servers/openenv/configs/openenv_maze.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"

# Terminal 2: Collect rollouts
source .venv/bin/activate
ng_collect_rollouts \
  +agent_name=openenv_maze_simple_agent \
  +input_jsonl_fpath=resources_servers/openenv/data/maze/example.jsonl \
  +output_jsonl_fpath=output_maze.jsonl \
  +num_samples_in_parallel=5
```

## Adding a New Environment

Adding a new OpenEnv environment requires **no Python code** -- only a YAML config, JSONL data, and the environment's package in `requirements.txt`.

### Step 1: Create a YAML Config

Create `resources_servers/openenv/configs/openenv_<name>.yaml`. The inner key under `resources_servers:` must be `openenv` (matching the directory name):

```yaml
openenv_<name>_resources_server:
  resources_servers:
    openenv:
      entrypoint: app.py
      domain: <domain>
      verified: false
      description: "<description>"
      env_class: "<installed.package.EnvironmentClass>"
      action_class: "<installed.package.ActionClass>"
      is_mcp: false           # true for MCP environments
      # reset_kwargs: {}      # optional kwargs for env.reset()

openenv_<name>_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: openenv_<name>_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: example
        type: example
        jsonl_fpath: resources_servers/openenv/data/<name>/example.jsonl
```

**Config fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `env_class` | Yes | Dotted import path to the installed `Environment` subclass (e.g., `echo_env.server.echo_environment.EchoEnvironment`). Use the **installed package** import path, not the OpenEnv repo path. |
| `action_class` | Yes | Dotted import path to the `Action` model. For MCP environments, use `openenv.core.env_server.mcp_types.CallToolAction` |
| `is_mcp` | No | Default `false`. Set `true` for MCP environments -- the adapter will discover tools at startup via `ListToolsAction` and register per-tool endpoints |
| `reset_kwargs` | No | Default `{}`. Dict of keyword arguments passed to `env.reset()` on each new session |

### Step 2: Create JSONL Example Data

Create `resources_servers/openenv/data/<name>/example.jsonl` with 5+ examples:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "developer", "content": "System prompt describing the environment and available tools."},
      {"role": "user", "content": "The task for the agent to complete."}
    ],
    "tools": [
      {
        "name": "step",
        "type": "function",
        "description": "What the tool does",
        "parameters": {
          "type": "object",
          "properties": {"param": {"type": "string", "description": "Param description"}},
          "required": ["param"],
          "additionalProperties": false
        },
        "strict": true
      }
    ]
  }
}
```

- **Non-MCP environments:** Define a single `"step"` tool whose parameters match the Action model's fields.
- **MCP environments:** Define one tool per MCP tool, matching the names and schemas the environment exposes.

### Step 3: Add Dependencies and Run

Add the environment's package to `resources_servers/openenv/requirements.txt`:

```
openenv-my-env @ git+https://github.com/meta-pytorch/OpenEnv.git#subdirectory=envs/my_env
```

Then start servers and collect rollouts:

```bash
# Terminal 1: Start servers
source .venv/bin/activate
ng_run "+config_paths=[resources_servers/openenv/configs/openenv_<name>.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"

# Terminal 2: Collect rollouts (after servers are ready)
source .venv/bin/activate
ng_collect_rollouts \
  +agent_name=openenv_<name>_simple_agent \
  +input_jsonl_fpath=resources_servers/openenv/data/<name>/example.jsonl \
  +output_jsonl_fpath=output_<name>.jsonl \
  +num_samples_in_parallel=5
```

## How It Works

1. **`POST /seed_session`** — Creates an environment instance and calls `env.reset()`.
2. **`POST /<tool_name>`** (MCP) or **`POST /step`** (non-MCP) — Calls `env.step(action)` and accumulates reward.
3. **`POST /verify`** — Returns accumulated reward, closes the environment, cleans up session. If `verifier_metadata.expected_output` is present, checks stdout match instead.

For MCP environments, tool endpoints are discovered at startup via `ListToolsAction`.

## Tests

```bash
python -m pytest resources_servers/openenv/tests/test_app.py -v
```
