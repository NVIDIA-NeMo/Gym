(configuration-concepts)=

# Configuration System

NeMo Gym uses YAML configuration files to define your [Model, Resources, and Agent servers](./core-abstractions). Each server gets its own configuration block, giving you modular control over your entire training environment.

## How Servers Connect

A training environment typically includes all three server types working together. The Agent server config specifies which Model and Resources servers to use by referencing their server IDs. This wiring is what ties your training environment together — the Agent knows which Model to call and which Resources to use.

## Config File Locations

Each server type has a dedicated directory with its implementations and their configs:

```text
# Model Server Config
responses_api_models/
  └── openai_model/
      └── configs/openai_model.yaml

# Resources Server Config
resources_servers/
  └── example_simple_weather/
      └── configs/simple_weather.yaml

# Agent Server Config
responses_api_agents/
  └── simple_agent/
      └── configs/simple_agent.yaml
```

## Server Block Structure

Each config file defines a server using this structure:

```yaml
server_id:              # Your unique name for this server
  server_type:          # Model, resources, or agent
    implementation:     # Which implementation directory to use
      entrypoint:       # The code file to run
```

| Field | Permitted Values |
|-------|------------------|
| **Server ID** | Any name you choose |
| **Server Type** | `responses_api_models`, `resources_servers`, or `responses_api_agents` |
| **Implementation** | Must match a directory name inside the specified server type directory |
| **Entrypoint** | A Python file in the implementation directory (e.g., `app.py`) |

Different server types have additional required fields (e.g., `domain` for resources servers, `resources_server` and `model_server` for agents). See {doc}`/reference/configuration` for complete field specifications.

In many config files in NeMo Gym, you'll see the same name used for both server ID and implementation:

```yaml
example_simple_weather:        # ← Server ID
  resources_servers:
    example_simple_weather:    # ← Implementation
```

These serve different purposes:

- **Server ID** (`example_simple_weather` on line 1): Your chosen identifier for this server instance. Used in API requests and when other servers reference it. You could name it `my_weather` or `weather_prod` instead.

- **Implementation** (`example_simple_weather` on line 3): Must match the folder `resources_servers/example_simple_weather/`. This tells NeMo Gym which code to run.

Examples often use matching names for simplicity, but the two values are independent choices.

## Policy Model Variables

NeMo Gym provides three standard placeholders for "the model being trained":

- `policy_base_url` - Model API endpoint
- `policy_api_key` - Authentication key
- `policy_model_name` - Model identifier

These let you reference the training model consistently across configs without hardcoding values. Define them once in `env.yaml` at the repository root, then use `${variable_name}` syntax anywhere you need them.

---

:::{seealso}
- {doc}`/reference/configuration` for complete syntax and field specifications
- {doc}`/troubleshooting/configuration` for common errors
:::

