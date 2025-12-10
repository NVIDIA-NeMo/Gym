(configuration-concepts)=

# Configuration System

NeMo Gym uses YAML configuration files to define [Model, Resources, and Agent servers](./core-components.md). Each server gets its own configuration block, providing modular control over the entire training environment.

## How Servers Connect

A training environment typically includes all three server types working together. The Agent server config specifies which Model and Resources servers to use by referencing their server IDs. These references connect each training environment together — the Agent knows which Model to call and which Resources to use.

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
server_id:                    # Your unique name for this server
  server_type:                # responses_api_models | resources_servers | responses_api_agents
    implementation:           # Directory name inside the server type directory
      entrypoint: app.py      # Python file to run
      # ... additional fields vary by server type
```

Different server types have additional required fields (e.g., `domain` for resources servers, `resources_server` and `model_server` for agents). See {doc}`/reference/configuration` for complete field specifications.

Config files in NeMo Gym often use the same name for both server ID and implementation:

```yaml
example_simple_weather:        # ← Server ID
  resources_servers:
    example_simple_weather:    # ← Implementation
```

These serve different purposes:

- **Server ID** (`example_simple_weather` on line 1): Your chosen identifier for this server instance. Used in API requests and when other servers reference it. You could name it `my_weather` or `weather_prod` instead.

- **Implementation** (`example_simple_weather` on line 3): Must match the folder `resources_servers/example_simple_weather/`. This tells NeMo Gym which code to run.

Examples often use matching names for simplicity, but the two values are independent choices.

---

:::{seealso}
- {doc}`/reference/configuration` for complete syntax and field specifications
- {doc}`/troubleshooting/configuration` for troubleshooting configuration related errors
:::

