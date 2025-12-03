(configuration-concepts)=

# Configuration System

NeMo Gym servers are defined in YAML configuration files. Each file specifies one or more servers that can be started together.

## Config File Structure

Every config file follows this pattern:

```yaml
server_id:                    # Unique name (used in API requests)
  server_type:                # One of: responses_api_models, resources_servers, responses_api_agents
    implementation:           # Folder name inside the server type directory
      entrypoint: app.py      # Python file to run
      # Additional fields vary by server type...
```

**Example** - a complete resources server config:

```yaml
simple_weather_resource:
  resources_servers:
    example_simple_weather:
      entrypoint: app.py
      domain: agent
      description: "Weather lookup for agent training"
```

## Server Types

The three server types map directly to top-level folders in the repository:

| Server Type | Repository Folder | Purpose |
|-------------|-------------------|---------|
| `responses_api_models` | `responses_api_models/` | LLM inference endpoints |
| `resources_servers` | `resources_servers/` | Tools and verification logic |
| `responses_api_agents` | `responses_api_agents/` | Orchestration between models and resources |

Config files live in each implementation's `configs/` subfolder:

```text
resources_servers/
  example_simple_weather/
    configs/
      simple_weather.yaml    ← config file
    app.py                   ← entrypoint
```

## Policy Model Variables

NeMo Gym provides standard placeholders for the model being trained:

- `policy_base_url` - Model API endpoint
- `policy_api_key` - Authentication key
- `policy_model_name` - Model identifier

These enable consistent references to "the model being trained" across different servers. Define them in `env.yaml`:

```yaml
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-key
policy_model_name: gpt-4o-2024-11-20
```

Reference them in config files using `${variable_name}`:

```yaml
openai_base_url: ${policy_base_url}
openai_api_key: ${policy_api_key}
openai_model: ${policy_model_name}
```

## Configuration Resolution

NeMo Gym resolves configuration from three sources in priority order:

```text
Server YAML Config Files  <  env.yaml  <  Command Line Arguments
    (lowest priority)                       (highest priority)
```

This separation enables:

- **Base configuration** in YAML files for shared, version-controlled settings
- **Secrets** in `env.yaml` for API keys and environment-specific values (never committed)
- **Runtime overrides** via command line for experimentation and testing

---

:::{tip}
Refer to {doc}`/reference/configuration` for complete syntax and options, or {doc}`/troubleshooting/configuration` if you encounter errors.
:::
