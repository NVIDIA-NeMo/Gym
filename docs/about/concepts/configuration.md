(configuration-concepts)=

# Configuration System

NeMo Gym servers are defined in YAML configuration files. Each file specifies one or more servers that can be started together.

## Config File Structure

Every config file follows this three-level nesting pattern:

```yaml
server_id:                    # Level 1: Unique name (used in API requests)
  server_type:                # Level 2: resources_servers, responses_api_models, or responses_api_agents
    implementation:           # Level 3: Folder name inside the server type directory
      entrypoint: app.py      # Server-specific fields...
```

**Example** - a complete resources server config:

```yaml
my_weather:                      # Server ID (you choose this)
  resources_servers:             # Server type (determines which folder)
    example_simple_weather:      # Implementation (must match folder name)
      entrypoint: app.py
      domain: agent
      description: "Weather lookup for agent training"
```

:::{note}
The server ID and implementation name are independent:

- **Server ID** (`my_weather`): Your chosen identifier, used in API requests and cross-references
- **Implementation** (`example_simple_weather`): Must match an existing folder in `resources_servers/`

In many examples, both names are the same, which can be confusing. They serve different purposes.
:::

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

These enable consistent references across servers. Define them in `env.yaml` and reference them in config files using `${variable_name}` syntax. Refer to {doc}`/reference/configuration` for complete syntax.

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
