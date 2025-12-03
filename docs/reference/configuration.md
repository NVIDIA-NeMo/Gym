(configuration-reference)=

# Configuration Reference

Complete syntax and options for NeMo Gym configuration.

:::{seealso}
{doc}`/about/concepts/configuration` for understanding the three-level config pattern.
:::

## File Locations

| File | Location | Version Control |
|------|----------|-----------------|
| Server configs | `<server_type>/<implementation>/configs/*.yaml` | ✅ Committed |
| env.yaml | Repository root (`./env.yaml`) | ❌ Gitignored (user creates) |

**Example paths**:

```text
resources_servers/example_simple_weather/configs/simple_weather.yaml
responses_api_models/openai_model/configs/openai_model.yaml
responses_api_agents/simple_agent/configs/simple_agent.yaml
./env.yaml  ← you create this
```

---

## Config File Structure

Each config file defines one or more server instances using three-level nesting:

```yaml
server_id:                       # Level 1: Your chosen name (used in API requests)
  server_type:                   # Level 2: resources_servers | responses_api_models | responses_api_agents
    implementation:              # Level 3: Must match a folder inside the server type directory
      entrypoint: app.py         # Required: Python file to run
      # Additional fields vary by server type...
```

**Example**:

```yaml
my_agent:                        # Server ID (you choose this)
  responses_api_agents:          # Server type
    simple_agent:                # Implementation (must match folder: responses_api_agents/simple_agent/)
      entrypoint: app.py
      # ...
```

:::{tip}
The server ID and implementation name are independent. Use descriptive server IDs that reflect your use case (for example, `math_tutor_agent`), while the implementation name must match an existing folder.
:::

### Server Types

| Type | Purpose | Required Fields |
|------|---------|-----------------|
| `responses_api_models` | LLM inference endpoints | `entrypoint` |
| `resources_servers` | Tools and verification logic | `entrypoint`, `domain` |
| `responses_api_agents` | Orchestration between models and resources | `entrypoint`, `resources_server`, `model_server` |

### Resources Server Fields

```yaml
my_resource:
  resources_servers:
    my_implementation:
      entrypoint: app.py
      domain: math              # Required (see domain values below)
      verified: false           # Optional: marks server as production-ready
      description: "Short description"  # Optional
```

**Domain values**: `math`, `coding`, `agent`, `knowledge`, `instruction_following`, `long_context`, `safety`, `games`, `e2e`, `other`

### Agent Server Fields

Agent servers reference other servers using the `type` and `name` pattern:

```yaml
my_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_resource       # Must match a loaded server ID
      model_server:
        type: responses_api_models
        name: policy_model      # Must match a loaded server ID
      datasets:                 # Optional: define associated datasets
        - name: example
          type: example         # example | train | validation
          jsonl_fpath: path/to/data.jsonl
```

### Model Server Fields

```yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_base_url: ${policy_base_url}
      openai_api_key: ${policy_api_key}
      openai_model: ${policy_model_name}
```

---

## env.yaml Options

Store secrets and environment-specific values that should not be committed to version control.

```yaml
# Required for OpenAI-compatible models
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-api-key
policy_model_name: gpt-4o-2024-11-20

# Optional: store config paths for reuse
my_config_paths:
  - responses_api_models/openai_model/configs/openai_model.yaml
  - resources_servers/example_simple_weather/configs/simple_weather.yaml

# Optional: validation behavior
error_on_almost_servers: true   # Default: true (exit on invalid configs)
```

---

## Command Line Overrides

Override any configuration value at runtime using Hydra syntax. Command line arguments have the highest priority.

```bash
# Load config files
ng_run "+config_paths=[config1.yaml,config2.yaml]"

# Override nested values (use dot notation after server ID)
ng_run "+config_paths=[config.yaml]" \
    +my_server.resources_servers.my_impl.domain=coding

# Override model selection
ng_run "+config_paths=[config.yaml]" \
    +policy_model_name=gpt-4o-mini

# Use paths stored in env.yaml
ng_run '+config_paths=${my_config_paths}'

# Disable strict validation
ng_run "+config_paths=[config.yaml]" +error_on_almost_servers=false
```

---

## Policy Model Variables

Standard placeholders resolved from `env.yaml` for consistent model references across configurations.

| Variable | Description | Example |
|----------|-------------|---------|
| `policy_base_url` | Model API endpoint | `https://api.openai.com/v1` |
| `policy_api_key` | Authentication key | `sk-...` |
| `policy_model_name` | Model identifier | `gpt-4o-2024-11-20` |

Reference in config files using `${variable_name}`:

```yaml
openai_base_url: ${policy_base_url}
openai_api_key: ${policy_api_key}
openai_model: ${policy_model_name}
```

---

## Dataset Configuration

Define datasets associated with servers for training data collection.

```yaml
datasets:
  - name: my_dataset
    type: example           # example | train | validation
    jsonl_fpath: path/to/data.jsonl
    num_repeats: 1          # Optional: repeat dataset (default: 1)
    license: "Apache 2.0"   # Required for train/validation types
```

**License values**: `Apache 2.0`, `MIT`, `Creative Commons Attribution 4.0 International`, `Creative Commons Attribution-ShareAlike 4.0 International`, `TBD`

**Dataset types**:
- `example`: For testing and development
- `train`: Requires `license` and GitLab identifier
- `validation`: Requires `license` and GitLab identifier

---

:::{seealso}
{doc}`/troubleshooting/configuration` for common configuration errors and solutions.
:::
