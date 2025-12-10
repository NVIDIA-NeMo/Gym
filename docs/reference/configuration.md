(configuration-reference)=

# Configuration Reference

Complete syntax and field specifications for NeMo Gym configuration files.

:::{seealso}
{doc}`/about/concepts/configuration` for understanding how configs are structured and how servers connect.
:::

## File Locations

| File | Location | Version Control |
|------|----------|-----------------|
| Server configs | `<server_type>/<implementation>/configs/*.yaml` | ✅ Committed |
| env.yaml | Repository root (`./env.yaml`) | ❌ Gitignored (user creates) |

---

## Server Configuration

All servers share this structure:

```yaml
server_id:                    # Your unique name for this server
  server_type:                # responses_api_models | resources_servers | responses_api_agents
    implementation:           # Directory name inside the server type directory
      entrypoint: app.py      # Python file to run
      # ... additional fields vary by server type
```

### Model Server Fields

```yaml
policy_model:                                 # Server ID (use "policy_model" — agent configs expect this name)
  responses_api_models:                       # Server type (must be "responses_api_models" for model servers)
    openai_model:                             # Implementation (use "openai_model", "vllm_model", or "azure_openai_model")
      entrypoint: app.py                      # Python file to run
      openai_base_url: ${policy_base_url}     # API endpoint URL
      openai_api_key: ${policy_api_key}       # Authentication key
      openai_model: ${policy_model_name}      # Model identifier
```

:::{tip}
Keep the server ID as `policy_model` — agent configs reference this name by default. The `${policy_base_url}`, `${policy_api_key}`, and `${policy_model_name}` placeholders should be defined in `env.yaml` at the repository root, allowing you to change model settings in one place.
:::

### Resources Server Fields

```yaml
my_resource:                                  # Server ID (your choice — agents reference this name)
  resources_servers:                          # Server type (must be "resources_servers" for resources servers)
    example_simple_weather:                   # Implementation (must match a directory in resources_servers/)
      entrypoint: app.py                      # Python file to run
      domain: agent                           # Server category (see values below)
      verified: false                         # Passed reward profiling and training checks (default: false)
      description: "Short description"        # Server description
      value: "What this improves"             # Training value provided
```

**Domain values:** `math`, `coding`, `agent`, `knowledge`, `instruction_following`, `long_context`, `safety`, `games`, `e2e`, `other` (see {py:class}`~nemo_gym.config_types.Domain`)

### Agent Server Fields

Agent servers must include both a `resources_server` and `model_server` block to specify which servers to use.

```yaml
my_agent:                                     # Server ID (your choice — used in API requests)
  responses_api_agents:                       # Server type (must be "responses_api_agents" for agent servers)
    simple_agent:                             # Implementation (must match a directory in responses_api_agents/)
      entrypoint: app.py                      # Python file to run
      resources_server:                       # Specifies which resources server to use
        type: resources_servers               # Always "resources_servers"
        name: my_resource                     # Server ID of the resources server
      model_server:                           # Specifies which model server to use
        type: responses_api_models            # Always "responses_api_models"
        name: policy_model                    # Server ID of the model server
      datasets:                               # Optional: define for training workflows
        - name: train                         # Dataset identifier
          type: train                         # example | train | validation
          jsonl_fpath: path/to/data.jsonl     # Path to data file
          license: Apache 2.0                 # Required for train/validation
```

#### Dataset Configuration

Define datasets associated with agent servers for training and evaluation.

```yaml
datasets:
  - name: my_dataset
    type: train
    jsonl_fpath: path/to/data.jsonl
    license: Apache 2.0
    num_repeats: 1
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Dataset identifier |
| `type` | Yes | `example`, `train`, or `validation` |
| `jsonl_fpath` | Yes | Path to data file |
| `license` | For train/validation | License identifier (see values below) |
| `num_repeats` | No | Repeat dataset n times (default: `1`) |

**Dataset types:**
- `example` — For testing and development
- `train` — Training data (requires `license`)
- `validation` — Evaluation data (requires `license`)

**License values:** `Apache 2.0`, `MIT`, `Creative Commons Attribution 4.0 International`, `Creative Commons Attribution-ShareAlike 4.0 International`, `TBD` (see {py:attr}`~nemo_gym.config_types.DatasetConfig.license`)

---

## Local Configuration (env.yaml)

Store secrets and local settings at the repository root. This file is gitignored.

```yaml
# Policy model (required for most setups)
# Reference these variables in server configs using `${variable_name}` syntax (e.g., `${policy_base_url}`)
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-api-key
policy_model_name: gpt-4o-2024-11-20

# Optional: store config paths for reuse
my_config_paths:
  - responses_api_models/openai_model/configs/openai_model.yaml
  - resources_servers/example_simple_weather/configs/simple_weather.yaml

# Optional: validation behavior
error_on_almost_servers: true   # Exit on invalid configs (default: true)
```

---

## Command Line Usage

Run servers using `ng_run`, NeMo Gym uses [Hydra](https://hydra.cc/) for configuration management."

### Loading Configs

```bash
# Load one or more config files
ng_run "+config_paths=[config1.yaml,config2.yaml]"

# Use paths stored in env.yaml
ng_run "+config_paths=${my_config_paths}"
```

### Overriding Values

```bash
# Override nested values (use dot notation after server ID)
ng_run "+config_paths=[config.yaml]" \
    +my_server.resources_servers.my_impl.domain=coding

# Override policy model
ng_run "+config_paths=[config.yaml]" \
    +policy_model_name=gpt-4o-mini

# Disable strict validation
ng_run "+config_paths=[config.yaml]" +error_on_almost_servers=false
```

---

## Troubleshooting

:::{seealso}
{doc}`/troubleshooting/configuration` for common configuration errors and solutions.
:::
