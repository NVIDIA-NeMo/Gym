(configuration-management)=

# Configuration Management

NeMo Gym uses a powerful configuration system with three sources that are resolved in this order:

```
Server YAML Config Files  <  env.yaml  <  Command Line Arguments
    (lowest priority)                       (highest priority)
```

This allows for:
- Base configuration in YAML files (shared settings)
- Secrets and environment-specific values in `env.yaml` 
- Runtime overrides via command line arguments

::::{tab-set}

:::{tab-item} 1. Server YAML Config Files

These are your base configurations that define server structures and default values. Later files override earlier files.

Example: Multi-Server Configuration
```bash
# Define which config files to load
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_simple_weather/configs/simple_weather.yaml,\
responses_api_agents/simple_agent/configs/simple_agent.yaml"

ng_run "+config_paths=[${config_paths}]"
```

Every config file defines **server instances** with this hierarchy:

```yaml
# Server ID - unique name used in requests and references
simple_weather_simple_agent:
  # Server type - must be one of: responses_api_models, resources_servers, responses_api_agents
  # These match the 3 top-level folders in NeMo-Gym
  responses_api_agents:
    # Implementation type - must match a folder name inside responses_api_agents/
    simple_agent:
      # Entrypoint - Python file to run (relative to implementation folder)
      entrypoint: app.py
      # Server-specific configuration (varies by implementation)
      resources_server:
        type: resources_servers               # What type of server to reference
        name: simple_weather                  # Which specific server instance
      model_server:
        type: responses_api_models
        name: policy_model                    # References the model server
```

:::
:::{tab-item} 2. env.yaml

Your `env.yaml` file contains **secrets and environment-specific values** that should never be committed to version control.

### Basic env.yaml

```yaml
# API credentials (never commit these!)
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-actual-api-key-here
policy_model_name: gpt-4o-2024-11-20
```

### Advanced env.yaml with Config Paths

```yaml
# Store complex config paths for convenience
simple_weather_config_paths:
  - responses_api_models/openai_model/configs/openai_model.yaml
  - resources_servers/example_simple_weather/configs/simple_weather.yaml

# Different environments
dev_model_name: gpt-4o-mini
prod_model_name: gpt-4o-2024-11-20

# Custom server settings
custom_host: 0.0.0.0
custom_port: 8080
```

**Usage with stored config paths**:
```bash
ng_run '+config_paths=${simple_weather_config_paths}'
```

:::

:::{tab-item} 3. Command Line Arguments

**Runtime overrides** using Hydra syntax for maximum flexibility. These runtime command line have the highest priority, meaning they can override any previous setting set in the config.yaml or env.yaml files.

Basic Overrides
```bash
# Override a specific model
ng_run "+config_paths=[config.yaml]" \
    +policy_model.responses_api_models.openai_model.openai_model=gpt-4o-mini

# Point agent to different resource server
ng_run "+config_paths=[config.yaml]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=different_weather
```

Advanced Overrides
```bash
# Multiple overrides for testing
ng_run "+config_paths=[${config_paths}]" \
    +policy_model_name=gpt-4o-mini \
    +simple_weather.resources_servers.simple_weather.host=localhost \
    +simple_weather.resources_servers.simple_weather.port=9090
```

:::
::::

## Special Policy Model Variables

NeMo Gym provides standard placeholders for the {term}`Policy Model` being trained:

```yaml
# These variables are available in any config file
policy_base_url: https://api.openai.com/v1    # Model API endpoint
policy_api_key: sk-your-key                   # Authentication
policy_model_name: gpt-4o-2024-11-20          # Model identifier
```

**Why these exist**: When training agents, you need consistent references to "the model being trained" across different resource servers and agents.

**Usage in config files**:
```yaml
policy_model:
  responses_api_models:
    openai_model:
      openai_base_url: ${policy_base_url}     # Resolves from env.yaml
      openai_api_key: ${policy_api_key}       # Resolves from env.yaml
      openai_model: ${policy_model_name}      # Resolves from env.yaml
```


## Troubleshooting

NeMo Gym validates your configuration and provides helpful error messages.

:::{dropdown} "Missing mandatory value"
```
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: policy_api_key
```
**Fix**: Add the missing value to `env.yaml` or command line.
:::

:::{dropdown} "Could not find X in the list of available servers"
```
AssertionError: Could not find type='resources_servers' name='typo_weather' 
in the list of available servers: [simple_weather, math_with_judge, ...]
```
**Fix**: Check your server name spelling and ensure the config is loaded.
:::

:::{dropdown} "Almost-Servers Detected"
Example:
```bash
═══════════════════════════════════════════════════
Configuration Warnings: Almost-Servers Detected
═══════════════════════════════════════════════════

  Almost-Server Detected: 'example_simple_agent'
  This server configuration failed validation:

- ResourcesServerInstanceConfig -> resources_servers -> example_server -> domain: Input should be 'math', 'coding', 'agent', 'knowledge', 'instruction_following', 'long_context', 'safety', 'games', 'e2e' or 'other'

  This server will NOT be started.
```
**What this means**: Your server configuration has the correct structure (entrypoint, server type, etc.) but contains invalid values that prevent it from starting.

**Common causes**:
- Invalid `license` enum values in datasets (must be one of the allowed options).
  - see the `license` field in `DatasetConfig` in `config_types.py`.
- Missing or invalid `domain` field for resources servers (math, coding, agent, knowledge, etc.)
  - see the `Domain` class in `config_types.py`.
- Malformed server references (wrong type or name)

**Fix**: Update the configuration based on the validation errors shown. The warning will detail exactly which fields are problematic.

#### Strict Validation Mode

By default, invalid servers will throw an error. You can bypass strict validation and just show a warning:

**In env.yaml:**
```yaml
error_on_almost_servers: false  # Will not error on invalid config
```

**Via command line:**
```bash
ng_run "+config_paths=[config.yaml]" +error_on_almost_servers=false
```

**Default behavior** (`error_on_almost_servers=true`):
- All configuration issues are detected and warnings are printed
- NeMo Gym exits with an error, preventing servers from starting with invalid configs

**When disabled** (`error_on_almost_servers=false`):
- All configuration issues are still detected and warnings are printed
- NeMo Gym continues execution despite the invalid configurations
- Invalid servers are skipped, and valid servers will attempt to start

:::
