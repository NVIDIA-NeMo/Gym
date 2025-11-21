(concepts-configuration-system)=
# Configuration System

NeMo Gym's configuration system provides a flexible, three-tier architecture for managing settings across development, testing, and production environments. Understanding this system is essential for deploying agents effectively.

This document explains how configuration resolution works, why the three-tier design exists, and how to structure your configurations for different deployment scenarios.

---

## Why Three Tiers?

The three-tier design separates concerns for secure, flexible deployments. Choose the perspective most relevant to your role:

::::{tab-set}

:::{tab-item} Developer Perspective
**What it means for you**:

- **Rapid iteration**: Test different models or configurations with a single command-line override—no file editing required
- **Safe experimentation**: Try changes without affecting your team's shared configs
- **Easy rollback**: Command-line experiments don't persist; just remove the override

**Example workflow**:
```bash
# Quick test with cheaper model
ng_run "+config_paths=[${config}]" +policy_model_name=gpt-4o-mini

# Back to normal - just remove the override
ng_run "+config_paths=[${config}]"
```
:::

:::{tab-item} DevOps Perspective
**What it means for you**:

- **Environment-specific deployments**: Same codebase, different env.yaml for dev/staging/prod
- **Secrets management**: API keys never in version control, different credentials per environment
- **CI/CD integration**: Override configurations via environment variables in pipelines
- **Infrastructure as code**: YAML configs define server architecture in version control

**Example deployment**:
```bash
# CI/CD pipeline with environment-specific overrides
ng_run "+config_paths=[${base_config}]" \
    +policy_api_key=${PROD_API_KEY} \
    +policy_model_name=${MODEL_VERSION}
```
:::

:::{tab-item} Security Perspective
**What it means for you**:

- **Zero secrets in git**: All sensitive values go in git-ignored env.yaml
- **Variable interpolation**: YAML files reference variables, never hardcode keys
- **Environment isolation**: Each environment has its own credentials
- **Audit trail**: Configuration changes tracked in git (structure) without exposing secrets (values)

**Security model**:
```yaml
# In git (public)
openai_api_key: ${policy_api_key}  # Variable reference

# In env.yaml (git-ignored, private)
policy_api_key: sk-actual-secret-key
```
:::

::::

---

## How the Layers Work Together

Each layer serves a specific purpose and overrides the previous one. Here's how they compare:

```{list-table}
:header-rows: 1
:widths: 20 27 27 26

* - 
  - **Layer 1: YAML Files**
  - **Layer 2: env.yaml**
  - **Layer 3: Command Line**
* - **Priority**
  - Lowest (foundation)
  - Middle (overrides YAML)
  - Highest (overrides everything)
* - **Purpose**
  - Define server structure and architecture
  - Store secrets and environment-specific values
  - Temporary overrides for testing
* - **Location**
  - `responses_api_models/`, `resources_servers/`, `responses_api_agents/` directories
  - `env.yaml` in project root
  - Arguments passed to `ng_run`
* - **Version Control**
  - ✅ Committed to git
  - ❌ In `.gitignore` (never commit)
  - ❌ Not persisted
* - **Contains**
  - Server hierarchy, entrypoints, structure, variable references
  - API keys, secrets, environment-specific values, config collections
  - Any configuration override, experiment parameters
* - **When to Use**
  - Defining your system architecture
  - Different credentials per environment (dev/staging/prod)
  - Quick experiments without editing files
* - **Example Content**
  - `openai_base_url: ${policy_base_url}`
  - `policy_api_key: sk-real-key`
  - `+policy_model_name=gpt-4o-mini`
```

---

## Configuration Structure

### 1. Server YAML Config Files

These are your base configurations that define server structures and default values. Later files in the `config_paths` list override earlier files.

**Example: Multi-Server Configuration**

```bash
# Define which config files to load
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_simple_weather/configs/simple_weather.yaml,\
responses_api_agents/simple_agent/configs/simple_agent.yaml"

ng_run "+config_paths=[${config_paths}]"
```

Every config file defines **server instances** with a specific hierarchy:

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

### 2. env.yaml

Your `env.yaml` file contains **secrets and environment-specific values** that should never be committed to version control.

**Basic env.yaml**:

```yaml
# API credentials (never commit these!)
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-actual-api-key-here
policy_model_name: gpt-4o-2024-11-20
```

**Advanced env.yaml with Config Paths**:

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

### 3. Command Line Arguments

**Runtime overrides** using Hydra syntax for maximum flexibility. These have the highest priority and can override any setting in `config.yaml` or `env.yaml`.

**Basic Overrides**:

```bash
# Override a specific model
ng_run "+config_paths=[config.yaml]" \
    +policy_model.responses_api_models.openai_model.openai_model=gpt-4o-mini

# Point agent to different resource server
ng_run "+config_paths=[config.yaml]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=different_weather
```

**Advanced Overrides**:

```bash
# Multiple overrides for testing
ng_run "+config_paths=[${config_paths}]" \
    +policy_model_name=gpt-4o-mini \
    +simple_weather.resources_servers.simple_weather.host=localhost \
    +simple_weather.resources_servers.simple_weather.port=9090
```

---

## Special Policy Model Variables

NeMo Gym provides standard placeholders for the model being trained. When training agents, you need consistent references to "the model being trained" across different resource servers and agents.

These variables are available in any config file:

```yaml
policy_base_url: https://api.openai.com/v1    # Model API endpoint
policy_api_key: sk-your-key                   # Authentication
policy_model_name: gpt-4o-2024-11-20          # Model identifier
```

**Usage in config files**:

```yaml
policy_model:
  responses_api_models:
    openai_model:
      openai_base_url: ${policy_base_url}     # Resolves from env.yaml
      openai_api_key: ${policy_api_key}       # Resolves from env.yaml
      openai_model: ${policy_model_name}      # Resolves from env.yaml
```

---

## How Configuration Gets Resolved

When you run `ng_run`, NeMo Gym merges configuration in this order:

1.  **Parse command-line arguments**: Extracts `config_paths` and any overrides
2.  **Load env.yaml**: Loads secrets and environment-specific values
3.  **Load YAML files**: Loads each file in `config_paths`, resolving variables like `${policy_api_key}`
4.  **Merge with priority**: Combines all layers: YAML files → env.yaml → CLI (highest priority)
5.  **Validate and cache**: Verifies server references exist, populate defaults, cache for session

**Priority order**: CLI overrides env.yaml, which overrides YAML files.

**Example**: If `policy_model_name` is in both env.yaml (`gpt-4o-2024-11-20`) and CLI (`+policy_model_name=gpt-4o-mini`), CLI wins.

---

## Troubleshooting

NeMo Gym validates your configuration and provides helpful error messages.

### Common Errors

**"Missing mandatory value"**

```text
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: policy_api_key
```

**Fix**: Add the missing value to `env.yaml` or command line arguments.

**"Could not find X in the list of available servers"**

```text
AssertionError: Could not find type='resources_servers' name='typo_weather' 
in the list of available servers: [simple_weather, math_with_judge, ...]
```

**Fix**: Check your server name spelling and ensure the config file defining it is included in `config_paths`.

**"Almost-Servers Detected"**

```text
═══════════════════════════════════════════════════
Configuration Warnings: Almost-Servers Detected
═══════════════════════════════════════════════════
  Almost-Server Detected: 'example_simple_agent'
  This server configuration failed validation:
- ResourcesServerInstanceConfig -> resources_servers -> example_server -> domain: Input should be ...
```

**What this means**: Your server configuration has the correct structure but contains invalid values (e.g., invalid enum, missing field).

**Fix**: Update the configuration based on the validation errors shown.

### Strict Validation Mode

By default, NeMo Gym exits with an error if any server configuration is invalid (`error_on_almost_servers=true`). You can bypass this strict validation to skip invalid servers and only start valid ones.

**In env.yaml**:
```yaml
error_on_almost_servers: false
```

**Via command line**:
```bash
ng_run "+config_paths=[config.yaml]" +error_on_almost_servers=false
```
