# Configuration Management

**Goal**: Master NeMo Gym's flexible configuration system to handle different environments, secrets, and deployment scenarios.

## The Three Configuration Sources

NeMo Gym uses a powerful configuration system with three sources that are resolved in this order:

```
Server YAML Config Files  <  env.yaml  <  Command Line Arguments
    (lowest priority)                       (highest priority)
```

This allows you to:
- Base configuration in YAML files (shared settings)
- Secrets and environment-specific values in `env.yaml` 
- Runtime overrides via command line arguments

## 1. Server Configuration Files

These are your base configurations that define server structures and default values.

### Example: Multi-Server Configuration

```bash
# Define which config files to load
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_simple_weather/configs/simple_weather.yaml,\
responses_api_agents/simple_agent/configs/simple_agent.yaml"

ng_run "+config_paths=[${config_paths}]"
```

### Config File Structure

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

## 2. Environment Configuration (env.yaml)

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

## 3. Command Line Arguments

**Runtime overrides** using Hydra syntax for maximum flexibility.

### Basic Overrides

```bash
# Override a specific model
ng_run "+config_paths=[config.yaml]" \
    +policy_model.responses_api_models.openai_model.openai_model=gpt-4o-mini

# Point agent to different resource server
ng_run "+config_paths=[config.yaml]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=different_weather
```

### Advanced Overrides

```bash
# Multiple overrides for testing
ng_run "+config_paths=[${config_paths}]" \
    +policy_model_name=gpt-4o-mini \
    +simple_weather.resources_servers.simple_weather.host=localhost \
    +simple_weather.resources_servers.simple_weather.port=9090
```

## Special Policy Model Variables

NeMo Gym provides standard placeholders for the model being trained:

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

## Configuration Resolution Process

When you run `ng_run` or `nemo_gym_run`, NeMo Gym resolves configuration in this order:

### Step 1: Load Server YAML Configs
```bash
ng_run "+config_paths=[model.yaml,weather.yaml]"
```
- Loads base configurations
- Later files override earlier files
- Creates the foundation configuration

### Step 2: Apply env.yaml
```yaml
# env.yaml values override Server YAML config values
policy_api_key: sk-real-key-from-env
custom_setting: override-value
```

### Step 3: Apply Command Line
```bash
ng_run "+config_paths=[...]" +policy_model_name=different-model
```
- Command line has highest priority
- Can override any previous setting
- Perfect for runtime customization

## Practical Configuration Scenarios

### Scenario 1: Development vs Production

**env.yaml** (shared secrets):
```yaml
policy_api_key: sk-your-key
```

**Development**:
```bash
ng_run "+config_paths=[dev-config.yaml]" +policy_model_name=gpt-4o-mini
```

**Production**:
```bash
ng_run "+config_paths=[prod-config.yaml]" +policy_model_name=gpt-4o-2024-11-20
```

### Scenario 2: Multi-Resource Testing

```bash
# Test with math resources
ng_run "+config_paths=[base.yaml]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=math_with_judge

# Test with weather resources  
ng_run "+config_paths=[base.yaml]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=simple_weather
```

### Scenario 3: Custom Server Ports

```bash
# Avoid port conflicts in multi-user environments
ng_run "+config_paths=[config.yaml]" \
    +simple_weather.resources_servers.simple_weather.port=8001 \
    +policy_model.responses_api_models.openai_model.port=8002
```

## Troubleshooting

NeMo Gym validates your configuration and provides helpful error messages:

### Problem: Missing Values
```
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: policy_api_key
```
**Fix**: Add the missing value to `env.yaml` or command line.

### Problem: Invalid Server References
```
AssertionError: Could not find type='resources_servers' name='typo_weather' 
in the list of available servers: [simple_weather, math_with_judge, ...]
```
**Fix**: Check your server name spelling and ensure the config is loaded.

### Problem: Port Conflicts
```
OSError: [Errno 48] Address already in use
```
**Fix**: Override ports via command line or use `+port=0` for auto-assignment.

### Problem: Almost-Server Detected (Configuration Validation)
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

### Strict Validation Mode

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

## Best Practices

### 1. Keep Secrets in env.yaml

**✅ Good - secrets in env.yaml:**
```yaml
# env.yaml (git-ignored, never committed)
policy_api_key: sk-actual-secret-key-here
policy_base_url: https://api.openai.com/v1
```

**❌ Bad - secrets in committed config files:**
```yaml
# responses_api_models/openai_model/configs/openai_model.yaml (committed to git!)
policy_model:
  responses_api_models:
    openai_model:
      openai_api_key: sk-actual-secret-key-here  # Don't do this!
      openai_base_url: https://api.openai.com/v1
```

**✅ Good - use placeholders in committed config files:**
```yaml
# responses_api_models/openai_model/configs/openai_model.yaml (committed to git)
policy_model:
  responses_api_models:
    openai_model:
      openai_api_key: ${policy_api_key}    # Resolves from env.yaml
      openai_base_url: ${policy_base_url}  # Resolves from env.yaml
```

### 2. Use Descriptive Config Collections
```yaml
# env.yaml - organize related configs
math_training_config_paths:
  - responses_api_models/openai_model/configs/openai_model.yaml
  - resources_servers/math_with_judge/configs/math_with_judge.yaml
  - responses_api_agents/simple_agent/configs/simple_agent.yaml

weather_demo_config_paths:
  - responses_api_models/openai_model/configs/openai_model.yaml  
  - resources_servers/example_simple_weather/configs/simple_weather.yaml
```

### 3. Document Your Overrides
```bash
# Clear, documented overrides for different scenarios
ng_run "+config_paths=[${base_config}]" \
    +policy_model_name=gpt-4o-mini \        # Use cheaper model for dev
    +simple_agent.host=0.0.0.0 \            # Allow external connections
    +limit=10                               # Limit rollouts for testing
```

### 4. Environment-Specific env.yaml Files
```bash
# Load different env files for different environments
cp env.dev.yaml env.yaml    # Development settings
cp env.prod.yaml env.yaml   # Production settings
```

## What You've Learned

NeMo Gym's configuration system provides:

- **Flexible deployment** - same code, different configurations
- **Secure secrets** - keep sensitive data out of version control  
- **Runtime customization** - override anything from command line
- **Easy testing** - swap components without code changes
- **Validation** - helpful error messages when something's wrong

This configuration mastery enables you to handle complex deployment scenarios, multi-environment setups, and rapid experimentation.