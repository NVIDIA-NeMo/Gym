(tutorial-configuration)=

# Configuration Management

Master NeMo Gym's flexible configuration system to handle different environments, secrets, and deployment scenarios—essential for moving from development to production.

:::{card}

**Goal**: Understand and apply NeMo Gym's three-tier configuration system for flexible deployments.

^^^

**In this tutorial, you will**:

1. Understand the three configuration sources and their priority
2. Set up environment-specific configurations
3. Use command-line overrides for runtime customization
4. Apply best practices for secrets management

:::

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New to NeMo Gym? Start with Get Started
:::

## Before You Start

This tutorial assumes you've completed the [Get Started](../get-started/index.md) series and understand how to run basic NeMo Gym commands. For hands-on practice, work through [Setup and Installation](../get-started/setup-installation.md) first.


---

## The Three Configuration Sources

NeMo Gym uses a powerful configuration system with three sources that are resolved in priority order:

```{list-table}
:header-rows: 1
:widths: 25 35 40

* - Source
  - Priority
  - Best For
* - **Server YAML Files**
  - Lowest (base layer)
  - Shared settings, server structure, defaults
* - **env.yaml**
  - Middle (overrides YAML)
  - Secrets, environment-specific values (never commit!)
* - **Command Line**
  - Highest (overrides all)
  - Runtime customization, testing, quick experiments
```

**Resolution order**: Server YAML → env.yaml → Command Line (later sources override earlier ones)

This layered approach gives you:
- **Reusable base configurations** in version-controlled YAML files
- **Secure secrets management** in git-ignored env.yaml
- **Flexible runtime overrides** via command line for any scenario

---

## 1. Server Configuration Files

These are your base configurations that define server structures and default values.

### Example: Multi-Server Configuration

```bash
# Define which config files to load
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/simple_weather/configs/simple_weather.yaml,\
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

---

## 2. Environment Configuration (env.yaml)

Your `env.yaml` file contains **secrets and environment-specific values** that should never be committed to version control.

Choose your setup complexity:

::::{tab-set}

:::{tab-item} Basic env.yaml

**Use case**: Simple deployments with minimal configuration needs

**Essential secrets only**:
```yaml
# API credentials (never commit these!)
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-actual-api-key-here
policy_model_name: gpt-4o-2024-11-20
```

**When to use**: 
- Single environment deployments
- Getting started with NeMo Gym
- Simple testing scenarios

:::

:::{tab-item} Advanced env.yaml

**Use case**: Complex deployments with multiple environments and reusable configurations

**Organized configuration collections**:
```yaml
# Store complex config paths for convenience
simple_weather_config_paths:
  - responses_api_models/openai_model/configs/openai_model.yaml
  - resources_servers/simple_weather/configs/simple_weather.yaml

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

**When to use**:
- Multi-environment setups (dev/staging/prod)
- Frequently switching between different configurations
- Team environments with shared conventions

:::

::::

---

## 3. Command Line Arguments

**Runtime overrides** using Hydra syntax for maximum flexibility.

Choose your complexity level:

::::{tab-set}

:::{tab-item} Basic Overrides

**Use case**: Simple, single-value overrides for common scenarios

**Override a specific model**:
```bash
ng_run "+config_paths=[config.yaml]" \
    +policy_model.responses_api_models.openai_model.openai_model=gpt-4o-mini
```

**Point agent to different resource server**:
```bash
ng_run "+config_paths=[config.yaml]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=different_weather
```

**When to use**: Quick testing, single parameter changes, learning the system

:::

:::{tab-item} Advanced Overrides

**Use case**: Multiple simultaneous overrides for complex scenarios

**Multiple overrides for testing**:
```bash
ng_run "+config_paths=[${config_paths}]" \
    +policy_model_name=gpt-4o-mini \
    +simple_weather.resources_servers.simple_weather.host=localhost \
    +simple_weather.resources_servers.simple_weather.port=9090
```

**Combining variable references with overrides**:
```bash
ng_run '+config_paths=${math_training_config_paths}' \
    +policy_model_name=gpt-4o-2024-11-20 \
    +limit=100 \
    +num_samples_in_parallel=10
```

**When to use**: Production runs, integration testing, complex environment setups

:::

::::

---

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

---

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

---

## Practical Configuration Scenarios

Choose the scenario that matches your use case:

::::{tab-set}

:::{tab-item} Development vs Production

**Use case**: Switch between cheaper dev models and production models

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

**Key insight**: Same code, same secrets, different models based on environment.

:::

:::{tab-item} Multi-Resource Testing

**Use case**: Test the same agent with different resource servers

**Switch to math resources**:
```bash
ng_run "+config_paths=[base.yaml]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=library_judge_math
```

**Switch to weather resources**:
```bash
ng_run "+config_paths=[base.yaml]" \
    +simple_agent.responses_api_agents.simple_agent.resources_server.name=simple_weather
```

**Key insight**: Command-line overrides let you swap components without editing config files.

:::

:::{tab-item} Custom Server Ports

**Use case**: Avoid port conflicts in multi-user or multi-environment setups

**Override default ports**:
```bash
ng_run "+config_paths=[config.yaml]" \
    +simple_weather.resources_servers.simple_weather.port=8001 \
    +policy_model.responses_api_models.openai_model.port=8002
```

**Auto-assign available ports**:
```bash
ng_run "+config_paths=[config.yaml]" \
    +simple_weather.resources_servers.simple_weather.port=0 \
    +policy_model.responses_api_models.openai_model.port=0
```

**Key insight**: Use `+port=0` for automatic port assignment to avoid conflicts entirely.

:::

::::

---

## Best Practices

:::{dropdown} 1. Keep Secrets in env.yaml
:icon: lock

**Principle**: Never commit secrets to version control—use env.yaml for sensitive data.

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

**Why it matters**: Accidentally committing API keys can lead to unauthorized usage and security breaches.

:::

:::{dropdown} 2. Use Descriptive Config Collections
:icon: list-unordered

**Principle**: Group related configurations for easy switching between scenarios.

**Implementation**:
```yaml
# env.yaml - organize related configs
math_training_config_paths:
  - responses_api_models/openai_model/configs/openai_model.yaml
  - resources_servers/library_judge_math/configs/library_judge_math.yaml
  - responses_api_agents/simple_agent/configs/simple_agent.yaml

weather_demo_config_paths:
  - responses_api_models/openai_model/configs/openai_model.yaml  
  - resources_servers/simple_weather/configs/simple_weather.yaml
```

**Usage**:
```bash
ng_run '+config_paths=${math_training_config_paths}'
ng_run '+config_paths=${weather_demo_config_paths}'
```

**Why it matters**: Reduces typing and errors when switching between different setups.

:::

:::{dropdown} 3. Document Your Overrides
:icon: comment

**Principle**: Use inline comments to explain why each override is needed.

**Implementation**:
```bash
# Clear, documented overrides for different scenarios
ng_run "+config_paths=[${base_config}]" \
    +policy_model_name=gpt-4o-mini \        # Use cheaper model for dev
    +simple_agent.host=0.0.0.0 \            # Allow external connections
    +limit=10                               # Limit rollouts for testing
```

**Why it matters**: Makes it easy for others (and future you) to understand configuration choices.

:::

:::{dropdown} 4. Environment-Specific env.yaml Files
:icon: file

**Principle**: Maintain separate env.yaml files for different environments and swap as needed.

**Setup**:
```bash
# Create environment-specific files
env.dev.yaml    # Development settings (gpt-4o-mini, test keys)
env.prod.yaml   # Production settings (gpt-4o, production keys)
env.staging.yaml  # Staging settings
```

**Usage**:
```bash
# Switch to development
cp env.dev.yaml env.yaml
ng_run "+config_paths=[...]"

# Switch to production
cp env.prod.yaml env.yaml
ng_run "+config_paths=[...]"
```

**Pro tip**: Add env.yaml to .gitignore but commit env.example.yaml as a template.

**Why it matters**: Prevents accidentally using production credentials in development and vice versa.

:::

---

## Troubleshooting

NeMo Gym validates your configuration and provides helpful error messages:

:::{dropdown} Problem: Missing Values
:icon: alert
:color: warning

**Error message**:
```text
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: policy_api_key
```

**What this means**: A required configuration value wasn't provided in any of the three sources.

**Solutions**:

1. **Add to env.yaml** (recommended for secrets):
   ```yaml
   policy_api_key: sk-your-actual-key
   ```

2. **Override via command line** (for testing):
   ```bash
   ng_run "+config_paths=[...]" +policy_api_key=sk-test-key
   ```

:::

:::{dropdown} Problem: Invalid Server References
:icon: search
:color: warning

**Error message**:
```text
AssertionError: Could not find type='resources_servers' name='typo_weather' 
in the list of available servers: [simple_weather, library_judge_math, ...]
```

**What this means**: You're referencing a server that doesn't exist or isn't loaded.

**Solutions**:

1. **Check spelling**: Verify the server name matches exactly (case-sensitive)
2. **Ensure config is loaded**: Add the server's config file to `+config_paths`
3. **List available servers**: Check the error message for valid server names

:::

:::{dropdown} Problem: Port Conflicts
:icon: plug
:color: danger

**Error message**:
```text
OSError: [Errno 48] Address already in use
```

**What this means**: Another process is already using the port you're trying to bind to.

**Solutions**:

1. **Override with available port**:
   ```bash
   ng_run "+config_paths=[...]" +simple_weather.resources_servers.simple_weather.port=8001
   ```

2. **Use auto-assignment** (recommended):
   ```bash
   ng_run "+config_paths=[...]" +simple_weather.resources_servers.simple_weather.port=0
   ```
   The system will automatically find and assign an available port.

3. **Check running processes**:
   ```bash
   lsof -i :8000  # Check what's using port 8000
   ```

:::


---

## What You've Learned

You now have hands-on experience with:

- ✓ Understanding the three-tier configuration system and priority order
- ✓ Using env.yaml for secrets and environment-specific values
- ✓ Applying command-line overrides for runtime customization
- ✓ Following best practices for secure configuration management

**Key insight**: NeMo Gym's layered configuration system gives you flexibility without complexity—base settings in YAML, secrets in env.yaml, and runtime tweaks via command line.

---

## Next Steps

You've mastered configuration management! Continue exploring:

- **[Offline Training with Rollouts](offline-training-w-rollouts.md)**: Apply your configuration skills to training workflows
- **[Concepts](../about/concepts/index.md)**: Deep dive into NeMo Gym architecture

Or return to the [Tutorials Overview](index.md) to explore other topics.