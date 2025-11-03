(concepts-configuration-system)=
# Configuration System

NeMo Gym's configuration system provides a flexible, three-tier architecture for managing settings across development, testing, and production environments. Understanding this system is essential for deploying agents effectively.

This document explains how configuration resolution works, why the three-tier design exists, and how to structure your configurations for different deployment scenarios.

---

## Why Three Tiers?

Different parts of your configuration have different needs:

```{list-table}
:header-rows: 1
:widths: 25 35 40

* - What You're Configuring
  - Where It Should Live
  - Why
* - **Server structure and defaults**
  - YAML files (version controlled)
  - Team shares these; they define the architecture
* - **Secrets and credentials**
  - env.yaml (git-ignored)
  - Never commit API keys; each environment has different credentials
* - **Runtime experiments**
  - Command line arguments
  - Quick testing without editing files; temporary overrides
```

This separation enables:
- **Version control** for shared configurations without exposing secrets
- **Environment-specific** settings (dev/staging/prod) with the same code
- **Rapid iteration** via command-line overrides during development
- **Secure deployment** with proper secrets management

---

## The Three Configuration Layers

### Layer 1: Server YAML Files (Foundation)

**Purpose**: Define the structure and default values for your servers.

**Location**: `responses_api_models/`, `resources_servers/`, `responses_api_agents/` directories

**Example** (`responses_api_models/openai_model/configs/openai_model.yaml`):
```yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_base_url: ${policy_base_url}
      openai_api_key: ${policy_api_key}
      openai_model: ${policy_model_name}
```

**Key characteristics**:
- Version controlled (committed to git)
- Uses variable interpolation (`${variable_name}`) for secrets
- Defines server hierarchy: server ID → server type → implementation → settings
- Multiple YAML files can be loaded; later files override earlier ones

**Evidence**: Configuration resolution in `nemo_gym/global_config.py:194`
```python
config_paths, extra_configs = self.load_extra_config_paths(config_paths)
```

### Layer 2: env.yaml (Secrets and Environment-Specific Values)

**Purpose**: Store secrets, API keys, and environment-specific settings that should never be committed.

**Location**: `env.yaml` in the project root (must be in `.gitignore`)

**Example** (`env.yaml`):
```yaml
# API credentials (never commit!)
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-actual-api-key-here
policy_model_name: gpt-4o-2024-11-20

# Config path collections for convenience
simple_weather_config_paths:
  - responses_api_models/openai_model/configs/openai_model.yaml
  - resources_servers/simple_weather/configs/simple_weather.yaml
```

**Key characteristics**:
- Loaded early so variables can be used in config paths
- Overrides values from YAML files
- Can store config path collections for convenience
- Each environment (dev/staging/prod) has its own env.yaml

**Evidence**: env.yaml loading in `nemo_gym/global_config.py:183-187`
```python
# Load the env.yaml config. We load it early so that people can use it to 
# conveniently store config paths.
dotenv_path = parse_config.dotenv_path or Path(PARENT_DIR) / "env.yaml"
if dotenv_path.exists() and not parse_config.skip_load_from_dotenv:
    dotenv_extra_config = OmegaConf.load(dotenv_path)
```

### Layer 3: Command Line Arguments (Runtime Overrides)

**Purpose**: Temporary overrides for testing and experimentation without editing files.

**Usage**:
```bash
ng_run "+config_paths=[configs.yaml]" \
    +policy_model_name=gpt-4o-mini \
    +simple_weather.resources_servers.simple_weather.port=8001
```

**Key characteristics**:
- Highest priority—overrides everything
- Uses Hydra syntax (`+key=value`)
- Can override any nested configuration value
- Perfect for one-off experiments and CI/CD pipelines

**Evidence**: Command line priority in `nemo_gym/global_config.py:199-201`
```python
# Merge config dicts
# global_config_dict is the last config arg here since we want command line args 
# to override everything else.
global_config_dict = OmegaConf.merge(*extra_configs, global_config_dict)
```

---

## Configuration Resolution Process

Let's trace what happens when you run this command:

```bash
ng_run "+config_paths=[model.yaml]" +policy_model_name=gpt-4o-mini
```

With an `env.yaml` file containing secrets:

```yaml
policy_api_key: sk-real-key
policy_model_name: gpt-4o-2024-11-20
```

1. **Parse Command Line → Configuration Dictionary**
   
   Hydra extracts arguments:
   > `{config_paths: ["model.yaml"], policy_model_name: "gpt-4o-mini"}`

2. **Load env.yaml → Secrets Layer**
   
   If `env.yaml` exists, it's loaded into a separate dictionary:
   > `{policy_api_key: "sk-real-key", policy_model_name: "gpt-4o-2024-11-20"}`

3. **Resolve config_paths → Determine Which Files to Load**
   
   System merges CLI and env.yaml to resolve any variable references in config paths:
   ```python
   # From global_config.py:189
   merged_config_for_config_paths = OmegaConf.merge(dotenv_extra_config, global_config_dict)
   config_paths = merged_config_for_config_paths.get(CONFIG_PATHS_KEY_NAME) or []
   ```
   
   This enables using config collections: `ng_run '+config_paths=${weather_config_paths}'`

4. **Load YAML Files → Server Configurations**
   
   Each file in `config_paths` is loaded in order:
   > First file: `{policy_model: {openai_model: {...}}}`
   >
   > Second file overrides/extends: `{simple_weather: {resources_servers: {...}}}`

5. **Final Merge → Single Configuration**
   
   All layers merge with priority order:
   ```python
   # From global_config.py:201
   global_config_dict = OmegaConf.merge(
       *extra_configs,        # YAML files (lowest priority)
       dotenv_extra_config,   # env.yaml (middle priority)
       global_config_dict     # CLI args (highest priority)
   )
   ```
   
   **Result**: `policy_model_name` from CLI (`gpt-4o-mini`) overrides env.yaml's value (`gpt-4o-2024-11-20`), while `policy_api_key` from env.yaml is preserved.

6. **Validate and Populate Defaults → Ready to Run**
   
   Final validation ensures the configuration is complete:
   - Verify all server references exist
   - Populate missing host values (default: `127.0.0.1`)
   - Assign available ports if not specified
   - Cache for the session
   
   Configuration is now ready for `ng_run` to start servers.

**Evidence**: Complete resolution logic in `nemo_gym/global_config.py:132-201`

---

## Configuration Structure

### Server Instance Config Format

Every server in NeMo Gym follows this hierarchy:

```yaml
server_id:                    # Unique identifier for this server instance
  server_type:                # One of: responses_api_models, resources_servers, responses_api_agents
    implementation:           # Folder name under server_type/
      entrypoint: app.py      # Python file to run
      # Implementation-specific settings
      setting_1: value
      setting_2: ${variable}  # Variable interpolation
```

**Example** (Agent configuration):
```yaml
simple_weather_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      max_steps: 10
      resources_server:
        type: resources_servers
        name: simple_weather
      model_server:
        type: responses_api_models
        name: policy_model
```

### Server References

Agents and resources reference each other using this format:

```yaml
resources_server:
  type: resources_servers      # Server type to reference
  name: simple_weather         # Server ID to connect to
```

The system validates these references during configuration resolution to ensure all referenced servers exist.

**Evidence**: Reference validation in `nemo_gym/global_config.py:145-153`

### Policy Model Variables

NeMo Gym provides three standard variables for the model being trained:

```yaml
policy_base_url: https://api.openai.com/v1    # Model API endpoint
policy_api_key: sk-your-key                   # Authentication
policy_model_name: gpt-4o-2024-11-20          # Model identifier
```

**Why they exist**: When training agents, you need consistent references to "the model being trained" across different components. These variables provide a standard way to specify the policy model regardless of which resource servers or agents are being used.

**Evidence**: Variable definitions in `nemo_gym/global_config.py:54-56`

---

## Environment-Specific Deployments

### Approach 1: Multiple env.yaml Files

Maintain separate environment files:

```bash
env.dev.yaml      # Development: gpt-4o-mini, test keys
env.staging.yaml  # Staging: gpt-4o, staging keys
env.prod.yaml     # Production: gpt-4o, production keys
```

Switch environments by copying:
```bash
cp env.prod.yaml env.yaml
ng_run "+config_paths=[${config}]"
```

### Approach 2: Environment-Specific YAML Configs

Use different YAML configs per environment:

```bash
# Development
ng_run "+config_paths=[configs/dev.yaml]" +policy_model_name=gpt-4o-mini

# Production
ng_run "+config_paths=[configs/prod.yaml]" +policy_model_name=gpt-4o-2024-11-20
```

### Approach 3: Command-Line Overrides

Keep a single config and override at runtime:

```bash
# CI/CD pipeline
ng_run "+config_paths=[${base_config}]" \
    +policy_model_name=${MODEL} \
    +policy_api_key=${API_KEY} \
    +limit=${LIMIT}
```

---

## Technical Implementation

### Configuration Caching

The configuration is resolved once per process and cached:

```python
# From global_config.py:280-282
global _GLOBAL_CONFIG_DICT
if _GLOBAL_CONFIG_DICT is not None:
    return _GLOBAL_CONFIG_DICT
```

This ensures consistent configuration throughout the process lifetime.

### Child Process Configuration

When NeMo Gym spawns child processes (for individual servers), the parent passes configuration via environment variable:

```python
# From global_config.py:284-290
nemo_gym_config_dict_str_from_env = getenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME)
if nemo_gym_config_dict_str_from_env:
    global_config_dict = OmegaConf.create(nemo_gym_config_dict_str_from_env)
```

This avoids re-parsing configuration in every subprocess.

### OmegaConf Integration

NeMo Gym uses [OmegaConf](https://omegaconf.readthedocs.io/) for configuration management, which provides:
- Variable interpolation: `${variable_name}`
- Type validation
- Hierarchical merging
- Structured configs

---

## Related Concepts

- **[Core Abstractions](core-abstractions.md)**: Understanding the three servers that configuration connects
- **[Rollout Collection Fundamentals](rollout-collection-fundamentals.md)**: How configuration affects rollout generation

---

## Summary

NeMo Gym's three-tier configuration system provides:

1. **YAML Files**: Version-controlled structure and defaults
2. **env.yaml**: Environment-specific secrets and settings
3. **Command Line**: Runtime overrides for experimentation

Configuration is resolved once at startup with this priority order (highest last):
```
YAML files → env.yaml → Command Line
```

This design enables secure, flexible deployments across different environments while maintaining a single codebase.

For hands-on practice with configuration, see the [Setup and Installation](../../get-started/setup-installation.md) tutorial.

