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

This separation enables secure, flexible deployments. Choose the perspective most relevant to your role:

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

**Merge priority** (highest last): YAML files → env.yaml → Command Line

---

### Detailed Examples by Layer

See concrete examples of how each layer is structured and used:

::::{tab-set}

:::{tab-item} Layer 1: YAML Files
**Example** (`responses_api_models/openai_model/configs/openai_model.yaml`):

```yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_base_url: ${policy_base_url}      # Variable reference
      openai_api_key: ${policy_api_key}        # Variable reference
      openai_model: ${policy_model_name}       # Variable reference
```

**Key points**:
- Uses variable interpolation (`${variable_name}`) instead of hardcoded values
- Defines server hierarchy: server ID → server type → implementation → settings
- Multiple YAML files can be loaded; later files override earlier ones
- Committed to git—team shares these files

**Evidence**: Configuration resolution in `nemo_gym/global_config.py:194`
```python
config_paths, extra_configs = self.load_extra_config_paths(config_paths)
```
:::

:::{tab-item} Layer 2: env.yaml
**Example** (`env.yaml` in project root):

```yaml
# API credentials (NEVER commit this file!)
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-actual-api-key-here
policy_model_name: gpt-4o-2024-11-20

# Config path collections for convenience
simple_weather_config_paths:
  - responses_api_models/openai_model/configs/openai_model.yaml
  - resources_servers/simple_weather/configs/simple_weather.yaml
```

**Key points**:
- Loaded early so variables can be used in config paths
- Each environment (dev/staging/prod) has its own env.yaml
- Can store config path collections for convenience
- Must be in `.gitignore`

**Evidence**: env.yaml loading in `nemo_gym/global_config.py:183-187`
```python
# Load the env.yaml config. We load it early so that people can use it to 
# conveniently store config paths.
dotenv_path = parse_config.dotenv_path or Path(PARENT_DIR) / "env.yaml"
if dotenv_path.exists() and not parse_config.skip_load_from_dotenv:
    dotenv_extra_config = OmegaConf.load(dotenv_path)
```
:::

:::{tab-item} Layer 3: Command Line
**Example** (command-line usage):

```bash
ng_run "+config_paths=[configs.yaml]" \
    +policy_model_name=gpt-4o-mini \
    +simple_weather.resources_servers.simple_weather.port=8001
```

**Key points**:
- Highest priority—overrides everything from YAML and env.yaml
- Uses Hydra syntax (`+key=value`)
- Can override any nested configuration value with dot notation
- Perfect for one-off experiments and CI/CD pipelines
- Changes don't persist (temporary overrides only)

**Evidence**: Command line priority in `nemo_gym/global_config.py:199-201`
```python
# Merge config dicts
# global_config_dict is the last config arg here since we want command line args 
# to override everything else.
global_config_dict = OmegaConf.merge(*extra_configs, global_config_dict)
```
:::

::::


---

## How Configuration Gets Resolved

When you run `ng_run`, NeMo Gym merges configuration in this order:

1. **Parse command-line arguments** — Extract `config_paths` and any overrides
2. **Load env.yaml** — Load secrets and environment-specific values
3. **Load YAML files** — Load each file in `config_paths`, resolving variables like `${policy_api_key}`
4. **Merge with priority** — Combine all layers: YAML files → env.yaml → CLI (highest priority)
5. **Validate and cache** — Verify server references exist, populate defaults, cache for session

**Priority order**: CLI overrides env.yaml, which overrides YAML files.

**Example**: If `policy_model_name` is defined in both env.yaml (`gpt-4o-2024-11-20`) and CLI (`+policy_model_name=gpt-4o-mini`), the CLI value wins.

:::{dropdown} See the Final Merged Result
:icon: code-square

After all layers merge:

```yaml
policy_model_name: gpt-4o-mini                     # CLI overrode env.yaml
policy_api_key: sk-real-key                        # From env.yaml
policy_base_url: https://api.openai.com/v1         # From env.yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py                           # From YAML
      openai_base_url: https://api.openai.com/v1   # Variable resolved
      openai_api_key: sk-real-key                  # Variable resolved
      openai_model: gpt-4o-mini                    # Variable resolved with override
```

Variable interpolation (`${...}`) happens during YAML loading, so references already see the merged CLI + env.yaml values.

:::

:::{dropdown} Resolution Implementation Details
:icon: tools

The merge happens in this order:

```python
# From global_config.py:201
global_config_dict = OmegaConf.merge(
    *extra_configs,        # YAML files (lowest priority)
    dotenv_extra_config,   # env.yaml (middle priority)
    global_config_dict     # CLI args (highest priority)
)
```

Complete resolution logic: `nemo_gym/global_config.py:132-201`

:::


---

## Understanding Configuration Structure

Once configuration is resolved, it follows a standard structure for defining servers and their relationships. These patterns apply regardless of which layer defines the values:

:::{dropdown} Server Instance Config Format
:icon: code

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

**Key points**:
- Server ID must be unique across all servers
- Server type determines which directory to look in
- Implementation matches a folder name under that type
- Entrypoint specifies which Python file to run

:::

:::{dropdown} Server References
:icon: link

Agents and resources reference each other using this format:

```yaml
resources_server:
  type: resources_servers      # Server type to reference
  name: simple_weather         # Server ID to connect to
```

**Validation**: The system validates these references during configuration resolution to ensure all referenced servers exist.

**Example error** (if reference is invalid):
```text
AssertionError: Could not find type='resources_servers' name='typo_weather' 
in the list of available servers: [simple_weather, library_judge_math, ...]
```

**Evidence**: Reference validation in `nemo_gym/global_config.py:145-153`

:::

:::{dropdown} Policy Model Variables
:icon: gear

NeMo Gym provides three standard variables for the model being trained:

```yaml
policy_base_url: https://api.openai.com/v1    # Model API endpoint
policy_api_key: sk-your-key                   # Authentication
policy_model_name: gpt-4o-2024-11-20          # Model identifier
```

**Why they exist**: When training agents, you need consistent references to "the model being trained" across different components. These variables provide a standard way to specify the policy model regardless of which resource servers or agents are being used.

**Usage pattern**:
- Define once in env.yaml (the actual values)
- Reference everywhere with `${policy_model_name}`, `${policy_api_key}`, etc.
- Override via command line for experiments

**Evidence**: Variable definitions in `nemo_gym/global_config.py:54-56`

:::


---

## Deployment Strategies Across Environments

The three-tier system supports multiple deployment patterns. Different deployment scenarios call for different configuration strategies—choose the approach that best fits your workflow:

::::{tab-set}

:::{tab-item} Multiple env.yaml Files
**Best for**: Teams with distinct environments and different deployment processes

**Structure**:
```bash
env.dev.yaml      # Development: gpt-4o-mini, test keys
env.staging.yaml  # Staging: gpt-4o, staging keys
env.prod.yaml     # Production: gpt-4o, production keys
```

**Usage**:
```bash
# Switch environments by copying
cp env.prod.yaml env.yaml
ng_run "+config_paths=[${config}]"
```

**Advantages**:
- Clear separation between environments
- Easy to see what differs per environment
- Each file can have environment-specific collections

**Considerations**:
- Need to remember to copy before running
- Can accidentally run wrong environment if you forget
:::

:::{tab-item} Environment-Specific YAML Configs
**Best for**: Infrastructure-as-code workflows where configs are managed separately per environment

**Structure**:
```bash
configs/dev.yaml      # Dev server configurations
configs/staging.yaml  # Staging configurations
configs/prod.yaml     # Production configurations
```

**Usage**:
```bash
# Development
ng_run "+config_paths=[configs/dev.yaml]" +policy_model_name=gpt-4o-mini

# Production
ng_run "+config_paths=[configs/prod.yaml]" +policy_model_name=gpt-4o-2024-11-20
```

**Advantages**:
- Explicit environment selection in command
- Configs can be version controlled (except secrets)
- Works well with container orchestration

**Considerations**:
- Secrets still need env.yaml or environment variables
- More files to maintain
:::

:::{tab-item} Command-Line Overrides
**Best for**: CI/CD pipelines and automated deployments with environment variables

**Structure**:
- Single base configuration
- Environment-specific values passed at runtime

**Usage**:
```bash
# CI/CD pipeline
ng_run "+config_paths=[${base_config}]" \
    +policy_model_name=${MODEL} \
    +policy_api_key=${API_KEY} \
    +limit=${LIMIT}
```

**Advantages**:
- No environment-specific files to manage
- Secrets injected from CI/CD secrets manager
- Single source of truth for structure

**Considerations**:
- Longer command lines
- Need to pass all environment-specific values explicitly
- Best combined with scripting or CI/CD tools
:::

::::


---

## Technical Implementation Details

Understanding how NeMo Gym implements configuration resolution can help you optimize performance and debug issues:

:::{dropdown} Configuration Caching
:icon: cache

The configuration is resolved once per process and cached:

```python
# From global_config.py:280-282
global _GLOBAL_CONFIG_DICT
if _GLOBAL_CONFIG_DICT is not None:
    return _GLOBAL_CONFIG_DICT
```

**Why**: Configuration resolution involves parsing command-line args, loading multiple YAML files, merging layers, and validation. This is expensive to do repeatedly.

**Implication**: Once resolved, configuration is immutable for the process lifetime. To change configuration, restart the process.

**Benefit**: Consistent configuration across all components; no surprises from configuration changing mid-execution.

:::

:::{dropdown} Child Process Configuration
:icon: versions

When NeMo Gym spawns child processes (for individual servers), the parent passes configuration via environment variable:

```python
# From global_config.py:284-290
nemo_gym_config_dict_str_from_env = getenv(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME)
if nemo_gym_config_dict_str_from_env:
    global_config_dict = OmegaConf.create(nemo_gym_config_dict_str_from_env)
```

**Why**: Each server runs in its own process. Without this optimization, every subprocess would need to re-parse command-line args, load YAML files, and perform resolution.

**How it works**:
1. Parent process resolves configuration once
2. Serializes configuration to string
3. Passes to child via `NEMO_GYM_CONFIG_DICT` environment variable
4. Child deserializes and uses directly

**Benefit**: Fast subprocess startup; no redundant file I/O or parsing.

:::

:::{dropdown} OmegaConf Integration
:icon: tools

NeMo Gym uses [OmegaConf](https://omegaconf.readthedocs.io/) for configuration management, which provides:

**Core features**:
- **Variable interpolation**: `${variable_name}` references
- **Hierarchical merging**: Later configs override earlier ones
- **Type validation**: Ensures configuration values have correct types
- **Structured configs**: Supports nested dictionaries and lists
- **Dot notation**: Access nested values with `key.nested.value`

**Example of interpolation**:
```yaml
# Define once
policy_model_name: gpt-4o-2024-11-20

# Reference everywhere
model_1: ${policy_model_name}
model_2: ${policy_model_name}
```

**Hydra integration**: NeMo Gym uses Hydra (built on OmegaConf) for command-line parsing, enabling the `+key=value` override syntax.

:::
