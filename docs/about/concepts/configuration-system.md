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

## How Configuration Gets Resolved

When you run `ng_run`, NeMo Gym merges configuration in this order:

1. **Parse command-line arguments**: Extracts `config_paths` and any overrides
2. **Load env.yaml**: Loads secrets and environment-specific values
3. **Load YAML files**: Loads each file in `config_paths`, resolving variables like `${policy_api_key}`
4. **Merge with priority**: Combines all layers: YAML files → env.yaml → CLI (highest priority)
5. **Validate and cache**: Verifies server references exist, populate defaults, cache for session

**Priority order**: CLI overrides env.yaml, which overrides YAML files.

**Example**: If `policy_model_name` is in both env.yaml (`gpt-4o-2024-11-20`) and CLI (`+policy_model_name=gpt-4o-mini`), CLI wins.

---

## Technical Details

Understanding how NeMo Gym implements configuration resolution can help you optimize performance and debug issues:

:::{dropdown} Configuration Caching
:icon: cache

The configuration is resolved once per process and cached:

```python
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
