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
