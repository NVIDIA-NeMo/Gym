(setup-config)=

# Configuration Management

Practical guides for managing NeMo Gym configurations across different environments, handling secrets, and configuring multiple servers.

:::{tip}
**New to NeMo Gym's configuration system?** Read {doc}`../../about/concepts/configuration-system` first to understand how the three-tier architecture (YAML → env.yaml → CLI) works conceptually.
:::

---

## Configuration Guides

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Environments
:link: environments
:link-type: doc

**How-to guide** for setting up separate dev, test, and production configurations.
+++
{bdg-secondary}`how-to` {bdg-secondary}`dev-test-prod`
:::

:::{grid-item-card} {octicon}`shield-lock;1.5em;sd-mr-1` Secrets Management
:link: secrets
:link-type: doc

**How-to guide** for securely managing API keys and sensitive configuration values.
+++
{bdg-secondary}`how-to` {bdg-secondary}`security`
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Multi-Server Setup
:link: multi-server
:link-type: doc

**How-to guide** for configuring multiple models, resource servers, and agents.
+++
{bdg-secondary}`how-to` {bdg-secondary}`architecture`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Troubleshooting
:link: troubleshooting
:link-type: doc

**Reference** for debugging configuration issues and validating your setup.
+++
{bdg-secondary}`reference` {bdg-secondary}`debugging`
:::

::::

---

## Configuration Hierarchy Reminder

NeMo Gym loads configuration from three layers (lowest to highest priority):

```
1. YAML Files          → Base configuration (structure)
2. env.yaml            → Secrets and environment-specific values
3. Command-Line Args   → Runtime overrides
```

**Most common pattern**:
- **YAML files**: Define server architecture (what servers exist)
- **env.yaml**: Store API keys and environment-specific settings
- **CLI**: Temporary overrides for testing

---

## Quick Configuration Patterns

### Pattern 1: Single Environment

**Use when**: Local development, simple deployments

```yaml
# config.yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: responses_api_models/openai_model/app.py
      openai_api_key: ${policy_api_key}  # References env.yaml
      model_name: gpt-4o-2024-11-20
```

```yaml
# env.yaml
policy_api_key: sk-your-actual-key
```

```bash
ng_run "+config_paths=[config.yaml]"
```

---

### Pattern 2: Multiple Environments

**Use when**: Dev, staging, production deployments

```bash
# Development
ng_run "+config_paths=[config.yaml]" "+dotenv_path=env.dev.yaml"

# Staging
ng_run "+config_paths=[config.yaml]" "+dotenv_path=env.staging.yaml"

# Production
ng_run "+config_paths=[config.yaml]" "+dotenv_path=env.prod.yaml"
```

**Guide**: {doc}`environments` for complete setup

---

### Pattern 3: Quick Testing Override

**Use when**: Testing different models or settings without changing files

```bash
# Test with cheaper model
ng_run "+config_paths=[config.yaml]" +policy_model_name=gpt-4o-mini

# Test with different temperature
ng_run "+config_paths=[config.yaml]" +responses_create_params.temperature=0.8
```

---

### Pattern 4: CI/CD Deployment

**Use when**: Automated deployments with environment variables

```bash
# In CI/CD pipeline
ng_run "+config_paths=[${CONFIG_PATH}]" \
    +policy_api_key=${PROD_API_KEY} \
    +policy_model_name=${MODEL_VERSION} \
    +default_host=0.0.0.0
```

**Guide**: {doc}`environments` for CI/CD patterns

---

## Configuration Best Practices

### ✅ Do

- **Use env.yaml for secrets** - Never commit API keys to git
- **Use YAML for structure** - Define server architecture in version control
- **Use CLI for testing** - Temporary overrides don't persist
- **Document your configs** - Add comments explaining why settings exist
- **Version your configs** - Track changes in git (except env.yaml)

### ❌ Don't

- **Don't hardcode secrets in YAML** - Use `${variable}` references
- **Don't commit env.yaml** - Add to `.gitignore`
- **Don't edit configs manually in prod** - Use configuration management tools
- **Don't mix concerns** - Keep structure (YAML) separate from secrets (env.yaml)

---

## Common Configuration Tasks

```{list-table}
:header-rows: 1
:widths: 40 60

* - Task
  - Guide
* - Set up dev/test/prod environments
  - {doc}`environments`
* - Manage API keys securely
  - {doc}`secrets`
* - Configure multiple models
  - {doc}`multi-server`
* - Debug configuration issues
  - {doc}`troubleshooting`
* - Override config for testing
  - Use CLI: `ng_run "+config_paths=[...]" +key=value`
* - Validate configuration
  - `ng_dump_config "+config_paths=[...]"`
```

---

## Next Steps

**Most users should start with**:

:::{button-ref} environments
:color: primary
:outline:
:ref-type: doc

Set Up Environments →
:::

**For secure deployments**:

:::{button-ref} secrets
:color: secondary
:outline:
:ref-type: doc

Manage Secrets →
:::

---

```{toctree}
:hidden:
:maxdepth: 1

environments
secrets
multi-server
troubleshooting
```

