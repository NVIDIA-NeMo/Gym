(setup-config)=

# Configuration Management

Manage NeMo Gym's three-tier configuration system for different environments, secrets, and multi-server deployments.

:::{button-ref} concepts-configuration-system
:color: primary
:outline:
:ref-type: ref

← Review Configuration System Concepts
:::

## Quick Configuration Patterns

Common configuration patterns for different deployment scenarios:

::::{tab-set}

:::{tab-item} Single Environment

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

:::

:::{tab-item} Multiple Environments

**Use when**: Dev, staging, production deployments

```bash
# Development
ng_run "+config_paths=[config.yaml]" "+dotenv_path=env.dev.yaml"

# Staging
ng_run "+config_paths=[config.yaml]" "+dotenv_path=env.staging.yaml"

# Production
ng_run "+config_paths=[config.yaml]" "+dotenv_path=env.prod.yaml"
```

**Pattern**: Use separate `env.yaml` files for each environment

:::

:::{tab-item} Quick Testing

**Use when**: Testing different models or settings without changing files

```bash
# Test with cheaper model
ng_run "+config_paths=[config.yaml]" +policy_model_name=gpt-4o-mini

# Test with different temperature
ng_run "+config_paths=[config.yaml]" +responses_create_params.temperature=0.8
```

**Pattern**: Use CLI overrides for temporary testing

:::

:::{tab-item} CI/CD Pipeline

**Use when**: Automated deployments with environment variables

```bash
# In CI/CD pipeline
ng_run "+config_paths=[${CONFIG_PATH}]" \
    +policy_api_key=${PROD_API_KEY} \
    +policy_model_name=${MODEL_VERSION} \
    +default_host=0.0.0.0
```

**Pattern**: Use environment variables in automated deployments

:::

::::

---

## Topics

Choose the topic that matches your current need:

::::{grid} 1 1 2 2
:gutter: 3


:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Multi-Server Configuration
:link: multi-server
:link-type: doc

Run multiple resource servers simultaneously for diverse training scenarios and production deployments.
+++
{bdg-secondary}`multi-server` {bdg-secondary}`patterns` {bdg-secondary}`production`
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Configuration Reference
:link: reference
:link-type: doc

Complete configuration anatomy with schemas, fields, and examples for all server types.
+++
{bdg-secondary}`reference` {bdg-secondary}`schema` {bdg-secondary}`fields`
:::

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Configuration Debugging
:link: debugging
:link-type: doc

Debug and validate configurations with `ng_dump_config`, troubleshoot common errors, and verify variable substitution.
+++
{bdg-secondary}`debugging` {bdg-secondary}`validation` {bdg-secondary}`troubleshooting`
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

multi-server
reference
debugging

```
