(setup-config)=

# Configuration Management

Practical guides for managing NeMo Gym configurations across different environments, handling secrets, and configuring multiple servers.

:::{tip}
**New to NeMo Gym's configuration system?** Read {doc}`../../about/concepts/configuration-system` first to understand how the three-tier architecture (YAML → env.yaml → CLI) works conceptually.
:::

---

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

## Configuration Debugging

### Inspect Resolved Configuration

View the fully resolved configuration after all three layers are merged:

```bash
# Basic usage
ng_dump_config "+config_paths=[config.yaml]"

# Grep for specific values
ng_dump_config "+config_paths=[config.yaml]" | grep policy_api_key

# With multiple configs
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml"
ng_dump_config "+config_paths=[$config_paths]"
```

**Use `ng_dump_config` to**:
- Debug configuration issues before running servers
- Verify variable substitution from `env.yaml` works correctly
- Confirm CLI overrides apply as expected
- Understand the final configuration NeMo Gym sees
- Troubleshoot server startup problems

:::{tip}
Run `ng_dump_config` before `ng_run` to catch configuration errors early. It uses the exact same config resolution logic as `ng_run`.
:::

### Common Debugging Scenarios

**Check if env.yaml variables are resolved**:

```bash
# env.yaml contains: policy_api_key: sk-abc123
ng_dump_config "+config_paths=[config.yaml]" | grep api_key
# Should show: openai_api_key: sk-abc123
```

**Verify CLI overrides work**:

```bash
ng_dump_config "+config_paths=[config.yaml]" +policy_model_name=gpt-4o-mini | grep model_name
# Should show the overridden value
```

**Identify port conflicts**:

```bash
ng_dump_config "+config_paths=[config.yaml]" | grep port
# Shows all assigned ports
```

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

**Pattern**: Use separate env files for each environment

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

**Pattern**: Use environment variables in automated deployments

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

---

## Multi-Server Configuration

NeMo Gym supports running multiple resource servers (training environments) simultaneously.

### Running Multiple Resource Servers

Combine multiple resource servers by listing all configuration files:

```bash
# Single resource server (math)
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml"
ng_run "+config_paths=[$config_paths]"

# Multiple resource servers (math + search)
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml"
ng_run "+config_paths=[$config_paths]"
```

**How it works**:
- Each YAML config defines a uniquely named server
- Configs are merged together (later configs override earlier ones)
- Each server maintains isolated configuration scope
- Servers can reference each other by name

### Multi-Server Use Cases

**Training with multiple environments**:

```bash
# Train agent across coding, math, and search tasks
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/comp_coding/configs/comp_coding.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml"
ng_run "+config_paths=[$config_paths]"
```

**Testing agent across capabilities**:

```bash
# Evaluate agent on diverse tasks
ng_collect_rollouts +agent_name=multi_task_agent \
    +input_jsonl_fpath=data/multi_task_test.jsonl \
    +output_jsonl_fpath=results/multi_task_rollouts.jsonl
```

**Production deployment with diverse capabilities**:

```bash
# Deploy comprehensive agent environment
config_paths="$MODEL_CONFIG,$MATH_CONFIG,$SEARCH_CONFIG,$WEATHER_CONFIG"
ng_run "+config_paths=[$config_paths]" +default_host=0.0.0.0
```

:::{seealso}
The same multi-server pattern applies to data preparation and training. Use the same `config_paths` for `ng_prepare_data` and downstream training frameworks.
:::

---

## Common Configuration Tasks

```{list-table}
:header-rows: 1
:widths: 40 60

* - Task
  - Method
* - Set up dev/test/prod environments
  - Use separate env files: `ng_run ... "+dotenv_path=env.prod.yaml"`
* - Manage API keys securely
  - Store in `env.yaml` (never commit to git)
* - Configure multiple resource servers
  - Add all YAML paths to `config_paths` (see Multi-Server section above)
* - Debug configuration issues
  - Use `ng_dump_config "+config_paths=[...]"` to inspect resolved config
* - Override config for testing
  - Use CLI: `ng_run "+config_paths=[...]" +key=value`
* - Validate configuration before running
  - `ng_dump_config "+config_paths=[...]"`
```



