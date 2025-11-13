(tutorials-configuration-management)=

# Configuration Management

Master NeMo Gym's three-layer configuration system for managing secrets, environment-specific settings, and runtime overrides across development and production deployments.

:::{card}

**What you'll learn**:

1. Three-layer configuration resolution (YAML → env.yaml → CLI)
2. Secure secrets management with git-ignored `env.yaml`
3. Runtime overrides for experimentation and A/B testing
4. Environment-specific configuration patterns
5. Config path shortcuts and reusable presets

^^^

**Prerequisites**: Completed {doc}`Your First Agent <../get-started/first-agent>` • OpenAI API key

:::

:::{button-ref} /get-started/index
:color: secondary
:outline:
:ref-type: doc

← New to NeMo Gym? Start with Get Started
:::

---

## Three-Layer Configuration Resolution

Configuration sources are resolved in priority order:

1. **YAML Files** (lowest) - Shared structure and architecture (committed to git)
2. **env.yaml** (middle) - Secrets and environment-specific values (git-ignored)
3. **Command Line** (highest) - Runtime overrides and experiments (ephemeral)

**Resolution**: CLI overrides `env.yaml` overrides YAML files. This enables version-controlled structure, secure per-developer secrets, and safe runtime experimentation without file modifications

:::{tip}
**Architecture details**: See {doc}`Configuration System <../about/concepts/configuration-system>` for design rationale and advanced patterns.
:::

---

## 1. Start with a Base Configuration

Run the weather agent example with YAML configs only:

```bash
ng_run "+config_paths=[responses_api_models/openai_model/configs/openai_model.yaml,resources_servers/example_simple_weather/configs/simple_weather.yaml,responses_api_agents/simple_agent/configs/simple_agent.yaml]"
```

**Expected error**:

```text
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: policy_api_key
    full_key: policy_model.responses_api_models.openai_model.openai_api_key
```

The YAML files reference `${policy_api_key}` but don't define it—secrets belong in `env.yaml`, not committed config files

---

## 2. Provide Secrets via env.yaml

Create `env.yaml` in your project root with your credentials:

```yaml
# API credentials (git-ignored)
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-actual-api-key-here  # Replace with your key
policy_model_name: gpt-4o-2024-11-20
```

Run the same command—servers now start successfully:

```bash
ng_run "+config_paths=[responses_api_models/openai_model/configs/openai_model.yaml,resources_servers/example_simple_weather/configs/simple_weather.yaml,responses_api_agents/simple_agent/configs/simple_agent.yaml]"
```

**Expected output**:

```text
✓ Started: policy_model (responses_api_models) on http://0.0.0.0:8000
✓ Started: simple_weather (resources_servers) on http://0.0.0.0:8001
✓ Started: simple_agent (responses_api_agents) on http://0.0.0.0:8002
```

Variables like `${policy_api_key}` in YAML files now resolve from `env.yaml`. Each developer maintains their own `env.yaml` with personal credentials

---

## 3. Runtime Overrides via CLI

Override any setting at runtime without modifying files. Test with a cheaper model:

```bash
ng_run "+config_paths=[responses_api_models/openai_model/configs/openai_model.yaml,resources_servers/example_simple_weather/configs/simple_weather.yaml,responses_api_agents/simple_agent/configs/simple_agent.yaml]" \
    +policy_model_name=gpt-4o-mini
```

**Verify in startup logs**:

```text
✓ Started: policy_model (responses_api_models)
  Model: gpt-4o-mini
  Base URL: https://api.openai.com/v1
```

CLI arguments override both `env.yaml` and YAML files. Overrides are ephemeral—remove the argument to revert to the `env.yaml` default

---

## 4. Config Path Shortcuts

Define reusable config path combinations in `env.yaml`:

```yaml
# API credentials
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-actual-api-key-here
policy_model_name: gpt-4o-2024-11-20

# Reusable config combinations
weather_config_paths:
  - responses_api_models/openai_model/configs/openai_model.yaml
  - resources_servers/example_simple_weather/configs/simple_weather.yaml
  - responses_api_agents/simple_agent/configs/simple_agent.yaml
```

Reference the shortcut (escape `$` to prevent shell expansion):

```bash
ng_run "+config_paths=\${weather_config_paths}"
```

This pattern enables project-specific presets and reduces command-line verbosity.

---

## 5. Environment-Specific Presets

Define environment presets in `env.yaml`:

```yaml
# Shared credentials
policy_base_url: https://api.openai.com/v1
policy_api_key: sk-your-actual-api-key-here

# Shared config paths
weather_config_paths:
  - responses_api_models/openai_model/configs/openai_model.yaml
  - resources_servers/example_simple_weather/configs/simple_weather.yaml
  - responses_api_agents/simple_agent/configs/simple_agent.yaml

# Environment presets
dev_model: gpt-4o-mini
dev_limit: 5

prod_model: gpt-4o-2024-11-20
prod_limit: 100
```

**Development** (fast iteration, lower cost):

```bash
ng_run "+config_paths=\${weather_config_paths}" \
    +policy_model_name=\${dev_model} \
    +limit=\${dev_limit}
```

**Production** (full scale):

```bash
ng_run "+config_paths=\${weather_config_paths}" \
    +policy_model_name=\${prod_model} \
    +limit=\${prod_limit}
```

Switch environments by referencing different variable sets—same codebase, same YAML structure, different runtime behavior.

---

## 6. Practical Rollout Collection

Run rollouts with different configurations to validate the workflow.

**Development** (5 rollouts, gpt-4o-mini, ~$0.05):

```bash
nemo_gym_run "+config_paths=\${weather_config_paths}" \
    +policy_model_name=\${dev_model} \
    +limit=5 \
    +output_dir=outputs/dev-test
```

**Production** (100 rollouts, gpt-4o-2024-11-20, ~$5.00):

```bash
nemo_gym_run "+config_paths=\${weather_config_paths}" \
    +policy_model_name=\${prod_model} \
    +limit=100 \
    +output_dir=outputs/prod-run
```

**Verify outputs**:

```bash
ls -l outputs/dev-test/   # 5 files
ls -l outputs/prod-run/   # 100 files
```

Same command structure, different behavior via configuration variables—no code changes required.

---

## Troubleshooting

**Missing mandatory value errors**: Add the referenced variable to `env.yaml` or provide via CLI:

```yaml
policy_api_key: sk-your-key-here
```

**Server not found errors** (`Could not find type='resources_servers' name='...'`): Verify the server config is in your `config_paths` and the name matches the YAML definition.

**Port conflicts**: Override with `+server_name.resources_servers.server_name.port=8001` or use `port=0` for auto-assignment.

**Variable expansion issues**: Escape `$` with `\${var_name}` to prevent shell expansion—NeMo Gym handles the resolution.

