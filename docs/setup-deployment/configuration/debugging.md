# Configuration Debugging

(config-debugging)=

Validate configuration and diagnose issues before running servers.

---

## Quick Start

View your resolved configuration to catch errors early:

```bash
ng_dump_config "+config_paths=[config.yaml]"
```

This shows the final configuration after merging all layers:

1. Configuration files (from `config_paths`)
2. Environment variables (from `env.yaml`)
3. Command-line overrides (from `+key=value`)

```{tip}
Always run `ng_dump_config` before `ng_run` to validate configuration. It uses identical resolution logic.
```

---

## Core Debugging Workflows

::::{tab-set}

:::{tab-item} Variable Resolution

Verify `env.yaml` variables are substituted correctly:

```bash
# Check specific variable
ng_dump_config "+config_paths=[config.yaml]" | grep api_key

# Expected output:
# openai_api_key: sk-abc123
```

**Common issues**:

- Variable undefined in `env.yaml`
- Typo in variable name (`${policy_key}` vs `${policy_api_key}`)
- Wrong env file loaded (specify with `+dotenv_path=env.prod.yaml`)

:::

:::{tab-item} Server References

Validate server cross-references:

```bash
# List all server references
ng_dump_config "+config_paths=[config.yaml]" | grep -A 2 "server:"
```

**Valid reference structure**:

```yaml
my_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        type: responses_api_models  # Must match server category
        name: policy_model          # Must match top-level server name
```

**Common issues**:

- Non-existent server name
- Wrong server type
- Server defined after reference (check `config_paths` order)

:::

:::{tab-item} Port Conflicts

Check port assignments across all servers:

```bash
ng_dump_config "+config_paths=[config.yaml]" | grep "port:"
```

**Resolution strategies**:

1. **Auto-assign** (recommended) - Omit `port` key entirely:

   ```yaml
   my_server:
     responses_api_models:
       openai_model:
         host: "127.0.0.1"  # Port auto-assigned by OS
   ```

2. **Explicit ports** - Set unique ports manually:

   ```yaml
   policy_model:
     responses_api_models:
       openai_model:
         port: 8080
   
   judge_model:
     responses_api_models:
       openai_model:
         port: 8081
   ```

:::

:::{tab-item} CLI Overrides

Test command-line overrides apply correctly:

```bash
ng_dump_config "+config_paths=[config.yaml]" \
    +policy_model.responses_api_models.openai_model.model_name=gpt-4o-mini \
    | grep model_name
```

**Override syntax**:

- Top-level: `+key=value`
- Nested: `+server.section.key=value`
- With spaces: `+key="value with spaces"`
- Multiple: `+key1=value1 +key2=value2`

:::

::::

---

## Multi-Config Debugging

When merging multiple configuration files:

```bash
config_paths="base.yaml,dev.yaml,overrides.yaml"
ng_dump_config "+config_paths=[$config_paths]"
```

**Merge order**: Files merge left-to-right, with later files overriding earlier ones.

**Priority**: `config_paths[0]` < `config_paths[1]` < ... < `env.yaml` < CLI args

### Compare Configs

```bash
# Dump each config separately
ng_dump_config "+config_paths=[base.yaml]" > /tmp/base.yaml
ng_dump_config "+config_paths=[dev.yaml]" > /tmp/dev.yaml

# See what changed
diff /tmp/base.yaml /tmp/dev.yaml
```

```{seealso}
See {doc}`multi-server` for multi-file configuration patterns.
```

---

## Pre-Deployment Checklist

Validate configuration before running:

```bash
# 1. Dump resolved config
ng_dump_config "+config_paths=[$config_paths]" > /tmp/config.yaml

# 2. Verify required variables present
grep "api_key" /tmp/config.yaml

# 3. Check all servers configured
grep "entrypoint:" /tmp/config.yaml

# 4. Validate port assignments
grep "port:" /tmp/config.yaml

# 5. Confirm dataset files exist
grep "jsonl_fpath:" /tmp/config.yaml
```

---

## Common Errors

::::{dropdown} Server reference not found

**Error**:

```text
ValueError: Server reference {'type': 'responses_api_models', 'name': 'policy_model'} not found
```

**Cause**: Referenced server doesn't exist in merged configuration.

**Fix**:

1. Dump config and search for server: `ng_dump_config ... | grep policy_model`
2. Check spelling matches exactly
3. Ensure server defined before reference (check `config_paths` order)

::::

::::{dropdown} API key not set

**Error**:

```text
ValueError: openai_api_key is required but not set
```

**Cause**: Variable `${policy_api_key}` not resolved from `env.yaml`.

**Fix**:

1. Verify `env.yaml` exists and contains the variable
2. Check exact spelling (case-sensitive)
3. Test resolution: `ng_dump_config ... | grep api_key`
4. Specify env file: `ng_dump_config "+dotenv_path=env.yaml" ...`

::::

::::{dropdown} Port already in use

**Error**:

```text
OSError: [Errno 48] Address already in use
```

**Cause**: Port conflict with another service.

**Fix**:

1. Identify conflicting port: `ng_dump_config ... | grep port`
2. **Best practice**: Remove `port` key (auto-assign)
3. **Alternative**: Change to unused port
4. **Temporary**: Kill process: `lsof -ti:8000 | xargs kill`

::::

::::{dropdown} Dataset file not found

**Error**:

```text
FileNotFoundError: [Errno 2] No such file or directory: 'data/train.jsonl'
```

**Cause**: Dataset path doesn't exist or is incorrect.

**Fix**:

1. Check path relative to repo root: `ls data/train.jsonl`
2. Verify path in config: `ng_dump_config ... | grep jsonl_fpath`
3. Download dataset: `ng_download_dataset_from_gitlab ...`

::::

::::{dropdown} Configuration merge conflicts

**Symptom**: Wrong values after merging multiple configs.

**Cause**: Incorrect merge order or unintended overwrites.

**Fix**:

```bash
# Bad: base overwrites dev settings
config_paths="dev.yaml,base.yaml"  # ❌

# Good: dev overrides base defaults
config_paths="base.yaml,dev.yaml"  # ✅

# Verify merge result
ng_dump_config "+config_paths=[base.yaml,dev.yaml]" > /tmp/merged.yaml
```

Remember: Later files override earlier ones.

::::

---

## Advanced Techniques

### Environment-Specific Validation

Compare resolved configs across environments:

```bash
# Development
ng_dump_config "+dotenv_path=env.dev.yaml" > /tmp/dev.yaml

# Production
ng_dump_config "+dotenv_path=env.prod.yaml" > /tmp/prod.yaml

# Compare
diff /tmp/dev.yaml /tmp/prod.yaml
```

---

### Extract Specific Server Config

```bash
# View single server configuration
ng_dump_config "+config_paths=[config.yaml]" | grep -A 20 "^policy_model:"

# View all model servers
ng_dump_config "+config_paths=[config.yaml]" | grep -A 10 "responses_api_models:"
```

---

### CI/CD Validation

```bash
# Validate in pipeline before deployment
ng_dump_config "+config_paths=[${CONFIG_PATH}]" > resolved.yaml

# Custom validation script
python scripts/validate_config.py resolved.yaml

# Check for required keys
if ! grep -q "policy_api_key" resolved.yaml; then
    echo "Error: Missing policy_api_key"
    exit 1
fi
```

---

## Next Steps

- **{doc}`index`** - Configuration system overview
- **{doc}`multi-server`** - Multi-server configuration patterns
- **{doc}`../operations/index`** - Debug running servers with logging and profiling
