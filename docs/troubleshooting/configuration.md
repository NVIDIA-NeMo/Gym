(troubleshooting-configuration)=

# Configuration Errors

These errors appear when running `ng_run` or `ng_collect_rollouts`. NeMo Gym validates configuration at startup before launching servers.

:::{seealso}
{doc}`/reference/configuration` for complete configuration syntax and options.
:::

---

## Startup Errors

Errors that prevent servers from starting.

### Missing Mandatory Value

```
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: policy_api_key
```

**When**: A required variable (usually in `env.yaml`) is not defined.

**Fix**: Add the missing value to `env.yaml`:

```yaml
policy_api_key: sk-your-api-key
```

Or pass via command line:

```bash
ng_run "+config_paths=[config.yaml]" +policy_api_key=sk-your-api-key
```

### Server Reference Not Found

```
AssertionError: Could not find ResourcesServerRef(type='resources_servers', name='typo_weather') 
in the list of available servers: [ResourcesServerRef(...), ...]
```

**When**: A server config references another server that doesn't exist or isn't loaded.

**Common causes**:
- Typo in the server name
- Referenced server's config file not included in `+config_paths`
- Server defined in a different config file that wasn't loaded

**Fix**: 
1. Check server name spelling in your config
2. Ensure all required config files are in `+config_paths`:

```bash
ng_run "+config_paths=[model.yaml,resource.yaml,agent.yaml]"
```

---

## Validation Errors

Errors where config structure is correct but values are invalid.

### Almost-Servers Detected

```
Configuration Warnings: Almost-Servers Detected

  Almost-Server Detected: 'example_simple_agent'
  This server configuration failed validation:

- ResourcesServerInstanceConfig -> resources_servers -> example_server -> domain: 
  Input should be 'math', 'coding', 'agent', 'knowledge', 'instruction_following', 
  'long_context', 'safety', 'games', 'e2e' or 'other'
```

**When**: Config has the right structure (server type, entrypoint) but contains invalid field values.

**Common causes**:
- Invalid `domain` value for resources servers
- Invalid `license` value in dataset configs
- Missing required fields for the server type

**Fix**: Check the validation error path (for example, `resources_servers -> example_server -> domain`) and update the field with a valid value. Refer to {ref}`configuration-reference` for valid field values.

#### Bypass Strict Validation

To continue with invalid configs (invalid servers will be skipped):

```yaml
# In env.yaml
error_on_almost_servers: false
```

```bash
# Or via command line
ng_run "+config_paths=[config.yaml]" +error_on_almost_servers=false
```

:::{warning}
Bypassing validation means invalid servers won't start. Use this only for debugging, not production.
:::
