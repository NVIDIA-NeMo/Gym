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
* - Understand config structure
  - See Configuration Reference section below
```

---

## Configuration Reference

Complete anatomy of a NeMo Gym configuration file.

### Three-Level Naming Structure

NeMo Gym configurations use a three-level hierarchy:

```yaml
<unique_server_name>:              # Level 1: Unique identifier at runtime
  <server_type>:                   # Level 2: Type (agents, models, or resources)
    <server_implementation>:       # Level 3: Implementation type
      entrypoint: app.py
      # ... configuration
```

**Level 1: Unique Server Name**
- Must be unique across your entire configuration
- Used by other servers to reference this server
- Example: `library_judge_math`, `policy_model`, `my_agent`

**Level 2: Server Type**
- One of three types:
  - `responses_api_models` - Model inference servers
  - `resources_servers` - Training environment servers
  - `responses_api_agents` - Agent servers
  
**Level 3: Server Implementation**
- Specific implementation of that server type
- Example: `openai_model`, `vllm_model`, `simple_agent`, `library_judge_math`
- Can run multiple instances with different names at Level 1

### Complete Configuration Example

```yaml
# Resource Server Configuration
library_judge_math_resources_server:
  resources_servers:
    library_judge_math:
      entrypoint: app.py
      # Server-specific configuration
      judge_model_server:
        type: responses_api_models
        name: judge_model
      judge_responses_create_params:
        input: []
      should_use_judge: false

# Model Server Configuration  
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_api_key: ${policy_api_key}
      model_name: gpt-4o-2024-11-20
      base_url: https://api.openai.com/v1

# Agent Server Configuration
library_judge_math_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      # Server references
      resources_server:
        type: resources_servers
        name: library_judge_math_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      # Dataset configuration
      datasets:
      - name: train
        type: train
        jsonl_fpath: resources_servers/library_judge_math/data/train.jsonl
        num_repeats: 1
        gitlab_identifier:
          dataset_name: bytedtsinghua_dapo17k
          version: 0.0.1
          artifact_fpath: train.jsonl
        license: Apache 2.0
      - name: validation
        type: validation
        jsonl_fpath: resources_servers/library_judge_math/data/validation.jsonl
        num_repeats: 1
        gitlab_identifier:
          dataset_name: bytedtsinghua_dapo17k
          version: 0.0.1
          artifact_fpath: validation.jsonl
        license: Apache 2.0
      - name: example
        type: example
        jsonl_fpath: resources_servers/library_judge_math/data/example.jsonl
        num_repeats: 1
```

### Server References

Servers reference each other using the `ServerRef` pattern:

```yaml
server_reference:
  type: <server_type>       # responses_api_models, resources_servers, or responses_api_agents
  name: <unique_name>       # Level 1 unique server name
```

**Example**: Agent referencing model and resource servers:

```yaml
my_agent:
  responses_api_agents:
    simple_agent:
      resources_server:
        type: resources_servers
        name: my_resources_server    # References Level 1 name
      model_server:
        type: responses_api_models
        name: my_model               # References Level 1 name
```

### Common Configuration Fields

**All Servers**:
- `entrypoint` (str): Path to server implementation (e.g., `app.py`)
- `host` (str): Host address (auto-assigned if not specified)
- `port` (int): Port number (auto-assigned if not specified)

**Model Servers**:
- `model_name` (str): Model identifier
- `base_url` (str): API endpoint URL
- `openai_api_key` (str): API key (use `${variable}` for env.yaml references)

**Resource Servers**:
- `domain` (str): Domain identifier for the resource server
- Server-specific configuration (varies by implementation)

**Agent Servers**:
- `resources_server` (ServerRef): Reference to resource server
- `model_server` (ServerRef): Reference to model server
- `datasets` (list): Dataset configurations (optional)

### Dataset Configuration

Datasets are configured under agent servers:

```yaml
datasets:
- name: train                    # Dataset identifier
  type: train                    # Type: train, validation, or example
  jsonl_fpath: path/to/data.jsonl
  num_repeats: 1                 # Number of times to repeat dataset
  start_idx: 0                   # Optional: Start index for slicing
  end_idx: 1000                  # Optional: End index for slicing
  gitlab_identifier:             # Required for train/validation
    dataset_name: dataset_name
    version: 0.0.1
    artifact_fpath: train.jsonl
  license: Apache 2.0            # Required for train/validation
```

**Dataset Types**:
- `train` - Training data (requires `gitlab_identifier` and `license`)
- `validation` - Validation data (requires `gitlab_identifier` and `license`)
- `example` - Example data (5 examples, committed to git, no `gitlab_identifier` needed)

### Global Configuration Options

Reserved top-level keys for global settings:

```yaml
# Global settings (outside server configurations)
config_paths: [...]              # List of YAML config files
default_host: "127.0.0.1"        # Default host for all servers
ray_head_node_address: "ray://..." # Custom Ray cluster address

# Head server configuration
head_server:
  host: "127.0.0.1"
  port: 8000

# Profiling configuration
profiling_enabled: false
profiling_results_dirpath: results/profiling

# HTTP client configuration
global_aiohttp_connector_limit: 102400
global_aiohttp_connector_limit_per_host: 1024
```

### Configuration Best Practices

**Naming conventions**:
- Resources servers: `<name>_resources_server`
- Agents: `<name>_simple_agent` or `<environment>_<name>_agent`
- Models: `policy_model`, `judge_model`, or descriptive names

**Structure**:
- One agent per configuration file (with its dependencies)
- Models and resources can be shared across agents
- Use separate files for different environments (dev, staging, prod)

**References**:
- Always use Level 1 unique names in server references
- Verify references exist before running (use `ng_dump_config`)
- Server references are validated at startup

**Datasets**:
- Example datasets (5 examples) must be committed to git
- Train/validation datasets must specify `gitlab_identifier`
- Use `num_repeats` for data augmentation during training

:::{seealso}
For live configuration validation, use `ng_dump_config "+config_paths=[...]"` to see the fully resolved configuration as NeMo Gym sees it.
:::

