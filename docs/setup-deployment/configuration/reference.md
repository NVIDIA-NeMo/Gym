(config-reference)=

# Configuration Reference

Complete schema reference for NeMo Gym configuration files.

```{tip}
Use `ng_dump_config "+config_paths=[config.yaml]"` to validate your configuration and see the final resolved values.
```

---

## Three-Level Naming Structure

All servers follow this hierarchy:

```yaml
<unique_server_name>:              # Level 1: Unique identifier
  <server_type>:                   # Level 2: Server category
    <server_implementation>:       # Level 3: Implementation
      entrypoint: app.py
```

```{list-table}
:header-rows: 1
:widths: 15 85

* - Level
  - Description
* - **Level 1**
  - Unique server name used for cross-references. Must be unique across entire configuration.
* - **Level 2**
  - Server type: `responses_api_models`, `resources_servers`, or `responses_api_agents`
* - **Level 3**
  - Implementation: `openai_model`, `vllm_model`, `simple_agent`, `library_judge_math`, etc.
```

---

## Server Configurations

Each server type has specific required and optional fields:

::::{tab-set}

:::{tab-item} Model Servers

Model servers provide LLM inference for agents.

**Required fields**:

```{list-table}
:header-rows: 1
:widths: 25 15 60

* - Field
  - Type
  - Description
* - `entrypoint`
  - str
  - Path to server implementation (e.g., `app.py`)
* - `model_name`
  - str
  - Model identifier (e.g., `gpt-4o-2024-11-20`)
* - `base_url`
  - str
  - API endpoint URL
* - `openai_api_key`
  - str
  - API key (use `${variable}` syntax for secrets)
```

**Optional fields**: `host`, `port` (auto-assigned if omitted)

**Example**:

```yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
      openai_api_key: ${policy_api_key}
      model_name: gpt-4o-2024-11-20
      base_url: https://api.openai.com/v1
```

:::

:::{tab-item} Resource Servers

Resource servers define training environments and evaluation logic.

**Required fields**:

```{list-table}
:header-rows: 1
:widths: 25 15 60

* - Field
  - Type
  - Description
* - `entrypoint`
  - str
  - Path to server implementation (e.g., `app.py`)
* - `domain`
  - enum
  - Domain: `math`, `coding`, `agent`, `knowledge`, `instruction_following`, `long_context`, `safety`, `games`, `e2e`, `other`
```

**Optional fields**: `host`, `port`, server-specific configuration

**Example**:

```yaml
math_resources_server:
  resources_servers:
    library_judge_math:
      entrypoint: app.py
      domain: math
      judge_model_server:
        type: responses_api_models
        name: judge_model
```

```{seealso}
Refer to individual resource server documentation for server-specific fields.
```

:::

:::{tab-item} Agent Servers

Agent servers orchestrate model and resource server interactions.

**Required fields**:

```{list-table}
:header-rows: 1
:widths: 25 15 60

* - Field
  - Type
  - Description
* - `entrypoint`
  - str
  - Path to server implementation (e.g., `app.py`)
* - `resources_server`
  - ServerRef
  - Reference to resource server (`type` + `name`)
* - `model_server`
  - ServerRef
  - Reference to model server (`type` + `name`)
```

**Optional fields**: `host`, `port`, `datasets`

**Example**:

```yaml
math_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: math_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
```

:::

::::

---

## Server References

Servers reference each other using `ServerRef` objects:

```yaml
server_reference:
  type: <server_type>    # Must match Level 2 category
  name: <unique_name>    # Must match Level 1 name
```

**Valid server types**: `responses_api_models`, `resources_servers`, `responses_api_agents`

```{important}
Referenced server names must exist in your configuration. Use `ng_dump_config` to validate references before running.
```

---

## Dataset Configuration

Configure datasets under agent servers. NeMo Gym supports three dataset types.

::::{tab-set}

:::{tab-item} Train

Training data for model optimization.

**Required fields**:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Field
  - Description
* - `name`
  - Dataset identifier (user-defined)
* - `type`
  - Must be `train`
* - `jsonl_fpath`
  - Path to JSONL file (relative to repo root)
* - `gitlab_identifier`
  - GitLab registry metadata (`dataset_name`, `version`, `artifact_fpath`)
* - `license`
  - Dataset license (e.g., `Apache 2.0`, `MIT`, `Creative Commons Attribution 4.0 International`)
```

**Optional**: `num_repeats` (default: 1)

**Example**:

```yaml
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
```

:::

:::{tab-item} Validation

Validation data for evaluation during training.

**Required fields**: Same as train dataset

**Example**:

```yaml
datasets:
- name: validation
  type: validation
  jsonl_fpath: resources_servers/library_judge_math/data/validation.jsonl
  gitlab_identifier:
    dataset_name: bytedtsinghua_dapo17k
    version: 0.0.1
    artifact_fpath: validation.jsonl
  license: Apache 2.0
```

:::

:::{tab-item} Example

Small example dataset (5 samples) for testing and documentation. Committed to git.

**Required fields**:

```{list-table}
:header-rows: 1
:widths: 30 70

* - Field
  - Description
* - `name`
  - Dataset identifier
* - `type`
  - Must be `example`
* - `jsonl_fpath`
  - Path to JSONL file (relative to repo root)
```

**Not required**: `gitlab_identifier`, `license` (commit examples to git)

**Example**:

```yaml
datasets:
- name: example
  type: example
  jsonl_fpath: resources_servers/library_judge_math/data/example.jsonl
```

:::

::::

```{seealso}
Use `ng_download_dataset_from_gitlab` to download train/validation datasets from the GitLab registry.
```

---

## Global Configuration Options

Reserved top-level keys for system-wide settings.

::::{tab-set}

:::{tab-item} Core Settings

```{list-table}
:header-rows: 1
:widths: 35 15 50

* - Field
  - Default
  - Description
* - `config_paths`
  - `[]`
  - List of YAML configuration files to merge
* - `default_host`
  - `127.0.0.1`
  - Default host for all servers
* - `head_server.host`
  - `127.0.0.1`
  - Head server host address
* - `head_server.port`
  - `11000`
  - Head server port
```

**Example**:

```yaml
default_host: "0.0.0.0"
head_server:
  host: "0.0.0.0"
  port: 11000
```

:::

:::{tab-item} Ray Cluster

```{list-table}
:header-rows: 1
:widths: 35 15 50

* - Field
  - Default
  - Description
* - `ray_head_node_address`
  - Auto
  - Custom Ray cluster address (e.g., `ray://127.0.0.1:10001`)
```

By default, NeMo Gym starts a local Ray cluster. Set `ray_head_node_address` to connect to an existing cluster.

**Example**:

```yaml
ray_head_node_address: "ray://192.168.1.100:10001"
```

:::

:::{tab-item} Performance

```{list-table}
:header-rows: 1
:widths: 35 15 50

* - Field
  - Default
  - Description
* - `global_aiohttp_connector_limit`
  - `102400`
  - Max concurrent HTTP connections
* - `global_aiohttp_connector_limit_per_host`
  - `1024`
  - Max connections per host
* - `profiling_enabled`
  - `false`
  - Enable performance profiling
* - `profiling_results_dirpath`
  - None
  - Directory for profiling results
```

**Example**:

```yaml
profiling_enabled: true
profiling_results_dirpath: results/profiling
global_aiohttp_connector_limit: 200000
```

:::

::::

---

## Variable Substitution

Reference values from `env.yaml` using `${variable}` syntax.

**In configuration file**:

```yaml
policy_model:
  responses_api_models:
    openai_model:
      openai_api_key: ${policy_api_key}
      model_name: ${policy_model_name}
```

**In env.yaml**:

```yaml
policy_api_key: sk-your-actual-key
policy_model_name: gpt-4o-2024-11-20
```

```{important}
Never commit `env.yaml` to git. Keep secrets in `env.yaml` and provide `env.yaml.example` for reference.
```

---

## Configuration Hierarchy

Configuration merges from three layers (later overrides earlier):

```text
1. YAML Files          → Base configuration structure
2. env.yaml            → Secrets and environment-specific values  
3. Command-Line Args   → Runtime overrides
```

**Example**:

```yaml
# base.yaml
policy_model:
  responses_api_models:
    openai_model:
      model_name: gpt-4o-2024-11-20
      temperature: 0.7

# env.yaml
policy_api_key: sk-abc123

# Command line
+policy_model.responses_api_models.openai_model.temperature=0.5

# Final result:
policy_model:
  responses_api_models:
    openai_model:
      model_name: gpt-4o-2024-11-20
      openai_api_key: sk-abc123
      temperature: 0.5  # Overridden by CLI
```

---

## Best Practices

Follow these conventions to maintain clear and consistent configurations:

::::{dropdown} Naming Conventions

**Resource servers**: `<name>_resources_server`

```yaml
math_resources_server:
  resources_servers:
    library_judge_math:
      entrypoint: app.py
```

**Agents**: `<name>_simple_agent` or `<environment>_<name>_agent`

```yaml
math_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
```

**Models**: `policy_model`, `judge_model`, or descriptive names

```yaml
policy_model:
  responses_api_models:
    openai_model:
      entrypoint: app.py
```

::::

::::{dropdown} Configuration Organization

**Recommended directory structure**:

```text
configs/
├── base/
│   ├── models/
│   │   ├── policy_model.yaml
│   │   └── judge_model.yaml
│   └── resources/
│       ├── math.yaml
│       └── search.yaml
├── dev.yaml
├── staging.yaml
└── prod.yaml
```

**Guidelines**:

- One agent per configuration file (with dependencies)
- Share models and resources across agents
- Use separate files for different environments

::::

::::{dropdown} Dataset Requirements

**Example datasets**:

- Commit to git
- Include 5 examples
- No `gitlab_identifier` or `license` required

**Train/validation datasets**:

- Must specify `gitlab_identifier`
- Must specify `license`
- Download via `ng_download_dataset_from_gitlab`

**Directory structure**:

```text
resources_servers/my_server/data/
├── example.jsonl          # 5 examples (committed to git)
├── example_metrics.json   # Metrics from ng_prepare_data
└── example_rollouts.jsonl # Rollouts from ng_collect_rollouts
```

::::
