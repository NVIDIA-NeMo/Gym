# Multi-Server Configuration

(config-multi-server)=

Run multi-server deployments to train agents across diverse task domains.

---

## Quick Start

Combine resource servers by listing all configuration files:

```bash
# Multi-server deployment (math + search)
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml"

ng_run "+config_paths=[$config_paths]"
```

**Result**: All servers start simultaneously, and agents can access any server by name.

---

## How Configuration Merging Works

When you provide multi-file configurations:

1. **Sequential loading** - NeMo Gym loads each file in order
2. **Key-level merging** - Later files override earlier ones for conflicting top-level keys
3. **Isolated scopes** - Each server (top-level key) maintains its own configuration
4. **Cross-references** - Servers reference each other using `type` and `name` fields

```{tip}
Later configuration files override earlier ones. Put environment-specific overrides at the end of your config list.
```

---

## Configuration Patterns

Choose the pattern that matches your use case.

::::{tab-set}

:::{tab-item} Shared Model

One model serves several resource servers (cost-effective for evaluation).

```yaml
# model_config.yaml
policy_model:
  responses_api_models:
    openai_model:
      model_name: gpt-4o-2024-11-20

# math_config.yaml
math_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        type: responses_api_models
        name: policy_model  # References shared model

# search_config.yaml
search_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        type: responses_api_models
        name: policy_model  # Same shared model
```

**Use when**:

- Cost-effective evaluation
- Consistent model across tasks
- Testing generalist capabilities

:::

:::{tab-item} Specialized Models

Different models optimized for specific domains.

```yaml
# Math-optimized model
math_model:
  responses_api_models:
    openai_model:
      model_name: gpt-4o-2024-11-20
      
# Coding-optimized model
coding_model:
  responses_api_models:
    vllm_model:
      model_name: codellama-34b

# Agents reference domain-specific models
math_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        name: math_model

coding_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        name: coding_model
```

**Use when**:

- Domain-specific optimization
- A/B testing models
- Different model sizes per domain

:::

:::{tab-item} Policy + Judge

Separate models for generation and evaluation (RL training pattern).

```yaml
# Generation model
policy_model:
  responses_api_models:
    vllm_model:
      model_name: training-model

# Evaluation model
judge_model:
  responses_api_models:
    openai_model:
      model_name: gpt-4o-2024-11-20

# Resource server uses judge
math_resources_server:
  resources_servers:
    library_judge_math:
      judge_model_server:
        type: responses_api_models
        name: judge_model

# Agent uses policy
math_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        name: policy_model
      resources_server:
        type: resources_servers
        name: math_resources_server
```

**Use when**:

- RL training with verification
- Separate policy and critic models
- Cost optimization (cheap policy, expensive judge)

See {doc}`../../tutorials/separate-policy-and-judge-models` for complete guide.

:::

::::

---

## Common Use Cases

Apply multi-server configurations to these typical deployment scenarios.

### Multi-Task Training

Train a single agent across diverse capabilities:

```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/comp_coding/configs/comp_coding.yaml,\
resources_servers/library_judge_math/configs/bytedtsinghua_dapo17k.yaml,\
resources_servers/google_search/configs/google_search.yaml"

ng_run "+config_paths=[$config_paths]"
```

**Benefits**: Single model learns diverse capabilities, better generalization, efficient multi-task training.

---

### Cross-Domain Evaluation

Assess agent performance across different task types:

```bash
# Same config for data prep and rollout collection
ng_collect_rollouts +agent_name=multi_task_agent \
    +input_jsonl_fpath=data/multi_task_test.jsonl \
    +output_jsonl_fpath=results/multi_task_rollouts.jsonl
```

**Benefits**: Comprehensive capability assessment, identify domain-specific strengths/weaknesses.

### Environment-Based Configuration

Use different server combinations per environment:

```bash
# Development: Minimal servers
export DEV_CONFIG="model.yaml,math.yaml"
ng_run "+config_paths=[$DEV_CONFIG]"

# Production: Full suite
export PROD_CONFIG="model.yaml,math.yaml,search.yaml,coding.yaml,weather.yaml"
ng_run "+config_paths=[$PROD_CONFIG]" +default_host=0.0.0.0
```

```{seealso}
See {doc}`../../training/handoff-to-training` for using multi-server configs with training frameworks.
```

---

## Configuration Organization

For complex deployments, organize configuration files by environment:

```text
configs/
├── base/
│   ├── models/
│   │   ├── policy.yaml
│   │   └── judge.yaml
│   └── servers/
│       ├── math.yaml
│       ├── search.yaml
│       └── coding.yaml
├── dev.yaml        # Minimal for fast iteration
├── staging.yaml    # Expanded for testing
└── prod.yaml       # Complete deployment
```

**Usage**:

```bash
# Development
ng_run "+config_paths=[configs/dev.yaml]"

# Production
ng_run "+config_paths=[configs/prod.yaml]" +default_host=0.0.0.0
```

```{tip}
Each YAML can reference other files using the `config_paths` key. NeMo Gym recursively loads all referenced configuration files.
```

---

## Validation and Debugging

Debug multi-server configurations by verifying merged settings and identifying conflicts.

### Verify Merged Configuration

View the final merged configuration:

```bash
ng_dump_config "+config_paths=[$config_paths]"
```

This shows the complete configuration after merging all files, with all references resolved.

### Common Issues

Watch out for these problems when merging multiple configuration files:

::::{dropdown} Server name conflicts

**Symptom**: Later configuration file overwrites earlier server unintentionally.

**Solution**: Each server needs a unique top-level key name:

```yaml
# ❌ Bad: Both files define "agent"
# math.yaml
agent:
  responses_api_agents:
    simple_agent: ...

# search.yaml  
agent:  # Overwrites math agent!
  responses_api_agents:
    simple_agent: ...

# ✅ Good: Unique names
# math.yaml
math_agent:
  responses_api_agents:
    simple_agent: ...

# search.yaml
search_agent:
  responses_api_agents:
    simple_agent: ...
```

::::

::::{dropdown} Server reference not found

**Symptom**: Error saying a referenced server doesn't exist.

**Solution**: Verify the referenced server exists in one of your configuration files:

```bash
# Check all server names in merged configuration
ng_dump_config "+config_paths=[$config_paths]" | grep "^[a-z_]*:" | head -20
```

Ensure your server references match these names:

```yaml
math_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        type: responses_api_models
        name: policy_model  # Must match a top-level key
```

::::

::::{dropdown} Configuration not loading in expected order

**Symptom**: Wrong values after merge.

**Solution**: Remember that later files override earlier ones. Put base configuration first, overrides last:

```bash
# ✅ Correct order: base → specific
config_paths="base_model.yaml,dev_overrides.yaml"

# ❌ Wrong order: specific → base (base overwrites dev settings)
config_paths="dev_overrides.yaml,base_model.yaml"
```

::::
