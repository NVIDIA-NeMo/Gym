---
name: gym-config
description: >
  Compose and validate Hydra YAML configurations for NeMo Gym. Use when setting up
  server configs, wiring agent-to-server references, configuring model endpoints,
  setting up multi-environment training, or debugging config composition errors.
  Covers Hydra/OmegaConf patterns, env.yaml, and ng_dump_config validation.
license: Apache-2.0
compatibility: Requires Python 3.12+ with NeMo Gym installed.
metadata:
  author: nvidia-nemo-gym
  version: "1.0"
allowed-tools: Bash(ng_*) Read Write Edit Grep Glob
---

# NeMo Gym Configuration

## Config anatomy

A NeMo Gym config defines server instances as top-level keys, each mapping to a server type + subdirectory:

```yaml
my_math_server:                    # Instance name (arbitrary, must be unique)
  resources_servers:               # Server type directory
    math_benchmark:                # Server subdirectory name
      entrypoint: app.py
      domain: math
      datasets:
      - name: example
        type: example
        jsonl_fpath: resources_servers/math_benchmark/data/example.jsonl
      # ... server-specific config fields
```

Agents reference their dependencies by instance name:

```yaml
my_math_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_math_server       # Must match the instance name above
      model_server:
        type: responses_api_models
        name: policy_model         # Must match a model server instance
```

## Step 1: Define server instances

For each component, create a top-level key with:
- A unique instance name
- The server type directory (`resources_servers`, `responses_api_models`, `responses_api_agents`)
- The server subdirectory name
- Server-specific configuration fields

## Step 2: Wire references

Verify that every `name` reference in agent configs points to an actual instance:
- `resources_server.name` must match a resources server instance
- `model_server.name` must match a model server instance
- If using multiple agents/servers, each cross-reference must be exact

## Step 3: Configure model endpoints

Model endpoint config goes in `env.yaml` at project root:

```yaml
policy_base_url: http://localhost:8000/v1
policy_api_key: your-key
policy_model_name: your-model
```

For multiple models (e.g. policy + reward model), add separate entries:
```yaml
reward_base_url: http://localhost:8001/v1
reward_api_key: your-key
reward_model_name: your-reward-model
```

## Step 4: Configure datasets

See the [gym-data](../gym-data/SKILL.md) skill for full dataset preparation. In config:

```yaml
datasets:
- name: train_dataset
  type: train
  jsonl_fpath: resources_servers/my_benchmark/data/train.jsonl
  gitlab_identifier:
    dataset_name: my_benchmark
    version: 0.0.1
    artifact_fpath: train.jsonl
  license: MIT
- name: example
  type: example
  jsonl_fpath: resources_servers/my_benchmark/data/example.jsonl
```

Rules:
- `train` and `validation` types need both `jsonl_fpath` and `gitlab_identifier`
- `example` type only needs `jsonl_fpath` (committed to git)
- `license` required for `train` and `validation`

## Step 5: Multi-environment training

To run multiple environments simultaneously, compose multiple config files:

```bash
ng_run "+config_paths=[
  resources_servers/math/configs/math.yaml,
  resources_servers/code_gen/configs/code_gen.yaml,
  responses_api_models/vllm_model/configs/vllm_model.yaml
]"
```

Each server gets its own instance name and port. Agents can reference different resources servers.

## Step 6: Validate

Always validate the merged config before running:

```bash
ng_dump_config "+config_paths=[resources_servers/my_benchmark/configs/my_benchmark.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

Check:
- All instance names resolve
- No OmegaConf interpolation errors (`${var}` references)
- Dataset paths exist (for example data) or gitlab_identifier is set (for train/validation)
- Port assignments don't conflict
- `verified: false` is present for new servers (pre-commit hook adds this)

## Server-specific config fields

Beyond the base fields documented in CLAUDE.md, individual servers define custom config fields. When configuring a server, read its `app.py` Config class to discover these. Common patterns:

### Concurrency and timeouts
Most servers that run subprocesses or external calls define:
```yaml
num_processes: 8              # asyncio.Semaphore value for parallel execution
max_concurrency: 32           # Alternative name for semaphore bound
unit_test_timeout_secs: 10    # Timeout for subprocess execution
max_execution_time: 10        # Alternative timeout field name
compilation_timeout: 30.0     # Compilation-specific timeout
sql_execution_timeout_s: 30.0 # SQL query timeout
```
These are NOT inherited from any base class — each server defines its own. Check the server's Config class.

### LLM-as-Judge configs
Servers using LLM judges (e.g., `equivalence_llm_judge`, `jailbreak_detection`) require a second model server reference:
```yaml
judge_model_server:
  type: responses_api_models
  name: judge_model              # Must match a model server instance
judge_responses_create_params:
  input: []
  temperature: 0.0
  max_output_tokens: 1024
judge_endpoint_max_concurrency: 64  # Rate-limit judge API calls
```
This means you need TWO model server instances in your config when using judge-based verification.

### Partial reward configs
Several servers support non-binary rewards for nuanced training signals:
```yaml
# jailbreak_detection
reward_if_safe: 1.0
reward_if_unsafe: 0.0
reward_if_unclear: 0.0
reward_if_quality_high: 1.0
reward_if_quality_low: 0.3     # Partial credit

# equivalence_llm_judge
reward_if_swap_fails: 0.0      # Can be -1.0 for penalty
reward_if_full_generation_succeeds: 0.5  # Partial credit on fallback
check_twice_swap: true          # Positional bias detection
```

### External service connections
Some servers connect to external services:
```yaml
sandbox_host: ${oc.env:SANDBOX_HOST,localhost}  # OmegaConf env var injection
sandbox_port: ${oc.env:SANDBOX_PORT,8080}
```
The `${oc.env:VAR_NAME,default}` pattern injects environment variables at config resolution time. This is the ONE place env vars are acceptable (for infra endpoints that vary per deployment).

### Agent-specific fields
```yaml
max_steps: 1                   # Override default conversation turns
max_correction_turns: 3        # For proof_refinement_agent
include_all_attempts: true     # Record all attempts in output
```

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Instance name mismatch between agent and server | Use exact same string in both places |
| Missing `env.yaml` | Create it at project root with model endpoint config |
| YAML indentation in nested `gitlab_identifier` | Use 4-space indent consistently |
| Hydra `+` prefix confusion | `+key=value` adds new keys, `key=value` overrides existing |
| Config path relative vs absolute | Paths in `config_paths` are relative to project root |
| Missing judge model server for judge-based benchmarks | Need TWO model server instances — one for policy, one for judge |
| Using bare env vars instead of `${oc.env:VAR,default}` | OmegaConf interpolation is the approved pattern for deployment-specific values |
| Forgetting `max_steps` in agent config | Defaults vary by agent — set explicitly for multi-turn |
