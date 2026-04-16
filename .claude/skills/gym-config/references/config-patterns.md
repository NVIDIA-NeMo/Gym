# NeMo Gym Configuration Patterns

Self-contained reference for Hydra/OmegaConf YAML configuration in NeMo Gym.

---

## Server instance structure

Every server instance is a top-level YAML key that maps to a server type directory and subdirectory:

```yaml
my_math_server:                    # instance name (your choice)
  resources_servers:               # server type directory
    code_gen:                      # server subdirectory (the implementation)
      entrypoint: app.py
      domain: coding
      verified: false
      timeout: 30
      num_processes: 4
      datasets:
      - name: math_example
        type: example
        jsonl_fpath: resources_servers/code_gen/data/example.jsonl
```

Three server types:
- `resources_servers` — verification and reward computation
- `responses_api_models` — LLM inference (openai, azure_openai, vllm, local_vllm)
- `responses_api_agents` — orchestration (simple_agent, proof_refinement_agent, custom)

---

## Agent-to-server wiring

Agents reference their resources and model servers by `type` + `name`:

```yaml
my_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_math_server          # Must match the instance name above
      model_server:
        type: responses_api_models
        name: policy_model            # Must match a model instance name
      datasets:
      - name: math_example
        type: example
        jsonl_fpath: resources_servers/code_gen/data/example.jsonl
```

**Common mistake:** The `name` field must match the *instance name* (top-level key), not the server subdirectory name. `name: code_gen` is wrong if the instance is called `my_math_server`.

---

## Model endpoint config (env.yaml)

Model endpoints are configured in `env.yaml` at the project root:

```yaml
# Policy model
policy_base_url: http://localhost:8000/v1
policy_api_key: your-key
policy_model_name: your-model

# Judge model (if using LLM-as-judge)
judge_base_url: http://localhost:8001/v1
judge_api_key: your-judge-key
judge_model_name: judge-model-name
```

Referenced in server configs via OmegaConf interpolation:
```yaml
policy_model:
  responses_api_models:
    openai_model:
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model_name: ${policy_model_name}
```

---

## LLM-as-judge configuration

When a resources server uses a judge model for verification:

```yaml
my_judge_benchmark:
  resources_servers:
    equivalence_llm_judge:
      entrypoint: app.py
      domain: math
      verified: false

      # Judge model reference
      judge_model_server:
        type: responses_api_models
        name: judge_model              # Points to a separate model instance

      # Judge inference parameters
      judge_responses_create_params:
        temperature: 0.0
        max_output_tokens: 1024

      # Concurrency control for judge calls
      judge_endpoint_max_concurrency: 16

      # Evaluation options
      check_twice_swap: true           # Check with swapped answer order (positional bias)
      reward_if_swap_fails: 0.0        # Reward when swap disagrees with original
      check_full_generation_on_fail: true
      reward_if_full_generation_succeeds: 0.5  # Partial credit from fallback
```

**Important:** The agent must reference the *policy* model, not the judge. The judge is referenced only by the resources server:

```yaml
# CORRECT
my_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        name: policy_model    # Agent talks to policy model

# WRONG — agent should not reference the judge
my_agent:
  responses_api_agents:
    simple_agent:
      model_server:
        name: judge_model     # Bug: agent generating with judge model
```

---

## Combined reward configuration

For benchmarks with multi-stage reward (e.g., jailbreak detection):

```yaml
my_jailbreak:
  resources_servers:
    jailbreak_detection:
      entrypoint: app.py
      domain: safety
      verified: false

      use_combined_reward: true
      reward_if_quality_low: 0.3       # Partial credit: safe but low quality

      # Judge for quality evaluation
      judge_model_server:
        type: responses_api_models
        name: judge_model
      judge_responses_create_params:
        temperature: 0.0
        max_output_tokens: 512
```

Combined reward formula: `reward = safety_reward * quality_reward`
- UNSAFE → safety_reward = 0.0 → final reward = 0.0
- SAFE + high quality → 1.0 * 1.0 = 1.0
- SAFE + low quality → 1.0 * 0.3 = 0.3

---

## Dataset entries

```yaml
datasets:
# Example (committed to git, 5 entries)
- name: my_example
  type: example
  jsonl_fpath: resources_servers/my_benchmark/data/example.jsonl

# Train (GitLab registry, NOT in git)
- name: my_train
  type: train
  jsonl_fpath: resources_servers/my_benchmark/data/train.jsonl
  gitlab_identifier:
    dataset_name: my_benchmark
    version: 0.0.1
    artifact_fpath: train.jsonl
  license: Apache-2.0

# Validation (GitLab registry, NOT in git)
- name: my_validation
  type: validation
  jsonl_fpath: resources_servers/my_benchmark/data/validation.jsonl
  gitlab_identifier:
    dataset_name: my_benchmark
    version: 0.0.1
    artifact_fpath: validation.jsonl
  license: Apache-2.0
```

**Rules:**
- `example` datasets: `jsonl_fpath` only (5 entries, committed to git)
- `train`/`validation` datasets: require `gitlab_identifier` + `license` + `jsonl_fpath`
- `jsonl_fpath` is the local download destination; `gitlab_identifier` is where to fetch from

---

## OmegaConf patterns

### Environment variable injection (deployment-specific only)

```yaml
# APPROVED — deployment-specific infrastructure values
sandbox_host: ${oc.env:SANDBOX_HOST,localhost}
ray_tmpdir: ${oc.env:RAY_TMPDIR,/tmp}
```

Always provide a default value. This pattern is ONLY for infrastructure values that vary per deployment (sandbox hosts, temp dirs). All benchmark config must go through YAML.

### Variable interpolation

```yaml
base_timeout: 30

my_server:
  resources_servers:
    code_gen:
      timeout: ${base_timeout}
      compilation_timeout: ${base_timeout}
```

### Merging configs

```bash
ng_run "+config_paths=[resources_servers/my_server/configs/my_server.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

Multiple YAML files are merged. Instance names must be unique across all files.

---

## Common mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Agent references judge model instead of policy | Agent generates garbage (wrong model) | Set `model_server.name` to policy instance |
| Instance name mismatch | "Server not found" at runtime | Ensure `name:` in references matches top-level key |
| Missing `gitlab_identifier` on train dataset | `ng_prepare_data` can't download | Add `gitlab_identifier` with dataset_name, version, artifact_fpath |
| Missing `license` on train dataset | Validation warning | Add `license: <SPDX-identifier>` |
| `verified: true` on new server | Pre-commit hook should catch | Set to `false` until baselining is complete |
| `os.environ` in Python for config | Config not reproducible | Use YAML config; `${oc.env:VAR,default}` only for infra values |
| Missing `judge_endpoint_max_concurrency` | Judge model overwhelmed | Set to reasonable value (8-32) |

---

## Validation

```bash
# Dump merged config to verify composition
ng_dump_config "+config_paths=[...]"

# Validate data preparation
ng_prepare_data "+config_paths=[...]" +output_dirpath=/tmp/prepare +mode=example_validation
```
