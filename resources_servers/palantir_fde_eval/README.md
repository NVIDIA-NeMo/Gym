# Palantir FDE Eval

Resource server evaluating LLM tool-calling accuracy on Palantir Foundry scenarios. Evaluates 315 synthetic scenarios covering 105 Foundry API tools across 6 task types and 3 complexity levels (105 x 3). Uses a three-layer composite binary reward: structure validation, tool name matching, and LLM judge parameter evaluation.

Ground truth is established by Claude Opus responses on the same prompts.

## Synthetic Data Generation

The 315 evaluation scenarios were generated in a two-phase pipeline. Foundry API tool definitions, extraction utilities, and the full generation pipeline live in the companion SDG repo: https://gitlab-master.nvidia.com/ai-sae/palantir-ai-fde-sdg

### Phase 1: Scenario Generation

An LLM (via NVIDIA Data Designer) generates realistic Foundry user requests from a seed matrix of **task type x complexity level**. For each combination, the LLM produces:
- A **system prompt** (Foundry persona context)
- **Context messages** (datasets, folders, resource identifiers)
- A **user request** (natural language ask)

### Phase 2: Ground Truth Collection

Each generated scenario is assembled into production-format messages and submitted to **Claude Opus** with the full 105-tool catalog. Claude's tool call response becomes the ground truth (`tool_call_response` column).

### Task Types (human-derived)

The 6 task types are **not synthetic** — they were derived from 6 real production traces provided by Palantir, representing actual user workflows on the Foundry platform:

| Task Type | Persona | Example |
|---|---|---|
| Data Query | Exploration | "Do any stations have multiple lift disruptions?" |
| Transform Authoring | Builder | "Build a dataset with a primary key, geopoint column, and priority score." |
| Ontology Creation | Builder | "Create an ontology for a todo app." |
| Function Authoring | Builder | "Create a functions repo and write a function that counts issues by state." |
| Diagram Generation | Exploration | "Draw me a diagram of how these datasets relate to each other." |
| Pipeline Building | Builder | "Create a pipeline that joins patient visits and diagnoses." |

### Complexity Levels (designed)

Three complexity levels (**Simple**, **Intermediate**, **Advanced**) were defined as a design axis to vary the difficulty of generated scenarios. The LLM was instructed to scale the number of constraints, resources, and multi-step reasoning required. Combined with the 105-tool ordered seed, this produces the 105 x 3 = 315 scenario matrix.

### Foundry API Tools

The evaluation uses **105 Foundry API tool definitions** extracted from the Palantir FDE platform. The canonical source is [`tools/tool_definitions.json`](https://gitlab-master.nvidia.com/ai-sae/palantir-ai-fde-sdg/-/blob/main/tools/tool_definitions.json) in the [SDG repo](https://gitlab-master.nvidia.com/ai-sae/palantir-ai-fde-sdg). A skinny version (anyOf type constraints only, ~36KB) is hosted in the GitLab ML registry and downloaded via `ng_download_dataset_from_gitlab` (see setup below). It is not committed to git.

## Quick Start

### Prerequisites

- `uv >= 0.9.30` (`uv self update` if needed)
- `ASTRA_API_KEY` env var (for NVIDIA inference API — judge/oracle models)
- Local vLLM server for policy model

### 1. Setup

```bash
cd nemo-gym
uv venv && uv sync --extra dev
source .venv/bin/activate
```

Create `env.yaml` at the nemo-gym root (not tracked in git). This file is loaded by NeMo Gym's OmegaConf config system and provides endpoints, API keys, and concurrency settings.

```yaml
# ── Policy model endpoints ──────────────────────────────────────────
# Three options — switch by changing the pltr_policy_* aliases below.

# local: vLLM on localhost (./scripts/serve.sh)
pltr_local_base_url: http://localhost:8088/v1
pltr_local_api_key: dummy
pltr_local_model_name: nemotron-super-rl-030326

# dev: NVIDIA internal inference API (requires ASTRA_API_KEY)
pltr_dev_base_url: https://inference-api.nvidia.com/v1
pltr_dev_api_key: ${oc.env:ASTRA_API_KEY}
pltr_dev_model_name: nvidia/nvidia/nemotron-3-super-preview

# public: NVIDIA public API (requires NVIDIA_API_KEY)
pltr_public_base_url: https://integrate.api.nvidia.com/v1
pltr_public_api_key: ${oc.env:NVIDIA_API_KEY}
pltr_public_model_name: nvidia/nemotron-3-super-120b-a12b

# ── Active policy endpoint (change these to switch) ────────────────
pltr_policy_base_url: ${pltr_public_base_url}
pltr_policy_api_key: ${pltr_public_api_key}
pltr_policy_model_name: ${pltr_public_model_name}

# ── Judge model (Opus 4.6 via NVIDIA inference API) ────────────────
pltr_judge_base_url: https://inference-api.nvidia.com/v1
pltr_judge_api_key: ${oc.env:ASTRA_API_KEY}
pltr_judge_model_name: aws/anthropic/bedrock-claude-opus-4-6

# ── Aliases for NeMo Gym framework (expects unprefixed names) ──────
policy_base_url: ${pltr_policy_base_url}
policy_api_key: ${pltr_policy_api_key}
policy_model_name: ${pltr_policy_model_name}

# ── GitLab ML registry (for dataset downloads) ─────────────────────
mlflow_tracking_uri: "https://gitlab-master.nvidia.com/api/v4/projects/191584/ml/mlflow/"
mlflow_tracking_token: <your-gitlab-personal-access-token>
```

To switch endpoints, change the 3 `pltr_policy_*` aliases (e.g., `${pltr_local_*}` for local vLLM, `${pltr_dev_*}` for internal API). Kill servers and restart after changing.

### 2. Prepare evaluation data

Both the 315-row eval dataset and the tool definitions are hosted in GitLab's ML registry. Neither is committed to git.

Requires MLflow credentials in `env.yaml`:

```yaml
mlflow_tracking_uri: "https://gitlab-master.nvidia.com/api/v4/projects/191584/ml/mlflow/"
mlflow_tracking_token: <your-gitlab-personal-access-token>
```

**a) Download tool definitions** (105 Foundry API tools, required for schema-aware validation):

```bash
ng_download_dataset_from_gitlab \
    +dataset_name=palantir_fde_eval \
    +version=0.0.1 \
    +artifact_fpath=tool_definitions.json \
    +output_fpath=resources_servers/palantir_fde_eval/data/tool_definitions.json
```

Source: [`tools/tool_definitions.json`](https://gitlab-master.nvidia.com/ai-sae/palantir-ai-fde-sdg/-/blob/main/tools/tool_definitions.json) in the SDG repo.

**b) Download the full eval dataset** (315 rows):

```bash
ng_prepare_data \
    "+config_paths=[resources_servers/palantir_fde_eval/configs/palantir_fde_eval.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    +output_dirpath=/tmp/palantir_prepare \
    +mode=train_preparation \
    +should_download=true \
    +data_source=gitlab
```

**c) Validate example data** (5 rows, committed — use for PR checks):

```bash
ng_prepare_data \
    "+config_paths=[resources_servers/palantir_fde_eval/configs/palantir_fde_eval.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
    +output_dirpath=/tmp/palantir_prepare \
    +mode=example_validation
```

**d) Regenerate from source** (requires the [SDG repo](https://gitlab-master.nvidia.com/ai-sae/palantir-ai-fde-sdg)):

```bash
cd ../palantir-ai-fde-sdg/evaluation/v2
python convert_to_nemo_gym.py --output palantir_fde_eval_v2.jsonl
cd ../../nemo-gym
```

### 3. Start servers

All wrapper scripts are in the aifde project root (`scripts/`), which symlink to this resource server where applicable.

**a) vLLM server** (policy model):
```bash
scripts/serve.sh                        # default: port 8088, 262K context
MAX_MODEL_LEN=131072 scripts/serve.sh   # shorter context (less GPU memory)
```

**b) NeMo Gym servers** (agent + judge + oracle + resource):
```bash
scripts/ng_run.sh
# or directly from nemo-gym root:
resources_servers/palantir_fde_eval/scripts/ng_run.sh
```

This starts 4 servers: policy_model, judge_model, palantir_fde_eval, palantir_fde_eval_simple_agent. Wait for `ng_status` to show all healthy.

### 4. Run evaluation

```bash
# Smoke test (1 sample)
scripts/run_eval.sh --num_prompts 1

# Full 315 rows with pre-flight smoke
scripts/run_eval.sh --num_prompts 315 --smoke-first

# Custom sample size with explicit input
scripts/run_eval.sh --num_prompts 20 --input data/full_315_with_tools.jsonl
```

Set `EVAL_DATASET_PATH` in `eval.env` (at nemo-gym root) to point at your dataset, or use `--input`. Results go to `results/rollouts.jsonl`.

### 5. Read results

Metrics printed to stdout after collection:

```
mean/reward              — overall pass rate (0.0-1.0)
mean/structure_valid     — % passing structure validation
mean/tool_name_match     — % picking the correct tool(s)
mean/num_predicted       — avg tool calls per row
mean/num_expected        — avg expected tool calls per row
```

Per-task breakdown:

```bash
ng_reward_profile \
  +input_jsonl_fpath=<input.jsonl> \
  +rollouts_jsonl_fpath=results/rollouts.jsonl \
  +output_jsonl_fpath=results/profiled.jsonl \
  +materialized_inputs_jsonl_fpath=results/rollouts_materialized_inputs.jsonl \
  +pass_threshold=1.0
```

### 6. Shutdown

```bash
scripts/ng_kill.sh     # or: ray stop --force
```

## Testing

```bash
ng_test +entrypoint=resources_servers/palantir_fde_eval
```

## Reward Computation

The `verify()` method implements a composite three-layer reward with short-circuit evaluation. Reward is binary: 1.0 (pass) or 0.0 (fail).

### Special Cases (bypass all layers)

- No tool call expected and none predicted: reward = 1.0
- No tool call expected but model predicted one: reward = 0.0
- Tool call expected but model predicted none: reward = 0.0

### Layer 1: Structure Validation

Detects invalid parameter structures before expensive downstream checks.

1. **Stringified parameter detection** — Recursively scans tool call arguments for string values that parse as JSON objects or arrays. Catches double-serialized parameters (e.g., `"filters": "{\"field\": \"value\"}"` instead of `"filters": {"field": "value"}`). Only flags dict/list; primitive JSON values are allowed as strings.

2. **Schema-aware anyOf validation** — When `tool_schemas_fpath` is configured, validates top-level parameter types against `anyOf` union type constraints. Only checks top-level properties; does not recurse into nested schemas.

If structure is invalid: reward = 0.0, skip remaining layers.

### Layer 2: Tool Name Matching

Sorted name comparison between predicted and expected tool calls. Handles multi-tool-call scenarios by sorting both lists. Count mismatch also fails.

If names do not match: reward = 0.0, skip judge layer.

### Layer 3: LLM Judge

Calls the judge model to evaluate parameter semantic equivalence. For each predicted/expected call pair:

1. Formats judge prompt template with `json.dumps` of both parameter sets
2. Sends to judge model via `/v1/responses`
3. Parses verdict: whichever label (`[[PASS]]` or `[[FAIL]]`) appears first wins
4. No verdict label found → pair fails

All pairs must pass. Short-circuits on first failure. Final reward = 1.0 only if all three layers pass.

## Configuration

### Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `judge_model_server` | `ModelServerRef` | (required) | Model server for the LLM judge |
| `judge_responses_create_params` | `NeMoGymResponseCreateParamsNonStreaming` | `input: []` | Empty input template, populated at runtime |
| `judge_prompt_template_fpath` | `str` | `prompt_templates/tool_call_judge.txt` | Judge prompt template path (relative to server dir) |
| `judge_equal_label` | `str` | `[[PASS]]` | Verdict label for parameter equivalence |
| `judge_not_equal_label` | `str` | `[[FAIL]]` | Verdict label for parameter mismatch |
| `judge_endpoint_max_concurrency` | `Optional[int]` | `64` | `asyncio.Semaphore` limit for concurrent judge calls |
| `oracle_model_server` | `Optional[ModelServerRef]` | `None` | Optional model server (wired in config, not used in `verify()`) |
| `tool_schemas_fpath` | `Optional[str]` | `None` | Path to `tool_definitions.json` for schema-aware validation |

### Concurrency

| Setting | Controls | Default | Where it applies |
|---------|----------|---------|------------------|
| `pltr_rollout_concurrency` | Parallel policy model calls | 10 | CLI arg `+num_samples_in_parallel` |
| `judge_endpoint_max_concurrency` | Parallel judge model calls | 64 | Code default in `app.py`; override in YAML if needed |

### Example YAML Config

```yaml
palantir_fde_eval:
  resources_servers:
    palantir_fde_eval:
      entrypoint: app.py
      domain: agent
      verified: false
      judge_model_server:
        type: responses_api_models
        name: judge_model
      judge_responses_create_params:
        input: []
      judge_prompt_template_fpath: prompt_templates/tool_call_judge.txt
      tool_schemas_fpath: data/tool_definitions.json
      oracle_model_server:
        type: responses_api_models
        name: oracle_model
palantir_fde_eval_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: palantir_fde_eval
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: example
        type: example
        jsonl_fpath: resources_servers/palantir_fde_eval/data/example.jsonl
```

## Licensing

```
Code: Apache 2.0
Data: Apache 2.0
Dependencies: nemo_gym (Apache 2.0)
```
