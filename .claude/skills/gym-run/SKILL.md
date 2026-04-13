---
name: gym-run
description: >
  Run NeMo Gym benchmarks end-to-end — set up env.yaml, validate config, launch servers,
  collect rollouts, and hand off to profiling. Use when you have a configured benchmark
  and need to actually execute it: first run, smoke test, full rollout collection, or
  troubleshooting a failed launch. Covers ng_run, ng_status, ng_collect_rollouts, and
  ng_dump_config in the correct sequence.
license: Apache-2.0
compatibility: Requires Python 3.12+, NeMo Gym installed. Model endpoint must be reachable.
metadata:
  author: nvidia-nemo-gym
  version: "1.0"
allowed-tools: Bash(ng_*) Bash(curl:*) Bash(python:*) Bash(ps:*) Read Write Edit Grep Glob
---

# NeMo Gym Run

Run a benchmark from zero to profiled results. Follow these steps in order — each depends on the previous.

## Step 1: Set up env.yaml

Create `env.yaml` at the project root. This file provides model endpoint credentials and is gitignored — never commit it.

Minimal template (single model):

```yaml
policy_base_url: http://localhost:8000/v1
policy_api_key: your-key
policy_model_name: your-model
```

Extended template (with judge model for LLM-as-judge benchmarks):

```yaml
policy_base_url: http://localhost:8000/v1
policy_api_key: your-key
policy_model_name: your-model

judge_base_url: http://localhost:8001/v1
judge_api_key: your-key
judge_model_name: judge-model
```

These values are injected into config YAML via OmegaConf interpolation (`${policy_base_url}`, etc.).

## Step 2: Choose config paths

Every run needs at least two configs: one benchmark config + one model config.

| Endpoint type | Model config to use | When |
|---|---|---|
| OpenAI-compatible `/v1/responses` | `responses_api_models/openai_model/configs/openai_model.yaml` | GPT, Claude, NIM endpoints |
| vLLM `/v1/chat/completions` | `responses_api_models/vllm_model/configs/vllm_model.yaml` | Self-hosted vLLM |
| Azure OpenAI | `responses_api_models/azure_openai_model/configs/azure_openai_model.yaml` | Azure deployments |

The benchmark config lives in the resources server directory, e.g. `resources_servers/code_gen/configs/code_gen.yaml`. It defines the resources server, agent, and dataset entries.

## Step 3: Validate before launching

Always validate the merged config before starting servers:

```bash
ng_dump_config "+config_paths=[resources_servers/my_benchmark/configs/my_benchmark.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

Check the output for:
- All `${...}` interpolations resolved (no OmegaConf errors) — if not, `env.yaml` is missing keys
- Agent `resources_server.name` and `model_server.name` match top-level instance names
- Dataset `jsonl_fpath` paths exist for example data
- No port conflicts between server instances

If validation fails, hand off to **gym-config**.

## Step 4: Launch servers

```bash
ng_run "+config_paths=[resources_servers/my_benchmark/configs/my_benchmark.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

This starts the head server, which spawns each server instance as a subprocess with its own isolated venv. First run is slow (venv creation + dependency install). Subsequent runs reuse existing venvs.

To install venvs without starting servers:

```bash
ng_run "+config_paths=[...]" +dry_run=true
```

The command blocks and streams logs. Servers are ready when all health checks pass in the startup output.

> **HPC note**: On systems with long working directory paths (e.g. Lustre mounts), Ray socket paths can exceed the 107-byte Linux limit. Fix: `export RAY_TMPDIR=/tmp` before running.

## Step 5: Verify servers are healthy

In a separate terminal:

```bash
ng_status
```

All servers should show healthy status before collecting rollouts. If any server shows `connection_error` or `timeout`:

1. Check server logs in the terminal running `ng_run`
2. Look for import errors, missing dependencies, or port conflicts
3. Try `ng_run "+config_paths=[...]" +dry_run=true` to verify venv setup
4. If the issue persists, hand off to **gym-debug**

## Step 6: Smoke test

Before a full run, do a quick smoke test with example data:

```bash
ng_collect_rollouts \
  +agent_name=my_benchmark_simple_agent \
  +input_jsonl_fpath=resources_servers/my_benchmark/data/example.jsonl \
  +output_jsonl_fpath=results/smoke_test.jsonl \
  +limit=3 \
  +num_repeats=1 \
  "+responses_create_params={max_output_tokens: 4096, temperature: 1.0}"
```

Inspect the results:

```python
import json
with open("results/smoke_test.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        print(f"task {entry.get('task_index')}: reward={entry.get('reward')}, "
              f"failure={entry.get('failure_reason', 'none')}")
```

**If all rewards are 0.0, do NOT proceed to a full run.** Inspect the diagnostic fields (`extracted_model_code`, `failure_reason`, `result`, `extracted_sql`, etc.) to identify where the pipeline fails. Hand off to **gym-debug** for verification failures.

## Step 7: Full rollout collection

```bash
ng_collect_rollouts \
  +agent_name=my_benchmark_simple_agent \
  +input_jsonl_fpath=path/to/dataset.jsonl \
  +output_jsonl_fpath=results/rollouts.jsonl \
  +num_repeats=5 \
  "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"
```

Key parameters:

| Parameter | Purpose | Typical value |
|---|---|---|
| `num_repeats` | Statistical significance | 5 for profiling, 1 for smoke test |
| `limit` | Cap dataset size | Omit for full run, 3-50 for debugging |
| `resume_from_cache` | Resume interrupted runs | `true` after a crash |
| `max_output_tokens` | Model output budget | 4096-16384 depending on task complexity |
| `temperature` | Sampling temperature | 1.0 for RL profiling, 0.0 for deterministic |
| `num_repeats_add_seed` | Add unique seed per repeat | `true` for reproducibility |

## Step 8: Hand off to gym-profile

Once rollouts are collected, use **gym-profile** to analyze:

```bash
ng_reward_profile \
  +input_jsonl_fpath=path/to/dataset.jsonl \
  +rollouts_jsonl_fpath=results/rollouts.jsonl \
  +output_jsonl_fpath=results/profiled.jsonl \
  +pass_threshold=1.0
```

Then aggregate:

```bash
python scripts/print_aggregate_results.py +jsonl_fpath=results/profiled.jsonl
```

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ng_dump_config` interpolation error | Missing `env.yaml` or missing keys | Create env.yaml with required fields (see Step 1) |
| Server exits immediately | Import error or missing dependency | Check logs; run `ng_run +dry_run=true` to install venvs first |
| `ng_status` shows connection_error | Server hasn't finished starting, or crashed | Wait for startup to complete; check logs for errors |
| Rollout collection hangs at 0% | Model endpoint unreachable | `curl $policy_base_url/models` to verify connectivity |
| All rewards 0.0 | Verification failing on all inputs | Inspect diagnostic fields in smoke test output (Step 6) |
| OOM during rollouts | Too many parallel samples or large outputs | Reduce `max_output_tokens`; reduce concurrent requests |
| `AF_UNIX path too long` | Ray socket path exceeds 107 bytes on HPC | `export RAY_TMPDIR=/tmp` before running |
| 429 errors from model endpoint | Rate limiting | Reduce request concurrency |
| Partial run lost after crash | No cache enabled | Re-run with `+resume_from_cache=true` |

## Quick reference

```bash
ng_dump_config "+config_paths=[...]"           # Validate merged config
ng_run "+config_paths=[...]"                   # Launch all servers
ng_run "+config_paths=[...]" +dry_run=true     # Install venvs only
ng_status                                      # Check server health
ng_collect_rollouts +agent_name=... \
  +input_jsonl_fpath=... \
  +output_jsonl_fpath=... \
  +num_repeats=5                               # Collect rollouts
ng_reward_profile +input_jsonl_fpath=... \
  +rollouts_jsonl_fpath=... \
  +output_jsonl_fpath=...                      # Profile results
```
