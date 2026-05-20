---
name: evaluate-environments
description: >
  Guide for running evaluations for NeMo Gym environments/benchmarks. 
  This should not be used for creating a new environment or integration a new
  evaluation/environment. 
  Use this skill when a model, agent, or benchmark needs to be run or compared. 
  It also should be used for collecting rollouts/rewards. 
  Triggered by:
  "evaluate model", "evaluate agent", "run benchmark", "collect rollouts", 
  "reward profiling", "benchmark results", "compare models", "compare agents", 
  "analyze results", "pass@k", "why is reward 0"
---

# Evaluate Environments
This is for running reliable evaluations and generating rollouts/getting rewards.

First always test and make sure that a single evaluation run works before scaling up.

## Pre-requisites
1. Install NeMo Gym or repo set up: `uv venv && uv sync` from project root if working in Github repo
2. You need a policy model. This can be a model endpoint or a self hosted model. 
env.yaml` at project root with model endpoint:
     ```yaml
     policy_base_url: https://api.openai.com/v1
     policy_api_key: <key>
     policy_model_name: gpt-4.1-2025-04-14
     ```
     For self-hosted / vLLM / Fireworks / OpenRouter, see [Configure Model docs](https://docs.nvidia.com/nemo/gym/latest/model-server).

## Running Evals/Rollouts
**Step 1 — Start servers.** NeMo Gym runs three coordinated server types; the agent name in `ng_collect_rollouts` must match the top-level instance key declared in the
environment config you load here.

```bash
ng_run "+config_paths=[<env_config>,<model_config>]"
```

Verify with `ng_status` in another terminal. You should see the resources server, the agent server, and the model server.

**Step 2 — Smoke test on `example.jsonl` (5 tasks, committed to git).**

```bash
ng_collect_rollouts \
+agent_name=<env>_simple_agent \
+input_jsonl_fpath=resources_servers/<env>/data/example.jsonl \
+output_jsonl_fpath=results/smoke_rollouts.jsonl \
+limit=5 \
+num_repeats=1
```

If smoke fails, do **not** scale up. Inspect `results/smoke_rollouts.jsonl` directly — a completed-with-reward-0 task is very different from a server/runtime error.

**Step 3 — Scale.** Use validation/train data (downloaded via `ng_prepare_data` if not local — see [Dataset 
Management](https://docs.nvidia.com/nemo/gym/latest/about/concepts/datasets)). Bump `num_repeats` for variance reduction.

```bash
ng_collect_rollouts \
+agent_name=<env>_simple_agent \
+input_jsonl_fpath=<full_dataset.jsonl> \
+output_jsonl_fpath=results/rollouts.jsonl \
+num_repeats=5 \
+num_samples_in_parallel=10 \
"+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"
```

`_aggregate_metrics.json` is written alongside `rollouts.jsonl` automatically. Headline numbers (`mean/reward`, `pass@1/accuracy`, etc.) print to stdout.

## Per-task pass rates & pass@k

```bash
ng_reward_profile \
+input_jsonl_fpath=<full_dataset.jsonl> \
+rollouts_jsonl_fpath=results/rollouts.jsonl \
+output_jsonl_fpath=results/profiled.jsonl \
+pass_threshold=1.0

python scripts/print_aggregate_results.py +jsonl_fpath=results/profiled.jsonl
```

- `pass@1 = avg_reward` across rollouts of a task.
- `pass@k` derived from `max_reward` across `k` repeats — only meaningful when reward is binary.
- For continuous reward, ignore pass@k and report distribution shifts + per-task means.

## Common Evaluation Patterns
**Compare models on the same env+agent:** chain multiple model configs and run `ng_collect_rollouts` once per `agent_name` that points at each. The agent's resources server
stays identical so any score delta is attributable to the model.
**Compare agents on the same env+model:** swap the agent config in `config_paths` and re-run. Hold dataset, `num_repeats`, and `responses_create_params` constant.
**No matter what, always change only one knob at a time** Mixing model/agent changes makes deltas uninterpretable. 

## Inspect saved results
- `results/<run>_rollouts.jsonl` — one line per (task, repeat) with `reward`, `response`, `task_index`, and any custom `VerifyResponse` fields.
- `results/<run>_aggregate_metrics.json` — array, one object per agent: `agent_ref`, `agent_metrics`, `key_metrics`, `group_level_metrics`.
- `results/<run>_materialized_inputs.jsonl` — the fully resolved inputs sent to the agent (useful for diffing prompts).
For benchmark-specific headline metrics, override `compute_metrics()` / `get_key_metrics()` on the resources server or agent — see [Aggregate Metrics 
  docs](https://docs.nvidia.com/nemo/gym/latest/environment-tutorials/aggregate-metrics). When debugging an unexpected score, read the rollout JSONL directly before re-running.

## Metrics interpretation 
1. **Binary vs continuous reward** — pass@k is only meaningful when reward is effectively {0, 1}. For continuous rewards, focus on distribution shifts and per-task means.
2. **Variance reduction** — keep increasing `num_repeats` until variance across runs of the same model is < 1%. Anything noisier and small score deltas are noise.
3. **Inspect samples before claiming regressions.** Aggregate numbers can hide a single broken task type swamping the average.
4. **Distinguish "completed rollout with low reward" from "runtime/server failure."** The latter shows up as exceptions in server logs and/or missing rollouts; the former is a
model/agent quality issue.

## Output format

When summarizing an evaluation run, return:

1. **Run configuration table** — env, agent_name, model, dataset, num_repeats, exact command line.
2. **Aggregate metrics** — `mean/reward`, `pass@1`, `pass@k` (if binary), per-task variance.
3. **Sample-level failure themes** — group the 0-reward rollouts by failure mode (parsing error, wrong answer, tool failure, timeout, etc.). Cite specific `task_index` values.