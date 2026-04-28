---
name: nemo-gym-reward-profiling
description: >-
  Use when designing, running, or modifying Nemo Gym reward profiling workflows. Covers
  ng_collect_rollouts, ng_reward_profile, repeated rollouts, materialized inputs, task/rollout
  identity, rollout JSONL artifacts, scaling collection, inflight reward profiling,
  raw reward-to-token/length mappings, and optional Nemo-RL ray.sub orchestration. For failures,
  prefer nemo-gym-debugging first.
---

# Nemo Gym Reward Profiling

## Invocation Check

Use this skill for reward profiling workflows and code paths around Nemo Gym. Start from the stable Gym flow:

`ng_run` starts the model/resource servers, `ng_collect_rollouts` writes rollout JSONL plus materialized inputs, and `ng_reward_profile` profiles those rollouts. Repeated rollouts make per-task variability meaningful, but `num_repeats=1` is valid.

Some checkouts also support inflight reward profiling through `ng_collect_rollouts +inflight_reward_profile=True`. When present, collection writes profile rows while the run is active, then rewrites the final output through the same canonical `RewardProfiler` path used by `ng_reward_profile`.

First classify the target:

- **Current Gym profiling path**: prefer `nemo_gym/rollout_collection.py`, `nemo_gym/reward_profile.py`, `ng_collect_rollouts`, and `ng_reward_profile`.
- **Checkout-specific launcher or extension**: inspect the target branch before relying on local scripts, inflight flags, or launcher-specific environment variables.

If the user is only asking why a job failed, use `$nemo-gym-debugging` instead or alongside this skill.

## Core Workflow

1. Inspect the target repo before changing behavior:
   - `nemo_gym/rollout_collection.py`
   - `nemo_gym/reward_profile.py`
   - `tests/unit_tests/test_reward_profile.py`
   - relevant CLI docs or entrypoints
2. Validate the command shape in that checkout. Some docs/branches expose different `ng_reward_profile` flags; trust the code and CLI help over memory.
3. Treat `ng_collect_rollouts` as the collection lifecycle:
   - source data JSONL
   - materialized inputs JSONL
   - rollout output JSONL
4. Treat `ng_reward_profile` as the reusable profiling path:
   - common input contract: materialized inputs plus rollout results
   - common output contract: task-level aggregate rows, rollout-level compact info, and agent/global metrics
   - current task-level rows preserve `_ng_task_index`, `num_rollouts`, and `rollout_infos`
5. Preserve identity through profiling outputs and extensions:
   - task index identifies the original sample/task
   - rollout index identifies repeated samples for that task
   - never compute reward/length/token mappings without a task and rollout join key
6. If adding checkout-specific fields, clearly mark them as extensions:
   - raw reward/length/token mappings
   - profiling rows written during collection
   - launcher conventions

## Reference Loading

Load references only when needed:

- Read `references/collection-lifecycle.md` for the concrete `ng_run -> ng_collect_rollouts -> ng_reward_profile` flow, scaling rollout collection, cache/resume behavior, materialized inputs, and rollout identity.
- Read `references/reward-profile-api.md` when changing profiling metrics, output files, `RewardProfiler`, aggregate metrics, or raw reward/length/token extensions.
- Read `references/inflight-reward-profiling.md` when using `+inflight_reward_profile=True`, validating that collection-written profiles match `ng_reward_profile`, or running the current StructEval smoke wrapper.
- Read `references/nemo-rl-ray-sub.md` if the user wants to do reward profiling using Nemo-RL to orchestrate the Slurm/Ray allocation, start vLLM on Ray, or add optional sandbox sidecars.

## Practical Defaults

- Prefer upstream `ng_reward_profile` for reusable profiling behavior.
- Prefer `RewardProfiler.profile_from_data(...)` as the single canonical code path when adding new profile outputs.
- Keep launcher-specific behavior out of shared claims unless the user explicitly asks about that launcher.
- Validate changes with the repo's reward-profile and rollout-collection unit tests when available.
- For new profiling fields, add focused tests that cover repeated rollouts, missing usage fields, and stable task/rollout identity.
