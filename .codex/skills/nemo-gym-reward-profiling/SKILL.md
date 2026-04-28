---
name: nemo-gym-reward-profiling
description: >-
  Use to help users get started with Nemo Gym reward profiling. Covers the basic
  ng_run, ng_collect_rollouts, and ng_reward_profile workflow, repeated rollouts,
  materialized inputs, rollout JSONL artifacts, task and rollout identity, output
  inspection, and simple scaling knobs. For failed jobs, prefer nemo-gym-debugging.
---

# Nemo Gym Reward Profiling

## Invocation Check

Use this skill when the user wants to run, understand, or lightly modify Nemo Gym reward profiling. Keep the first answer oriented around the normal workflow:

`ng_run` starts model/resource servers, `ng_collect_rollouts` writes rollout artifacts and reward profiling by default, and `ng_reward_profile` can rerun profiling from completed artifacts.

If the user is primarily debugging a failed job or stack trace, use `$nemo-gym-debugging` first.

## Basic Workflow

1. Identify the environment config paths and input JSONL.
2. Start Gym servers with `ng_run`.
3. Collect rollouts with `ng_collect_rollouts`; by default this writes `*_reward_profiling.jsonl` during collection and rewrites the final file through the canonical profiler.
4. Optionally run `ng_reward_profile` on the materialized inputs and rollout JSONL to regenerate or validate the profile.
5. Inspect line counts and profile rows before scaling up.

Repeated rollouts are the main profiling lever. `num_repeats=1` is valid, but per-task averages, variance, and pass-rate-style interpretation are only meaningful with multiple rollouts per task.

## Core Concepts

- `*_materialized_inputs.jsonl`: expanded collection inputs after repeat expansion, agent defaults, and task/rollout id assignment.
- `rollouts.jsonl`: one completed rollout/result per materialized input row.
- `*_reward_profiling.jsonl`: one summarized profile row per original task.
- `_ng_task_index`: original task/sample id.
- `_ng_rollout_index`: repeated rollout id for that task.
- `rollout_infos`: compact per-rollout info inside each task profile row, including reward and token usage when available.

Keep reward-to-length or reward-to-token analysis keyed by both `_ng_task_index` and `_ng_rollout_index`.

## Reference Loading

Load references only when the user needs that detail:

- Read `references/quick-start.md` for a generic command template and the minimal run sequence.
- Read `references/output-format.md` to explain materialized inputs, rollout JSONL, reward profile rows, `rollout_infos`, and pass-rate recovery.
- Read `references/scaling-and-validation.md` for `num_repeats`, `num_samples_in_parallel`, cache/resume checks, and smoke-test validation.

## Practical Defaults

- Treat inflight reward profiling as the default collection behavior in current Gym.
- Use `+inflight_reward_profile=False` only when a rollout-only collection is needed.
- Use post-hoc `ng_reward_profile` to regenerate or validate profile output from completed artifacts.
- Trust the target checkout's CLI help and `nemo_gym/reward_profile.py` over memory if flags differ.
- For code changes, validate with reward-profile and rollout-collection unit tests when available.
