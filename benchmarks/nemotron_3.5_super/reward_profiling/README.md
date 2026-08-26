# Nemotron 3.5 Super reward profiling

Reward-profiles the RL training blend by running the policy over every training dataset and
summarizing per-task reward with `gym eval profile`.

`manifests/no_judge_no_sandbox.yaml` covers the **26 no-judge, no-sandbox environments** (623,006 rows). The 10
environments that need an LLM judge or a sandbox are intentionally excluded; they are tracked
separately because they need extra infrastructure.

The sweep machinery itself is generic and lives in `nemo_gym/sweep/`. Only these manifests are
Nemotron-specific.

```
manifests/         input: what to profile. Hand-edited.
outputs/         everything a run produces: sweeps/<nickname>/ and slurm-logs/. Gitignored.
scripts/           launchers.
```

## Why one job instead of one job per environment

Rollout collection dispatches each row to the agent named in its own `agent_ref`
(`nemo_gym/rollout_collection.py`). Training rows already carry that field, so datasets for
different environments can be concatenated into a single input file and served by one Gym
deployment composed from the union of their configs.

## Flow

```bash
R=benchmarks/nemotron_3.5_super/reward_profiling

# 1. validate + expand repeats in parallel. Once per (manifest, checkpoint).
MANIFEST=$R/manifests/no_judge_no_sandbox.yaml \
bash $R/scripts/prepare_sweep.sh

# 2. one job: vLLM endpoint + Gym sweep driver + reward profile
MODEL=<checkpoint-path> \
VLLM_CONFIG=benchmarks/nemotron_3.5_super/vllm_configs/nemotron_3.5_super.sh \
SWEEP_DIR=$R/outputs/sweeps/<nickname> \
NUM_PREFILL_NODES=1 NUM_DECODE_NODES=2 \
SBATCH_ACCOUNT=<account> SBATCH_PARTITION=batch SBATCH_GRES=gpu:4 \
CONTAINER=<reward-profiling sqsh> \
MOUNTS=/lustre:/lustre \
bash $R/scripts/sbatch_reward_profiling.sh
```

`run_rollouts.sh` does the same collection against an **already-running** vLLM job
(`VLLM_JOBID=<jobid>`), which is the faster loop when iterating against a warm endpoint.

`prepare_sweep.sh` validates the manifest, then materializes it. Validation fails loudly if a
dataset's `agent_ref` disagrees with the agent its paired config declares, which is what stops a
mispairing from silently scoring rollouts with the wrong verifier.

`run_rollouts.sh` starts every Gym server for the sweep, collects rollouts resuming from the
materialized inputs, and runs `gym eval profile` at the end. It runs the driver on `nodes[1]` of
the vLLM allocation, off the node serving prefill and the router.

Artifacts land in `outputs/sweeps/<nickname>/`:

| file | |
|---|---|
| `rollouts_materialized_inputs.jsonl` | expanded inputs; the name Gym derives for `--resume` |
| `rollouts.jsonl` | completed rollouts |
| `rollouts_failures.jsonl` | failure sidecar |
| `rollouts_reward_profiling.jsonl` | per-task reward profile |
| `sweep_report.json` | observed per-entry row counts |

## Notes

- `--no-serve` is required. Without it `--input` is silently replaced by the collated split.
- `num_repeats` resolves per agent, so entries sharing an agent share a repeat count. `validate`
  reports which entries those are. Set a per-entry `num_repeats` to override the global default.
- Set `num_samples_in_parallel` explicitly; unset means unbounded concurrency.
- Pass `--resume` and a stable output path. Rollout collection clears the output file otherwise,
  and a sweep of this size will not finish inside one `batch` allocation.
- `agent_ref_override` rewrites `agent_ref` while concatenating. Use it only to deliberately run a
  dataset through a different agent; the override is recorded in `build_report.json`.
