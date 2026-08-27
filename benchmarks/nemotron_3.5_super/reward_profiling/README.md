# Nemotron 3.5 Super reward profiling

Reward-profiles the RL training blend by running the policy over every training dataset and
summarizing per-task reward with `gym eval profile`.

`manifests/nemotron_3_ultra.yaml` is the whole sweep: **36 environments, 726,121 rows**, run as one Gym
deployment over one concatenated input. Rows carry their own `agent_ref` and rollout collection
dispatches per row, so judge-scored, sandbox-backed and plain environments coexist in one job.

Two of those groups need infrastructure the third does not. The judge entries interpolate
`${nv_inference_api_key}` from `env.yaml`, and the sandbox entries (`ns_tools`,
`math_formal_lean`) need `SANDBOX_CONTAINER` set -- without it they fail per rollout rather than
at startup, so watch the launcher's `/health` gate rather than the rollout count.

The sweep machinery itself is generic and lives in `nemo_gym/sweep/`. Only these manifests are
Nemotron-specific.

```
RATES.md         measured throughput and GPU-hour sizing. Read before allocating.
manifests/       input: what to profile. Hand-edited.
outputs/         everything a run produces: sweeps/<nickname>/ and slurm-logs/. Gitignored.
scripts/         numbered by run order; see below.
```

## Scripts, in order

| | | |
|---|---|---|
| `01_prepare_sweep.sh` | manifest -> one materialized input | always |
| `02_shard.sh` | deal into N sweep dirs, one per job | only to exceed one job's node count |
| `03_run.sh` | one job: vLLM + Gym + collect + profile | single-job runs, and what the sharded runner submits |
| `03_run_sharded.sh` | submit + watch + resubmit N shards, then merge and profile | sharded runs |
| `03_run_attached.sh` | collect against an already-running vLLM job | debugging against a warm endpoint |
| `04_merge_shards.sh` | unshard: merge rollouts back into the parent | after individually-launched shards, or before resharding |
| `05_profile.sh` | split by entry, profile each and the whole sweep | mid-run, or after a merge |

Single job: `01` then `03`. Sharded: `01`, `02`, `03_run_sharded` (which does `04` and `05` itself).
`04` and `05` are separate because both are useful mid-run -- the profiler handles partial
sweeps, so you can see per-entry rewards before a run finishes.

## Why one job instead of one job per environment

Rollout collection dispatches each row to the agent named in its own `agent_ref`
(`nemo_gym/rollout_collection.py`). Training rows already carry that field, so datasets for
different environments can be concatenated into a single input file and served by one Gym
deployment composed from the union of their configs.

## Flow

```bash
R=benchmarks/nemotron_3.5_super/reward_profiling

# 1. validate + expand repeats in parallel. Once per (manifest, checkpoint).
MANIFEST=$R/manifests/nemotron_3_ultra.yaml \
bash $R/scripts/01_prepare_sweep.sh

# 2. one job: vLLM endpoint + Gym sweep driver + reward profile
MODEL=<checkpoint-path> \
VLLM_CONFIG=benchmarks/nemotron_3.5_super/vllm_configs/nemotron_3.5_super.sh \
SWEEP_DIR=$R/outputs/sweeps/nemotron_3_ultra \
NUM_PREFILL_NODES=1 NUM_DECODE_NODES=2 \
CONTAINER=<reward-profiling sqsh> \
SANDBOX_CONTAINER=<nemo-skills sandbox sqsh> \
bash $R/scripts/03_run.sh
```

`SBATCH_ACCOUNT` defaults to `nemotron_n4_post` and `SBATCH_GRES` to `gpu:4`; both override.
`SANDBOX_CONTAINER` is required for the `ns_tools` and `math_formal_lean` entries -- only an arm64
build works on NVL72, and there is one:
`/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-0.7.1-arm64.sqsh`.

`03_run_attached.sh` does the same collection against an **already-running** vLLM job
(`VLLM_JOBID=<jobid>`), which is the faster loop when iterating against a warm endpoint.

`01_prepare_sweep.sh` validates the manifest, then materializes it. Validation fails loudly if a
dataset's `agent_ref` disagrees with the agent its paired config declares, which is what stops a
mispairing from silently scoring rollouts with the wrong verifier.

`03_run_attached.sh` starts every Gym server for the sweep, collects rollouts resuming from the
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
