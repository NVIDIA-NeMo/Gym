# Reward profiling — index

Run the policy over every training dataset, measure per-task reward across repeats.
36 environments, one Gym deployment, one input file.

## Layout

```
manifests/      what to profile. nemotron_3_ultra.yaml is the real sweep; example_*.yaml are
                one-entry samples (basic / judge / sandbox+judge).
configs/        container_config.yaml — union of every config, for baking venvs.
scripts/        numbered by run order.
outputs/        everything a run produces. Gitignored.
RATES.md        measured throughput and GPU-hour sizing.
SCALE_NOTES.md  full-scale prepare timings.
```

## Single job

```bash
R=benchmarks/nemotron_3.5_super/reward_profiling
MANIFEST=$R/manifests/nemotron_3_ultra.yaml bash $R/scripts/01_prepare_sweep.sh
MODEL=<ckpt> CONTAINER=<sqsh> SANDBOX_CONTAINER=<sandbox sqsh> \
  SWEEP_DIR=$R/outputs/sweeps/nemotron_3_ultra bash $R/scripts/03_run.sh
```

## Many jobs (past one job's node limit)

```bash
SWEEP_DIR=<sweep> NUM_SHARDS=16 bash $R/scripts/02_shard.sh
MODEL=... CONTAINER=... SWEEP_DIR=<sweep> NUM_SHARDS=16 bash $R/scripts/03_run_sharded.sh
```
`03_run_sharded.sh` submits one job per shard, resubmits any that die, then merges and profiles.

## Against an endpoint you already have

```bash
SWEEP_DIR=<sweep> POLICY_BASE_URL=http://host:8000/v1 POLICY_MODEL_NAME=<model> \
  POLICY_API_KEY=<key> bash $R/scripts/03_run_endpoint.sh   # no Slurm, no GPUs
```

## Scripts

| | |
|---|---|
| `01_prepare_sweep.sh` | manifest → materialized input |
| `02_shard.sh` | deal into N sweep dirs (optional) |
| `03_run.sh` | one job: serves the policy itself |
| `03_run_sharded.sh` | N jobs + watch + resubmit + merge + profile |
| `03_run_endpoint.sh` | you supply a URL; no Slurm |
| `03_run_attached.sh` | srun into a running vLLM job |
| `04_merge_shards.sh` | unshard |
| `05_profile.sh` | split by entry, profile each |

## Outputs

```
outputs/sweeps/<nickname>/
  rollouts_materialized_inputs.jsonl   inputs, tasks x num_repeats
  rollouts.jsonl                       collected rollouts (merged, if sharded)
  rollouts_reward_profiling.jsonl      one row per task: mean/std/min/max reward, rollout_infos
  rollouts_agent_metrics.json          per-agent aggregates
  by_label/<entry>/                    the above, split per manifest entry
  shards/shard_NNN/                    per-shard working dirs
  snapshots/<UTC>/                     parent state before each reshard
  slurm-logs/                          job + sandbox logs
```

## Gotchas

- **Sandbox** — `ns_tools` and `math_formal_lean` need `SANDBOX_CONTAINER`. Only one arm64 image
  exists: `/lustre/fsw/.../igitman/images/nemo-skills-sandbox-0.7.1-arm64.sqsh`.
- **Judges** — need `env.yaml` for `${nv_inference_api_key}`. Bind them in the manifest's
  `config_overlay`, not by editing upstream configs.
- **`--resume` needs pre-expanded inputs.** `NO_EXPAND=1` is 8x smaller and faster but cannot
  resume; `03_run.sh` rejects such a sweep at submit.
- **Merge before resharding.** `02_shard.sh` carries work from the parent, and snapshots it first.
- **The tail dominates.** Runs reach ~97% quickly then stall on a few environments (`lean`,
  `math_cot`). Profile partial results rather than waiting; `allow_partial_rollouts` handles it.
- **Some upstream configs are wrong for this sweep** — `ns_tools` registers one verifier,
  `abstention` caps its judge at 64 tokens. Fixed via `config_overlay`; see `Gym-ultra-3-rebased`.
