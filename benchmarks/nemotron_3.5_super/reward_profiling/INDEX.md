# Reward Profiling How-To

## Index
1. [(Optional) Create a container](#01---optional-create-a-container)
2. [Create a manifest](#02---create-a-manifest)
3. [Run the reward profiling job](#03---run-the-reward-profiling-job)
   - a. [Sharding / Unsharding data](#03a---sharding--unsharding-data)
   - b. [Starting, resuming and monitoring a profiling job](#03b---starting-resuming-and-monitoring-a-profiling-job)
4. [Postprocess reward profiling outputs](#04---postprocess-reward-profiling-outputs)
   - a. [Collating finished / unfinished data](#04a---collating-finished--unfinished-data)
   - b. [Running ng_reward_profile](#04b---running-ng_reward_profile)
   - c. [Re-creating profiled data to input shapes with reward profiled information](#04c---re-creating-profiled-data-to-input-shapes-with-reward-profiled-information)


## 01 - (Optional) Create a container
The reward profiling container pre-installs all resources_servers and responses_api_agents needed for reward profiling.
It follows the same flow from the [Super-v3.5 readme](../README.md), with one change:
1. New container config: `benchmarks/nemotron_3.5_super/reward_profiling/configs/container_config.yaml`

```bash
INPUT_CONTAINER=<vllm sqsh> \
OUTPUT_CONTAINER=<output sqsh> \
MOUNTS=/lustre:/lustre \
GYM_CONFIG=benchmarks/nemotron_3.5_super/reward_profiling/configs/container_config.yaml \
SKIP_PREPARE=1 \
sbatch benchmarks/nemotron_3.5_super/build_eval_container.sh
```

`container_config.yaml` is generated from the manifests, so it cannot drift from them:

```bash
python -m nemo_gym.sweep container-config manifests/*.yaml --out configs/container_config.yaml
```

It bakes one venv per server implementation the manifests reference. A server with no baked venv
installs at runtime and hangs the run behind connection retries instead of failing, so regenerate
this whenever you add an entry that pulls in a new server.

`SKIP_PREPARE=1` skips `gym eval prepare`, which downloads benchmark datasets. Reward profiling
supplies its own data via the manifest, so it is not needed.

You also need a **sandbox container** if any entry uses `ns_tools` or `math_formal_lean`. Only one
arm64 build exists: `/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-0.7.1-arm64.sqsh`

## 02 - Create a Manifest
The manifest is the highest-level config of what environments are being profiled, and parameterizes any judge, sandbox, or config overrides needed.
1. nickname: name of the reward profiling jobs
    - becomes the output directory: artifacts land in `<OUT_DIR>/<nickname>/`
2. defaults: set any reward profiling defaults.
    - can set: `num_repeats` (rollouts per task; the variance across these is the profile)
3. extra_configs: any other configs to be loaded
    - sweep-wide config paths, merged after every entry's own `configs`, so they win on conflict.
      Usually just the model server, e.g. `responses_api_models/vllm_model/configs/vllm_model.yaml`
4. config_overlay: Gym config written inline, spliced into the generated `sweep_config.yaml`
    - a config overrides anything its own `config_paths` pulled in, so these beat every file
    - this is how we route different judge models to one gym_env_start server
    - use it instead of editing upstream configs: the container is built from a Gym ref and does
      not contain this repo, so a repo-relative config path will not resolve inside it
5. entries: the environments to be profiled.
    - label (required): nickname of profiled env
    - agent (required): agent_ref of the data
    - configs (optional but effectively required): gym configs defining the agent and its
      resources server. The agent must be declared by at least one of them
    - data (required): jsonl path to the data with labelled agent_ref
    - owner (optional): owner of environment
    - num_repeats (optional): overrides the default. Resolved per agent, so entries sharing an
      agent share a value

Validate before running — it checks the configs exist, the agent is declared, and the data parses:

```bash
python -m nemo_gym.sweep validate manifests/<name>.yaml
```

See `manifests/example_basic.yaml`, `example_judge.yaml`, `example_sandbox_judge.yaml` for a
one-entry manifest of each shape.

## 03 - Run the reward profiling job

First expand the manifest into one input file. Every entry is concatenated, repeats are expanded,
and each row is stamped with a globally unique `_ng_task_index` plus its manifest label:

```bash
R=benchmarks/nemotron_3.5_super/reward_profiling
MANIFEST=$R/manifests/<name>.yaml bash $R/scripts/01_prepare_sweep.sh
# -> $R/outputs/sweeps/<nickname>/
```

Then run it. One job serves the policy itself:

```bash
MODEL=<ckpt> CONTAINER=<eval sqsh> SANDBOX_CONTAINER=<sandbox sqsh> \
  SWEEP_DIR=$R/outputs/sweeps/<nickname> bash $R/scripts/03_run.sh
```

Or against a policy already served somewhere — no Slurm, no GPUs:

```bash
SWEEP_DIR=<sweep> POLICY_BASE_URL=http://host:8000/v1 POLICY_MODEL_NAME=<model> \
  POLICY_API_KEY=<key> bash $R/scripts/03_run_endpoint.sh
```

### 03a - Sharding / Unsharding data

One job cannot go past ~16 nodes: `--segment` needs a topology-contiguous allocation and an NVL72
rack is 18 nodes. To go wider, split the input across N jobs.

```bash
SWEEP_DIR=<sweep> NUM_SHARDS=16 bash $R/scripts/02_shard.sh   # deal
SWEEP_DIR=<sweep> bash $R/scripts/04_merge_shards.sh          # unshard
```

Each shard directory is a complete SWEEP_DIR, so `03_run.sh` runs against one unmodified. Rows are
dealt round-robin, so every shard carries every environment and none inherits a whole slow one.

This works because `_ng_task_index` is stamped before sharding and Gym never rewrites it, so shard
rollouts concatenate without renumbering. Merge deduplicates on `(_ng_task_index,
_ng_rollout_index)` — the same key Gym resumes on — so a rerun shard cannot double-count.

Resharding to a different N is safe and lossless. `02_shard.sh` folds any collected rollouts back
into the parent and snapshots it to `snapshots/<UTC>/` before touching a shard directory, then
carries that work into the new layout.

### 03b - Starting, resuming and monitoring a profiling job

`03_run_sharded.sh` submits one job per shard and watches them. A shard whose job dies with work
outstanding is resubmitted, up to `MAX_ROUNDS` (4). It merges and profiles when all are done:

```bash
MODEL=<ckpt> CONTAINER=<sqsh> SANDBOX_CONTAINER=<sandbox sqsh> \
  SWEEP_DIR=<sweep> NUM_SHARDS=16 bash $R/scripts/03_run_sharded.sh
```

Resume is automatic and requires nothing: `--resume` reads the rollouts already written and
collects only what is missing. A killed job restarted against the same SWEEP_DIR reports

```
Resumed from cache. Found:
- 254 rows already done (in main jsonl)
- 322 rows that still need to be run
```

Monitor with `squeue`, the rollout count, and the per-shard logs:

```bash
wc -l <sweep>/shards/shard_*/rollouts.jsonl        # progress
tail -f <sweep>/slurm-logs/<jobid>-gym-*.log       # job
less <sweep>/env_start.log                          # the 63 Gym servers
```

Expect ~30s to a healthy sandbox, ~4 min to all servers ready, and ~16 min to the first rollout
(vLLM weight loading dominates). Runs reach ~95% quickly then stall on a few slow environments
(`lean`, `math_cot`); profile the partial result rather than waiting for the tail.

## 04 - Postprocess reward profiling outputs

### 04a - Collating finished / unfinished data

If sharded, merge first — this is also what makes a reshard safe:

```bash
SWEEP_DIR=<sweep> bash $R/scripts/04_merge_shards.sh
```

Then split the one concatenated file back into one directory per manifest entry:

```bash
python -m nemo_gym.sweep split <sweep>     # -> <sweep>/by_label/<label>/
```

This keys on `_ng_task_index`, not `agent_ref`, because entries share agents — `math_tir`,
`stem_mcqa_tools_ultra_0` and `stem_openqa_tools_ultra_0` all dispatch to `ns_tools_simple_agent`,
and splitting on the agent would merge three environments into one. `split_report.json` lists
per-label counts and names any label that collected nothing.

Unfinished data needs no special handling: partial groups are kept and reported.

### 04b - Running ng_reward_profile

`ng_reward_profile` is the legacy alias for `gym eval profile`. Either runs on a partial file:

```bash
gym eval profile --inputs <dir>/rollouts_materialized_inputs.jsonl \
                 --rollouts <dir>/rollouts.jsonl ++allow_partial_rollouts=True
```

```
Reward profile completion: 2151/2304 rollout rows (93.36%)
Input rows: 288 total; 232 complete; 56 partial; 0 without rollouts dropped from output.
```

`05_profile.sh` does the split and then profiles every entry plus the whole sweep:

```bash
SWEEP_DIR=<sweep> CONTAINER=<sqsh> VLLM_JOBID=<any live job> bash $R/scripts/05_profile.sh
```

The profiler needs no GPU; `VLLM_JOBID`/`CONTAINER` only get it a container with `gym` on PATH.

### 04c - Re-creating profiled data to input shapes with reward profiled information

`rollouts_reward_profiling.jsonl` is already one row per task — the same shape as the source data,
not the expanded input:

```
source tasks          288      <- what you started with
materialized inputs 2,304      tasks x num_repeats
rollouts            2,151      collected
profiling rows        288      <- one per task
```

Each row carries the reward distribution plus the original input under `sample`:

```json
{ "_ng_task_index": 0,
  "mean/reward": 0.25, "std/reward": 0.46, "min/reward": 0.0, "max/reward": 1.0,
  "mean/input_tokens": 5412, "mean/output_tokens": 373,
  "num_rollouts": 8, "expected_num_rollouts": 8, "reward_profile_completion_pct": 100.0,
  "rollout_infos": [ { "rollout_id": "0:0", "reward": 1.0, "input_tokens": 5412, ... } ],
  "sample": { the original input row, unchanged } }
```

To rebuild a dataset with the new pass rates, take `sample` and write `mean/reward` over the
existing `pass_rate` field. Datasets that carry one (e.g. `tau_pivot`) hold the *previous*
checkpoint's rate, so `sample.pass_rate` vs `mean/reward` in the same row is the before/after
comparison — no joining required.

There is no script for this rewrite yet; it is per-dataset, since not every dataset carries
`pass_rate` and the field name varies.

`rollouts_agent_metrics.json` holds the same aggregated per agent.
