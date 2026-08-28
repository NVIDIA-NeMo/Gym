# Nemotron 3.5 Super — Reward Profiling

Runs the policy over every dataset in the RL training blend and summarizes per-task reward, so you
can see which environments the checkpoint has saturated and which still carry signal.

`manifests/nemotron_3_ultra.yaml` is the full sweep: **36 environments, 726,121 source rows**, run
as **one** Gym deployment over one concatenated input. Each row carries its own `agent_ref` and
rollout collection dispatches per row, so judge-scored, sandbox-backed and plain environments all
coexist in a single job. The machinery is generic and lives in `nemo_gym/sweep/`; only the
manifests here are Nemotron-specific.

To contribute an environment, see [CONTRIBUTING.md](./CONTRIBUTING.md).

## Index
0. [Setup](#00---setup)
1. [(Optional) Create a container](#01---optional-create-a-container)
2. [Create a manifest](#02---create-a-manifest)
3. [Run the reward profiling job](#03---run-the-reward-profiling-job)
   - a. [Starting, resuming and monitoring a profiling job](#03a---starting-resuming-and-monitoring-a-profiling-job)
4. [Postprocess reward profiling outputs](#04---postprocess-reward-profiling-outputs)
   - a. [Collating finished / unfinished data](#04a---collating-finished--unfinished-data)
   - b. [Running gym eval profile](#04b---running-gym-eval-profile)
   - c. [Re-creating profiled data to input shapes with reward profiled information](#04c---re-creating-profiled-data-to-input-shapes-with-reward-profiled-information)
5. [Reference](#05---reference)
6. Appendix — [sharding by hand](#a1---sharding-and-unsharding-by-hand), [building the container](#a2---building-the-reward-profiling-container)

## Quick start

Three commands, from the repo root. Everything else on this page is a variation on them.

```bash
R=benchmarks/nemotron_3.5_super/reward_profiling
uv venv && uv sync --extra dev && source .venv/bin/activate

# 1. manifest -> one materialized input file
MANIFEST=$R/manifests/nemotron_3_ultra.yaml bash $R/scripts/01_materialize.sh

# 2. shard across N jobs, submit them, resubmit any that die, merge and split when done.
#    Model, containers, account, qos and timelimit all come from the manifest, so there is
#    nothing else to pass. Runs for hours in the foreground -- detach it, and start only one.
setsid nohup bash -lc "SWEEP_DIR=$R/outputs/sweeps/nemotron_3_ultra NUM_SHARDS=16 \
  bash $R/scripts/03_run_sharded.sh" > watcher.log 2>&1 &

# 3. profile the whole sweep (safe on a partial run)
SWEEP_DIR=$R/outputs/sweeps/nemotron_3_ultra bash $R/scripts/05_profile.sh
```

`NUM_SHARDS` is jobs, not nodes: each is `NUM_PREFILL_NODES + NUM_DECODE_NODES` (1 + 2 by
default), so 16 shards is 48 nodes. Size it from [RATES.md](RATES.md).

For a single job instead of N, swap step 2 for `03_run_single.sh` — same arguments, no sharding.
`MODEL=<ckpt>` overrides the manifest's checkpoint; so does `CONTAINER`, `SANDBOX_CONTAINER`, or
any `SBATCH_*`.

Add `LIMIT_PER_ENTRY=8` to step 1 for a smoke run that exercises every code path in minutes —
N rows from *each* entry, which is not what `gym eval run --limit` does ([why](#gotchas)).

## 00 - Setup

Everything outside the profiling job — validate, prepare, shard, merge, split, profile — is
CPU-only and runs from a venv. Once, in the repo root:

```bash
uv venv && uv sync --extra dev
source .venv/bin/activate
```

A venv older than the branch's `openai==2.44.0` pin imports `nemo_gym` fine and then fails inside
`gym eval profile`, so a stale one is worse than none.

You also need an `env.yaml` in the repo root — it is gitignored, so it does not come with a
checkout. Judge entries resolve their key through it, and the launchers refuse to submit if any
variable it names is unexported:

```yaml
nv_inference_api_key: ${oc.env:NVI_KEY_EVALUATOR}
```

`export NVI_KEY_EVALUATOR=...` must be a real export — a bare assignment in `.bashrc` is a shell
variable that sbatch never sees, and `export` there only reaches a login shell.

## 01 - (Optional) Create a container

The manifest already names a container that exists, so a normal run needs nothing here. Build a new
one only when an entry pulls in a server the image does not have — see
[A2](#a2---building-the-reward-profiling-container), and
[CONTRIBUTING](CONTRIBUTING.md#01---creating-a-container) for how to tell.

## 02 - Create a Manifest
The manifest is the highest-level config of what environments are being profiled, and parameterizes any judge, sandbox, or config overrides needed.
Seven top-level keys. Every block but `nickname` and `entries` is named for the command it
configures, so where a setting goes follows from which command consumes it:

1. **nickname** — names the run; artifacts land in `<OUT_DIR>/<nickname>/`
2. **materialize** — which rows are in the sweep at all, read by `01_materialize.sh`
    - `limit_per_entry`: rows to take from every entry. Unset means all
    - `sample`: `head` (default) or `random`. `head` stops reading at the limit, so it is what a
      smoke run wants; `random` must read the whole file, but the first N rows of a sorted dataset
      are a biased subset, so prefer it when the limit is a deliberate subset
    - `seed`: seeds `random`, mixed with each entry's label so adding an entry does not reshuffle
      its neighbours and invalidate their resume keys
3. **sbatch** — Slurm settings for the launchers, which shell out to `sbatch`
    - free-form: each key becomes `SBATCH_<KEY>`, which is how sbatch takes them.
      `account`, `partition`, `qos`, `gres`, `timelimit`, `reservation`, `comment`, and anything
      else listed under INPUT ENVIRONMENT VARIABLES in `man sbatch`
    - applied only where the launcher's own environment does not already set the variable
4. **srun** — containers, for `srun --container-image`
    - `container`, `sandbox_container`, `mounts`. The container belongs here because it is built
      *from* this manifest, so the two stay together. `MODEL` stays a launcher argument: the
      checkpoint is what you are profiling, not part of what the sweep is
5. **gym_env_start** — becomes `sweep_config.yaml`, passed as `--config`
    - `config_paths`: sweep-wide configs merged ahead of every entry's own. Usually just the model
      server, e.g. `responses_api_models/vllm_model/configs/vllm_model.yaml`
    - any other key: ordinary Gym config, spliced in verbatim. A config overrides whatever its own
      `config_paths` pulled in, so these beat every file — which is how a judge gets rebound
      without editing an upstream config. Do it here rather than in a file: the container is built
      from a Gym ref and has no copy of this repo, so a repo-relative path will not resolve inside it
6. **gym_eval_run** — runtime settings, emitted as `++key=value`
    - `num_repeats`: rollouts per task; the spread across them is the profile. Applied by
      01_materialize.sh, which writes this many copies of each row into the materialized inputs,
      so collection itself runs with `++num_repeats=1`
    - anything else Gym takes, e.g. `num_samples_in_parallel`
    - precedence, lowest to highest: **manifest → script env var → command line**. A launcher
      passes `++` only when its env var is set, so these are defaults rather than something
      silently clobbered. `num_samples_in_parallel` is the exception: the launcher computes
      `512 × decode_nodes`, since only it knows the job's shape
7. **entries** — the environments to be profiled
    - `label` (required): nickname of profiled env
    - `agent` (required): `agent_ref` of the data
    - `configs`: gym configs defining the agent and its resources server. The agent must be
      declared by at least one of them
    - `data` (required): jsonl path to the data with labelled `agent_ref`
    - `owner` (optional): owner of environment
    - `num_repeats` (optional): overrides the default. Resolved per agent, so entries sharing an
      agent share a value
    - `limit` (optional): rows to take from this dataset, overriding `materialize.limit_per_entry`.
      Committed, unlike `LIMIT_PER_ENTRY`, which caps a single smoke run and never raises this

```yaml
nickname: my_sweep

gym_env_start:
  config_paths:
    - responses_api_models/vllm_model/configs/vllm_model.yaml
  math_judge_model:              # bind a judge without touching upstream configs
    responses_api_models: {...}

materialize:
  limit_per_entry: 10000         # optional; unset means every row
  sample: random                 # head (default) | random
  seed: 0

sbatch:                          # optional; each key becomes SBATCH_<KEY>
  account: nemotron_n3_post
  qos: interactive
  timelimit: "04:00:00"

srun:                            # optional
  container: /lustre/.../eval.sqsh
  sandbox_container: /lustre/.../nemo-skills-sandbox-0.7.1-arm64.sqsh

gym_eval_run:
  num_repeats: 8
  num_samples_in_parallel: 512

entries:
  - label: my_env
    agent: my_simple_agent
    configs: [resources_servers/my_env/configs/my_env.yaml]
    data: /path/to/data.jsonl
    limit: 500                   # optional; overrides materialize.limit_per_entry
```

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
MANIFEST=$R/manifests/<name>.yaml bash $R/scripts/01_materialize.sh
# -> $R/outputs/sweeps/<nickname>/
```

Then run it. `03_run_sharded.sh` is the normal path: it shards, submits one job per shard,
resubmits any that die, and merges when they finish. `NUM_SHARDS=1` is legitimate — it is how a
single-job run gets resubmission.

```bash
SWEEP_DIR=$R/outputs/sweeps/<nickname> NUM_SHARDS=16 bash $R/scripts/03_run_sharded.sh
```

Two variants, both taking the same manifest-supplied settings:

```bash
# one job, no watcher, no resubmission
SWEEP_DIR=$R/outputs/sweeps/<nickname> bash $R/scripts/03_run_single.sh

# against a policy already served somewhere -- no Slurm, no GPUs
SWEEP_DIR=<sweep> POLICY_BASE_URL=http://host:8000/v1 POLICY_MODEL_NAME=<model> \
  POLICY_API_KEY=<key> bash $R/scripts/03_run_endpoint.sh
```

### 03a - Starting, resuming and monitoring a profiling job

`03_run_sharded.sh` submits one job per shard and watches them. A shard whose job dies with work
outstanding is resubmitted, up to `MAX_ROUNDS` (4). It merges and splits when all are done;
each shard's own job profiles itself, so run `05_profile.sh` after for the whole sweep:

```bash
MODEL=<ckpt> SWEEP_DIR=<sweep> NUM_SHARDS=16 bash $R/scripts/03_run_sharded.sh
```

It runs in the foreground for hours, so detach it — and start exactly one, since each watcher
resubmits independently and several will pile up jobs against the node limit:

```bash
setsid nohup bash -lc "... bash $R/scripts/03_run_sharded.sh" > watcher.log 2>&1 &
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
wc -l <sweep>/shards/shard_*/rollouts.jsonl               # progress
tail -f <sweep>/shards/shard_000/slurm-logs/*-gym-*.log   # one shard's job
less <sweep>/shards/shard_000/env_start.log               # its 63 Gym servers
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
per-label counts and names any label that collected nothing — a label at zero means that lane
failed rather than scored badly, and that silence is usually the finding.

Unfinished data needs no special handling: partial groups are kept and reported.

### 04b - Running gym eval profile

It runs on a partial file:

```bash
gym eval profile --inputs <dir>/rollouts_materialized_inputs.jsonl \
                 --rollouts <dir>/rollouts.jsonl ++allow_partial_rollouts=True
```

```
Reward profile completion: 2243/2304 rollout rows (97.35%)
Input rows: 288 total; 263 complete; 25 partial; 0 without rollouts dropped from output.
```

`05_profile.sh` splits, then profiles every entry plus the whole sweep. No GPU, no Slurm:

```bash
source .venv/bin/activate
SWEEP_DIR=<sweep> bash $R/scripts/05_profile.sh
```

`VLLM_JOBID`/`CONTAINER` are an alternative, not a requirement — they borrow a container from any
live allocation to get `gym` on PATH, which is how `03_run_single.sh` calls this at the end of its job:

```bash
SWEEP_DIR=<sweep> CONTAINER=<sqsh> VLLM_JOBID=<any live job> bash $R/scripts/05_profile.sh
```

Labels run concurrently — each is a separate `gym` process and on Lustre interpreter start
dominates (~45s vs ~5s in the container). 36 labels take ~4m45s at `PROFILE_JOBS=12` (default 8).

It fails fast on the two things that otherwise error identically in all 36 `profile.txt` files: a
venv older than the `openai==2.44.0` pin, and an unexported `${oc.env:VAR}` from `env.yaml`.

### 04c - Re-creating profiled data to input shapes with reward profiled information

`rollouts_reward_profiling.jsonl` is already one row per task — the same shape as the source data,
not the expanded input:

```
source tasks          288      <- what you started with
materialized inputs 2,304      tasks x num_repeats
rollouts            2,243      collected
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

## 05 - Reference

### Layout

```
RATES.md         per-environment rates, GPU-hour and disk sizing. Read before allocating.
CONTRIBUTING.md  how to add an environment.
manifests/       input: what to profile. Hand-edited.
                 nemotron_3_ultra.yaml is the real sweep; example_*.yaml are minimal
                 one-entry manifests for basic / judge / sandbox+judge.
configs/         generated container config. Reproducible from the manifests.
outputs/         everything a run produces: sweeps/<nickname>/. Gitignored.
scripts/         numbered by run order; see below.
```

### Scripts, in order

| script | does | when |
|---|---|---|
| `01_materialize.sh` | manifest -> one materialized input | always |
| `02_shard.sh` | deal into N sweep dirs, one per job | only to exceed one job's node count |
| `03_run_single.sh` | one job: vLLM + Gym + collect + profile | single-job runs, and what the sharded runner submits |
| `03_run_sharded.sh` | submit + watch + resubmit N shards, then merge and split | sharded runs |
| `03_run_endpoint.sh` | collect against any OpenAI-compatible URL, no Slurm | an endpoint someone else is serving |
| `03_run_attached.sh` | collect inside an already-running vLLM Slurm job | debugging against a warm allocation |
| `04_merge_shards.sh` | unshard: merge rollouts back into the parent | after individually-launched shards, or before resharding |
| `05_profile.sh` | split by entry, profile each and the whole sweep | mid-run, or after a merge |
| `debug_selftest.sh` | assert the shard/reshard/merge/split/profile invariants | after changing the sweep machinery |

The `03_` variants differ on two axes, not one:

| | serves the policy | jobs |
|---|---|---|
| `03_run_single.sh` | yes, its own P/D stack | 1 |
| `03_run_sharded.sh` | yes, via `03_run_single.sh` | N |
| `03_run_endpoint.sh` | no, you give it a URL | 0 -- no Slurm at all |
| `03_run_attached.sh` | no, uses a running job's | 0 -- srun into yours |

Single job: `01` then `03_run_single`. Sharded: `01` then `03_run_sharded` (it shards and merges
itself; run `05` after).
Against an endpoint you already have: `01` then `03_run_endpoint`. `04` and `05` are separate
because both are useful mid-run — the profiler handles partial sweeps, so you can see per-entry
rewards before a run finishes.

### Common knobs

| variable | script | |
|---|---|---|
| `MANIFEST`, `OUT_DIR` | 01 | input manifest, where `<nickname>/` lands |
| `OVERWRITE=1` | 01 | replace an existing materialized file. Refused if rollouts were already collected |
| `LIMIT_PER_ENTRY` | 01 | take N source rows from *each* entry. Use `8` for a smoke run. Not the same as `gym eval run --limit` |
| `NUM_SHARDS` | 02, 03_run_sharded | how many jobs to split across |
| `MODEL`, `CONTAINER` | 03 | checkpoint path, eval sqsh |
| `SANDBOX_CONTAINER` | 03 | required by `ns_tools` and `math_formal_lean` |
| `NUM_PREFILL_NODES`, `NUM_DECODE_NODES` | 03 | P/D split; nodes = P + D |
| `SBATCH_ACCOUNT`, `SBATCH_GRES` | 03 | default `nemotron_n4_post`, `gpu:4` |
| `WALLTIME`, `MAX_ROUNDS` | 03 | per-job limit; resubmissions per shard (4) |
| `PROFILE_JOBS` | 05 | concurrent label profiles (8) |

### Artifacts, in `outputs/sweeps/<nickname>/`

| file | |
|---|---|
| `rollouts_materialized_inputs.jsonl` | expanded inputs; the name Gym derives for `--resume` |
| `rollouts.jsonl` | completed rollouts |
| `rollouts_failures.jsonl` | failure sidecar |
| `rollouts_reward_profiling.jsonl` | per-task reward profile — the output you want |
| `rollouts_agent_metrics.json` | the same, aggregated per agent |
| `sweep_report.json` | per-entry row counts and `task_index_range` |
| `by_label/<label>/` | the above, split per manifest entry |
| `shards/shard_NNN/` | each a complete SWEEP_DIR |
| `snapshots/<UTC>/` | parent state before a reshard |

### Gotchas

- **Run exactly one `03_run_sharded.sh` per sweep.** Each watcher resubmits dead shards
  independently, so a second one doubles the submissions and they pile up against the account's
  node limit. It now takes a lock on the shards directory and refuses to start twice, but it still
  runs in the foreground — detach it with `setsid nohup ... &`.
- **Keep `policy_model.num_workers` at 16.** Gym's FastAPI servers default to one worker
  (`server_utils.py:595`), and every rollout crosses the policy server, so at 512+ concurrency that
  worker caps throughput regardless of GPU capacity. `server_utils.py:159` divides the aiohttp
  connector limit by the worker count, so the two are tuned together.
- `--no-serve` is required for collection. Without it `--input` is silently replaced by the
  collated split.
- **Do not use `gym eval run --limit` to smoke-test a sweep.** It takes the first N rows of the
  whole file, and materialize lays entries out as contiguous blocks in manifest order, so
  `--limit 72` here covers 2 of 36 environments — all of them `tau_pivot`. `LIMIT_PER_ENTRY` at
  prepare takes N rows from *each* entry, so the same 72 tasks cover all 36. The failure is silent:
  you get rollouts, rewards and a clean profile for one environment and conclude the sweep works.
- Pass `--resume` and a stable output path. Rollout collection clears the output file otherwise,
  and a sweep of this size will not finish inside one `batch` allocation.
- `num_repeats` resolves per agent, so entries sharing an agent share a repeat count. `validate`
  reports which entries those are.
- Set `num_samples_in_parallel` explicitly; unset means unbounded concurrency.
- `export` in `.bashrc` only reaches a login shell, and a bare `VAR=value` reaches nothing. Judge
  keys read via `${oc.env:VAR}` need a real export; `03_run_single.sh` and `05_profile.sh` both check.
- `agent_ref_override` rewrites `agent_ref` while concatenating. Use it only to deliberately run a
  dataset through a different agent; the override is recorded in the build report.

## Appendix

### A1 - Sharding and unsharding by hand

One job cannot go past ~16 nodes: `--segment` needs a topology-contiguous allocation and an NVL72
rack is 18 nodes. To go wider, split the input across N jobs — which `03_run_sharded.sh` does for
you, so you rarely run these two directly:

```bash
SWEEP_DIR=<sweep> NUM_SHARDS=16 bash $R/scripts/02_shard.sh   # deal
SWEEP_DIR=<sweep> bash $R/scripts/04_merge_shards.sh          # unshard
```

Reach for them only to launch shards by hand, or to reshard between runs. Each shard directory is a
complete SWEEP_DIR, so `03_run_single.sh` runs against one unmodified. Rows are dealt round-robin,
so every shard carries every environment and none inherits a whole slow one.

This works because `_ng_task_index` is stamped before sharding and Gym never rewrites it, so shard
rollouts concatenate without renumbering. Merge deduplicates on `(_ng_task_index,
_ng_rollout_index)` — the same key Gym resumes on — so a rerun shard cannot double-count.

Resharding to a different N is safe and lossless: collected rollouts are folded back into the
parent and snapshotted to `snapshots/<UTC>/` before any shard directory is touched, then carried
into the new layout.

### A2 - Building the reward profiling container
The reward profiling container pre-installs all resources_servers and responses_api_agents needed for reward profiling.
It follows the same flow from the [Super-v3.5 readme](../README.md), with one change: new container config, created from the manifests: `benchmarks/nemotron_3.5_super/reward_profiling/configs/container_config.yaml`

```bash
# 0. make container config
python -m nemo_gym.sweep container-config manifests/*.yaml --out configs/container_config.yaml

# 1. make vllm container
mkdir results/vllm
CONTAINER_IMAGE_PATH=vllm/vllm-openai:v0.27.1
mkdir -p "$(dirname "$CONTAINER_IMAGE_PATH")"
enroot import -o "results/$CONTAINER_IMAGE_PATH" "docker://${CONTAINER_IMAGE_PATH}"

SLURM_ACCOUNT=nemotron_n3_post \
SBATCH_PARTITION=batch \
SBATCH_QOS=interactive \
BASE_IMAGE=$(pwd)/results/vllm/vllm-openai:v0.27.1 \
bash benchmarks/nemotron_3.5_super/build-super-vl-evals-v0271-thin.sh \
    $(pwd)/results/vllm/vllm-openai:v0.27.1___tomer.sqsh

# 2. make reward profiling container from vllm container
SBATCH_ACCOUNT=nemotron_n3_post \
SBATCH_PARTITION=batch \
SBATCH_QOS=interactive \
SBATCH_GRES=gpu:4 \
INPUT_CONTAINER=$(pwd)/results/vllm/vllm-openai:v0.27.1___tomer.sqsh \
OUTPUT_CONTAINER=$(pwd)/results/vllm/vllm-openai:v0.27.1___tomer_with_gym_all.sqsh \
MOUNTS=$(pwd)/env.yaml:/opt/Gym/env.yaml:x-create=file \
GYM_CONFIG=benchmarks/nemotron_3.5_super/reward_profiling/configs/container_config.yaml \
SKIP_PREPARE=1 \
NEMO_GYM_GIT_REF=main \
sbatch benchmarks/nemotron_3.5_super/build_eval_container.sh
```

Two things differ from the stock eval container build:

- **`GYM_CONFIG`** points at the generated `container_config.yaml`, not `eval_container_config.yaml`.
  Step 0 unions every `config_paths` across the manifests *and* their `gym_env_start` overlays, then
  adds dummy `policy_*` / `nv_inference_api_key` values so the config resolves at build time without
  secrets. The overlay part matters: the judge model servers exist only there, and a server with no
  baked venv installs at runtime and hangs the run behind connection retries rather than failing.
  Regenerate whenever an entry pulls in a new server; the file is reproducible from the manifest.
- **`SKIP_PREPARE=1`** skips `gym eval prepare`, which downloads benchmark datasets. Reward
  profiling supplies its own data via the manifest, so preparation would fail on environments that
  have no registered benchmark split.

Building all three lanes' servers gives 63 baked venvs. Building only a subset and then running a
manifest that needs more is the failure above -- it presents as a hang, not an error.

You also need a **sandbox container** if any entry uses `ns_tools` or `math_formal_lean`. Only one
arm64 build exists: `/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-0.7.1-arm64.sqsh`
