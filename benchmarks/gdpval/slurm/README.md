# GDPVal rollout orchestration on Slurm

Reference scripts for collecting GDPVal rollouts on a Slurm cluster with a
self-hosted vLLM endpoint, across more wall-clock time than one allocation
gives you.

They exist because a GDPVal rollout batch does not fit in a single job. Tasks
run for tens of minutes each, allocations are time-boxed, and a run that is
merely *restarted* after an allocation expires silently loses or duplicates
work. These scripts keep the model server and the rollout client in separate
jobs, hand off between allocations, and validate that a resume is safe before
it is allowed to proceed.

Nothing here is required to run GDPVal — `gym eval run` is. Reach for these
when a batch is too large for one allocation.

## Configuration

Three layers, so swapping any one of them does not touch the others:

| Layer | File | Holds |
|---|---|---|
| Site | `slurm/cluster.env` | Slurm account, partition, filesystem paths, weights + image location |
| Model | `models/<model>.env` | Parallelism, context, tool/reasoning parsers, KV dtype |
| Batch | `datasets/<batch>.env` | Input JSONL, expected count/hash, task-id pattern, turn budget |

```bash
cd benchmarks/gdpval/slurm
cp cluster.env.example cluster.env
$EDITOR cluster.env                          # account, model path, container image

set -a
. ./cluster.env
. ../models/glm-5.2-fp8.env                  # or your own profile
. ../datasets/my_batch.env
set +a
```

Serving a different model is a new file in `models/`, not an edit to any
script. Start from `models/example.env`. The three GLM-5.2 profiles are worked
examples of the same weights deployed three ways:

| Profile | Shape | For |
|---|---|---|
| `glm-5.2-fp8.env` | 2 nodes, TP=8, 1 replica | Lowest per-request latency |
| `glm-5.2-fp8-dp7.env` | 7 nodes, TP=4, 7 replicas | Aggregate throughput |
| `glm-5.2-bf16.env` | 16 nodes, TP=4, 16 replicas | Throughput, unquantized |

For rollout collection the binding constraint is replicas, not the agent loop:
aggregate throughput on a single replica was still scaling at ~0.84 efficiency
per doubling of agent concurrency with no sign of flattening. Reach for a
data-parallel profile before tuning concurrency, and set concurrency to roughly
16 per replica so the replicas stay fed.

Four settings are required and deliberately have no default —
`GDPVAL_SLURM_ACCOUNT`, `GDPVAL_MODEL_PATH`, `GDPVAL_VLLM_IMAGE` and
`GDPVAL_MODEL_BASE_NAME` (the last comes from the model profile). A default
that looks plausible but points at the wrong model or account fails hours later
as bad numbers instead of immediately as an error.

## Running a batch

```bash
# 1. one-time: build an isolated checkout + runtime so an in-flight run is
#    never disturbed by edits to your working checkout
./bootstrap_gdpval_runtime.sh

# 2. submit the model server and wait for /v1/models to serve the expected name
./submit_model_server.sh

# 3. submit exactly one rollout client against that endpoint
./monitor_gdpval_pipeline.sh

# 4. when the allocation is nearly up, chain the next one
./submit_gdpval_continuation.sh
```

`run_parallel_vllm.sh` is the underlying multi-instance vLLM launcher;
`submit_model_server.sh` drives it and validates the endpoint before any rollout
client is released. Run either with `--help` for its own flags.

## Dataset profiles

Per-dataset settings (input JSONL, expected row count, task-id pattern, model
name, concurrency) live in `benchmarks/gdpval/datasets/*.env` (see `example.env`) and are selected
with `GDPVAL_ENV_FILE`. Adding a new batch means adding a profile, not editing
these scripts.

## Resuming safely

`validate_gdpval_resume_state.py` is the gate. It checks that a partial run's
state is internally consistent — no duplicate or missing task indices, launch
and source files unchanged since the run started — before another allocation is
allowed to continue it.

Editing the launcher or validator mid-run changes their hashes and the
continuation gate will refuse to resume. That is deliberate: silently resuming
against changed code produces a batch that is two experiments stitched
together.

## What each script does

| Script | Purpose |
|---|---|
| `bootstrap_gdpval_runtime.sh` | Build an isolated checkout + venv so a running batch is insulated from edits |
| `submit_model_server.sh` | Submit the vLLM server, wait for readiness, validate the served model name |
| `monitor_gdpval_pipeline.sh` | Wait for the endpoint, then submit exactly one rollout client |
| `run_gdpval_client.sbatch` | The rollout client job itself (CPU-only) |
| `submit_gdpval_continuation.sh` | Chain the next allocation once the current one is nearly spent |
| `run_gdpval_continuation_gate.sbatch` | Validate resume state, then release the next hop |
| `run_gdpval_continuation_monitor.sbatch` | Passive watcher for the continuation chain |
| `validate_gdpval_resume_state.py` | Standalone resume-safety check |
| `run_parallel_vllm.sh` | Multi-instance vLLM launcher, shaped entirely by the model profile |
| `wait_for_slurm_account.sh` | Poll until a Slurm account/QOS association exists; never submits or cancels |
| `smoke_apptainer_worker.sh` | Verify Apptainer works on a compute node before committing an allocation |
| `rpm2cpio_stdin.sh` | Shim: BusyBox `rpm2cpio` rejects `-` for stdin, which the unprivileged Apptainer installer needs |

## Portability

Written against Slurm with pyxis/enroot or Apptainer, and tested on one
cluster. Job shapes (partition names, QOS, GPUs per node) are configurable but
the defaults reflect that cluster; expect to adjust `cluster.env` for yours.
