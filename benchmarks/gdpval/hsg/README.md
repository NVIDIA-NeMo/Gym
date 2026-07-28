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

## Setup

```bash
cd benchmarks/gdpval/hsg
cp cluster.env.example cluster.env
$EDITOR cluster.env                 # account, model path, container image
set -a; . ./cluster.env; set +a
```

Three settings are required and deliberately have no default —
`GDPVAL_SLURM_ACCOUNT`, `GDPVAL_MODEL_PATH`, `GDPVAL_VLLM_IMAGE`. A default
that looks plausible but points at the wrong model or account fails hours
later as bad numbers instead of immediately as an error.

## Running a batch

```bash
# 1. one-time: build an isolated checkout + runtime so an in-flight run is
#    never disturbed by edits to your working checkout
./bootstrap_gdpval_runtime.sh

# 2. submit the model server and wait for /v1/models to serve the expected name
./submit_gdpval_glm52.sh

# 3. submit exactly one rollout client against that endpoint
./monitor_gdpval_pipeline.sh

# 4. when the allocation is nearly up, chain the next one
./submit_gdpval_continuation.sh
```

`run_parallel_{fp8,bf16}-glm52-vllm.sh` are the underlying multi-instance vLLM
launchers; `submit_gdpval_glm52.sh` drives the fp8 one by default. Run either
directly with `--help` for its own flags.

## Dataset profiles

Per-dataset settings (input JSONL, expected row count, task-id pattern, model
name, concurrency) live in `benchmarks/gdpval/datasets/*.env` and are selected
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
| `submit_gdpval_glm52.sh` | Submit the vLLM server, wait for readiness, validate the served model name |
| `monitor_gdpval_pipeline.sh` | Wait for the endpoint, then submit exactly one rollout client |
| `run_gdpval_client.sbatch` | The rollout client job itself (CPU-only) |
| `submit_gdpval_continuation.sh` | Chain the next allocation once the current one is nearly spent |
| `run_gdpval_continuation_gate.sbatch` | Validate resume state, then release the next hop |
| `run_gdpval_continuation_monitor.sbatch` | Passive watcher for the continuation chain |
| `validate_gdpval_resume_state.py` | Standalone resume-safety check |
| `run_parallel_fp8-glm52-vllm.sh` | Multi-instance vLLM launcher (fp8) |
| `run_parallel_bf16-glm52-vllm.sh` | Multi-instance vLLM launcher (bf16) |
| `wait_for_slurm_account.sh` | Poll until a Slurm account/QOS association exists; never submits or cancels |
| `smoke_apptainer_worker.sh` | Verify Apptainer works on a compute node before committing an allocation |
| `rpm2cpio_stdin.sh` | Shim: BusyBox `rpm2cpio` rejects `-` for stdin, which the unprivileged Apptainer installer needs |

## Portability

Written against Slurm with pyxis/enroot or Apptainer, and tested on one
cluster. Job shapes (partition names, QOS, GPUs per node) are configurable but
the defaults reflect that cluster; expect to adjust `cluster.env` for yours.
