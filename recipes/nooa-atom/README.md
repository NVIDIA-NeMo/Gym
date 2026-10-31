# Atom (NOOA) recipes

One pinned command per benchmark for running Atom through the sandboxed NOOA adapter, the same way
for a one-row smoke, a full evaluation, or a NeMo RL rollout environment.

| Recipe | Agent | Tasks | Command |
|---|---|---|---|
| `terminal-bench-2.1` | `nooa_atom.gym:AtomHarness`, `handle_batch` | Terminal-Bench 2.1, 89 tasks | `recipes/nooa-atom/run.sh terminal-bench-2.1` |
| `swebench-pro` | `nooa_atom.gym:AtomHarness`, `handle_batch` | SWE-bench Pro, the 120-task roster in `swebench-pro-swe120.txt` | `recipes/nooa-atom/run.sh swebench-pro` |
| `gdpval` | `nooa_atom.gym:AtomHarness`, `handle_batch` | GDPval, 220 tasks, rubric mode with a judge panel (needs JUDGE_API_KEY, HF_TOKEN, TAVILY_API_KEY, GDPVAL_SANDBOX_IMAGE, PERSIST_DELIVERABLES_DIR) | `recipes/nooa-atom/run.sh gdpval` |

A recipe's `.setup.sh` runs after `gym eval prepare` and may pin a task repository (Terminal-Bench). A roster
recipe (SWE-bench Pro) names `benchmarks/swebench/pro/prepare_roster.py` as its prepare script with the roster
file and its sha256 in `prepare_script_args`.

Every recipe file carries three pins: `gym_commit` (this checkout), `nooa_commit` (what
`responses_api_agents/nooa_agent/requirements.txt` installs) and the task-set commit. `preflight.sh`
refuses to run when any of them disagrees with what is actually on disk.

## Setup, once per checkout

```bash
git checkout <gym_commit from the recipe>
uv venv && uv sync --frozen
(cd responses_api_agents/nooa_agent && uv pip install -r requirements.txt)   # NOOA at the pinned revision
cp recipes/nooa-atom/env.yaml.example env.yaml                                # model endpoint and name
cp .env.example .env                                                          # POLICY_API_KEY, OPENSANDBOX_DOMAIN, OPENSANDBOX_API_KEY
```

### Runtime

Task sandboxes never install from the network. The controller prepares an offline NOOA runtime once per
Gym commit and platform, and the recipe requires it (`runtime_wheelhouse_required: true`):

```bash
python -m responses_api_agents.nooa_agent.offline_runtime --output <dir>/x86_64-gnu --target x86_64/gnu
export NOOA_RUNTIME_WHEELHOUSE=<dir>/x86_64-gnu
```

Building needs Docker on the machine that runs the command (with `linux/amd64` emulation when that
machine is not x86_64). The bundle is bound to the checkout's source fingerprint: a new Gym commit needs a
new bundle, and `preflight.sh` validates the bundle against the checkout before every run.

## Preflight, before every run

```bash
recipes/nooa-atom/preflight.sh terminal-bench-2.1
```

In order: pins match (checkout, requirements, installed NOOA distributions, task repo); environment
names present; offline runtime validates; the exact config resolves and every file it names exists;
then one real row runs through `run.sh` itself. `SKIP_ROW=1` stops after the offline checks. The
resolved config and its hash land in `results/<recipe>-preflight/`. Two preflights from clean
checkouts at the same pins must produce the same `resolved.sha256`.

## Run

```bash
recipes/nooa-atom/run.sh terminal-bench-2.1                 # full benchmark
LIMIT=5 PARALLEL=2 recipes/nooa-atom/run.sh terminal-bench-2.1
ONLY_TASKS='code-from-image,fp8-*' recipes/nooa-atom/run.sh terminal-bench-2.1   # named tasks only
RESUME=1 recipes/nooa-atom/run.sh terminal-bench-2.1        # continue an interrupted run
```

`ONLY_TASKS` takes comma-separated globs on task ids (`task_id`, `task_name` or `instance_id` of the dataset rows).
The run filters the prepared dataset file to the matching rows (`only_tasks.py`) and points Gym at the copy.

Output: `results/<recipe>/evaluator_rollouts.jsonl` with the resolved config saved beside it by Gym, and
the failures sidecar. `gym eval export atif --rollouts results/<recipe>/evaluator_rollouts.jsonl` writes
ATIF trajectories.

### What an Atom shell sees in the sandbox

```bash
python recipes/nooa-atom/probe_runtime.py --image alexgshaw/code-from-image:20251031 \
  --wheelhouse "$NOOA_RUNTIME_WHEELHOUSE" --wheelhouse-musl "$NOOA_RUNTIME_WHEELHOUSE_MUSL"
```

Starts one sandbox from the task image, stages the runtime as the agent server does, and reports what `command -v`
resolves under the entrypoint's PATH policy. `ACCEPTED` means `python3` and `pip` are the image's own (or absent),
`rg` is `<runtime>/shell-tools/rg`, and nothing else of the runtime is on PATH. Use it after changing the runtime
or the entrypoint, before a benchmark run.

### Arms

`NOOA_ATOM_TURN_METHOD` selects Atom's root turn method per submission: `handle_batch` (default, one agent
unattended) or `handle_orchestrate` (the root may delegate to child sessions). Everything else in the recipe is
identical, so two submissions differing only in that variable are a matched pair; the resolved config saved beside
the rollouts records the value each run used.

### On Slurm

```bash
sbatch --account=<acct> --partition=cpu --job-name=atom-tb21 \
  recipes/nooa-atom/slurm/run.sbatch terminal-bench-2.1 <attempts dir>
```

One allocation runs preflight then the full recipe into a fresh attempt directory. The job name is the
idempotency key: check `squeue --name=atom-tb21` before submitting again. The controller node only
needs the Gym venv; sandboxes come from OpenSandbox, so the node's architecture does not matter.

Size the time limit from the rows, not the task count: at `PARALLEL=32` a full SWE-bench Pro or
Terminal-Bench 2.1 run is four to six waves of rows, and a wave lasts as long as its slowest row (p90 around
50 min for SWE-bench Pro, 20 min for Terminal-Bench, with a tail above two hours), so a four-hour queue
limit usually ends the job with a few rows unfinished. Resume the attempt in place instead of starting over:

```bash
RESUME_ATTEMPT=<attempts dir>/atom-tb21-<timestamp>-<jobid> sbatch --account=<acct> --partition=cpu \
  --job-name=atom-tb21-resume recipes/nooa-atom/slurm/run.sbatch terminal-bench-2.1 <attempts dir>
```

Completed rows are kept. Rows the time limit killed left no record and run again; a row whose attempt
failed gets exactly one more attempt. Report the first attempt's rows as the primary result and the
resumed rows separately: they are infrastructure retries, and `evaluator_rollouts_failures.jsonl` keeps
every failed attempt so the two are separable after the fact.

## RL

The same configs are the environment for NeMo RL. In the RL config, under `env.nemo_gym.config_paths`,
list `responses_api_models/vllm_model/configs/vllm_model_for_training.yaml` instead of the plain vLLM
config, then the recipe's `benchmark_config` entries and the recipe file. Atom routes every model call
through Gym's model server and the adapter declares the root agent's terminal response, so token-id
capture attributes the scored chain. Run a trajectory collection (`is_trajectory_collection: true`) first
and check `n_calls` and the `mask_sample` fraction on the collected rollouts.
