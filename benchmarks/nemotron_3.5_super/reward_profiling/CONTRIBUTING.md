# How to Contribute an environment / dataset.

## Index
1. Creating a container.
2. Creating a single manifest file.
    a. Non-judge environments.
    b. Judge environments.
    c. Sandbox environments.
3. Running a single manifest job.
    a. Non-judge environments.
    b. Judge environments.
    c. Sandbox environments.
4. Adding a manifest entry into manifests/nemotron_3_5_super.yaml

Contribute your environment on its own first, then add it to the shared manifest. A one-entry
manifest fails in seconds; the same mistake in the shared one costs a full spin-up on every shard.

See [INDEX.md](./INDEX.md) for the full pipeline. `R=benchmarks/nemotron_3.5_super/reward_profiling`
throughout.

## 01 - Creating a container.

Only needed if your environment's dependencies are not already in the reward profiling container.
Check first:

```bash
python -m nemo_gym.sweep container-config $R/manifests/<yours>.yaml
```

If it resolves, reuse the existing container. Otherwise see [INDEX.md § 01](./INDEX.md#01---optional-create-a-container).

## 02 - Creating a single manifest file.

Copy the closest example in `manifests/` and change `entries`. Every entry needs `label`, `agent`,
`configs`, `data`, and `owner` — `owner` is you, because the /lustre path stops identifying anyone
once the data is copied.

`agent` must be the `agent_ref.name` your rows actually carry. It is declared rather than inferred
so a mispaired dataset and config fails loudly instead of scoring against the wrong verifier.
Validate before running:

```bash
python -m nemo_gym.sweep validate $R/manifests/<yours>.yaml
```

### 02a - Non-judge environments.

Nothing beyond the entry itself. See [`manifests/example_basic.yaml`](./manifests/example_basic.yaml).

### 02b - Judge environments.

Shipped configs name a judge but never an endpoint, so you must bind one under `gym_env_start`.
Skip this and the composed config fails to resolve — the run dies at parse, before any rollout.

Set `max_concurrent_requests`. The transport retries 429s with a flat sleep and extends its retry
budget on rate limits, so a saturated judge becomes a retry storm rather than backpressure.

See [`manifests/example_judge.yaml`](./manifests/example_judge.yaml).

### 02c - Sandbox environments.

Two extra things, in [`manifests/example_sandbox_judge.yaml`](./manifests/example_sandbox_judge.yaml):

- Register every `verifier_type` your rows use under `ns_tools.verifiers`. An unregistered name is
  a hard `ValueError` that reaches the driver as a bare 500, so the lane produces zero rollouts.
- If the verifier is judge-backed, turn the judge on explicitly — `math_with_judge` verifies
  symbolically by default and only falls back to the judge.

## 03 - Running a single manifest job.

```bash
MANIFEST=$R/manifests/<yours>.yaml OUT_DIR=$R/outputs/sweeps/<name> bash $R/scripts/01_prepare_sweep.sh
SWEEP_DIR=$R/outputs/sweeps/<name>/<nickname> MODEL=<hf path> CONTAINER=<sqsh> bash $R/scripts/03_run.sh
```

Use `LIMIT_PER_ENTRY=2` for a first pass — it exercises every code path in minutes.

Then profile, which needs no GPU (see [§ 00](./INDEX.md#00---setup) for the venv):

```bash
SWEEP_DIR=<sweep> bash $R/scripts/05_profile.sh
```

Check `by_label/split_report.json` for `labels_without_rollouts`. A label at zero means the lane
failed rather than scored badly, and that silence is usually the finding.

### 03a - Non-judge environments.

Nothing extra.

### 03b - Judge environments.

Export the keys `env.yaml` interpolates, e.g. `NVI_KEY_EVALUATOR`. A bare `VAR=value` in `.bashrc`
is a shell variable, not an environment one, so sbatch never sees it. `03_run.sh` and
`05_profile.sh` both check this at submit and name what is missing.

### 03c - Sandbox environments.

Pass `SANDBOX_CONTAINER=<nemo-skills sqsh>`. Without it `ns_tools` falls back to `127.0.0.1:6000`,
where nothing listens, and every rollout fails with a bare 500. `03_run.sh` starts one and waits
for `/health` before collecting. The sandbox is one node only — sessions pin to a worker by
`X-Session-ID`, so it does not span nodes.

## 04 - Adding a manifest entry into manifests/nemotron_3_5_super.yaml

Once your entry runs clean on its own, move it into
[`manifests/nemotron_3_ultra.yaml`](./manifests/nemotron_3_ultra.yaml):

- Append to `entries`. Order sets task indices, so appending keeps existing `--resume` keys valid;
  inserting in the middle invalidates every entry after it.
- Merge your `gym_env_start` keys into the shared block, reusing the judge anchors already there
  rather than declaring another endpoint.
- Re-run `validate`, then a `LIMIT_PER_ENTRY=2` pass over the whole manifest — your entry can
  resolve alone and still collide on a port or judge binding once composed with 35 others.
