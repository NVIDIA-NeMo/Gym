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

Contribute your environment as a one-entry manifest first, prove it runs, then add it to the shared
manifest. A one-entry manifest fails in seconds; the same mistake in the shared one costs a full
spin-up on every shard.

See [README.md](./README.md) for the pipeline itself. `R=benchmarks/nemotron_3.5_super/reward_profiling`
throughout, and all paths below are relative to the repo root.

### Files you touch

| file | when | what |
|---|---|---|
| your data `.jsonl` | always | one row per task, on `/lustre`. Not committed |
| `resources_servers/<env>/` | only if your env is not already in Gym | verifier + config. See the [environment guide](https://docs.nvidia.com/nemo/gym/latest/contribute/environments) |
| `$R/manifests/<yours>.yaml` | always | your one-entry manifest. Committed |
| `$R/manifests/nemotron_3_ultra.yaml` | step 04 | add your entry to the shared sweep |
| `$R/configs/container_config.yaml` | only if you pulled in a new server | regenerate, do not hand-edit |

Nothing else. In particular you do not edit anything under `nemo_gym/`, and you do not edit
upstream configs under `resources_servers/*/configs/` to bind a judge — that goes in your manifest
([02b](#02b---judge-environments)).

### What your data rows need

```json
{ "agent_ref": { "name": "my_simple_agent" },
  "responses_create_params": { "input": [ ... ] },
  "verifier_metadata": { ... } }
```

`agent_ref.name` is what rollout collection dispatches on — it is why 36 environments can share one
deployment. It must match the `agent` your manifest entry declares, and that agent must be declared
by one of the entry's `configs`. `validate` checks all three agree.

## 01 - Creating a container.

Only needed if your environment's servers are not already baked into the reward profiling
container. A server with no baked venv installs at runtime and hangs the run behind connection
retries rather than failing, so check first:

```bash
python -m nemo_gym.sweep container-config $R/manifests/<yours>.yaml --out /tmp/mine.yaml
python -c "
import yaml
have = set(yaml.safe_load(open('$R/configs/container_config.yaml'))['config_paths'])
mine = set(yaml.safe_load(open('/tmp/mine.yaml'))['config_paths'])
print('\n'.join(sorted(mine - have)) or 'nothing new; reuse the existing container')
"
```

A plain `diff` is not the check — the committed config covers 36 entries and yours covers one, so
they always differ. What matters is whether your `config_paths` are a subset of it. Anything
printed is a server the container does not have: regenerate `configs/container_config.yaml` over
all manifests and rebuild, per [README.md § 01](./README.md#01---optional-create-a-container).

## 02 - Creating a single manifest file.

Copy the closest `manifests/example_*.yaml` and change `entries`. Every entry needs `label`,
`agent`, `configs`, `data`, and `owner` — `owner` is you, because the `/lustre` path stops
identifying anyone once the data is copied or a blend is re-cut.

```bash
python -m nemo_gym.sweep validate $R/manifests/<yours>.yaml
```

This checks the configs exist, the agent is declared by one of them, the data parses, and each
row's `agent_ref` matches. That last check is what stops a mispaired dataset and config from
silently scoring rollouts with the wrong verifier.

### 02a - Non-judge environments.

Nothing beyond the entry itself. See [`manifests/example_basic.yaml`](./manifests/example_basic.yaml).

### 02b - Judge environments.

Shipped configs name a judge but never an endpoint, so bind one under `gym_env_start` in your
manifest. Skip this and the composed config fails to resolve — the run dies at parse, before a
single rollout.

Two keys, both in [`manifests/example_judge.yaml`](./manifests/example_judge.yaml): the model server
itself, and the binding that tells your resources server to use it.

Set `max_concurrent_requests`. The transport retries 429s with a flat sleep and extends its retry
budget on rate limits, so a saturated judge degrades into a retry storm rather than backpressure.

### 02c - Sandbox environments.

Two extra things, in [`manifests/example_sandbox_judge.yaml`](./manifests/example_sandbox_judge.yaml):

- Register every `verifier_type` your rows use under `ns_tools.verifiers`. Rows carry it per-row,
  and an unregistered name is a hard `ValueError` that reaches the driver as a bare 500 — the lane
  produces zero rollouts rather than bad ones.
- If the verifier is judge-backed, enable the judge explicitly. `math_with_judge` verifies
  symbolically via math-verify by default and only consults the judge as a fallback.

## 03 - Running a single manifest job.

```bash
MANIFEST=$R/manifests/<yours>.yaml OUT_DIR=$R/outputs/sweeps/<name> LIMIT_PER_ENTRY=8 \
  bash $R/scripts/01_prepare_sweep.sh

MODEL=<ckpt> CONTAINER=<eval sqsh> \
  SWEEP_DIR=$R/outputs/sweeps/<name>/<nickname> bash $R/scripts/03_run.sh
```

`LIMIT_PER_ENTRY=8` exercises every code path in minutes. Then profile and read the result:

```bash
SWEEP_DIR=$R/outputs/sweeps/<name>/<nickname> bash $R/scripts/05_profile.sh
cat $R/outputs/sweeps/<name>/<nickname>/by_label/<label>/profile.txt
```

Check `by_label/split_report.json` for `labels_without_rollouts`. Your label at zero means the lane
failed rather than scored badly — read `outputs/.../slurm-logs/` for the 500.

Reward is not the only signal worth a look. A `std/reward` of 0 across every task usually means the
verifier is returning a constant, not that the model is perfectly consistent.

### 03a - Non-judge environments.

Nothing extra.

### 03b - Judge environments.

Export the keys `env.yaml` interpolates, e.g. `NVI_KEY_EVALUATOR`. A bare `VAR=value` in `.bashrc`
is a shell variable, not an environment one, so sbatch never sees it — and `export` in `.bashrc`
only reaches a login shell. `03_run.sh` and `05_profile.sh` both check this up front and name what
is missing.

### 03c - Sandbox environments.

Pass `SANDBOX_CONTAINER=<nemo-skills sqsh>`. Without it `ns_tools` falls back to `127.0.0.1:6000`,
where nothing listens, and every rollout fails with a bare 500. `03_run.sh` starts one and waits
for `/health` before collecting, so watch that gate rather than the rollout count.

The sandbox is one node only — sessions pin to a worker by `X-Session-ID` consistent hashing, so it
does not span nodes. Only one arm64 build exists:
`/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-0.7.1-arm64.sqsh`

## 04 - Adding a manifest entry into manifests/nemotron_3_5_super.yaml

Once your entry runs clean alone, move it into
[`manifests/nemotron_3_ultra.yaml`](./manifests/nemotron_3_ultra.yaml):

- **Append to `entries`, never insert.** Order assigns task indices, so appending keeps every
  existing `--resume` key valid; inserting in the middle renumbers every entry after yours and
  invalidates work already collected.
- **Merge your `gym_env_start` keys into the shared block**, reusing the judge anchors already
  defined there rather than declaring a second endpoint for the same model.
- **Regenerate the container config** if you added a server, and rebuild.
- **Re-validate, then re-run with `LIMIT_PER_ENTRY=2` over the whole manifest.** Your entry can
  resolve alone and still collide once composed with 35 others — a port, a judge binding, or a
  server that was never baked into the container.

Open the PR against this branch with the `LIMIT_PER_ENTRY` profile output for your label.
