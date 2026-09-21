# NeMo Gym harness on Modal

Runs a Gym evaluation campaign in Modal containers instead of on a laptop: one container
per model, results on a Volume, and a web dashboard over the lot.

This hosts the **harness** — head server, resources server, agent server, model proxy and
their Ray cluster — not the models. Models stay on their own deployments. The workload is
pure I/O orchestration, so the containers are CPU-only and cheap.

Nothing here is ASB-specific. A campaign is `(config paths, agent, input, model endpoint)`.

## Why bother

A Gym stack is about five processes plus a Ray cluster. Several campaigns on one machine
contend for RAM, collide on ports, and tear down each other's servers — three stacks on one
Mac produced twenty server processes and one working eval. Separate containers share no
ports, no Ray cluster and no memory, so models genuinely run in parallel.

Measured on the same model, same harness, same day: Super-VL managed 0.12–0.47 rollouts/s on
a contended laptop and **1.71 rollouts/s** in a container.

## Use it for your benchmark

```python
from scripts.modal_harness.campaign import app, run_campaign

handle = run_campaign.spawn(
    slug="my-model",                       # names the output file
    namespace="mybench",                   # directory on the volume
    config_paths=["resources_servers/mybench/configs/mybench.yaml"],
    agent="mybench_agent",
    input_path="resources_servers/mybench/data/all.jsonl",
    policy_base_url="https://.../v1",
    policy_model="org/Model-Name",
    policy_token_var="MODAL_PROXY_TOKEN",  # key inside the campaign secret
    judge_server_name="mybench_judge_model",   # must match your YAML's server name
    expected_rows=5000,
    concurrency=96,
    git_ref="your-branch",
    prepare_module="benchmarks.mybench.prepare",  # optional; materializes inputs in-container
)
print(handle.get())
```

Then launch several at once, one per model — see `asb.py` for a worked example with a
`modal run` entrypoint:

```bash
MODAL_ENVIRONMENT=FDR modal run --detach -m scripts.modal_harness.asb --models supervl,qwen
```

`--detach` matters: without it the run dies with your terminal.

## One-time setup

```bash
# Secret with your endpoint tokens. Reads .env and passes values via a temp file,
# never argv — `modal secret create NAME KEY=VALUE` puts credentials in shell history.
python scripts/modal_harness/bootstrap_secret.py --env-file /path/to/.env

# Deploy (registers the runner and the dashboard)
MODAL_ENVIRONMENT=FDR modal deploy -m scripts.modal_harness.dashboard
```

Add your own token keys to `WANTED` in `bootstrap_secret.py` if you need endpoints beyond
the ones ASB uses.

## Bring existing rollouts with you

Upload before the first run, or the campaign starts from zero:

```bash
modal volume put nemo-gym-campaign-results --env=FDR \
  results/mybench/my-model.jsonl /mybench/my-model.jsonl --force
```

**Upload the sidecars too.** Resume matches output rows against
`<output>_materialized_inputs.jsonl` and reads prior attempts from
`<output>_failures.jsonl`. Without them resume has nothing to match against and silently
re-runs every row — the progress bar reads the full count instead of what is outstanding.

## Dashboard

Deployed alongside the runner, HTTP Basic against the `nemo-gym-dashboard-auth` secret:

- `/` every campaign on the volume, state and coverage
- `/run/{ns}/{slug}` progress, status, latest rollouts
- `/run/{ns}/{slug}/rollout/{i}` one trajectory — the messages the model actually saw
- `/run/{ns}/{slug}/logs` server log tail
- `/run/{ns}/{slug}/export` download rollouts or logs
- `/api/runs` JSON

Your runs appear automatically; discovery is driven by the rollout files, so a run whose
container died still shows up with whatever it collected.

## Stopping a cell

```bash
modal run -m scripts.modal_harness.stop --namespace mybench --slugs cell-a,cell-b
modal run -m scripts.modal_harness.stop --namespace mybench --slugs cell-a --force
```

Cancels those runs and nothing else; `modal app stop` takes down every cell in the app.
Collected rows are safe — the publisher writes under the no-shrink rule every 45s, so a
cancel loses at most that interval and the next launch resumes.

`--force` is for a cell with no recorded call id (any run started before that was added).
It clears the slug's status entry so the idempotency guard respawns it. The stuck container
keeps its slot until its own timeout but cannot affect the rows: publishing floors against
the volume's current count, and `landed` is monotonic per slug. Use it only on a genuinely
stuck cell — on a slow one it just spawns a redundant second container.

**`modal container logs` will not identify an old container.** It returns a short recent
tail, and the `[slug] attempt N` line is printed at container start — so grepping containers
for a slug works right after a spawn and silently fails for exactly the long-running cell
you want to stop. Use `--force` instead.

## Things that cost me hours

Each of these was found by running it, not by reading the code.

- **A failed `gym eval run --resume` truncates its output file.** Five failed attempts
  destroyed 6,298 already-collected rollouts. The runner now works on a container-local
  copy and publishes to the volume only when an attempt ends with at least as many rows as
  it began with; a shrink is rolled back rather than published. If you write your own
  runner, do not point the eval at your canonical results file.
- **Resume needs its sidecars beside the output file** (above).
- **Wait for every server, not just the head.** The head answers in seconds, but
  `gym env start` then builds a separate venv per server directory, which takes minutes on
  a fresh container. Polling the head alone declares readiness while the agent server does
  not exist, and the eval burns thousands of retries against a dead port. Count `✓` marks
  from `gym env status`.
- **A Modal Volume cannot back uv's cache.** uv persists through atomic rename/link and the
  volume returns EPERM. Keep the cache container-local.
- **`debian_slim` has no `procps`**, so `pkill` silently does nothing and every retry leaks
  a server stack.
- **Volume writes only persist on `commit()`.** Anything written by a shell redirect is lost
  if the container dies before the next commit — keep logs container-local and copy them up.
- **Stream long commands, don't capture them.** `capture_output=True` makes a multi-hour
  eval print nothing until it returns, so a working run and a wedged one look identical.
- **Endpoint saturation is silent.** An overloaded deployment queues rather than erroring:
  zero 503s, zero 429s, just 20–30x latency inflation. Measure latency under load before
  concluding a run is client-bound. On a single replica, *more* client concurrency makes
  throughput worse; on three, more is better. The right ceiling is a property of the
  deployment, not a constant.
