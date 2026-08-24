# ViBench

[ViBench](https://github.com/ViBench/vibench-public) evaluates whether a model can build a
working web application from a product requirements document. A task hands the agent a PRD
and an empty container; grading stands the finished app up for real, seeds it with data
through its own UI, and then drives it in a browser against a human-written test plan.

ViBench is **"case 2"** in [#2082](https://github.com/NVIDIA-NeMo/Gym/issues/2082): the
rollout copies an artifact out of its own box and the verifier grades it in a fresh one. So
this server never touches the agent's sandbox. That is not a stylistic choice — sharing a box
is case 3, which needs the (unmerged) sandbox server, and node-local providers like Docker
deliberately cannot support it. Only OpenSandbox implements `serialize()`/`connect()`.

## Shape

| Stage | Owner |
| --- | --- |
| Task rows (`app`, `artifact`, PRD paths, test-plan paths) | `prepare.py` → `data/*.jsonl` |
| PRD text + asset paths | `seed_session` in `app.py` (no sandbox, no handle) |
| Build sandbox, PRD staging, writing the app, harvesting it | `responses_api_agents/vibench_agent` |
| Seed → evaluate → score | `verify` in `app.py`, shelling into a ViBench checkout |

The agent writes a tarball of the built app into `artifact_dir` and passes the path to
`/verify`. A plain shared path is enough: grading already shells into a local Docker daemon,
so both processes are on one host either way.

One row is one `(app, artifact)` pair. Reward is the mean normalized score
(`score / full_points`) across that artifact's test plans, with the per-plan values exposed
in `reward_components`. Every plan counts toward the denominator, so a plan that fails to
seed pulls the mean down instead of dropping out of it.

Reward is **continuous, not binary**. A ViBench test plan is a list of scored steps, and
partial credit is the signal the benchmark is built around.

## Setup

Requires Docker and a ViBench checkout. ViBench's grading scripts invoke the legacy
`docker-compose` name, which Docker 29.x no longer ships; without it seeding fails in
seconds and reports a fully failed seeding rate, which reads like a bad app rather than a
missing binary.


```bash
git clone https://github.com/ViBench/vibench-public.git ~/vibench
cd ~/vibench && uv sync && cp .env.template .env   # fill in the grader's provider keys
docker build -f _harness/runner/docker/Dockerfile.base -t app-bench-base:latest .
```

`app-bench-base:latest` is the tag ViBench's own pipeline builds and reuses, so the build
sandbox and the grading stack share one base image. Its `WORKDIR` is `/app`, which is where
sandboxed agents land; override `app_workdir` only alongside a different image.

```bash
export VIBENCH_REPO_ROOT=~/vibench
export VIBENCH_ENV_FILE=~/vibench/.env
export VIBENCH_ARTIFACT_DIR=/tmp/vibench-artifacts
```

`VIBENCH_ENV_FILE` supplies `AGENT_SEEDING_LLM_*` and `AGENT_EVALUATION_LLM_*` for the
grader agents. Those are the **verifier's** models and are deliberately separate from the
policy model under test — do not point them at the same endpoint when profiling.

## Generate task rows

`prepare.py` renders ViBench's own `coding_prompt.j2` as each task's brief rather than
paraphrasing it. That prompt is a contract: it requires the app to ship
`setup-environment.sh` and `start-server.sh`, which the grading stack invokes. An app built
without them fails evaluation regardless of quality, and a reworded brief would change what
the benchmark measures. Rendering needs `jinja2` — from ViBench's own venv if present,
otherwise the interpreter running `prepare.py`.

```bash
python resources_servers/vibench/prepare.py \
    --vibench-root "$VIBENCH_REPO_ROOT" \
    --output resources_servers/vibench/data/vibench_mvp.jsonl
```

That yields 24 tasks across 74 test plans. `data/example.jsonl` holds five of them
(`notes`, `quiz`, `barber`, `wedding`, `market_place`).

P0 covers `mvp` artifacts only. `--artifacts feature1 feature1-on_mvp` is wired but needs a
reference-implementation starting tree that the public ViBench repo does not ship.

## Run

The second `--config` is required: `sandbox_provider: sandbox` is a reference that the
provider config binds, so without it startup fails with *"Sandbox provider reference
'sandbox' is not defined in the merged config"*. Swap that one path to move to another
provider (OpenSandbox, Fargate, Enroot) without editing this config.

Use `vibench_agent`'s docker config rather than the stock one. The harness inside the
sandbox reaches the policy model at `http://127.0.0.1:<port>` (`get_server_url`), which in a
bridged container points at the container itself — the harness then makes **zero** LLM calls
and exports an empty app with `"tokens": {"input": 0, "output": 0}`, with no error anywhere.
The host network makes that address mean the same thing in both places. Read the trade-off
note at the top of that file before using it on anything but a disposable box.

```bash
gym env start \
    --config resources_servers/vibench/configs/vibench.yaml \
    --config responses_api_agents/vibench_agent/configs/docker_host_network.yaml \
    --model-type openai_model

gym eval run --no-serve \
    --agent vibench_opencode_agent \
    --input resources_servers/vibench/data/example.jsonl \
    --output results/vibench_rollouts.jsonl \
    --limit 1 \
    --num-repeats 1
```

Start with `--limit 1`. A single rollout builds an app and then runs a full compose stack
per test plan; wall-clock is tens of minutes and the box needs headroom for
`max_concurrent_test_plans` simultaneous Postgres + app + Playwright stacks.

## Validation

Run end to end on a Docker host. A single `notes/mvp` task graded 3/3 test plans, and the
reward tracks model capability rather than merely completing:

| policy model | reward | steps passed | outcome |
| --- | --- | --- | --- |
| weaker | 0.0 | 0/19 | app built but unreachable (missing build output) |
| stronger | 1.0 | 19/19 | full marks on all three plans |

Same code, dataset and grader agents in both cases. `data/example_rollouts.jsonl` holds the
scoring run.

`verified: false` still stands: that flag means baselined and reviewed, which needs a
profiling sweep across many tasks, not one.

## P0 limitations

These are known and deliberate; each is a follow-up rather than a bug.

- **Grading runs on the resources server's Docker daemon**, not inside a Gym sandbox,
  because ViBench's grading stack is multi-container (app + postgres + code-browse). Folding
  it into one supervisord image is the prerequisite for running this on more than one host.
- **The agent and the resources server must share `artifact_dir`** (see above).
- **The verifier is itself an LLM agent** driving a browser, so reward is stochastic.
  Profile that variance — repeated grading of one fixed app — before treating this as a
  training signal. `REVERIFY_MODE` is `UNSUPPORTED` for the same reason: scores cannot be
  recomputed from stored rollouts, since grading depends on live app and database state.
- **`mvp` artifacts only.** See above.
- **Cost per rollout is high**: one coding agent, plus a seeding agent and an evaluation
  agent per test plan.

## Anti-cheat

The agent's sandbox receives the PRD and `prds/<app>/assets/` only. Test plans and
`test_assets/` are never staged into the build container — they are read at grade time by
the resources server. Dataset paths are resolved against `vibench_repo_root` and rejected
if they escape it.

## Licensing

ViBench is Apache 2.0. PRDs, test plans, and the runner harness come from the ViBench
repository; this server contains no ViBench data of its own — `prepare.py` reads a local
checkout.
