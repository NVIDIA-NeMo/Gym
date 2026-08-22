# ViBench

[ViBench](https://github.com/ViBench/vibench-public) evaluates whether a model can build a
working web application from a product requirements document. A task hands the agent a PRD
and an empty container; grading stands the finished app up for real, seeds it with data
through its own UI, and then drives it in a browser against a human-written test plan.

This server is the **P0** integration: it owns the build sandbox and delegates grading to
ViBench's existing `run-seed-then-evaluate.py` pipeline.

## Shape

| Stage | Owner |
| --- | --- |
| Task rows (`app`, `artifact`, PRD paths, test-plan paths) | `prepare.py` → `data/*.jsonl` |
| Build sandbox + PRD staging | `seed_session` in `app.py` |
| Writing the app | any Gym agent that consumes `sandbox_handle` (P0 pairs with `opencode_sandboxed_agent`) |
| Seed → evaluate → score | `verify` in `app.py`, shelling into a ViBench checkout |

One row is one `(app, artifact)` pair. Reward is the mean normalized score
(`score / full_points`) across that artifact's test plans, with the per-plan values exposed
in `reward_components`. Every plan counts toward the denominator, so a plan that fails to
seed pulls the mean down instead of dropping out of it.

Reward is **continuous, not binary**. A ViBench test plan is a list of scored steps, and
partial credit is the signal the benchmark is built around.

## Setup

Requires Docker and a ViBench checkout.

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
```

`VIBENCH_ENV_FILE` supplies `AGENT_SEEDING_LLM_*` and `AGENT_EVALUATION_LLM_*` for the
grader agents. Those are the **verifier's** models and are deliberately separate from the
policy model under test — do not point them at the same endpoint when profiling.

## Generate task rows

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

```bash
gym env start \
    --config resources_servers/vibench/configs/vibench.yaml \
    --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
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

## P0 limitations

These are known and deliberate; each is a follow-up rather than a bug.

- **Grading runs on the resources server's Docker daemon**, not inside a Gym sandbox,
  because ViBench's grading stack is multi-container (app + postgres + code-browse). Folding
  it into one supervisord image is the prerequisite for OpenSandbox/Fargate/Enroot and for
  running this anywhere but a single fat host.
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
