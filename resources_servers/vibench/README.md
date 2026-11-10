# ViBench

[ViBench](https://github.com/ViBench/vibench-public) evaluates whether a model can build a
working web application from a product requirements document. A task hands the agent a PRD
and an empty container; grading stands the finished app up for real, seeds it with data
through its own UI, and then drives it in a browser against a human-written test plan.

This environment covers ViBench 1.0 (`v1/` in vibench-public): 24 apps, each with an MVP
PRD and feature PRDs. ViBench 2.0, which builds 17 apps feature by feature in one conversation
on its own Harbor harness, is not integrated.

## Shape

Two configs run this server:

- `configs/vibench_sessions.yaml` runs the episode through the `single_agent_turn_legacy` Environment
  Server and the agent-session protocol. This server owns the build sandbox and lends it; the agent
  only borrows it. The rest of this README describes this path unless it says otherwise.
- `configs/vibench.yaml`, the default, keeps the legacy relay path through `vibench_agent`; see
  [Legacy relay path](#legacy-relay-path). It stays the default until both paths are graded on the
  same tasks, since their scores are not directly comparable.

| Stage | Owner |
| --- | --- |
| Task rows (`app`, `artifact`, PRD paths, test-plan paths) | `prepare.py` → `data/*.jsonl` |
| Build sandbox, PRD and asset staging, `SandboxAccess` | `seed_session` in `app.py` |
| Writing the app in the lent sandbox | `responses_api_agents/opencode_agent` (an agent session) |
| Harvesting the app, stopping the sandbox, seed → evaluate → score | `verify` in `app.py`, shelling into a ViBench checkout |
| Stopping a sandbox that was never verified | `close_session` in `app.py` |

`verify` tars the app directory inside the sandbox, downloads it and stops the sandbox before
grading starts. Grading then runs on the copy, in fresh compose stacks on this server's Docker
daemon; the box the model controlled is never graded in place. A second `verify` of the same
session regrades the same copy. The harvest holds the session's lock, so a concurrent `verify` or a
`close` waits for it, and each `verify` grades its own copy outside the lock. A missing app directory, or one tar cannot read, is a build
failure. A provider that fails to run the harvest or transfer the archive is the environment's
failure, not the model's: the sample is masked with that failure kind, and the sandbox is kept so
a resent `verify` can harvest again (close stops it). Sessions are process-local, so the server
runs with one worker.

The sandbox provider has to support reattaching (`serialize()`/`connect()`), since the agent
borrows the sandbox from another process. OpenSandbox and Docker both do.

One row is one `(app, artifact)` pair. Reward is the mean normalized score
(`score / full_points`) across that artifact's test plans, with the per-plan values exposed
in `reward_components`. Every plan counts toward the denominator, so a plan that fails to
seed pulls the mean down instead of dropping out of it.

Reward is **continuous, not binary**. A ViBench test plan is a list of scored steps, and
partial credit is the signal the benchmark is built around.

## Setup

The resources server's host needs Docker and a vibench-public checkout, because grading runs
there. ViBench's grading scripts invoke the legacy `docker-compose` name, which Docker 29.x no
longer ships; without it seeding fails in seconds and reports a fully failed seeding rate,
which reads like a bad app rather than a missing binary. The build sandbox can run on any
provider that supports reattaching.

Since ViBench 2.0 the 1.0 materials live under `v1/` and the shared runner under `_harness/`.
`v2.0.7` is the tag this environment was checked against; a checkout from before ViBench 2.0
(`prds/` at the root) also works.

```bash
git clone --branch v2.0.7 https://github.com/ViBench/vibench-public.git ~/vibench
cd ~/vibench/v1 && uv sync && cp .env.template .env   # fill in the grader's provider keys
cd ~/vibench && docker build -f _harness/runner/docker/Dockerfile.base -t app-bench-base:latest .
```

`app-bench-base:latest` is the tag ViBench's 1.0 pipeline builds and reuses, so the build
sandbox and the grading stack share one base image. Its `WORKDIR` is `/app`, which is where the
builder lands; override `app_workdir` only alongside a different image. For a remote provider
such as OpenSandbox, push the image to a registry the provider can pull and set
`VIBENCH_BUILD_IMAGE` to it.

```bash
export VIBENCH_REPO_ROOT=~/vibench
export VIBENCH_ENV_FILE=~/vibench/v1/.env
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

That yields 24 tasks across 74 test plans. Row paths are relative to the checkout root
(`v1/prds/<app>/...`). `data/example.jsonl` holds five of them (`notes`, `quiz`, `barber`,
`wedding`, `market_place`).

P0 covers `mvp` artifacts only. `prepare.py` already resolves the PRD chain and test plans
for feature artifacts, but `seed_session` only stages a PRD into an empty `/app` — there is no
path yet to stage the existing codebase a feature task starts from.

`feature-ri` — building a feature on top of the reference implementation — is a follow-up
rather than a blocked one: vibench-public ships the starting trees at
`v1/results/<app>/RI_MVP/app/` for 21 of the 24 apps (not `barber`, `canary` or `quiz`), and
`seed_session` would upload one into the sandbox the way it uploads the PRD assets.
`feature-mvp` (`featureN-on_mvp`) is harder, because it starts from the model's own MVP output
and so depends on a prior rollout's artifact.

## Run

The second `--config` is required: `sandbox_provider: sandbox` is a reference that the
provider config binds, so without it startup fails with *"Sandbox provider reference
'sandbox' is not defined in the merged config"*. Swap that one path to move to another
provider without editing this config.

OpenCode in the build sandbox calls the policy model through Gym's model server, so the
sandbox must reach that server by address: run with `use_absolute_ip=true` and
`absolute_ip_server_types=[responses_api_models]`. Without the first, OpenCode is told the model
is on loopback, which inside the sandbox is the sandbox itself, so it makes **zero** LLM calls and
the build leaves an empty app. Without the second, every server listens on the host's address,
including the one whose config endpoint returns the merged config and this resources server, which
holds the grader's keys.

With OpenSandbox, start Gym on a host the sandboxes can route to:

```bash
gym env start \
    --config resources_servers/vibench/configs/vibench_sessions.yaml \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --model-type openai_model \
    ++use_absolute_ip=true \
    '++absolute_ip_server_types=[responses_api_models]'
```

With Docker, use this environment's `configs/docker.yaml` instead of the stock Docker provider
config. It sets both settings, so only the model server listens on the host's address, which
the default bridge can route to — and with it the run's token-capture path. Single-tenant hosts
only, or firewall the port; see the comment at the top of that file. The stock config's 180s
exec timeout would also kill the build, which OpenCode runs as one long exec. Do not use
`network: host`: that puts model-written code on the host network namespace.

```bash
gym env start \
    --config resources_servers/vibench/configs/vibench_sessions.yaml \
    --config resources_servers/vibench/configs/docker.yaml \
    --model-type openai_model
```

Then, in either case:

```bash
gym eval run --no-serve \
    --agent vibench_opencode_agent \
    --input resources_servers/vibench/data/example.jsonl \
    --output results/vibench_rollouts.jsonl \
    --limit 1 \
    --num-repeats 1
```

The recipe keeps the legacy relay path's `vibench_agent` settings where it can: OpenCode 1.17.11,
a 5,400-second build, no subagents. Two differ. The output budget is 32,768 tokens per model call
and the context window 229,376, so a call fits a 262,144-token served context; the legacy
recipe asks for the whole window on every call, which an OpenAI-compatible server such as vLLM
refuses, and OpenCode then ends the build with no model call at all. OpenCode also runs with
`opencode_agent`'s command deny-list (`configs/permissions/default.yaml`), which also denies Git
network commands, Git history mining and `curl`/`wget` to GitHub; the legacy recipe denies only
process killing and `rm -rf /`. The recipe loads that deny-list but not `opencode_agent.yaml`:
OpenCode 1.17.11 places the keys of a `tools` map, such as the one that file sets, before the
permission map's `"*"` rule, and the last matching rule wins, so with that map none of the denials
apply. Web, subagent and skill tools are turned off through `permission` instead.

Start with `--limit 1`. A single rollout builds an app and then runs a full compose stack
per test plan; wall-clock is tens of minutes and the box needs headroom for
`max_concurrent_test_plans` simultaneous Postgres + app + Playwright stacks.

## Legacy relay path

`responses_api_agents/vibench_agent` still runs ViBench behind the `legacy_agent` relay, from
`configs/vibench.yaml`, the default config. That agent starts its own build sandbox from the same image, runs
OpenCode in it with the legacy recipe above, and writes the app as a tarball into `artifact_dir`, a
directory it shares with this server; `seed_session` hands it only the PRD text and asset
directories, and `verify` unpacks the tarball and grades it the same way. The shared directory puts
the agent and this server on one host. With Docker, use the agent's own `configs/docker.yaml`; with
OpenSandbox, set `sandbox_model_base_url` on the agent. See
[its README](../../responses_api_agents/vibench_agent/README.md).

```bash
gym env start \
    --config resources_servers/vibench/configs/vibench.yaml \
    --config responses_api_agents/vibench_agent/configs/docker.yaml \
    --model-type openai_model
```

The two paths differ in output budget and command deny-list, so their scores are not directly
comparable. Making the sessions path the default, and removing `vibench_agent`, waits until both
have been graded on the same tasks.

## Validation

Run end to end on a Docker host with the legacy relay path's agent, `vibench_agent`, which owns the
build sandbox and hands the app over through a shared directory. Grading is unchanged since. All five
example tasks graded, 16/16 test plans, no build or seeding failures:

| app | reward | plans graded |
| --- | --- | --- |
| barber | 1.00 | 3/3 |
| market_place | 0.97 | 3/3 |
| notes | 0.93 | 3/3 |
| quiz | 0.67 | 3/3 |
| wedding | 0.36 | 4/4 |

Mean 0.785. The spread matters more than the mean: `wedding` well below the rest shows the
scale is not saturated. `data/example_rollouts.jsonl` holds this run, with its task paths
updated to the `v1/` layout.

The reward tracks model capability. An earlier run with a weaker policy model scored 0.0 on
`notes` with 0/19 steps, its app unreachable because the build produced no output, while the
stronger model scores near the top of the range on the same task with the same graders.

Reward varies across repeats of the same model, because the model does not build the same app
twice: `notes` and `wedding` have each spanned most of the range across runs. That is a
property to quantify during reward profiling rather than a defect, and profiling with repeats
is what would separate build-to-build variance from any variance in the LLM-driven verifier.

`verified: false` still stands: that flag means baselined and reviewed, which needs a
profiling sweep across many tasks and repeats, not five single rollouts.

## P0 limitations

These are known and deliberate; each is a follow-up rather than a bug.

- **Grading runs on the resources server's Docker daemon**, not inside a Gym sandbox,
  because ViBench's grading stack is multi-container (app + postgres + code-browse). Folding
  it into one supervisord image is the prerequisite for running this on more than one host.
- **Sessions are process-local**: the resources server runs with one worker and holds every
  episode's build sandbox until its verify or close.
- **The verifier is itself an LLM agent** driving a browser, so reward is stochastic.
  Profile that variance — repeated grading of one fixed app — before treating this as a
  training signal. `REVERIFY_MODE` is `UNSUPPORTED` for the same reason: scores cannot be
  recomputed from stored rollouts, since grading depends on live app and database state.
- **Grading has not been re-run on `v2.0.7`.** The grader's base image and runner scripts now
  come from that tag's `_harness/`; the results under Validation were graded on a checkout
  from before ViBench 2.0.
- **`mvp` artifacts only.** Feature tasks need sandbox staging of a starting codebase; see above.
- **Cost per rollout is high**: one coding agent, plus a seeding agent and an evaluation
  agent per test plan.

## Anti-cheat

The build sandbox receives the PRD and `v1/prds/<app>/assets/` (`prds/<app>/assets/` before
ViBench 2.0) only. Test plans and `test_assets/` are never staged into it — they are read at
grade time by the resources server. Dataset paths are resolved against `vibench_repo_root` and
rejected if they escape it. The harvested tarball is untrusted: unpacking refuses members and
links that escape the app directory.

## Licensing

ViBench is Apache 2.0. PRDs, test plans, and the runner harness come from the ViBench
repository; this server contains no ViBench data of its own — `prepare.py` reads a local
checkout.
