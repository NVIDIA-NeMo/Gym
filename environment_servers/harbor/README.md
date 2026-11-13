# Harbor Environment Server

The `harbor` Environment Server runs task folders in the [Harbor](https://github.com/harbor-framework/harbor) task
format: a folder with `task.toml`, `instruction.md`, `environment/`, `tests/test.sh` and an optional `solution/`.
Harbor is the input format; Gym owns execution. This page maps the code; the format, the configuration keys and
the troubleshooting steps live in the pages under [Documentation map](#documentation-map).

## What it does

One `/run` call is one whole episode, and no Resources Server takes part. For each row the server runs:

1. **Seed.** Check the row's digest against the taskset mapping and the folder, start the task's image as a
   sandbox through the configured sandbox provider (a sandbox group for a Compose task, or for any task when the
   job-level `compose_overlay` names `files`), run an overlay-mode Dockerfile's `RUN` lines, create the working
   directory, start the image's `ENTRYPOINT` and wait for the health checks.
2. **Agent.** Open a session on the configured Agent Server with a `SandboxAccess` for that sandbox, post the
   task's `responses_create_params`, then close the session and keep what the agent observed. A failed response
   whose harness names its cause goes through `agent_outcome_policy` before verification; by default a transport or
   unknown cause is retried ungraded. When the agent is `harbor_oracle_agent`, the seed first uploads the task's
   `solution/` to `/solution`; a `[[steps]]` task instead gets each step's `steps/<name>/solution/` (else the
   task's) before that step.
3. **Verify.** Run the `[[verifier.collect]]` hooks. In shared mode, empty `/tests` and `/logs/verifier` in the
   agent's sandbox, keeping only the task's `[verifier].inputs`, and upload `tests/`. In separate mode (also forced
   by `separate_verifier_for_artifacts` for a task with `artifacts`), collect `/logs/artifacts` and the task's
   `artifacts`, stop the agent's sandbox and restore them into a fresh one, which gets `tests/` only if its image
   has no `/tests/test.sh` or a Python verifier grades the task. Then write the agent's transcript to
   `/logs/agent/trajectory.json` (ATIF), run `test.sh` and read `reward.json`, else `reward.txt`.
4. **Cleanup.** Stop every sandbox of the episode, whatever happened before.

Once `test.sh` has run, the sample counts: a missing or invalid reward file scores 0 with a `failure_kind`. A Gym-side
failure (a lost sandbox, a failed transfer) sets `mask_sample`. So do a failed ATIF export, an image without `bash` or
`timeout`, an image `ENTRYPOINT` that exits non-zero during the rollout and an overlay or entrypoint that fails in a
separate verifier's sandbox. A task that cannot be set up fails at `stage: seed`: terminally when the server refuses
the task, row or configuration (such as a failed overlay `RUN` line or `ENTRYPOINT`), and retryably otherwise, even
for a health check that never passes or an image that cannot be pulled. A task with `[[steps]]` runs one such turn
per step in the same sandbox. It stops early after a step whose preparation fails, or one that is masked, leaves no
valid reward or misses its `min_reward`.

## How it fits

[`fern/versions/latest/pages/about/architecture.mdx`](../../fern/versions/latest/pages/about/architecture.mdx)
describes the usual split: an Environment Server orders the episode, and for JSONL rows a Resources Server holds the
tools and the verifier. Task folders are the second way to define a task: the folder is the whole task, verifier
included, so this server seeds the sandbox from the folder and runs the task's own verifier itself. `gym eval run`
and `gym dataset validate` load the folders with `nemo_gym.data.tasks.harbor`, write one row per task and a run
configuration with a per-taskset instance of this server, and bind it to a harness: the one `gym eval run` names
with `--agent`, or `harbor_oracle_agent` for `gym dataset validate`. The harness must accept a sandbox in its
session seed and leave the sandbox to the server when the session closes. A task with a `skills_dir`, or with MCP
servers not granted through `task_mcp_grants: required`, also needs a harness that reports
`supports_task_tool_context`, or it fails terminally at `stage: agent`; `turn_sessions: continue` needs one that
declares multi-turn continuation. `harbor_agent` and `harbor_agent_general` are separate Agent Servers that run
Harbor's own runtime; they are not this path.

## Run it

The committed entry `environments/harbor/terminal_bench_2_1/` pins Terminal-Bench 2.1 in Harbor's package store.
Pull it once, check the reference solutions with the oracle, then run a model:

```bash
gym dataset pull terminal_bench_2_1
gym dataset validate terminal_bench_2_1 --sandbox opensandbox
gym eval run terminal_bench_2_1 --agent terminus_2_sandboxed_agent --sandbox opensandbox \
  --model-type openai_model --model <model> --model-url <url> --model-api-key <key>
```

`gym dataset validate` plays each task's `solution/` through `harbor_oracle_agent` and exits non-zero unless every
selected task loads and every solution scores 1; a task without `solution/` is listed as unvalidated and does not fail
the command, unless no selected task has one. Two Terminal-Bench 2.1 tasks fail their own reference solutions as
published, so the full set exits non-zero and names them; see
[`fern/versions/latest/pages/evaluation-tutorials/terminal-bench-2-1-harbor.mdx`](../../fern/versions/latest/pages/evaluation-tutorials/terminal-bench-2-1-harbor.mdx).
`--limit N` or `--only-tasks <names>` narrows either command. A task folder, a folder of task folders or a `harbor:`
reference also works as the target.

## Configuration

The server block is [`configs/harbor.yaml`](configs/harbor.yaml), instance `harbor_environment_server`; every key,
its default and the sandbox provider setup are documented in
[`fern/versions/latest/pages/agent-interactions/harbor.mdx`](../../fern/versions/latest/pages/agent-interactions/harbor.mdx)
under "Configure the server". Dockerfiles that need a build are built at prepare and configured by the top-level
`builds:` block of the merged configuration (`env.yaml`, a `--config` file or `++builds.<setting>=<value>`), not by
this server's block. Its settings are `tool` (buildah, podman or docker; buildah and podman need a `repository`;
default: the first usable one on `PATH`), `repository` (default `$NEMO_GYM_BUILD_REPOSITORY`), `platform` (default
`linux/amd64`), `enabled` and `force`; they are documented in
[`fern/versions/latest/pages/reference/configuration.mdx`](../../fern/versions/latest/pages/reference/configuration.mdx)
under "Harbor image builds". The job-level `compose_overlay` and `compose_variables` are documented in
[`fern/versions/latest/pages/data/task-folders/compose-tasks.mdx`](../../fern/versions/latest/pages/data/task-folders/compose-tasks.mdx).

## Test it

The server's tests and the oracle agent's tests use fakes, so they need no sandbox provider or model:

```bash
pytest environment_servers/harbor/tests responses_api_agents/harbor_oracle_agent/tests
pytest tests/unit_tests/test_harbor_*.py tests/unit_tests/test_tb4_harbor_sandbox_overlay.py
```

Both lines need Jinja2. The first needs the server's [`requirements.txt`](requirements.txt) installed
(`nemo-gym[dev,sandbox]` and Jinja2). The second covers `nemo_gym/data/tasks/harbor/` and runs with the rest of
`tests/unit_tests/`; its Compose overlay prompt-template tests fail without Jinja2, which comes with the `ray` extra
(`uv sync --extra dev --extra ray`, as CI installs). To run the server's suite in its own venv, as the CI server
suite does:

```bash
gym env test +entrypoint=environment_servers/harbor
```

Unit tests are not enough for a behavior change: also run
`gym dataset validate environments/hello_world --sandbox docker`, then a model run.

## Code map

`environment_servers/harbor/`:

| Module | Role |
| --- | --- |
| `app.py` | The server, its config model and `/run`: seed, agent, verify and cleanup, multi-step roll-up |
| `sandboxes.py` | Provision the agent's sandbox, a Compose group or a separate verifier's sandbox |
| `session.py` | The per-episode session and the workspace the agent and the verifier share |
| `overlay.py` | Apply an overlay-mode Dockerfile's `RUN` lines at seed |
| `entrypoint.py` | Run the image `ENTRYPOINT` beside the provider's keepalive and gate the seed on health checks |
| `users.py` | Which user an exec runs as: the image's `USER`, `[agent].user`, root on every provider |
| `steps.py` | Prepare a `[[steps]]` turn: its `workdir/`, `setup.sh` and health check |
| `verifier.py` | The `test.sh` wrapper, the reward files and the verifier sandbox |
| `network_policy.py` | A `[verifier]` `network_mode` and `allowed_hosts` as an OpenSandbox egress policy |
| `task_data.py` | The row's task-data schema: the reserved `_ng_digest` and `_ng_dataset_digest` keys |
| `configs/harbor.yaml` | The shipped server block |
| `data/` | The example tasks and their recorded rows, rollouts and metrics |

`nemo_gym/data/tasks/harbor/` (loading and prepare; `gym eval run` and `gym dataset validate` on a task target
live in `nemo_gym/cli/eval.py`):

| Module | Role |
| --- | --- |
| `prepare.py` | Find, pull (`gym dataset pull`) and prepare a taskset, and build its run config |
| `task.py` | Load one task folder, or a folder of task folders |
| `models.py` | Pydantic mirror of Harbor's `task.toml` |
| `harbor_compat.py` | Environment hashing and image naming vendored from Harbor v0.22.0 |
| `dataset_config.py` | The `[gym]` table of `dataset.toml` |
| `task_data.py` | A dataset's `task_data.py` model for typed task fields |
| `missing_reward.py` | A dataset's `missing_reward.py` classifier of a verifier run that wrote no reward |
| `parameterized.py` | Render a template folder plus `tasks.jsonl` into task folders |
| `python_verifier.py` | Python verifiers and the `test.sh` shim that runs them |
| `scaffold.py` | `gym dataset init` |
| `publish.py` | `gym dataset publish --harbor` |
| `materialize.py` | Turn loaded tasks into rows and the generated run config |
| `digest.py` | The whole-folder task digest |
| `manifest.py` | `manifest.toml`, a dataset folder's derived identity |
| `entries.py` | Committed entries under `environments/harbor/` and where a pull lands |
| `hub.py` | Resolve `harbor:<dataset>[@<version>]` through the Harbor registry |
| `package_store.py` | Fetch `harbor:<org>/<name>[@<tag> \| @sha256:<digest>]` from the package store |
| `dockerfile.py` | Recognize pull-mode and overlay-mode Dockerfiles and resolve them |
| `verifier_env.py` | Harbor's separate-verifier rules |
| `registry.py` | Read an image's configuration from its OCI registry |
| `image_configs.py` | Record the OCI configuration of the images a task runs |
| `records.py` | `built-images.json` and `compose-images.json` |
| `build_options.py` | The `builds:` block |
| `build_plan.py` | Plan the image builds of a task folder |
| `build_compose.py` | The task's Compose file as `docker compose build` reads it |
| `build_tools.py` | Drive buildah, podman or docker for one build |
| `build_records.py` | Decide when a recorded image can be reused |
| `image_build.py` | Build the images an overlay cannot express |
| `prepare_builds.py` | The build step of prepare and its Compose field checks |
| `compose_variables.py` | Compose interpolation for task folders |
| `compose_overlay.py` | The job-level Compose overlay |
| `run_summary.py` | The per-run `<rollouts stem>_run_summary.json` |
| `validation_report.py` | The `gym dataset validate` report |
| `errors.py` | `HarborTaskError` |

`responses_api_agents/harbor_oracle_agent/` is the harness `gym dataset validate` uses: it runs the task's
reference solution in the sandbox instead of calling a model.

## Example data

`data/tasks/` contains five small Apache-2.0 terminal tasks: copying a file, numeric sorting,
deduplicating lines, uppercasing text, and selecting the first line. Each uses `ubuntu:24.04`, a
60-second agent limit, and a 30-second verifier limit. The verifier compares the output byte for
byte with a fixture held outside the agent workdir.

`data/example.jsonl` contains their materialized rows. `data/example_rollouts.jsonl` records actual
Qwen2.5-0.5B-Instruct commands, execution results, and verifier outputs; `data/example_metrics.json`
records their measured scores and runtime. These are smoke examples from a single-command harness,
not a Hermes benchmark. They used llama.cpp on CPU and Enroot sandboxes with an in-process handoff
because Enroot cannot serialize sandbox access. Provider expiration and the distributed handoff
were not exercised by these examples.

To run the bundled tasks with Hermes and a configured sandbox provider and model:

```bash
gym eval run environment_servers/harbor/data/tasks --agent hermes_agent --sandbox opensandbox \
  --model-type openai_model --model <model> --model-url <url> --model-api-key <key>
```

The loader generates the task mapping and dataset rows for that run.

## Documentation map

All under `fern/versions/latest/pages/`:

| Page | What it covers |
| --- | --- |
| [`data/task-folders/harbor-tasks.mdx`](../../fern/versions/latest/pages/data/task-folders/harbor-tasks.mdx) | Tutorial: write, validate, run and publish task folders; coming from Harbor |
| [`data/task-folders/task-folder-format.mdx`](../../fern/versions/latest/pages/data/task-folders/task-folder-format.mdx) | The canonical spec: layout, `task.toml`, `dataset.toml`, Dockerfile modes, Harbor compatibility, what the loader refuses |
| [`data/task-folders/task-folder-verifiers.mdx`](../../fern/versions/latest/pages/data/task-folders/task-folder-verifiers.mdx) | The verifier contract: reward files, Python verifiers, reference answers, sidecar inputs, shared and separate verification |
| [`data/task-folders/multi-step-tasks.mdx`](../../fern/versions/latest/pages/data/task-folders/multi-step-tasks.mdx) | `[[steps]]` tasks, the reward roll-up and `turn_sessions: continue` |
| [`data/task-folders/compose-tasks.mdx`](../../fern/versions/latest/pages/data/task-folders/compose-tasks.mdx) | Compose tasks, OpenSandbox networking, template variables, volumes and the job-level overlay |
| [`data/task-folders/task-folder-patterns.mdx`](../../fern/versions/latest/pages/data/task-folders/task-folder-patterns.mdx) | Parameterized tasks, typed task fields, task MCP tools, shared verifier code |
| [`agent-interactions/harbor.mdx`](../../fern/versions/latest/pages/agent-interactions/harbor.mdx) | The episode, the harnesses, the server's keys and what it writes |
| [`reference/task-folder-storage.mdx`](../../fern/versions/latest/pages/reference/task-folder-storage.mdx) | Pulled datasets, image records, the run summary, digests and what is safe to delete |
| [`reference/configuration.mdx`](../../fern/versions/latest/pages/reference/configuration.mdx) | The `builds:` block and the Harbor environment variables |
| [`troubleshooting/task-folders.mdx`](../../fern/versions/latest/pages/troubleshooting/task-folders.mdx) | Symptoms and fixes: registries, builds, timeouts, rewards, changed folders |
