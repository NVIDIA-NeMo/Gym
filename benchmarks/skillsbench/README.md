# Official SkillsBench tasks in NeMo Gym

This integration reads the default runnable tasks from
[`benchflow-ai/skillsbench`](https://github.com/benchflow-ai/skillsbench). It does
not use the separate `skillsbench-nemogym.zip` Harbor task package.

The official repository stores tasks as BenchFlow `task.md` packages. Both preparation scripts map each package to
Harbor's `instruction.md`, `task.toml`, `environment/`, `tests/`, and `solution/` layout. They preserve the task
prompt, verifier files, Docker build context, timeouts, CPU/memory/storage limits, and per-task network policy.

There are two ways to run it:

- **On the `harbor` environment server (default).** Each skill condition is a Harbor dataset entry,
  [`skillsbench`](../../environments/harbor/skillsbench/README.md) and
  [`skillsbench_no_skill`](../../environments/harbor/skillsbench_no_skill/README.md), that `gym eval run` resolves
  by name. The [`harbor` environment server](../../environment_servers/harbor/README.md) owns
  the sandbox, the skills and the verifier; any harness that accepts a borrowed task sandbox and declares
  `supports_task_tool_context` works in it, at this revision `miniswe_sandboxed_agent`.
- **OpenHands through `harbor_agent_general`** (`skillsbench/harbor`, `skillsbench/harbor_opensandbox`). A whole
  Harbor job with an OpenHands SDK agent runs behind a `legacy_agent` relay. See
  [OpenHands through `harbor_agent_general`](#openhands-through-harbor_agent_general).

Scores from the two paths are not comparable: OpenHands loads skills into its `invoke_skill` tool, while the
environment server gives the model the folder and its shell.

## Run on the `harbor` environment server

### Prepare the datasets

From the Gym repository root:

```bash
python -m benchmarks.skillsbench.prepare_harbor_env
```

Preparation sparse-clones the official `tasks/` tree at the pinned revision
`12ebd8653093e0734c7902835cb20c054196026c` (87 default tasks) into `benchmarks/skillsbench/data/source`, the same
checkout the OpenHands path uses, and reuses it later. It writes each condition's task folders into its entry under
`environments/harbor/`, beside the entry's committed `README.md` and `dataset.toml`; the task folders are ignored by
Git, and a new preparation replaces them:

| Condition | Dataset entry | Skills |
| --- | --- | --- |
| with-skill | `environments/harbor/skillsbench` | `task.toml` declares `skills_dir = "/skills"`; the environment server uploads the task's `environment/skills/` there at seed |
| no-skill | `environments/harbor/skillsbench_no_skill` | no `skills_dir` and no `environment/skills/`; a task image that builds skills into the standard agent skill locations has them removed before the agent starts |

`SKILLSBENCH_SKILL_MODE=with-skill` or `no-skill` prepares one condition only. Each task declares the official
published `linux/amd64` image for the pinned revision, digest-pinned in `benchmarks/skillsbench/images.json`, so
no image is built. Both conditions share those images: in the no-skill condition a task whose Dockerfile copies
skills into the image runs that image with an overlay `RUN` that deletes the nine agent skill locations listed in
`AGENT_SKILL_PATHS` in `prepare_harbor_env.py` (`/root/.agents/skills`, `/root/.claude/skills`, `/root/.codex/skills`
and the like), which the server applies at seed. A copy to any other path stays in the image. At the pinned revision
one task does this, `radar-vital-signs`, and all seven of its copies land in that list.

To use another checkout, set `SKILLSBENCH_TASKS_DIR` to its `tasks/` directory and record its commit with any
reported score. A custom checkout builds every task from its Dockerfile (see "Dockerfile builds" in the
environment server's README) unless `SKILLSBENCH_IMAGE_MANIFEST` names a manifest of its published images (see
the image builder below). `SKILLSBENCH_IMAGE_MANIFEST=none` builds the pinned checkout's tasks from their
Dockerfiles too.

### Run

```bash
gym eval run skillsbench --agent miniswe_sandboxed_agent --sandbox opensandbox \
  --skip-unloadable --exclude-tasks fix-visual-stability --model-type openai_model --model <model> --model-url <url>
gym eval run skillsbench_no_skill --agent miniswe_sandboxed_agent --sandbox opensandbox \
  --skip-unloadable --exclude-tasks fix-visual-stability --model-type openai_model --model <model> --model-url <url>
```

`--skip-unloadable` runs the tasks the environment server loads and records the ones it refuses in
`skipped.json`; `--exclude-tasks fix-visual-stability` leaves out a task whose Compose service has to be built (see below). On a host without an
image builder that is 83 of the 87 tasks. Add `--only-tasks <name> --concurrency 1` for a first check.
`gym dataset validate skillsbench --sandbox opensandbox --exclude-tasks fix-visual-stability` plays every reference solution through the same
server, leaving out the tasks it cannot load. Eight official reference solutions import scripts from a task skill.
Four of them (`3d-scan-calc`, `exceltable-in-ppt`, `pptx-reference-formatting`, `video-silence-remover`) search
`/skills`; the other four (`jax-computing-basics`, `pddl-airport-planning`, `pddl-tpp-planning`, `travel-planning`)
search only `/app/skills`, where BenchFlow's oracle mode puts skills, so the with-skill oracle wrapper links
`/app/skills` to `/skills` while `solve.sh` runs and removes the link afterwards. The agent and the verifier see one
skills folder. All eight fail validation in the no-skill condition, which has no skills at all. Set
`OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` for the OpenSandbox provider. A sandboxed harness calls Gym's model server from inside the
task sandbox, so run Gym where the sandboxes can reach it (`gym eval run` advertises the model server's
absolute address for such a harness).

Rows, results and verifier logs are labelled by the entry's name (`skillsbench` or
`skillsbench_no_skill`), and the run summary beside the rollouts records both dataset digests.

### Skills and harnesses

Skills stay plain files: in the with-skill condition the environment server places each task's skills at
`/skills` (the location Harbor's own skills example uses, outside every task's working directory) and writes
the path to `/tmp/.nemo-gym/task.json` in the sandbox. A harness that declares `supports_task_tool_context` reads it
there and points its model at the folder, so loading and the tools the model uses do not depend on the harness. A
task with `skills_dir` is refused to a harness that does not declare it, rather than run without its skills; at
this revision `miniswe_sandboxed_agent` declares it and tells the model where the skills are. The no-skill
condition declares no `skills_dir` and runs on any sandboxed harness.

Three tasks' skills (`3d-scan-calc`, `lean4-proof`, `video-silence-remover`) name their own scripts under
`/root/.claude/skills/`; the skills reach the model under `/skills/`, so the model must follow the
folder it was given, as with any non-Claude harness in the official runner.

### Differences from the official runner

- One task (`bike-rebalance` at the pinned revision) requests `no-network`. The environment server does not
  restrict an agent sandbox's network yet, so its loader refuses that task instead of running it with open
  egress; `--skip-unloadable` leaves it out of a run and names it in `skipped.json`.
- Three tasks ship Compose files written for Docker whose services are built from the task's own Dockerfiles
  (`drone-planning-control`, `fix-visual-stability`, `react-performance-debugging`); the published images cover only
  each task's main image. Preparation keeps the files as they are. Without an image builder (see "Dockerfile builds"
  in the environment server's README) the server refuses `drone-planning-control` and `react-performance-debugging`,
  which `--skip-unloadable` leaves out and names in `skipped.json`, while `fix-visual-stability`'s build fails the
  run's preparation outright, hence `--exclude-tasks fix-visual-stability`. With a builder configured the server
  builds those services; that setup is untested here.
- The harness and its prompt are Gym's, not one of the official repository's published agent configurations.
  Record the harness, its version and settings with any reported score.

## OpenHands through `harbor_agent_general`

Like Terminal-Bench and SWE-bench, this path registers a prepared JSONL,
config, and manifest for `gym eval run --benchmark`. Its execution path follows
the `harbor_agent_general` integration: SkillsBench supplies Dockerfiles
for each task, while Terminal-Bench's native resources server expects an
already-built `docker_image` in each prepared row.

### Prepare the task source

From the Gym repository root:

```bash
gym eval prepare --benchmark skillsbench/harbor
```

Preparation (`benchmarks/skillsbench/prepare.py`) clones the same pinned
revision (matching the official `skillsbench-v1.1-*-12ebd8653093` task images)
into `benchmarks/skillsbench/data/source`, and reuses that checkout later. To use
another checkout, set `SKILLSBENCH_TASKS_DIR` to its `tasks/` directory and
record its commit with any reported score. Converted tasks (`data/tasks`) and the
index (`data/skillsbench_official.jsonl`) are generated under
`benchmarks/skillsbench/data/`, which is ignored by Git, beside the environment
server's dataset folders.

`SKILLSBENCH_SKILL_MODE` defaults to `with-skill`. For local Docker builds,
preparation adds skill-installation lines to each task Dockerfile. With the
official prebuilt images on OpenSandbox, the SkillsBench environment uploads
each task's `environment/skills/` at sandbox startup to the standard agent
skill locations. Set `SKILLSBENCH_SKILL_MODE=no-skill` to skip installation.
Prepare and run the conditions separately; preparation replaces the previous
generated task tree.

Background processes remain available between terminal actions. If an agent or
terminal command reaches its deadline after a terminal action has started, the
sandbox is discarded before verification. The adapters cannot confirm complete
process ownership across detached sessions: terminating a wrapper or its process
group is not sufficient, even for an action that appears to be foreground-only.
Cancellation before any terminal action starts can retain the sandbox. Discarded
sandboxes must not be treated as verified task scores.

Per-rollout `temperature`, `top_p`, and `max_output_tokens` overrides reach
OpenHands; omitted values retain the profile defaults. These overrides do not
change the configured input-context limit.

### Run on Slurm through OpenSandbox

Use `benchmarks/skillsbench/harbor_opensandbox.yaml` with `gym eval submit`.
The general Harbor adapter runs OpenHands in a Ray worker alongside Gym's model
server; terminal commands execute in OpenSandbox, so the worker needs no Docker
daemon. Set `OPENSANDBOX_DOMAIN` and `OPENSANDBOX_API_KEY` in the driver job.
Preparation installs Ray in the driver environment, as required by the general
adapter. Set `skillsbench_max_input_tokens` to match the policy model context
limit; this avoids indexing the native Harbor agent list in flat launcher overrides.
The SkillsBench entrypoint resolves the named provider and passes credentials
only to the worker, outside Harbor's retained job and trial configurations.

OpenSandbox needs pullable task images. The official SkillsBench repository
publishes a complete 87-task `linux/amd64` image set to
`ghcr.io/benchflow-ai/skillsbench-task-env`. Gym ships its digest-pinned
manifest as `benchmarks/skillsbench/images.json`: set
`SKILLSBENCH_IMAGE_MANIFEST=benchmarks/skillsbench/images.json` in the driver
job (a relative path is read from the Gym checkout). For a source revision
without published images, use Gym's image builder on a Docker machine:

```bash
python responses_api_agents/harbor_agent/custom_envs/singularity/scripts/build_and_push_images.py \
  --input benchmarks/skillsbench/data/tasks \
  --registry <registry>/<namespace> \
  --image-tag <source-commit-and-skill-mode> \
  --manifest-out /shared/skillsbench/images.json
```

Then export `SKILLSBENCH_IMAGE_MANIFEST=/shared/skillsbench/images.json` and set
`SKILLSBENCH_TASKS_DIR` to the matching source checkout. `gym eval prepare`
writes each image reference into the generated Harbor task, and the manifest
must cover every task. Both skill modes use the same images; the OpenSandbox
environment controls skill installation.

One official task requests `no-network`. Gym's OpenSandbox Harbor adapter does
not enforce network isolation, so this config explicitly permits its execution
with cluster-default egress. That condition differs from the official sandbox
policy and must be resolved before claiming comparable scores.

### Run locally

Docker and an OpenAI-compatible policy endpoint are required. Run one task first:

```bash
gym eval run \
  --benchmark skillsbench/harbor \
  --model-type openai_model \
  --split benchmark --limit 1 --concurrency 1 \
  --model "$POLICY_MODEL" \
  --model-url "$POLICY_BASE_URL" \
  --model-api-key "$POLICY_API_KEY"
```

Remove `--limit 1` to run the whole default set, and size `--concurrency` for
the machine. All tasks request one CPU and 4 GiB; some Docker builds are large.
The config leaves `workdir` unset so each Docker image supplies its own.

This uses OpenHands SDK 1.54.0. The agent runs in a Ray worker beside the
Gym model server, and its terminal tool executes commands in each
OpenSandbox task image. The with-skill condition loads the task's skills into
OpenHands' `invoke_skill` catalog and installs their files in the sandbox. The
no-skill condition clears baked-in agent skill paths. This harness differs from
the official repository's published agent configurations, so record its version
and settings with any reported score. The converter keeps the official prompts
and task files, adding skill installation to local Dockerfiles and mount-path
wrappers for Harbor's verifier and oracle.

The prepared rows use `task_name`, as required by `harbor_agent_general`.
Older SkillsBench configurations using `harbor_agent`, `instance_id`, or
`harbor_*` forwarding fields must be re-prepared and migrated to the native
Harbor `job` configuration shown here. Existing benchmark scores should retain
the adapter version with their provenance; this migration is not a new baseline.

## Runtime recovery and qualification

OpenHands uses its native LLM summarizing condenser with the same model and
rollout capture URL as task execution. Self-hosted model aliases retain
interleaved reasoning, and seeded repeats forward their seed to the SDK.
`skillsbench_max_output_tokens` controls the per-request output limit (default
32768); `skillsbench_llm_timeout_s` retains the existing 3600-second request
budget. The task's official agent deadline and resource limits do not change.

Terminal observations carry completion status and the current directory in the
SDK metadata. Terminal action timeouts are soft waits: the command keeps running,
returns status -1, and an empty terminal action polls its eventual output. A new
foreground command is refused until that result is collected. Output is buffered
until completion; interactive key input is unavailable. The official task deadline
still deletes the sandbox and joins the worker, including a command left pending
when the conversation finishes. Provider timeouts remain infrastructure failures.
A removed current directory is a recoverable tool error. A sandbox
transport failure stops the conversation and disposes the sandbox; it cannot be
converted into an agent-deadline score. Cancellation also stops further model
retries and joins the SDK worker before returning.

OpenSandbox uses process-level OOM killing and checks `memory.oom.group=0` at
startup so an agent command cannot take down the exec daemon with it. Task CPU,
memory, disk, and time limits stay official. A typed provider OOM report
does not establish task-limit versus node-pressure attribution at this Harbor
boundary: it stays unscored and terminal for review. The provider retains its
shared failure contract and trusted current-container evidence checks. Disk
exhaustion, node eviction, and transport errors remain infrastructure failures
unless Harbor reports a normal graded outcome; text in a command-status HTTP
410 body cannot establish a model fault. An unknown command exit (-1) discards
the sandbox before verification. A narrowly recognized transient Maven transfer failure in a failed
verifier assertion is also infrastructure failure; its original result is kept
as `result.failed.json`.

The OpenSandbox environment records the native verifier command's actual exit
on the host, outside sandbox-downloadable logs. Only a record inside the trial's
completed verification phase supplies test-execution evidence; rewards and genuine
nonzero command exits are preserved. Historical runs without this record remain
unverified. SDK action/message response IDs identify policy calls for trajectory
collection, while the wrapped LLM is explicitly registered with OpenHands' token
accounting. This registration preserves the SDK's built-in prompt and condenser.

Qualification must retain the resolved config, aggregate metrics, rollout health
summary, and evidence-gate report from the exact candidate revision. Historical
runs of a private branch are supporting regression evidence, not certificates
for a new pin. OpenHands/Harbor is not registered in the P0 conformance registry;
the evidence-gate result is informational and any gaps still require explicit
acceptance by the benchmark owner.
