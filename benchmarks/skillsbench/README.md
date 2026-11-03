# Official SkillsBench tasks in NeMo Gym

This integration reads the default runnable tasks from
[`benchflow-ai/skillsbench`](https://github.com/benchflow-ai/skillsbench). It does
not use the separate `skillsbench-nemogym.zip` Harbor task package.

The official repository stores tasks as BenchFlow `task.md` packages. The
preparation script maps each package to Harbor's `instruction.md`, `task.toml`,
`environment/`, `tests/`, and `solution/` layout, then writes the Gym JSONL
index. It preserves the task prompt, verifier files, Docker build context,
timeouts, CPU/memory/storage limits, and per-task network policy.

Like Terminal-Bench and SWE-bench, this benchmark registers a prepared JSONL,
config, and manifest for `gym eval run --benchmark`. Its execution path follows
the `harbor_agent_general` integration: SkillsBench supplies Dockerfiles
for each task, while Terminal-Bench's native resources server expects an
already-built `docker_image` in each prepared row.

## Prepare the task source

From the Gym repository root:

```bash
gym eval prepare --benchmark skillsbench/harbor
```

Preparation sparse-clones the official `tasks/` tree at the pinned revision
`12ebd8653093e0734c7902835cb20c054196026c` (87 default tasks, matching the
official `skillsbench-v1.1-*-12ebd8653093` task images) into
`benchmarks/skillsbench/data/source`, and reuses that checkout later. To use
another checkout, set `SKILLSBENCH_TASKS_DIR` to its `tasks/` directory and
record its commit with any reported score. Converted tasks and the index are
generated under `benchmarks/skillsbench/data/`, which is ignored by Git.

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

## Run on Slurm through OpenSandbox

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

## Run locally

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
