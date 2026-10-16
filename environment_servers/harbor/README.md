# Harbor tasks

Runs tasks written in the [Harbor](https://github.com/harbor-framework/harbor) task format: a folder with
`task.toml`, `instruction.md`, `environment/`, `tests/test.sh` and an optional `solution/`. Harbor is the input
format; Gym owns execution. The supported subset is listed below.

## What the server does

This is an environment server: one `/run` call owns one whole episode, and no resources server takes part.

- Seed: checks the row's digest against the taskset mapping, snapshots and checks the task files once per
  digest, starts the task image as a sandbox through the configured provider, and creates the working
  directory. Concurrent episodes share the snapshot. Later source edits cannot change an already cached task;
  re-materialize and restart the server to use edited tasks.
- Agent: opens a session on the configured agent server with a `SandboxAccess` for that sandbox (the same
  seed request every sandboxed harness already accepts), posts the task's `responses_create_params` to it,
  then closes the session and keeps the harness's observations for the result.
- Verify: clears `/tests` and `/logs/verifier`, uploads the snapshot's `tests/`, runs `bash /tests/test.sh`
  in the same sandbox, and downloads the verifier logs.
- Cleanup: stops the sandbox after the episode, whatever happened before. Cancellation and the episode
  deadline also reach this cleanup; a failed stop keeps the sandbox owned by the server for a later retry
  or shutdown, and a redelivered episode cannot provision while an owned sandbox of the same episode remains.

A task that cannot be set up is a handled episode failure with `stage: seed`: terminal when the task or row
is at fault (unknown task, changed folder, unsupported settings), retryable when the sandbox provider failed.
A harness that cannot be seeded, fails its turn or fails to close is `stage: agent` or `stage: cleanup`, with
its observations and any usable response attached. Verification never runs before the harness has let go.

When `test.sh` ran, the sample counts. A missing or invalid reward file scores 0 with a `failure_kind`.
Only a required infrastructure failure (sandbox lost, reward transfer failed) sets `mask_sample`.
Diagnostic logs are best effort: unsafe archive entries are skipped securely, and a diagnostic
transfer failure cannot discard an already measured reward. Verifier timeouts score zero and retain
available logs. An existing `reward.json`
takes precedence even if malformed; `reward.txt` is read only when JSON is absent. Boolean JSON rewards
are numeric, as in Harbor. Gym selects the `reward` key or the only key; multiple keys without `reward`
are an authoring error (a terminal `stage: verification` failure) because Gym requires one scalar score.

## Supported task behavior

Supported today: single-step tasks with a prebuilt `docker_image`, a pull-mode Dockerfile (`FROM` plus
`WORKDIR`/`ENV`/`USER`/`LABEL`) or an overlay-mode Dockerfile (the same plus `RUN` lines, which the server
runs in the pulled base image at seed, before the healthcheck and the agent, each with the `ENV`, `WORKDIR`
and `USER` in effect at that line; `ENV`, `WORKDIR` and `USER` resolve at prepare from the base image's
recorded configuration the way `docker build` resolves them, so preparing fails when the registry cannot
resolve the base image. A `RUN` line that exits non-zero or outruns `[environment].build_timeout_sec` is a
422 with the command and the tail of its output; a provider that cannot run one is a 503, and in a separate
verifier's sandbox either masks the sample with `harbor:overlay_failed`); shared verifier mode, and
separate verifier mode when
`[verifier.environment]` names a prebuilt image (the agent's `/logs/artifacts` and `artifacts` entries are
copied into the verifier sandbox first, sidecar hooks and artifacts included). Compose environments start as
a sandbox group when a `compose-images.json` with the sidecar images' recorded OCI configuration sits next to
the task folders, written by `gym dataset fetch`; on OpenSandbox the provider block also needs
`networking.enabled: true` (with `loopback_forwarding`) and `runtime_requirements.shm_size_metadata_key`, as
`benchmarks/terminal_bench_4/resources.yaml` shows. `[environment.healthcheck]` is polled at seed.
Tasks that declare GPUs run on the `gpu_sandbox_provider` block when one is set. The task's
`[[environment.mcp_servers]]` and `skills_dir` are written to `/tmp/.nemo-gym/task.json` in the agent's
sandbox for harnesses that drive task MCP servers from inside it (mini-SWE does). Dockerfiles outside pull
and overlay mode are built at prepare, see below. Multi-step tasks come in a later milestone.

## Dockerfile builds

Prepare decides one of three image modes per distinct Dockerfile.

- Pull. A base-image-only file (`FROM` plus `WORKDIR`, `ENV`, `USER`, `LABEL`). The sandbox starts from the
  base image.
- Overlay. A pullable `FROM` plus `RUN` lines. The server runs the lines in the pulled base image at seed.
  No build and no registry.
- Build. Everything else: `COPY`, `ADD`, heredocs, BuildKit flags, multi-stage files. The task's
  `environment/` folder is the build context. Prepare builds it once, records it, and the task then runs in
  pull mode from the built image. A `build:` or `pull_policy` the task's Compose file gives `main` (build args,
  a target, a `dockerfile_inline`) applies to that build; at seed the recorded image stands in for them, so the
  server drops the two keys from `main` before the Compose adapter, which starts images only, sees the
  document. A task in pull or overlay mode keeps whatever its file says.

The image is named the way Harbor names it, `hb__<hash>`, where the hash is Harbor's 32-character content
hash of the `environment/` folder, its environment id (vendored from Harbor v0.22.0 in
`nemo_gym/tasks/harbor/harbor_compat.py`), so identical contexts get the same image name in Harbor and in
Gym. Builds are configured by a top-level `builds:` block in the merged configuration (`env.yaml`, a
`--config` file, or `++builds.<setting>=<value>` on the command line; explicit settings apply over the
defaults below):

```yaml
builds:
  tool: buildah            # buildah, podman or docker (docker compose build); default: the first on PATH that fits
  repository: registry/path  # push built images under this path; default: $NEMO_GYM_BUILD_REPOSITORY
  platform: linux/amd64    # the platform asked of the tool; default linux/amd64
  enabled: true            # false never builds; tasks that need a build are skipped with a message
```

With `builds.repository` (or the `NEMO_GYM_BUILD_REPOSITORY` variable) the image is named
`registry/path/hb__<hash>` and pushed, which is what OpenSandbox needs. The build then runs with the first
of `buildah`, `podman` and `docker` found on PATH, or the tool named by `builds.tool`.
buildah runs rootless, with the `vfs` storage driver and `chroot` isolation, under a node-local storage
root (`/raid/scratch/<user>/nemo-gym-buildah` when that disk exists, else the system temp folder). Without
a repository the image stays local as `hb__<hash>`, and only `docker` can build it: the Docker
daemon is the one local store a sandbox provider starts images from, so buildah and podman are refused
without a repository. That is the laptop path with the Docker provider.

buildah and podman run with a `containers.conf` override. The host's containers configuration may give every
container a default bind mount (this deployment's `/etc/containers/containers.conf.d/30-mounts.conf` mounts
`/etc/localtime` read-only into each one), and both tools apply those mounts to build steps too, so any
Dockerfile that installs `tzdata` (Ubuntu with `python3` pulls it in) fails with
`mv: cannot move '/etc/localtime.dpkg-new' to '/etc/localtime': Device or resource busy`. The builder writes
`containers.conf` with `[containers]` `volumes = []` under its storage root, once, and sets
`CONTAINERS_CONF_OVERRIDE` to it for every buildah and podman command, the layer containers-common applies over
the host's files. A `CONTAINERS_CONF_OVERRIDE` already set in the environment is left as it is, so a deployment
that needs its own override keeps it.

A proxy the build tool inherits can break the registry push. The tool reads `http_proxy`, `https_proxy`,
`no_proxy` and their upper-case forms from its environment and from its own configuration
(`containers.conf` `[engine].env`, `~/.docker/config.json` `proxies`, the Docker daemon), and a proxy that
answers 403 to the registry ping fails the push even when the gym process itself has no proxy variables.
`no_proxy` (and `NO_PROXY`) has to include the registry host, or the proxy variables have to be cleared for the
build. Nothing is hard-coded: when a build, push or ping fails with a 403, the builder logs the names of the
proxy variables present in the tool's environment (names only, never values), or says that there are none and
that the tool's own configuration is the place to look, so the cause is visible in the failure.

With `docker`, the build is Harbor's build step. Prepare writes the Compose file Harbor's Docker environment
builds a task with (service `main`, `build.context` the task's `environment/` folder, Harbor's
`pull_policy: build` and keep-alive `command`), adds the task's own `environment/docker-compose.yaml` after it
when the task ships one, and runs
`docker compose --project-directory <environment/> -f <harbor file> [-f <task file>] build`, as
`DockerEnvironment.start()` does (`harbor/environments/docker/docker.py` at v0.22.0, cited by permalink in
`nemo_gym/tasks/harbor/image_build.py`). The command's environment is merged as Harbor merges it: the process
environment, then the task's `[environment].env` (a whole value `${VAR}` or `${VAR:-default}` read from the
host, as Harbor's `resolve_env_vars` reads it; an unset `${VAR}` with no default fails the build), then Harbor's
infrastructure variables, which win: `MAIN_IMAGE_NAME`, `CONTEXT_DIR`, `CPUS` and `MEMORY` (`<n>M`) from the
task's resources, and the `HOST_*_PATH` / `ENV_*_PATH` log mounts. So `main`'s build args and target in the
task file, interpolated or literal, apply as they do in Harbor. Two deliberate additions over Harbor's literal
file and command: Harbor's template sets no `image` (its build is tagged `<session>-main` unless the task file
says `image: ${MAIN_IMAGE_NAME}`), and Gym sets `image: hb__<hash>`, the `MAIN_IMAGE_NAME` value, so the result
can be inspected, pushed and recorded; and `DOCKER_DEFAULT_PLATFORM` carries `builds.platform`. `main`'s
`image`, when the task file sets it, has to be `${MAIN_IMAGE_NAME}`.

The task's Compose file is third-party content that `docker compose build` runs on the build host, which often
holds the push credentials, so prepare checks it before the command runs. The check reads the file as Compose
reads it: interpolated against the build's own environment (the merged environment above, with
`environment/.env` under it, as Compose reads `.env` from the project directory), with Compose's `${VAR}`,
`${VAR:-default}`, `${VAR-default}`, `${VAR:?error}`, `${VAR:+alt}` and `$VAR` forms, so `network: ${X:-host}` is
refused like `network: host` and `cache_to: [type=gha,${X:-token}=abc]` like a literal token. `.env` values are
read as Compose's dotenv reader reads them: an unquoted or double-quoted value is itself interpolated against
the build's environment and the lines above it (so `NET=${X:-host}` in `.env` is `host`), a single-quoted value
is literal, and `COMPOSE_ENV_FILES` set in the build's environment fails the build, as it would point Compose at
files the check does not read. A variable the
build cannot resolve is refused rather than left to Compose's empty default, so the file checked is the file
built. Build-time-only variables, the build host's environment and `environment/.env`, do not exist at seed,
where the server interpolates the file from the task's `[environment].env` and Harbor's infrastructure
variables only, so a Compose file must not rely on them outside `main`'s build section. Prepare refuses, naming the service and key: build `secrets`, `ssh`, `extra_hosts`, `network: host`, `additional_contexts` other than `service:`
references, `cache_from` / `cache_to` entries carrying credentials, a `build.context` or `dockerfile` that does
not resolve under `environment/` (symlinks resolved; `${CONTEXT_DIR}` is allowed), a top-level `include` and an
`extends` with a `file`. A sidecar with a `build:` section is refused too: sidecars name a prebuilt `image:`
(the loader records it from the registry and refuses one without), and no provider starts a locally built
sidecar, so only `main` is built, from `environment/Dockerfile` or the task file's `dockerfile_inline`. Harbor
reads a task's Compose file from `environment/docker-compose.yaml` only; the loader also accepts
`docker-compose.yml`, `compose.yaml` and `compose.yml` at run time, and the build refuses those spellings so the
file a task builds with is the file it runs with. `docker build` is never called, so the Compose v2 plugin
(`docker-compose-plugin`) is required and its absence fails prepare with a message that names it. buildah and
podman are the OpenSandbox path: they build the Dockerfile directly and push it.

Images are built for `linux/amd64`, the platform the sandboxes run. The tool is asked for that platform
(`docker compose` through `DOCKER_DEFAULT_PLATFORM`), and an image that comes out for another one is
refused before it is pushed or recorded. Building on an
arm64 host therefore needs qemu with binfmt registered, or an x86 node. `builds.platform: OS/ARCH` overrides
the target. `builds.enabled: false` never builds. Tasks that need a build and have none recorded are then
skipped with a message that says so.

The hash is SHA-256 over the sorted relative paths and bytes of the regular files in the build context,
exactly as Harbor computes it: symbolic links and anything under `__pycache__`, `.git` or `.DS_Store` are
skipped, and file modes are not part of it. Two tasks with identical `environment/` folders share one
image and a changed file gets a new one. `built-images.json` next to the task folders maps each context
hash to the image, the tasks that use it, the tool and whether it was pushed. Deleting a context's entry
from that file forces a rebuild of it at the next prepare; there is no flag for that, since a changed
Dockerfile or context is a new hash and builds on its own. The built image's
configuration lands in `compose-images.json` beside it, so the loader reads the Dockerfile's final
`WORKDIR`, `ENV` and `USER` from it like any pulled image. A recorded image is reused when it fits the run:
a pushed image serves every run, a local one only runs that push nothing, and the platform has to match.
Otherwise it is rebuilt and the record replaced, and prepare says why. An image is recorded only after it
was built, pushed and inspected, and each one as soon as it is done, so an interrupted prepare resumes
where it stopped. A failed build stops the preparation and names the tasks. Prepare also writes the
distinct images of the prepared tasks to `results/harbor/<taskset>/images.json`, for pre-pulling.

The build tool pushes with its own login. Pulling a pushed image into a sandbox uses the registry
credentials of the sandbox provider block (`images.auth`), the same way as any private image. The sandbox
is created with the Dockerfile's final `USER` (or `[agent].user`) as the default for every command, on every
provider, not only for the overlay `RUN` lines. An overlay task starts without that default, because its
`RUN` lines may be what creates the user, and switches to it once they have run. A numeric `USER 1000` is
resolved to its name in the image with `getent` first, because OpenSandbox and PTY sessions switch user by
name. When the uid names no user, the number stays and the seed logs a warning: Docker accepts a uid as
it is, the other providers cannot run commands as it.

| Setting or feature | Behavior in this layer |
| --- | --- |
| Image | Prebuilt `environment.docker_image`, or a base-image-only Dockerfile (`FROM`, literal absolute `WORKDIR`, `ENV`, `LABEL`). Dockerfile builds are deferred. |
| Verification | Single-step, shared-sandbox verification. Explicit `verifier.environment_mode = "separate"` or a separate verifier environment is rejected at seed. |
| Users | The image's default user and `verifier.user` are supported. Explicit `agent.user` and Dockerfile `USER` overrides are rejected at seed because sandbox access cannot yet carry the agent user. This layer has no oracle-specific exemption. |
| Environment variables | Whole-value `${VAR}` and `${VAR:-default}` templates in environment and verifier variables resolve from the server's host environment at use time. Missing required values fail the episode before provisioning. Resolved values are not added to dataset rows, task snapshots, or the episode result. A task can still print its own environment into verifier logs. |
| Prebuilt-image input files | `environment/` contents are uploaded to the workdir when the task has a prebuilt image and no `Dockerfile` or Compose definition. File modes are preserved. |
| Network restrictions | Public networking only. Restricted network modes, host allowlists and deny rules are rejected by the loader rather than silently ignored. |
| Artifacts and hooks | Verifier logs are downloaded. Artifact `exclude` rules and shared-mode collect hooks are not implemented in this layer. |
| Image ENTRYPOINT | No Harbor-compatible entrypoint lifecycle is guaranteed; startup follows the selected Gym sandbox provider. Tasks must not rely on an entrypoint running. |
| Other features | Multi-step tasks, in-sandbox MCP tools, Windows and TPU environments are unsupported. |

Clearing `/tests` and `/logs/verifier` before verification is a deliberate difference from Harbor's
single-step shared mode, which merges tests. It prevents agent-planted reward files or test helpers
from being used. Linked verification roots (including `/logs`) are unlinked before preparation,
without deleting their former targets. Helpers baked into `/tests` in an image are also removed; ship required helpers in
the task's `tests/` directory instead. Preparation and transfers attempt root, with a fallback to the
default image user when switching identity is unavailable. Target directory ownership and modes are
preserved during extraction.

## Configuration and artifacts

See [`configs/harbor.yaml`](configs/harbor.yaml) for this layer's supported settings. Verifier logs default
to `${results_dir}/harbor/verifier/<generated-directory>`, using Gym's absolute results root. An explicit
relative `artifacts_dir` is relative to that root; an absolute path is used directly. The server reserves
a fresh directory for each verification; episode identifiers are never used as host paths.
New directories honor the server process's umask and inherited directory permissions, including group
access for shared results. Local log-directory creation or containment failures exclude the sample
with `failure_kind: persistence_failed` and identify the affected path.
Measured outcomes include an absolute `verifier_logs_dir` naming the collection location; that directory
can be empty if optional log collection fails.

Hermes honors the task's `[agent].timeout_sec` via the materialized `agent_timeout_sec` metadata,
using the smaller of that deadline and its configured runner limit. The verifier uses
`[verifier].timeout_sec`. Sandbox lifetime covers the episode budget (`default_episode_timeout_seconds`),
the final cleanup budget (`cleanup_timeout_seconds`) and `sandbox_ttl_slack_s`, so setup and queued agent
work also fit; both budgets must be finite. Sandboxes are normally stopped as soon as the episode ends.

Resource settings come from the task. `sandbox_resources_override`, `gpu_sandbox_provider`,
`request_gpu_type` and `verifier_grace_s` belong to later stack layers and are not configuration options
in this PR.

`harbor:<dataset>[@<version>]` resolves the Harbor registry and fetches task folders into Gym's shared
datasets folder (`$NEMO_GYM_DATASETS_DIR`, default `./datasets`). `harbor:<org>/<name>[@<tag> | @sha256:<digest>]`
fetches a package-store dataset instead (for example `harbor:terminal-bench/terminal-bench@4.0.0`): every task
package is downloaded by content hash into `<name>/` (`latest`), `<name>-<tag>/` or `<name>-<12 digest chars>/` (a digest pin), checked against Gym's task digest and recorded in
`manifest.toml`. A local task folder, or a folder of task folders, works the same way. The generated run config declares the materialized rows as the agent's
validation dataset and binds the per-taskset environment server to that agent. Running model episodes
requires a harness that accepts `sandbox_access` in its session seed and leaves the sandbox to the server
when its session is closed.

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

`harbor:<dataset>[@<version>]` resolves the Harbor registry and fetches the task folders into Gym's shared
datasets folder (`$NEMO_GYM_DATASETS_DIR`, default `./datasets`). `harbor:<org>/<name>[@<tag> | @sha256:<digest>]`
fetches a package-store dataset instead (for example `harbor:terminal-bench/terminal-bench@4.0.0`): every task
package is downloaded by content hash into `<name>/` (`latest`), `<name>-<tag>/` or `<name>-<12 digest chars>/` (a digest pin), checked against Gym's task digest and recorded in
`manifest.toml`. A local task folder, or a folder of task folders, works the same way.

## Compose template variables and `env_file`

A task's `environment/docker-compose.yaml` may use Compose interpolation. Prepare resolves it from the task
alone, never from the process environment, so a dataset resolves the same way on every deployment. The
server reads the file through the same resolver at seed.

Harbor's infrastructure variables are defined for every task and win over the task's own:

| Variable | Value |
|---|---|
| `MAIN_IMAGE_NAME` | the resolved task image |
| `CONTEXT_DIR` | the task's `environment/` folder |
| `PREBUILT_IMAGE_NAME` | `[environment].docker_image`, when declared |
| `CPUS`, `MEMORY` | `[environment].cpus` and `memory_mb` (as `<n>m`), when declared |
| `HOST_VERIFIER_LOGS_PATH`, `HOST_AGENT_LOGS_PATH`, `HOST_ARTIFACTS_PATH` | `/logs/verifier`, `/logs/agent`, `/logs/artifacts` |
| `ENV_VERIFIER_LOGS_PATH`, `ENV_AGENT_LOGS_PATH`, `ENV_ARTIFACTS_PATH` | the same `/logs` paths |

`[environment].env` supplies the task's own variables. Their values are interpolated against the
infrastructure variables first, so a value like `${ASK_HUMAN_BACKEND:-litellm_proxy}` resolves to its default.
The forms `${VAR}`, `${VAR:-default}`, `${VAR-default}`, `${VAR:?error}`, `${VAR:+alternative}`, `$VAR` and `$$`
are supported, and a default may itself contain an interpolation. A variable with neither a value nor a
default fails prepare with its name. A dollar that is not an interpolation is literal.

A service's `env_file` entries are read at prepare, after interpolation, so a path may use `${CONTEXT_DIR}`.
The file must sit under the task's `environment/` folder: an absolute path outside it, a path that climbs
out, or a symlink fails prepare with the path. Its `KEY=value` lines are inlined into the service's
`environment` and taken literally, quotes and inline comments handled as Compose does, and a `$` in a value
is never interpolated. The service's own `environment` wins over its files. Every service's `environment` becomes a mapping of
strings. A bare `NAME` entry in a list-form `environment` takes the task variable of that name and is
dropped when there is none. A missing
file fails prepare with its name unless the entry says `required: false`. Error messages name the file and
the line number and never repeat a line's content.

## Where settings live

Four homes, one owner each. A setting sits on this server's config only if every dataset on the same
deployment wants the same value; a unit test pins that list.

| Home | File | Owner | Holds |
|---|---|---|---|
| Task | `task.toml` | task author | image, resources, timeouts, env, verifier, agent user (read-only once fetched) |
| Dataset | `dataset.toml`, `[gym]` table | dataset author | `[gym.defaults]` `timeout_multiplier`, `resource_multiplier`; `[gym.tasks."<id>".environment]` overrides in Harbor's own fields (`env` merges) |
| Deployment | the `sandbox:` provider block | operator | image rewrites and registry credentials (`images`), root setup commands (`setup`), networking, shared-memory labels |
| Run | run config | the run | provider selection, GPU routing, repeats, output paths, `sandbox_resources_override` for parity runs |

`gym dataset init <name>` writes a dataset in this shape, and `gym dataset validate <folder>` with no extra
config is the test that a dataset is self-contained.
