# Harbor tasks

Runs tasks written in the [Harbor](https://github.com/harbor-framework/harbor) task format: a folder with
`task.toml`, `instruction.md`, `environment/`, `tests/test.sh` and an optional `solution/`. Harbor is the input
format; Gym owns execution. The supported subset is listed below.

## What the server does

This is an environment server: one `/run` call owns one whole episode, and no resources server takes part.

- Seed: checks the row's digest against the taskset mapping, snapshots and checks the task files once per
  digest, starts the task image as a sandbox through the configured provider, and creates the working
  directory. A declared `docker_image` starts as written; a pull-mode or overlay-mode Dockerfile's base image
  starts pinned to the multi-arch index digest recorded at prepare (`compose-images.json`), so a moved tag
  cannot change the image under the recorded configuration and each platform's sandbox still selects its own
  manifest. Concurrent episodes share the snapshot. Later source edits cannot change an already cached task;
  re-materialize and restart the server to use edited tasks.
- Agent: opens a session on the configured agent server with a `SandboxAccess` for that sandbox (the same
  seed request every sandboxed harness already accepts), posts the task's `responses_create_params` to it,
  then closes the session and keeps the harness's observations for the result. When the agent instance is
  implemented by `harbor_oracle_agent` (the key under its `responses_api_agents` block), the seed first
  uploads the snapshot's `solution/` to `/solution`, as root like its other uploads and readable by the task
  user, and the sandbox access carries the host-resolved `[solution].env` as `env`; that is where Harbor's
  oracle copies and runs the solution. No other harness receives the solution, and a task without
  `solution/` uploads nothing (the oracle reports it unvalidated).
- Verify: clears `/tests` and `/logs/verifier`, uploads the snapshot's `tests/`, runs `bash /tests/test.sh`
  in the same sandbox, and downloads the verifier logs.
- Cleanup: stops the sandbox after the episode, whatever happened before. Cancellation and the episode
  deadline also reach this cleanup; a failed stop keeps the sandbox owned by the server for a later retry
  or shutdown, and a redelivered episode cannot provision while an owned sandbox of the same episode remains.

A task that cannot be set up is a handled episode failure with `stage: seed`: terminal when the task or row
is at fault (unknown task, changed folder, unsupported settings), retryable when the sandbox provider failed.
A harness that cannot be seeded, fails its turn or fails to close is `stage: agent` or `stage: cleanup`, with
its observations and any usable response attached. A task that declares `[[environment.mcp_servers]]` or
`skills_dir` is a terminal `stage: agent` failure unless the harness declared `supports_task_tool_context` in
its seed response: running it without its tools would score a silent zero. Verification never runs before the
harness has let go.

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
separate verifier mode in `[verifier.environment].docker_image` or, without one, the task image (the agent's
`/logs/artifacts` and `artifacts` entries are copied into the verifier sandbox first, sidecar hooks and
artifacts included, each as the archive the agent's sandbox packed, so the verifier sees the modes the agent
left). Compose environments start as
a sandbox group when a `compose-images.json` with the sidecar images' recorded OCI configuration sits next to
the task folders, written by `gym dataset fetch`; on OpenSandbox the provider block also needs
`networking.enabled: true` (with `loopback_forwarding`) and `runtime_requirements.shm_size_metadata_key` (see
[Compose tasks on OpenSandbox](#compose-tasks-on-opensandbox)); each service's working directory is created once its sandbox is up, as Docker creates a missing `working_dir` or `WORKDIR` at container create. `[environment.healthcheck]` is polled at seed.
The server's `compose_variables` fill a Compose file's `${NAME}` references at the process environment's
level, below the task's `[environment].env` and Harbor's own variables: a literal `value`, a process `env`
variable, or a `model_server`, whose sandbox-reachable URL with the episode's rollout prefix becomes the value
(a sidecar's model calls are then recorded with the rollout and never become training tokens). An `env` source
the process lacks fails the seed terminally. With `compose_variables` set, each episode's document is written
to a private folder removed once the group starts, not under `artifacts_dir`.

```yaml
compose_variables:
  LITELLM_BASE_URL: {model_server: user_model}
  LITELLM_API_KEY: {env: USER_MODEL_API_KEY}
  SIM_USER_MODEL: {value: gpt-5.5}
```
Tasks that declare GPUs run on the `gpu_sandbox_provider` block when one is set. The task's
`[[environment.mcp_servers]]` and `skills_dir` are written to `/tmp/.nemo-gym/task.json` in the agent's
sandbox for harnesses that drive task MCP servers from inside it (mini-SWE does). With `task_mcp_grants:
optional` or `required`, the `streamable-http` URL entries are also granted to the agent as MCP tool
accesses, as declared; a Compose sidecar's URL names its service, which resolves inside the task's group.
Dockerfiles outside pull and overlay mode are built at prepare, see below.

## Multi-step tasks

A task with `[[steps]]` runs each step as one turn in the same sandbox, as Harbor's multi-step trial does
without `resume_trajectory`: a fresh agent session per turn, with the step's
`steps/<name>/instruction.md` as its only input and the step's `[steps.agent].timeout_sec` (else the task's)
as its budget. The task's top-level `instruction.md` is never sent; a multi-step task may keep material there
the agent must not see. After each turn the shared verifier runs the task's `tests/` with
`steps/<name>/tests/` on top, under the step's verifier timeout, user and environment over the task's.

The episode stops after a turn that is masked, writes no valid reward file, or misses the step's
`min_reward` (a number gates `reward`, a table gates each named key). `multi_step_reward_strategy` rolls the
turns up: `mean` (the default) averages the turns with a verifier result, and `final` takes the last turn that
ran. The result lists every turn under `turns`; its `response` and `ng_agent_observations` are the last
turn's. Per-step `workdir/`, `setup.sh`, healthchecks, artifacts and separate verification are refused by the
loader. A long task's turns can budget more than `default_episode_timeout_seconds`; the seed logs a warning,
because an expired episode is retried rather than scored.

## Dockerfile builds

Prepare decides one of three image modes per distinct Dockerfile.

- Pull. A base-image-only file (`FROM` plus `WORKDIR`, `ENV`, `USER`, `LABEL`). The sandbox starts from the
  base image.
- Overlay. A pullable `FROM` plus `RUN` lines. The server runs the lines in the pulled base image at seed.
  No build and no registry.
- Build. Everything else: `COPY`, `ADD`, heredocs, BuildKit flags, multi-stage files. The task's
  `environment/` folder is the build context. Prepare builds it once, records it, and the task then runs in
  pull mode from the built image. A `build:` the task's Compose file gives `main` (build args, a target, a
  `dockerfile_inline`) applies to that build; at seed the recorded image stands in for it, so the server drops
  the key from `main` before the Compose adapter, which starts images only, sees the document. A task in pull
  or overlay mode whose file only restates `main`'s build (`context: ${CONTEXT_DIR}` or `.`, `dockerfile:
  Dockerfile`, `pull`) has that section dropped too, as Harbor's own template already supplies it; a build key
  Gym cannot apply without a build (`args`, `target`, `dockerfile_inline`, another `dockerfile` or `context`)
  leaves the task in `skipped.json` naming the key. `pull_policy` is dropped from every
  service in every mode, as the sandbox has no pull step to configure. Prepare runs the adapter's field check
  on each Compose task's resolved document, so a field no provider starts (`deploy`, `extra_hosts`,
  `platform`, a sidecar `build:`) leaves that task in `skipped.json` with the service and field named instead
  of failing every episode at seed. The same check runs on the raw file of every task about to be built,
  before the build, so a refused task's context is neither built nor pushed and a build that would fail for
  it cannot stop the preparation.

A task that declares `[environment].docker_image` runs that image, unless `builds.force` is set: then its
Dockerfile is built anyway and the recorded build wins, as Harbor's `--force-build` does (SWE-Interact's
declared images lack the helper its Dockerfiles copy in). A Compose sidecar with a `build:` section is built
the same way, once per distinct context, from the context's own `Dockerfile` (a sidecar build takes only
`context`, inside `environment/`); its recorded image then stands in for the section. With docker, the
compose build of `main` names `main` alone when a sidecar builds on its own.

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
  force: false             # true also builds the Dockerfile of a task that declares docker_image
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

podman as the build tool needs a working rootless setup on the build host. On the cluster nodes used here, podman's
default `crun` runtime cannot create rootless build containers and `chroot` isolation breaks Dockerfile heredocs, while
`BUILDAH_ISOLATION=rootless` with the `runc` runtime (named in a `containers.conf` `[engine]` block) builds them. Point
`CONTAINERS_CONF_OVERRIDE` at such a file before `gym eval run` when the host's defaults do not work; the builder leaves
a preset override alone. Build contexts must be on local disk for heredoc builds: buildah's heredoc support overlay-mounts
the context, which fails on Lustre.

buildah and podman also run with a `registries.conf` of the builder's and build docker-format images. An
unqualified `FROM` (`FROM oven/bun:1`) resolves to docker.io under `docker compose build`; buildah and podman
resolve it through the host's `registries.conf`, and the stock Debian and Ubuntu file names no
unqualified-search registry, so the build fails with `short-name "oven/bun:1" did not resolve to an alias`. The
builder writes `registries.conf` with `unqualified-search-registries = ["docker.io"]` under its storage root and
sets `CONTAINERS_REGISTRIES_CONF` to it for every buildah and podman command. That file replaces the host's main
`registries.conf` (the drop-ins under `registries.conf.d`, short-name aliases included, still load), so a site
whose main file carries mirrors or blocked registries sets `CONTAINERS_REGISTRIES_CONF` itself and the builder
leaves it alone. The build runs with `--format docker`, the image format BuildKit builds: under the tools'
default OCI format `SHELL` is ignored by the `RUN` steps (`RUN [[ ... ]]` after `SHELL ["/bin/bash", "-c"]`
fails with `/bin/sh: [[: not found`) and `HEALTHCHECK` and `ONBUILD` are dropped from the image. The recorded
configuration in `compose-images.json` carries `Healthcheck` from the image under all three tools, and `Shell`
and `OnBuild` with buildah and docker (podman's inspect does not report them, although its image carries them).
docker needs neither setting.

Dockerfile heredocs (`RUN <<EOF`, `COPY <<EOF /file`, `RUN python - <<'PY'`) need a buildah or podman built with
openshift/imagebuilder 1.2.6 or later (upstream buildah 1.33.0 or later). Ubuntu 24.04's packaged buildah 1.33.7
and podman 4.9.3 are linked against imagebuilder 1.2.5 and read the heredoc body as instructions
(`Unknown instruction: "ECHO"`). Before the first Dockerfile with a heredoc is built, the builder runs a
preflight once: a two-line `FROM scratch` plus `COPY <<EOF /probe` build in a temporary context, no base image
pulled. When it fails, prepare stops with one error naming the tool and its version, the Dockerfile and the
remedy (a qualifying buildah or podman named with `builds.tool`, or `builds.tool: docker`) instead of one
`Unknown instruction` failure per task. A heredoc opened with a space before the delimiter
(`RUN cat > /x << 'EOF'`) is not a heredoc to BuildKit either and fails under every tool.

With buildah and podman, `main`'s `build` section of the task's `environment/docker-compose.yaml` applies to the
direct build the way `docker compose build` applies it over Harbor's template: `context` (a path or the string
form of the section) and `dockerfile` are resolved against `environment/` and have to stay under it, `args`
(mapping or `KEY=value` list; a bare `KEY` takes the variable's value from the build's environment) become
`--build-arg`, and `target` becomes `--target`; a mapping value of `null` (`KEY:`) is read from the build's
environment like a bare list `KEY`, and both are dropped when it is unset, so the Dockerfile's own `ARG` default
stands, as under Compose. The section is interpolated as Compose interpolates it, from the build's environment
(the process environment, the task's `[environment].env` and Harbor's variables, with `environment/.env` under
them), so `${VAR:?message}` with no value fails that task's build naming the task and the variable instead of
passing an empty build arg, while a bare `${VAR}` nobody sets is an empty string with one warning, as Compose
makes it. The task's Compose file is read at build time, not at planning, and checked on this path as on the
docker path before anything is built, so a file the loader would refuse (a sidecar with a `build:` section
among them) is never built and pushed under the `hb__` name, and a file one task gets wrong leaves that task
out without touching the rest of the dataset. `dockerfile_inline` is refused on the direct path, with
`builds.tool: docker` as the remedy. The image name and hash do not change: the hash is Harbor's hash of
`environment/`, which already covers the Compose file.

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
task's resources, the `HOST_*_PATH` / `ENV_*_PATH` log mounts, and the two the loader defines for the same file,
`TEST_DIR` and `COMPOSE_PROJECT_NAME` (the table below). So `main`'s build args and target in the
task file, interpolated or literal, apply as they do in Harbor. Two deliberate additions over Harbor's literal
file and command: Harbor's template sets no `image` (its build is tagged `<session>-main` unless the task file
says `image: ${MAIN_IMAGE_NAME}`), and Gym sets `image: hb__<hash>`, the `MAIN_IMAGE_NAME` value, so the result
can be inspected, pushed and recorded; and `DOCKER_DEFAULT_PLATFORM` carries `builds.platform`. `main`'s
`image`, when the task file sets it, has to resolve to `MAIN_IMAGE_NAME` (`${MAIN_IMAGE_NAME}`,
`${MAIN_IMAGE_NAME:-…}` or `${MAIN_IMAGE_NAME-…}`); a literal other image is refused.

The task's Compose file is third-party content that `docker compose build` runs on the build host, which often
holds the push credentials, so prepare checks it before the command runs. The check reads the file as Compose
reads it: interpolated against the build's own environment (the merged environment above, with
`environment/.env` under it, as Compose reads `.env` from the project directory), with Compose's `${VAR}`,
`${VAR:-default}`, `${VAR-default}`, `${VAR:?error}`, `${VAR:+alt}` and `$VAR` forms, so `network: ${X:-host}` is
refused like `network: host` and `cache_to: [type=gha,${X:-token}=abc]` like a literal token. `.env` values are
read as Compose's dotenv reader reads them: an unquoted or double-quoted value is itself interpolated against
the build's environment and the lines above it (so `NET=${X:-host}` in `.env` is `host`), a single-quoted value
is literal, and `COMPOSE_ENV_FILES` set in the build's environment fails the build, as it would point Compose at
files the check does not read. A bare `${VAR}` nobody sets is an empty string with one warning naming the task
and the variable, as `docker compose build` makes it and as the loader resolves the same file at seed, so the
file checked is the file built; `${VAR:?message}` fails the build with the message. The build and the loader
read the same sources (the process environment, the task's `[environment].env`, Harbor's variables and
`environment/.env`), so prepare and the environment server have to run with the same variables exported for the
file built to be the file started. Prepare refuses, naming the service and key: build `secrets`, `ssh`, `extra_hosts`, `network: host`, `additional_contexts` other than `service:`
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
where it stopped. Both record files are replaced atomically under a sidecar lock file next to them
(`built-images.json.lock`, `compose-images.json.lock`), so two Gym commands recording into one datasets
folder keep each other's entries and a reader never sees a half-written file. A failed build stops the
preparation and names the tasks. Prepare also writes the
distinct images of the prepared tasks to `results/harbor/<taskset>/images.json`, for pre-pulling.

The hash names the image; it is not the whole build. The configuration record of each built image in
`compose-images.json` also carries an `inputs_digest` over everything its build read, beside the
`config_digest` of what came out: every entry of the context tree with its mode (files by content, symbolic
links by target, folders by presence, the names Harbor's hash skips included) and `main`'s `build` section of
the task's Compose file as the build reads it (the whole section after interpolation, an inline Dockerfile and
labels included, its paths as resolved and its build arguments as the tool is given them). A recorded image is
reused only when the context's hash and the recorded inputs digest both match; a context whose inputs changed
without changing its hash (a mode flip, a new empty folder, a different link target, a variable the file
interpolates into the build section or a build argument taken from the environment) is rebuilt under the same
name and its records replaced, and prepare says why. With `builds.enabled: false` such a task is left out with
the reason. An image recorded before the digest was kept is rebuilt once.

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
| Verification | Single-step verification, shared or separate. A separate verifier runs in `[verifier.environment].docker_image`, else in the task image, for the episode-sized sandbox lifetime; a verifier that needs a `tests/Dockerfile` build is rejected at seed. |
| Users | The image's default user and `verifier.user` are supported. Explicit `agent.user` and Dockerfile `USER` overrides are rejected at seed because sandbox access cannot yet carry the agent user. This layer has no oracle-specific exemption. |
| Environment variables | Whole-value `${VAR}` and `${VAR:-default}` templates in environment and verifier variables resolve from the server's host environment at use time. Missing required values fail the episode before provisioning. Resolved values are not added to dataset rows, task snapshots, or the episode result. A task can still print its own environment into verifier logs. |
| Prebuilt-image input files | `environment/` contents are uploaded to the workdir when the task has a prebuilt image and no `Dockerfile` or Compose definition. File modes are preserved. |
| Network restrictions | Public networking only. Restricted network modes, host allowlists and deny rules are rejected by the loader rather than silently ignored. |
| Artifacts and hooks | Verifier logs are downloaded. `[[verifier.collect]]` hooks run in both modes, each in the service it names (a non-zero exit or a hook that runs past its `timeout_sec` warns, a hook the sandbox cannot run masks). In separate mode `/logs/artifacts` and the `artifacts` entries are copied into the verifier, each under its own staging root, with their `exclude` rules applied as Harbor passes them to `tar --exclude` (unanchored: `node_modules` at any depth), replacing what the verifier image ships at those paths. The verifier receives each artifact as the archive the agent's sandbox packed, so read-only, world-writable and set-id modes arrive as the agent left them; the host copy under the staging root is unpacked through a filter (widened owner modes, links kept, nothing that escapes and no device nodes) and is the gate the archive must pass plus a diagnostic, and the archive is removed once restored. A probe the sandbox cannot answer masks rather than counting as an absent artifact. |
| Image ENTRYPOINT | Runs for a plain (non-Compose) task whose image configuration is recorded: a built image, or a pull or overlay Dockerfile whose base image is recorded, unless the Dockerfile says `ENTRYPOINT []`. The sandbox starts with the provider's keepalive; the seed then starts `ENTRYPOINT + sh -c "sleep infinity"` beside it (what Harbor's Compose `command` leaves it with) as the image's `User` in its `WorkingDir` (created for a resolved Dockerfile when the task's workdir is another), after the working directory and `/logs` exist. It then waits as `up --wait` and Harbor's `run_healthcheck()` do: the image `HEALTHCHECK` first, then `[environment.healthcheck]`, both as the image's user (the image check in its `WorkingDir`, the task check in the task workdir; a `CMD-SHELL` check runs through the image's recorded `Shell`, `/bin/sh -c` when it sets none, as the daemon runs it, so `SHELL ["/bin/bash", "-o", "pipefail", "-c"]` fails `false | true` here too), probing the entrypoint's exit status before every attempt and once more before the hand-over. Exit non-zero before the hand-over fails the seed as a task error (terminal, with the output tail), as does a launcher whose own shell fails for the image's user or directory; a provider that cannot run the launcher is retried. Exit zero logs a warning once and the episode continues on the keepalive. At verification the status is read again: in the separate verifier's fresh sandbox a non-zero exit masks the sample as `harbor:entrypoint_failed` (under Harbor the container would have died and the trial errored); in the agent's sandbox it masks only when the exec holding the entrypoint returned on its own with that status, because the runtime folder is world-writable and the agent had the sandbox; otherwise it is recorded as `entrypoint-exit.txt` with an error in the log and the sample is scored. The holding exec lives as long as the provider allows (docker caps it at `exec.default_timeout_s`, OpenSandbox without `operations.background_exec` at `connection.request_timeout_s`); past that the entrypoint is untracked, logged once, and OpenSandbox deployments need `background_exec` for long-lived launchers. An entrypoint that is one `sleep` or `tail` command is not started. A separate verifier that runs in the task image gets that image's entrypoint and `HEALTHCHECK` as a substitute for Harbor's `tests/Dockerfile` verifier image, never the task's healthcheck. The output is kept as `entrypoint-stdout.txt` (and `verifier-entrypoint-stdout.txt`) next to the verifier logs. A prebuilt `docker_image` has no recorded configuration yet, so its entrypoint still does not run. |
| Other features | Windows and TPU environments are unsupported. Multi-step tasks run as turns (see above). |

Shared-mode verification does not stop or quiesce what the agent left running: `test.sh` runs in the same
sandbox beside those processes, as Harbor's shared mode runs it, so a test that talks to a service the agent
started keeps working, and a process the agent left behind can still interfere. Separate verifier mode is the
tamper-proof path: the verifier gets its own sandbox and its own image, and nothing the agent started is there
(see the Verification row for what this layer supports).

Clearing `/tests` and `/logs/verifier` before verification is a deliberate difference from Harbor's
single-step shared mode, which merges tests. It prevents agent-planted reward files or test helpers
from being used. Restored artifacts in separate mode are held to the same line: an artifact at `/tests`,
inside or over it, or at or inside `/logs/verifier` is refused when the task is loaded (its source is settled
to one leading slash first), collected on the host but never restored should it reach the server, one that a
link in the verifier sandbox would divert onto either directory is refused once its target is resolved
there, and a restore that covers `/logs/verifier` (`/logs`) is followed by that directory being emptied
again before `test.sh` runs. Linked verification roots (including `/logs`) are unlinked before preparation,
without deleting their former targets. Helpers baked into `/tests` in an image are also removed; ship required helpers in
the task's `tests/` directory instead. Preparation and transfers attempt root, with a fallback to the
default image user when switching identity is unavailable. Target directory ownership and modes are
preserved during extraction. Uploaded test files stay root-owned with the package's modes, as Harbor's
`docker cp` leaves them, and only `test.sh` gets `chmod +x` as root before it runs as the verifier user
(Harbor's `needs_chmod` rule for a `.sh` script), nothing is chowned or widened, so a package built under
`umask 077` behaves as in Harbor: its `0600` files are readable by a root verifier and by no other
`[verifier].user`.

In separate mode a `/tests/test.sh` baked into the verifier image wins, as in Harbor, which never uploads the
package's tests into a separate verifier (`skip_tests_upload`): the package's `tests/` is uploaded only when that
file is absent. The check is on `test.sh` alone, not on the directory. An image that ships a `/tests` folder without
a `test.sh` gets the package tree extracted into it, same-named files replaced and the rest merged in, so only a
baked `test.sh` is protected. To grade a task's own image with the package's tests, drop the baked `test.sh` from
that image, or ship a `tests/Dockerfile` so the verifier gets an image of its own (again, see the Verification row
for what this layer supports).

## Compose tasks on OpenSandbox

The shipped `nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml` (what `--sandbox opensandbox`
loads) has no networking block, and the Compose adapter refuses to start a group without one. Put an overlay
like this one after it in `--config` (or in a `+config_paths` entry). The three blocks are the deployment's:
service-to-service networking, the label the cluster reads a service's `shm_size` from, and the capability
labels and probes a task that needs `SYS_PTRACE` is scheduled with. The optional second provider block shows
how `gpu_sandbox_provider` routes the tasks that declare GPUs to a GPU pool while the rest stay on `sandbox`.

```yaml
sandbox:
  opensandbox:
    networking:
      enabled: true
      loopback_forwarding: true
      # Runs in every container the provider creates, sidecars included: it must no-op where it does not apply.
      setup_command: "command -v python3 >/dev/null 2>&1 || ! command -v apt-get >/dev/null 2>&1 || (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3)"
    runtime_requirements:
      shm_size_metadata_key: nemo.nvidia.com/shm
      capability_metadata:
        SYS_PTRACE:
          nemo.nvidia.com/ptrace: "true"
      capability_probes:
        SYS_PTRACE: "awk '/^CapEff:/ { print $2 }' /proc/self/status | { read caps; test $((0x$caps & (1 << 19))) -ne 0; }"

# Optional: a second provider block for GPU tasks, named by the server's `gpu_sandbox_provider`.
sandbox_gpu:
  opensandbox: ${sandbox.opensandbox}

harbor_environment_server:
  environment_servers:
    harbor:
      gpu_sandbox_provider: sandbox_gpu
```

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

The task's `[agent].timeout_sec` reaches the harness twice: in the agent-session seed as
`agent_timeout_sec`, next to `sandbox_access` (which also names the task's `[agent].user` as `user`, and
for the oracle the resolved `[solution].env` as `env`; the oracle's solve budget is that `agent_timeout_sec`),
and, until every harness reads the seed, as the materialized `agent_timeout_sec` metadata that Hermes
honors today, using the smaller of that deadline and its configured runner limit. The verifier uses
`[verifier].timeout_sec`. Sandbox lifetime covers the episode budget (`default_episode_timeout_seconds`),
the final cleanup budget (`cleanup_timeout_seconds`) and `sandbox_ttl_slack_s`, so setup and queued agent
work also fit; both budgets must be finite. Sandboxes are normally stopped as soon as the episode ends.

Resource settings come from the task. `sandbox_resources_override`, `gpu_sandbox_provider`,
`request_gpu_type` and `verifier_grace_s` belong to later stack layers and are not configuration options
in this PR.

Every sandbox the server creates (the agent's, a Compose group's `main` service and a separate verifier's,
sized by `[verifier.environment]`) is also told its CPU limit through the CPU-count hints `cpu_cap_env`
derives from it: `OMP_NUM_THREADS`, `PYTHON_CPU_COUNT`, `PYTEST_XDIST_AUTO_NUM_WORKERS`, `GOMAXPROCS` and
the other `CPU_CAP_ENV_VARS` in `nemo_gym/sandbox/utils.py`, each set to the limit's whole core count (an
override moves them with the limit). They are always set, with no switch: on many clusters a sandbox sees
the host's cores through `nproc` and `os.cpu_count()`, so build and test pools sized from them fan out far
past a one or two CPU limit and thrash or get OOM-killed, and the hints are the portable cap. They are
hints, not limits: the cgroup limit stays the provider's, and a variable the task declares itself, in
`[environment].env`, in its image's `ENV` or in the verifier environment's `env`, wins on a key clash.

Servers bind loopback (`127.0.0.1`) by default, the head server with its `/global_config_dict_yaml` endpoint
included. A harness that calls the model server from inside the task sandbox (`calls_model_from_sandbox = True`
on its agent class, or `sandbox_model_base_url` in its config) cannot reach loopback there, so
`gym eval run <target> --agent <harness>` sets `use_absolute_ip: true` together with
`absolute_ip_server_types: [responses_api_models]` in the generated run config: only the model server
advertises the node's resolved address; the head server, the agent server and this environment server stay on
loopback exactly as on a run without a sandboxed harness. The oracle (`gym dataset validate`), Terminus 2 and
mini-SWE run their commands in the sandbox but call the model from the host, so nothing moves for them. A
deployment that needs more servers reachable sets `absolute_ip_server_types` (or `default_host`, which applies
to every server) explicitly; a config file or command-line override wins over the generated default.
A registry fetch (`harbor:<dataset>`, below) reuses a task folder already in Gym's datasets folder when its
`manifest.toml` pin names the registry's `git_url` and `path` with a full commit SHA, the folder has a
`task.toml`, and no symlink in it leaves the folder. The recorded commit is kept on purpose when the registry's
entry moves (a deleted folder is fetched again at that commit), and the fetch does not hash the folder's content
(Harbor's own download reuses any non-empty folder without a check). Move the folder aside to fetch the
registry's current copy.

`harbor:<dataset>[@<version>]` resolves the Harbor registry and fetches task folders into Gym's shared
datasets folder (`$NEMO_GYM_DATASETS_DIR`, default `./datasets`). `harbor:<org>/<name>[@<tag> | @sha256:<digest>]`
fetches a package-store dataset instead (for example `harbor:terminal-bench/terminal-bench@4.0.0`): every task
package is downloaded by content hash into `<name>/` (`latest`), `<name>-<tag>/` or `<name>-<12 digest chars>/` (a digest pin), checked against the content hash Harbor published for it and recorded in
`manifest.toml`, which pins the package and the version the fetch resolved so a later fetch never mixes another package or a moved `latest` into the folder. A local task folder, or a folder of task folders, works the same way. The generated run config declares the materialized rows as the agent's
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

A task's `environment/docker-compose.yaml` may use Compose interpolation. Prepare resolves it the way Harbor's
`docker compose` resolves it, and the server reads the file through the same resolver at seed. Variables come
from these sources, the first that defines a name wins:

1. Gym's infrastructure variables, defined for every task (the table below);
2. the task's `[environment].env`, each value read as Harbor reads it: a whole value `${VAR}` or
   `${VAR:-default}` is taken from the process environment, the default standing in when the variable is unset,
   and one with neither fails prepare naming the task and the variable (Harbor fails the trial the same way);
   any other value is literal;
3. the process environment of the command;
4. `environment/.env` next to the Compose file, read as Compose's dotenv reader reads it (unquoted and
   double-quoted values are themselves interpolated against the sources above and the lines before them,
   single-quoted values are literal);
5. the default in the expression (`${VAR:-default}`, `${VAR-default}`);
6. otherwise an empty string, with one warning naming the task and the variable, as docker compose does.
   `${VAR:?message}` and `${VAR?message}` fail prepare with the message instead.

Source 3 is a trust boundary. `compose_variables()` includes the whole environment of the process that runs
prepare or the server (Harbor parity: Harbor interpolates with `include_os_env=True`), so a task's
`docker-compose.yaml` can reference any variable of that process, an API key or `HF_TOKEN` among them, and the
value lands in the sandbox environment and in the resolved `compose.yaml` written under `artifacts_dir`. For a
third-party hub dataset, run prepare and the server with a scrubbed environment that exports only what the
dataset needs; an allowlist of variable names is a later option.

| Variable | Value |
|---|---|
| `MAIN_IMAGE_NAME` | the resolved task image |
| `CONTEXT_DIR` | the task's `environment/` folder |
| `TEST_DIR` | `/tests`, where the task's tests land in the sandbox (Harbor leaves it empty; Harbor's own terminal-bench tasks read it) |
| `COMPOSE_PROJECT_NAME` | `hb__<hash>`, a stable per-task placeholder (Harbor's environment id of `environment/`, also the `main` image name). It is not the runtime project name: Harbor's `docker compose -p` sets that to the trial's session id, and Gym's Compose adapter names each episode's project `compose-<uuid>`. A value derived from it is consistent within one document but matches no Compose-generated container, network or project name. |
| `PREBUILT_IMAGE_NAME` | `[environment].docker_image`, when declared |
| `CPUS`, `MEMORY` | `[environment].cpus` and `memory_mb` (as `<n>M`, Harbor's spelling), when declared |
| `HOST_VERIFIER_LOGS_PATH`, `HOST_AGENT_LOGS_PATH`, `HOST_ARTIFACTS_PATH` | `/logs/verifier`, `/logs/agent`, `/logs/artifacts` |
| `ENV_VERIFIER_LOGS_PATH`, `ENV_AGENT_LOGS_PATH`, `ENV_ARTIFACTS_PATH` | the same `/logs` paths |

The forms `${VAR}`, `${VAR:-default}`, `${VAR-default}`, `${VAR:?error}`, `${VAR:+alternative}`, `$VAR` and `$$`
are supported, and a default may itself contain an interpolation. A dollar that is not an interpolation is
literal. The process environment is not part of the task digest: run prepare and the environment server with
the same variables exported, as a Harbor run does in its one process, or the document the sandbox starts
differs from the one prepare validated.

A task whose Compose file does not resolve, or whose image fails to build, is left out of the prepared rows
and recorded in `skipped.json` with the reason, like a task that does not load; the rest of the dataset still
prepares. `gym eval run` lists every task left out with its reason and stops, unless `--skip-unloadable` says
to run the loadable subset.

At seed `main`'s environment is layered as Harbor layers it: the image's `ENV` (baked into a prebuilt or built
image, or the Dockerfile's `ENV` lines the loader applies for a pull or overlay task) at the bottom, `main`'s
Compose `environment` over it, and the task's `[environment].env`, resolved on the host, over both, as Harbor's
env override file is (`harbor/environments/docker/docker.py` 363-370 and `environments/base.py` 278-285 at
v0.22.0). So a `${HF_TOKEN:-}` that interpolated to empty at prepare does not hide the host value, and a key the
file sets still beats the image's `ENV`. A sidecar's `${VAR}` is resolved again at seed with the same
host-resolved `[environment].env`, as Harbor interpolates the file for every service against it (`docker.py`
`_compose_env_vars`), so hil-bench's `ask-human` runs with the deployment's backend and keys rather than the
dataset-time defaults; only the sidecars' `environment` values change, never the structure prepare validated.
A `[environment].env` value with no default (N2) still fails prepare.

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
