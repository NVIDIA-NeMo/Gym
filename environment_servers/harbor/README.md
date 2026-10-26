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
422 with the command and the tail of its output; a provider that cannot run one is a 503); shared verifier mode, and
separate verifier mode (the agent's `/logs/artifacts` and `artifacts` entries are copied into the verifier sandbox
first, sidecar hooks and artifacts included, each as the archive the agent's sandbox packed, so the verifier sees the
modes the agent left). In separate mode the verifier's image is the one prepare built from
`tests/Dockerfile` when the task ships that file and its effective verifier environment names no image, as Harbor
builds it (see "Dockerfile builds" below); else `[verifier.environment].docker_image`; else the task's own
`[environment].docker_image` (Harbor's effective verifier environment is then a copy of `[environment]`). A
separate-mode task that names no verifier image and ships no `tests/Dockerfile` is refused as Harbor refuses it, and
one whose `tests/Dockerfile` has no recorded build is left out at prepare like an unrecorded `environment/` build;
the server refuses either at seed. Compose environments start as
a sandbox group when a `compose-images.json` with the sidecar images' recorded OCI configuration sits next to
the task folders, written by `gym dataset fetch`; on OpenSandbox the provider block also needs
`networking.enabled: true` (with `loopback_forwarding`) and `runtime_requirements.shm_size_metadata_key` (see
[Compose tasks on OpenSandbox](#compose-tasks-on-opensandbox)); each service's working directory is created once its sandbox is up, as Docker creates a missing `working_dir` or `WORKDIR` at container create. `[environment.healthcheck]` is polled at seed.
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
  pull mode from the built image. A `build:` the task's Compose file gives `main` (build args, a target, a
  `dockerfile_inline`) applies to that build; at seed the recorded image stands in for it, so the loader drops
  the section, and a `pull_policy: build` naming it, from every service prepare built before the Compose
  adapter, which starts images only, sees the document. A task in pull or overlay mode whose file only restates
  `main`'s build (`context: ${CONTEXT_DIR}` or `.`, `dockerfile: Dockerfile`, `pull`) has that section dropped
  too, as Harbor's own template already supplies it; a build key Gym cannot apply without a build (`args`,
  `target`, `dockerfile_inline`, another `dockerfile` or `context`) leaves the task in `skipped.json` naming the
  key.
  A `pull_policy: build` goes with the build it names and is dropped with it (a built service's, a restated
  `main`'s); every other `pull_policy` is the service's runtime pull policy and reaches the adapter (see the field
  table below). Prepare runs the adapter's field check on each Compose task's resolved document, so a field no
  provider starts (`deploy`, `extra_hosts`, `platform`, a sidecar `build:` prepare recorded no build for) leaves
  that task in `skipped.json` with the service and field named instead of failing every episode at seed. The
  check's first step, the key allow-list, also runs on the raw file of every task about to be built, before the
  build, so a task refused by key is neither built nor pushed and a build that would fail for it cannot stop the
  preparation. A value the check refuses (`count: all`, a `platform` other than the deployment's) is read from
  the resolved document once the tasks load, after their builds, and leaves the task out the same way.

Every other service of the task's `environment/docker-compose.yaml` with a `build:` section (a sidecar) is built
too, whatever `main`'s mode, as Harbor's `docker compose build` builds it: from its own context and Dockerfile
under `environment/`, with its args and target. Its image is named `hb__<hash>-<service>`, the form Compose gives
a service without `image:` under the project name Gym builds with (Harbor's is the per-trial session id, so its
sidecar tags are never published), and a sidecar's own `image:` spelling is accepted and overridden by that name.
Each built service is recorded by name under the context's `built-images.json` entry, and at seed the loader
substitutes the recorded image for every built sidecar and drops their `build:` and a `pull_policy: build`, so the
sidecar starts from the image prepare built with its recorded entrypoint, command and ports like any pulled image,
under its runtime `pull_policy` when it declares one. A sidecar with a `build:` section and no recorded build is
refused by the loader naming it, whatever its `image:` says (under Compose a service with `build:` is built, never
pulled, so that `image:` is only a tag), and so is one with neither key. `main`'s recorded image stands in only when
the task runs from it (build mode): a task that declares a `docker_image`, in `task.toml` or by a dataset override,
keeps that image even when the context's `main` was built for another task sharing it or before the override, as
Harbor's prebuilt mode runs the declared image. The loader hashes `environment/` once per task, when it loads, and
reads the record then; nothing is hashed per episode.

The separate verifier's image is built the same way. Harbor runs a separate verifier in a second environment whose
build context is the task's `tests/` folder and whose configuration is the effective verifier environment
(`[verifier.environment]` when declared, else a copy of `[environment]`); without a `docker_image` there it runs
`docker compose build` on `tests/Dockerfile`, whatever that file holds (a `FROM` plus `RUN` file is built, not
overlaid), and uploads no `tests/` into the image. Prepare plans that build for every task in separate mode with
`tests/Dockerfile` and no declared or inherited `docker_image`: the `tests/` folder is hashed on its own, as Harbor's
environment id hashes it, and the image is named `hb__<tests hash>-verifier` (the hash Harbor exports as
`MAIN_IMAGE_NAME` for its verifier environment, with the sidecar suffix naming the role) and recorded under that
hash as service `verifier` in `built-images.json`, its configuration in `compose-images.json`. The loader reads it,
the server grades in it and skips the `tests/` upload when the image holds `/tests/test.sh` (it uploads the folder
with a warning when it does not). A task whose verifier image fails to build is left out of the preparation like one
whose own image fails, and so is one whose `tests/Dockerfile` has no recorded build (`builds.enabled: false`, or a
record from before verifier builds): the loader refuses it the way it refuses an unrecorded `environment/` build,
naming the file, the record and the two remedies. Harbor's verifier environment dir is always `tests/` (there is no
fallback to `environment/`), so a task in separate mode with neither `tests/Dockerfile` nor a declared or inherited
`docker_image` fails Harbor's environment validation (`require_agent_environment_definition`), and the loader
refuses it with that wording. A `tests/docker-compose.yaml` may declare `main` only (its `build` section is
honoured): Harbor builds and starts any other service as the verifier's Compose project, this server starts a single
verifier sandbox, so such a task is refused by the loader and at prepare before any build. The same file is refused
at load (`gym dataset validate` and the seed alike) when it sets a runtime key on `main`, anything beside `image`,
`build` and `pull_policy` (`environment`, `env_file`, `volumes`, `command`, `network_mode`, `cap_drop`, ...), and,
beside a declared verifier image, `image`, `build` and `pull_policy: build` too; and when it carries a top-level key
beside `services`, `name`, `version` and `x-` fields (`networks`, `volumes`, `configs`, `secrets`, `include`, ...).
Harbor stacks the whole file on its verifier project (`_docker_compose_paths` in docker.py at v0.22.0), so every such
key shapes Harbor's verifier container, while this server starts the verifier from `task.toml`'s environment and the
verifier image and reads the file's `services` alone (`nemo_gym/tasks/harbor/verifier_env.py`); grading it without
the key would differ from Harbor without a word. Among the hub datasets, arc-crm-6's six separate-verifier tasks are
refused this way (their file sets `network_mode`, `pids_limit`, `security_opt` and `cap_drop` on `main`);
cadir-ci-benchmark's two ship the `main`-only shape this rule honours, but they are refused at load before the file
is read, by their `[agent]` allowlist and `[verifier.environment] network_mode = "no-network"` (the same
network-restriction check that refuses every terminal-bench-science task, all 70 of which carry that verifier
setting, the two overlay-mode ones named under "Upgrading from the pre-build layer" below among them).

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
docker path before anything is built, so a file the loader would refuse is never built and pushed under the
`hb__` name, and a file one task gets wrong leaves that task out without touching the rest of the dataset. Each
service with a `build:` section is one `buildah bud` or `podman build` of its own context and Dockerfile, `main`
first and the sidecars in file order, each inspected, pushed and recorded as it finishes, so an interrupted
prepare resumes at the next service. `dockerfile_inline` and `additional_contexts` are refused on the direct
path for any service, before the context's first build (the checked file already says so, so nothing is built or
pushed for a context that is then refused), with `builds.tool: docker` as the remedy; `no_cache` and `pull` are
ignored there. The
image name and hash do not change: the hash is Harbor's hash of `environment/`, which already covers the Compose
file.

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
`docker compose --project-directory <environment/> -f <harbor file> [-f <task file>] -f <images file> build`, as
`DockerEnvironment.start()` does (`harbor/environments/docker/docker.py` at v0.22.0, cited by permalink in
`nemo_gym/tasks/harbor/image_build.py`), with one generated override of Gym's own passed last, the way Harbor
passes its own generated overrides after the task's file: `docker-compose-images.yaml`, the `image` of every
service built (`hb__<hash>` for `main`, `hb__<hash>-<service>` per sidecar), so Gym's names win over a sidecar's
own `image:` without a `docker tag` step. One compose build builds every service with a `build:` section, as
Harbor's does; when some of a context's services are already recorded, or `main` is not built, the build names
the ones it builds. The command's environment is merged as Harbor merges it: the process
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
`${MAIN_IMAGE_NAME:-…}` or `${MAIN_IMAGE_NAME-…}`) when `main` is built; a literal other image is refused. A
sidecar-only build (a prebuilt, pull-mode or overlay `main`, or one already recorded) does not read or tag `main`'s
image, so its spelling is not checked, as Harbor's prebuilt mode never reads it.

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
file built to be the file started. Prepare refuses, naming the service and key: build `secrets`, `ssh`, `extra_hosts`, `entitlements` and `privileged` (BuildKit's
`--allow network.host` / `security.insecure`, which let a `RUN --network=host` line use the build host's network), `network: host`, `additional_contexts` other than `service:`
references, `cache_from` / `cache_to` entries carrying credentials, a `build.context` or `dockerfile` that does
not resolve under `environment/` (symlinks resolved; `${CONTEXT_DIR}` is allowed), a top-level `include` and an
`extends` with a `file`. The per-section checks run on every service with a `build:` section, sidecars included;
a sidecar's `image:` is accepted whatever it says, since the build is tagged with Gym's name for it and the seed
document is filled from the record by service name. Harbor
reads a task's Compose file from `environment/docker-compose.yaml` only; the loader also accepts
`docker-compose.yml`, `compose.yaml` and `compose.yml` at run time, and the build refuses those spellings so the
file a task builds with is the file it runs with. `docker build` is never called, so the Compose v2 plugin
(`docker-compose-plugin`) is required and its absence fails prepare with a message that names it. buildah and
podman are the OpenSandbox path: they build the Dockerfile directly and push it.

Images are built for `linux/amd64`, the platform the sandboxes run. The tool is asked for that platform
(`docker compose` through `DOCKER_DEFAULT_PLATFORM`), and an image that comes out for another one is
refused before it is pushed or recorded. Building on an
arm64 host therefore needs qemu with binfmt registered, or an x86 node. `builds.platform: OS/ARCH` overrides
the target; an amd64 micro-architecture level above the baseline (`linux/amd64/v3`) is refused, since images
are recorded by os/arch only and a later `linux/amd64` run would reuse what it built. `builds.enabled: false`
never builds. Tasks that need a build and have none recorded, the separate
verifier's `tests/Dockerfile` included, are then skipped with a message that says so.

The hash is SHA-256 over the sorted relative paths and bytes of the regular files in the build context,
exactly as Harbor computes it: symbolic links and anything under `__pycache__`, `.git` or `.DS_Store` are
skipped, and file modes are not part of it. Two tasks with identical `environment/` folders share one
image and a changed file gets a new one. `built-images.json` next to the task folders maps each context
hash to the images built from it, by service, with the tool that built each and whether it was pushed, and the
tasks that use the context:

```json
"<hash>": {
  "image": "registry/path/hb__<hash>@sha256:...",
  "tasks": ["cb100-015-ember-medical"],
  "tool": "buildah",
  "pushed": true,
  "services": {
    "main":  {"image": "registry/path/hb__<hash>@sha256:...", "tool": "buildah", "pushed": true},
    "world": {"image": "registry/path/hb__<hash>-world@sha256:...", "tool": "buildah", "pushed": true}
  }
}
```

`image`, `tool` and `pushed` at the top level are `main`'s, kept there for older readers; `image` is absent when
`main` was not built (a pull-mode `main` with a built sidecar), and the two flags are then the first recorded
service's. The flags are kept per service because one context's services may be built by differently configured
runs (a sidecar built locally after `main` was pushed, or the reverse), and each service is judged by its own. An
entry with `image` and no `services` was written before sidecars were
built and reads as `main` alone. Deleting a context's entry
from that file forces a rebuild of it at the next prepare; there is no flag for that, since a changed
Dockerfile or context is a new hash and builds on its own. Each built image's
configuration lands in `compose-images.json` beside it, so the loader reads the Dockerfile's final
`WORKDIR`, `ENV` and `USER` from it like any pulled image. A recorded image is reused when it fits the run:
a pushed image serves every run, a local one only runs that push nothing, and the platform has to match, each
service on its own record. Otherwise it is rebuilt and the record replaced, and prepare says why; a service
without a record is built and the context's other services are left as recorded. An image is recorded only after
it was built, pushed and inspected, and each one as soon as it is done, so an interrupted prepare resumes
where it stopped. Both record files are replaced atomically under a sidecar lock file next to them
(`built-images.json.lock`, `compose-images.json.lock`), so two Gym commands recording into one datasets
folder keep each other's entries and a reader never sees a half-written file. A failed build leaves its tasks
out with the reason and the others prepare. Prepare also writes every
distinct image a run of the prepared tasks starts to `results/harbor/<taskset>/images.json`, for pre-pulling: the
task images (built or pulled), the separate verifiers' images (declared, or built from `tests/Dockerfile`) and the
Compose sidecar images as their records pin them (built or from a registry); the records themselves sit in
`compose-images.json` (and `built-images.json` for the built ones) next to the task folders.

The hash names the image; it is not the whole build. The configuration record of each built image in
`compose-images.json` also carries an `inputs_digest` over everything its build read, beside the
`config_digest` of what came out: every entry of the context tree with its mode (files by content, symbolic
links by target, folders by presence, the names Harbor's hash skips included) and the service's `build` section
of the task's Compose file as the build reads it (the whole section after interpolation, an inline Dockerfile and
labels included, its paths as resolved and its build arguments as the tool is given them), so the images of one
context share the tree and differ in the section. A service that builds from another service's image
(`additional_contexts: {base: service:db}` with `FROM base`, which `docker compose build` honours) also carries
`db`'s inputs digest, transitively, so a change to `db`'s inputs rebuilds `db` and `main` both, as Compose would
build a different `main`. A recorded image is reused only when the context's hash and
its recorded inputs digest both match; a service whose inputs changed without changing the context's hash (a
mode flip, a new empty folder, a different link target, a variable the file interpolates into the build section
or a build argument taken from the environment) is rebuilt under the same name and its records replaced, the
context's other services left as recorded, and prepare says why. With `builds.enabled: false` such a task is
left out with the reason. An image recorded before the digest was kept is rebuilt once.

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
| Image | Prebuilt `environment.docker_image`; a pull-mode Dockerfile (`FROM` plus `WORKDIR`, `ENV`, `USER`, `LABEL`), started from the base image; an overlay-mode Dockerfile (the same plus `RUN` lines, run in the pulled base image at seed, no build); any other Dockerfile (`COPY`, `ADD`, heredocs, BuildKit flags, multi-stage) is built at prepare with buildah, podman or docker as `hb__<hash>` and the task runs from the built image, Compose sidecars with `build:` as `hb__<hash>-<service>`; with `builds.enabled: false` or no record such a task is skipped at prepare. The separate verifier's `tests/Dockerfile` is built the same way as `hb__<tests hash>-verifier`, whatever it holds (see "Dockerfile builds"). |
| Verification | Single-step verification, shared or separate. A separate verifier grades in `[verifier.environment].docker_image` when the task declares one, else in the task's own `[environment].docker_image` (Harbor's effective verifier environment is a copy of `[environment]`), else in the image prepare built from `tests/Dockerfile`, for the episode-sized sandbox lifetime; a declared or inherited image is never rebuilt from `tests/Dockerfile`, as Harbor's `should_use_prebuilt_docker_image` decides. A separate-mode task with neither is refused as Harbor refuses it; one whose `tests/Dockerfile` has no recorded build is left out at prepare and refused at seed. A `tests/docker-compose.yaml` may declare `main` only. |
| Users | The image's default user and `verifier.user` are supported. Explicit `agent.user` and Dockerfile `USER` overrides are rejected at seed because sandbox access cannot yet carry the agent user. This layer has no oracle-specific exemption. |
| Environment variables | Whole-value `${VAR}` and `${VAR:-default}` templates in environment and verifier variables resolve from the server's host environment at use time. Missing required values fail the episode before provisioning. Resolved values are not added to dataset rows, task snapshots, or the episode result. A task can still print its own environment into verifier logs. |
| Prebuilt-image input files | `environment/` contents are uploaded to the workdir when the task has a prebuilt image and no `Dockerfile` or Compose definition. File modes are preserved. |
| Network restrictions | Public networking only. Restricted network modes, host allowlists and deny rules are rejected by the loader rather than silently ignored. |
| Artifacts and hooks | Verifier logs are downloaded. `[[verifier.collect]]` hooks run in both modes, each in the service it names (a non-zero exit or a hook that runs past its `timeout_sec` warns, a hook the sandbox cannot run masks). In separate mode `/logs/artifacts` and the `artifacts` entries are copied into the verifier, each under its own staging root, with their `exclude` rules applied as Harbor passes them to `tar --exclude` (unanchored: `node_modules` at any depth), replacing what the verifier image ships at those paths. The verifier receives each artifact as the archive the agent's sandbox packed, so read-only, world-writable and set-id modes arrive as the agent left them; the host copy under the staging root is unpacked through a filter (widened owner modes, links kept, nothing that escapes and no device nodes) and is the gate the archive must pass plus a diagnostic, and the archive is removed once restored. A probe the sandbox cannot answer masks rather than counting as an absent artifact. |
| Image ENTRYPOINT | Runs for a plain (non-Compose) task whose image configuration is recorded: a built image, or a pull or overlay Dockerfile whose base image is recorded, unless the Dockerfile says `ENTRYPOINT []`. The sandbox starts with the provider's keepalive; the seed then starts `ENTRYPOINT + sh -c "sleep infinity"` beside it (what Harbor's Compose `command` leaves it with) as the image's `User` in its `WorkingDir` (created for a resolved Dockerfile when the task's workdir is another), after the working directory and `/logs` exist. It then waits as `up --wait` and Harbor's `run_healthcheck()` do: the image `HEALTHCHECK` first, then `[environment.healthcheck]`, both as the image's user (the image check in its `WorkingDir`, the task check in the task workdir; a `CMD-SHELL` check runs through the image's recorded `Shell`, `/bin/sh -c` when it sets none, as the daemon runs it, so `SHELL ["/bin/bash", "-o", "pipefail", "-c"]` fails `false | true` here too), probing the entrypoint's exit status before every attempt and once more before the hand-over. Exit non-zero before the hand-over fails the seed as a task error (terminal, with the output tail), as does a launcher whose own shell fails for the image's user or directory; a provider that cannot run the launcher is retried. Exit zero logs a warning once and the episode continues on the keepalive. At verification the status is read again: in the separate verifier's fresh sandbox a non-zero exit masks the sample as `harbor:entrypoint_failed` (under Harbor the container would have died and the trial errored); in the agent's sandbox it masks only when the exec holding the entrypoint returned on its own with that status, because the runtime folder is world-writable and the agent had the sandbox; otherwise it is recorded as `entrypoint-exit.txt` with an error in the log and the sample is scored. The holding exec lives as long as the provider allows (docker caps it at `exec.default_timeout_s`, OpenSandbox without `operations.background_exec` at `connection.request_timeout_s`); past that the entrypoint is untracked, logged once, and OpenSandbox deployments need `background_exec` for long-lived launchers. An entrypoint that is one `sleep` or `tail` command is not started. A separate verifier that runs in the task image gets that image's entrypoint and `HEALTHCHECK` because Harbor's effective verifier environment is then a copy of `[environment]` (its prebuilt verifier), never the task's healthcheck. A declared `[verifier.environment].docker_image` starts the same way when `compose-images.json` records it, which it does when that image is the base of a task's Dockerfile (this task's own, in pull or overlay mode, or another task's): its own `ENTRYPOINT` as its own `User` in its own `WorkingDir`, its own `HEALTHCHECK`, none of the Dockerfile's `RUN` lines, `USER` or `WORKDIR`, as Harbor's `up --wait` of the declared image leaves it. The separate verifier's sandbox is a plain one whatever the agent's environment: under a Compose task too, the verifier image's entrypoint runs there and its `HEALTHCHECK` gates `test.sh`, as Harbor's `up --wait` of the verifier image does. The output is kept as `entrypoint-stdout.txt` (and `verifier-entrypoint-stdout.txt`) next to the verifier logs. A prebuilt `docker_image` has no recorded configuration yet, so its entrypoint still does not run. |
| Other features | Multi-step tasks, in-sandbox MCP tools, Windows and TPU environments are unsupported. |

Upgrading from the pre-build layer (#4566): before Dockerfile builds, a separate verifier with no declared
image always ran in the task image (an overlay-mode task's `RUN` lines applied in the verifier's sandbox too,
masking as `harbor:overlay_failed` when they could not be), and a task shipping `tests/Dockerfile` was rejected at
seed. Now the task image is used only when Harbor's inherited verifier environment carries
`[environment].docker_image`; the verifier-side overlay and its mask are gone for a task that declares separate
mode (`separate_verifier_for_artifacts`, below, keeps both for the one shape it creates: a shared-mode task with an
overlay-mode Dockerfile forced into a fresh sandbox of the task image gets the agent's `RUN` lines there, within the
episode budget, and masks as `harbor:overlay_failed` when they cannot be applied), so a separate-mode task with an
overlay-mode Dockerfile grades in the image built from its `tests/Dockerfile` (every separate-mode hub task the
15 Oct scan labelled overlay ships one: 20 in tb3-preview-v2, tb4-preview and terminal-bench-4-no-prebuild, and
terminal-bench-science's two, which a `[verifier.environment]` network restriction refuses at load before that) and
is refused without one. A `tests/docker-compose.yaml` with services beside `main`, with a runtime key on `main`
(`environment`, `env_file`, `network_mode`, ...) or with a top-level `networks`, `volumes` or `include` is refused
at load (arc-crm-6's six separate-verifier tasks; see "Dockerfile builds" above), and a `tests/Dockerfile` verifier needs a
record in `built-images.json`, so a dataset prepared before this layer has to be prepared again (`gym dataset
fetch` or `gym dataset validate`) on a host with buildah, podman or docker installed. With `builds.enabled: false`
the tasks that need a verifier build are left out with the reason, so an existing dataset's eligible task set can
grow (verifiers now built from `tests/Dockerfile`) or shrink (shapes now refused) between the two layers.

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
`[verifier].timeout_sec`.

Every episode runs under its task's own time budget (`enforce_task_timeouts: true`). The deadline starts
when the sandbox is ready, after the image pull, its retries and the overlay, and covers
`harness_setup_allowance_s`, `[agent].timeout_sec`, `[verifier].timeout_sec` and `verifier_grace_s`, with a
second sandbox start for a separate verifier; the seed has its own bounds (`sandbox_ready_timeout_s` per
start, `image_pull_retries`) and is never charged against it. An episode that overruns it fails with
`failure_kind: harbor:task_deadline` (retryable), naming the stage it was in. The whole task budget (one
sandbox start, plus `[environment].build_timeout_sec` for an overlay, plus the deadline) never exceeds
`default_episode_timeout_seconds`, which stays the run-wide cap. Sandbox lifetime covers that budget, the
final cleanup budget (`cleanup_timeout_seconds`) and `sandbox_ttl_slack_s`, so setup and queued agent work
also fit; both budgets must be finite. Set `++<instance>.environment_servers.harbor.enforce_task_timeouts=false`
to run with the run-wide deadline alone. Sandboxes are normally stopped as soon as the episode ends.

Before `test.sh` runs, in either mode, the server writes the agent transcript to `/logs/agent/trajectory.json`
in ATIF v1.7 (the version Harbor 0.22 and 0.23 read), exported from the Gym transcript by
`nemo_gym.atif_export.transcript_to_atif`. Harnesses that feed terminal output back as user turns (Terminus)
export with those turns as the previous step's observation. Shapes a model causes still export and are
graded: a turn cut off inside its reasoning (a reasoning item no answer follows) becomes an agent step with
an empty message and `extra.nemo_gym.incomplete: max_output_tokens`, several reasoning segments join with
newlines, empty or encrypted reasoning exports without `reasoning_content`, tool arguments that are not
a JSON object become `{}` with the original string under the call's `extra.nemo_gym.raw_arguments`, and a
refusal part exports as the step's text, flagged `extra.nemo_gym.refusal` with reason `refusal` (Harbor's
Terminus records an empty step there and grades on; the text is kept here); the trajectory then says
`conversion.status: partial` with the `reasons`. A transcript the exporter cannot
represent at all (a copied assistant turn, a later system turn, a tool result for no call or a second
`function_call_output` for the same `call_id`, item types ATIF has no place for) masks the sample with
`failure_kind: harbor:trajectory_export_failed`, so a converter defect can never appear as a reward; a copy
of the file sits in the verification's log directory.

Operator secrets (judge API keys and the like) go in the server's `secrets` block, by environment variable
name. They fill `${NAME}` templates in a task's `[verifier].env`, which is the whole `test.sh` process
environment, and reach nothing else: a secret no template names is passed nowhere (Harbor's verifier
resolves its env the same way), and a `${NAME}` template in `[environment].env` or
`[verifier.environment].env` still resolves from the server's own environment only. Secrets never enter a
task folder or a row.

A sandbox start that fails on an image pull (the agent's sandbox, a Compose group or a separate verifier's
sandbox) is retried `image_pull_retries` times in the server, with jittered backoff and behind a per-server
gate, before the failure is reported as retryable. A slow
agent-session close that fails only by timeout no longer discards the finished turn: the episode is
verified and scored with `ng_agent_observations` unset.

`separate_verifier_for_artifacts: true` verifies every task that declares `artifacts` in a fresh sandbox even
when its `task.toml` keeps Harbor's shared mode, so an agent running as root cannot tamper with the grader. The
forced verifier runs in the task's own image, an overlay-mode Dockerfile's `RUN` lines applied again (counted in
the episode budget), and the sample masks as `harbor:overlay_failed` when they fail there. Tasks without artifacts
keep their declared mode.

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
`manifest.toml`, whose per-task pins carry the task's package and that content hash, so a later fetch never mixes another package or a moved `latest` into the folder. A local task folder, or a folder of task folders, works the same way; task folders may be
grouped in subfolders (the search is recursive, `task_id` stays the task folder's own name and must be unique
across the dataset, and `dataset.toml` and the recorded image configurations sit at the dataset root). The
taskset mapping records the relative path of a task found below the first level under `paths`. The fetch stages its clone in a hidden `.nemo-gym-harbor-*` directory inside the dataset folder, which the search skips. A fetch interrupted under a `main` build from before this change staged under an unhidden `nemo-gym-harbor-*` name; the recursive search sees that clone and refuses the dataset (a duplicate task ID, or a task the manifest does not record) rather than running it. Delete those directories once; nothing in them is needed. The generated run config declares the materialized rows as the agent's
validation dataset and binds the per-taskset environment server to that agent. `gym dataset validate` reports,
per task, the verifier's wall time (`verify_s`) and the network it ran under (`network`). Running model episodes
requires a harness that accepts `sandbox_access` in its session seed and leaves the sandbox to the server
when its session is closed.

### Dataset identity

`manifest.toml` beside the task folders holds derived facts only, never a copy of the name or version
(those live in `dataset.toml`): `[dataset]` `source` (`harbor-registry`, `harbor-package-store`, `git` or
`local`), `digest` (the dataset's identity: a hash over the sorted per-task digests and `dataset.toml`),
`dataset_toml_digest` on its own, and per task
`[tasks."<id>"]` `digest` (the folder digest rows carry as `_ng_digest`) with its upstream pin (`git_url`,
`git_commit_id`, `path`, or `package`, `content_hash`). The output is deterministic: fixed key order,
sorted tasks, no timestamps, so equal manifests mean identical inputs and two runs compare by `digest`.
Two datasets that differ only in a `[gym]` timeout override are different datasets, on purpose.

Fetches write the manifest and record only the tasks they fetched; a task folder beside them that neither the
source names nor a previous fetch recorded stops the fetch (move it aside). Prepare writes a manifest for a
local folder, unless git tracks the folder (such as `data/tasks/` here), whose identity is computed in memory.
Before any image is built or pushed, prepare recomputes the per-task digests from the folders and refuses a
fetched dataset whose folders no longer match (the folder was edited by hand or deleted, or its source moved):
the error names the tasks and says to delete the folder, or move it aside, and run again. A single task folder
inside a fetched dataset (`gym eval run <dataset>/<task>`, or a task moved deeper under the dataset folder) is
held the same way to the manifest that check holds it to, before any image is recorded or built: the nearest
fetched manifest within the dataset check's own reach (at most six levels up, through visible folders that are
not links; a `local` manifest on the way is passed over, and a manifest further up, another tool's `manifest.toml`
in a project root for one, is not read). Its digest is compared with the one recorded for it, and an edited
folder, one the fetch did not record, or one it never finished, is refused with the same recovery step; a
subfolder of a fetched dataset is no dataset of its own, so preparing it as one is refused, naming the dataset,
rather than writing a `local` manifest inside the fetched tree. Preparing one task or one subfolder is thus no way
around the check (a single task run reports the fetch's source, not `local`; nothing is written beside the task;
a task the dataset check never sees, deeper down, under a hidden or linked folder or inside another task folder,
or one with no fetched manifest in reach, is held to nothing). The manifest holds the task's content, not its settings: a single task
folder target is prepared with the `dataset.toml` and image records beside it, as any single task target is, so a
task moved into a subfolder runs without the dataset's `[gym]` table; to run it under that table, prepare the
dataset folder and select the task. A registry fetch
re-hashes the folders it keeps only when it also adds tasks (a package-store fetch compares every folder it finds
on each run), while prepare recomputes every task digest on each run, so a hand-edited folder a re-fetch kept is
caught at prepare. A manifest naming a
source this Gym does not know is refused too, never read as local. When only `dataset.toml` moved, the derived
facts are rewritten without a fetch. A `latest` package-store folder whose manifest was written for another
version of the dataset is refused the same way: the error names the version difference and says to delete the
folder, or move it aside, and run again to fetch the store's `latest`, or to pin the version you have with
`harbor:<org>/<name>@<tag>`. A manifest written by a newer Gym (a `schema_version` this Gym does not read) is
refused untouched, by fetch and prepare alike, until Gym is upgraded or the folder moved aside. The dataset digest is
recorded in the run config (`tasksets.<name>.dataset_digest` and `dataset_toml_digest`), in every materialized
row as `task_data._ng_dataset_digest`, and, because rollout collection copies a row's `_ng_*` task data onto the
record it writes, in every `rollouts.jsonl` row as `_ng_dataset_digest` (beside `_ng_digest`), so every results
file says which dataset ran.

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

## Compose service fields

Beyond images, commands, `environment`, `depends_on`, `healthcheck`, `expose`/`ports`, `shm_size` and
`cap_add`, a task's Compose services may use the fields below. Prepare gives each its meaning (or fails naming
the field) before any sandbox starts, and the server re-reads the file through the same check at seed. Error
messages read `<task>: environment/docker-compose.yaml: service '<name>' <field> ...`.

| Field | Meaning in a sandbox group |
|---|---|
| `deploy.resources` | `limits.cpus`/`limits.memory` size the service's sandbox (then `reservations`, then the legacy `cpus`/`mem_limit`). `reservations.devices` with the `gpu` capability and a numeric `count` (or `device_ids`) becomes the service's GPU count; `count: all`, other capabilities, `limits.pids` and device `options` fail prepare. `replicas` must be 1; `restart_policy` is recorded like `restart`; every other `deploy` key names swarm scheduling and fails prepare. |
| `restart` | Accepted (`no`, `always`, `on-failure[:n]`, `unless-stopped`) and recorded on the collection (`restart_policies`). Nothing restarts: the group starts each service once, and an exit is reported through the dependency gates, as before. |
| `pull_policy` | `always`, `missing` (also `if_not_present`, `daily`, `weekly`, `every_<n>`, since the group keeps no pull history) or `never`. Applied per sandbox where the provider can (OpenSandbox's image pull policy extension, Docker's `--pull`) and recorded on the collection (`pull_policies`) otherwise, so the deployment's default pull behaviour applies. `build` is dropped with the build it names (a service prepare built, a pull or overlay `main` restating Harbor's template) and fails prepare on any other service. |
| `platform` | Must match the deployment platform, which prepare and the server read from the same setting: the environment server's `sandbox_provider_options.platform` (`{os, arch}`, as `++harbor_environment_server.environment_servers.harbor.sandbox_provider_options.platform.arch=arm64` names it) when set, else `linux/amd64`. `x86_64` and `aarch64` are the usual aliases. Image configurations are still recorded for `linux/amd64` only, so a Compose task runs end to end on amd64 deployments today. |
| `networks` | Custom networks map onto the group's one pod network as name scopes: a service resolves itself, the services sharing a network with it (by name and by their aliases on that network) and nothing else. Network declarations take `name`, `driver`, `external`, `attachable` and `labels` with no further meaning; `internal: true`, `ipam` with a configuration, `driver_opts` and `enable_ipv6: true` ask for isolation the pod does not provide and fail prepare, as do static `ipv4_address`/`ipv6_address` entries on a service; `internal: false` and `enable_ipv6: false` (Compose's defaults written out, as every kumo task does) ask for nothing and are accepted. |
| `ports` | Short (`"8080"`, `"8080:80"`, `"127.0.0.1:8080:80/udp"`) and long syntax take one shape at prepare, so the adapter reads mappings. A service's ports are endpoints the provider allocates: a container port alone (or `published: 0`) is accepted; a fixed published port, a `host_ip`, a protocol other than TCP, a `mode` other than `ingress`, a port range or an unknown option fails prepare naming the service, the refusal the adapter used to make at seed for the long syntax only (a short `"8080:8080"` passed prepare and failed every episode). A service that only needs to be reached by its peers lists the port under `expose`. |
| `extra_hosts` | Hostname-to-IP entries (list or mapping form) added to the service's hosts. `host-gateway` fails prepare. A sidecar whose image runs as a non-root user gets no host injection (the group resolves its peers through its environment URLs instead), so it cannot take them: prepare fails naming the service once the image configurations are recorded, and the adapter refuses the same document at seed. |
| `post_start` | Hooks run once the service's command has started, in order, before the group waits for health (as Compose runs them). `command` (string through the shell, or argv), `user` (default: the service's), `working_dir` and `environment`. A hook that exits non-zero fails the seed with its index and output. `privileged` hooks and `pre_stop` fail prepare. |
| `tmpfs`, `ulimits`, `sysctls`, `security_opt` | Behind a provider capability check (next paragraph). |

GPU sidecars route per service: a service whose `deploy.resources` reserves GPUs runs on the `gpu_sandbox_provider`
block when one is set, with that block's `default_metadata` and, with `request_gpu_type`, the task's
`gpu_types[0]`, exactly as the main service does when `task.toml` declares GPUs. The task's own services keep
following the task's declaration, so a CPU task with one GPU sidecar puts only that sidecar on the GPU block. Both
blocks must be able to reach each other's sandboxes on the network. Everything about a sidecar on another block
goes through that block's provider: its sandbox, its `network_mode` forwarding relay, and its volumes, which are
seeded and mounted on that provider's `shared_storage`, so a volume shared with a service on the task's block
needs the same root on both.

**Capability check.** `tmpfs`, `ulimits`, `sysctls` and `security_opt` need a provider that implements
`SupportsSandboxServiceOptions` and names the field in its `SERVICE_OPTION_FIELDS`. Prepare reads the deployment's
provider names from the keys of the `sandbox:` block the command names (the `--sandbox` provider file and the
configuration's own block) and, when `harbor_environment_server.environment_servers.harbor.gpu_sandbox_provider`
names a GPU block, from that block's keys too (values are never resolved, so credentials are not needed). Each
service is checked against the provider it will run on, the GPU block's for the services of a GPU-declaring task
and for a sidecar that reserves GPUs, and prepare fails with
`service '<name>' <field> is not supported by the <provider> sandbox provider` when that provider lacks it. Docker
applies all four as `docker run` flags; OpenSandbox applies none (a pod exposes no tmpfs, ulimit, sysctl or
security option through its API). When prepare cannot tell the provider, the adapter runs the same check against
the live provider before creating any sandbox.

## Compose volumes

A service's `volumes` entries, short (`SOURCE:TARGET[:ro]`) or long syntax, are resolved at prepare and run
in one of three ways, with the same planner at prepare and at seed:

| Volume | How the sandbox group runs it |
|---|---|
| Bind mount of a task file or folder (`./seed.json:/app/seed.json`, `${CONTEXT_DIR}/data:/data`) | Uploaded into the service once it starts, before its command runs, so the process finds the file where the Compose file mounted it. The receiving folders are created when missing, and the target, every folder a file lands in (pre-existing or not: a `0700` folder the image ships at the target becomes readable, `0755`), the folders created for the upload and the uploaded files are made readable by every user (`chmod a+rX`, never recursive); what the image already holds under the target and the upload did not write keeps its mode; `ro` is not enforced: the copy is an ordinary tree in the service, owned and moded as the provider's upload leaves it, where Harbor's `:ro` bind is a mount the kernel keeps read-only for root too. An agent that can write the copy (root, or a user who owns the folder above it and can rename it aside) can therefore change any task data bound `:ro` into `main`, whatever folder it comes from (`environment/`, `tests/`, `solution/`, `steps/` or a top-level task file), and the verifier can read the changed copy in either mode: in shared mode `test.sh` runs in `main` itself after the agent's turn; in separate mode it runs in a fresh single sandbox that receives no Compose upload at all (the bind's copy is not there, only what the verifier image itself ships at that path, unless collected) and sees the agent's tree only through `/logs/artifacts` and the task's top-level `artifacts` entries (the `[verifier]` table carries no such key; a misplaced one is ignored), restored at their own paths with the modes the agent left, so a `:ro` target under a collected artifact path (`./reference:/app/reference:ro` with `artifacts = ["/app"]`) reaches the verifier as the agent left it, where Harbor restores artifacts the same way but its kernel-enforced `:ro` mount keeps the collected copy pristine. What `test.sh` must trust it reads from `/tests`, which is emptied and re-uploaded from the task's `tests/` after the agent's turn in shared mode and uploaded into the fresh sandbox in separate mode (when its image ships no `/tests`); a bind of a `tests/` path at another target (`../tests/golden:/golden:ro`) is an ordinary upload like the rest. A folder's tree is recreated, empty folders included, over whatever the image already has under the target (a merge; a bind mount would hide the image's folder, so a task that relies on hiding, say an `/etc/nginx/conf.d` with a default site, behaves differently), and the names the task digest skips (`__pycache__`, `*.pyc`, `.DS_Store`, editor swap and backup files) are not uploaded, so what reaches the service is what the digest covers. A task file or folder bound into several services is uploaded into each service separately: the copies are independent, so a write in one service is not seen by the others, where a Docker bind mount would share one host folder. A source the task folder does not contain, a link leaving it, a link with an absolute target (the server plans on a copy of the task, where it would leave the folder), a folder that contains a link, or a source outside what the task digest covers (`environment/`, `tests/`, `solution/`, `steps/` and the top-level task files, free of editor-leftover names such as `__pycache__`; a `../data/seed.json` could change without changing the task's identity) fails prepare with the path. The target must be an absolute path below `/` with no `..`. |
| Named volume declared under top-level `volumes:` (shared between services or not) | A folder on the provider's shared storage (`shared_storage` in the OpenSandbox block), mounted into every service that names it, copied up from the image's own content unless `nocopy`, removed when the group stops. When the `sandbox:` block's provider configures no shared storage (no `shared_storage.host_path`), a task that mounts one is left out at prepare naming the setting and the block (a GPU-reserving sidecar's volumes are answered by the `gpu_sandbox_provider` block), rather than failing at every seed; prepare reads the answer from the block's one provider key, so a block it cannot read leaves the decision to the seed. |
| `external: true` volume, or a bind mount of a host path outside the task | Resolved through `volume_sources` in the `sandbox:` block: a mapping of the volume name or host path to a path under the shared storage root. A volume with no entry fails prepare with its name. Only the entries a task's Compose file resolves this way reach the sandbox adapter, so an ordinary named volume that happens to share a mapped name stays the group's own, isolated per episode. |

```yaml
sandbox:
  opensandbox:
    shared_storage: { host_path: /mnt/shared }
  volume_sources:
    pgdata: datasets/pgdata        # volumes: { pgdata: { external: true } }
    /var/lib/models: models        # - /var/lib/models:/models:ro
```

`gym eval run` and `gym dataset validate` read `volume_sources` from the `--sandbox` provider file and the
configuration the command names; `prepare_target(..., volume_sources=...)` takes it directly. Without one,
every external volume and foreign bind mount is reported at prepare. When the configuration sets the server's
`gpu_sandbox_provider` (`harbor_environment_server.environment_servers.harbor.gpu_sandbox_provider`), a task
that declares GPUs is checked against that block's `volume_sources` instead, the block the server seeds it
from, and so is a sidecar that reserves GPUs through `deploy.resources` (the server routes it to the GPU
block, and its external volumes are seeded and mounted through that block's mapping), so a mapping has to be
present in each block whose services need it; `prepare_target(..., gpu_volume_sources=...)` takes that mapping
directly. A bind of Harbor's `/logs` tree (`${HOST_VERIFIER_LOGS_PATH}:/logs/verifier`) onto `main` is not a
mount: the sandbox adapter creates the path in the agent's sandbox before `main`'s command, hooks and
healthcheck run, as Docker creates a bind target at container create, and a different target
(`${HOST_ARTIFACTS_PATH}:/app/output`) becomes a link to it, since under Harbor both are one host folder. The
link is made after the overlay's `RUN` lines and before the uploads, where Harbor's bind sits: a folder the
image ships at the target (`RUN mkdir -p /app/output`) is replaced, with a warning in the server log when it
was not empty, as the bind would hide it, and a task file bound under the target lands in `/logs/artifacts`.
A sidecar that binds a path of the tree (kumo's `verifier` service binds `${HOST_VERIFIER_LOGS_PATH}:/logs/verifier`
and appends `actions.jsonl` there for `main`'s `tests/score.py` to read) shares it with `main` the way Harbor's one
host folder is shared: the path becomes a named volume on the provider's shared storage (declared as
`nemo-gym-logs-verifier`, `nocopy`, created and removed with the group), mounted at the sidecar's target and at the
path itself in `main`, whether or not `main` binds it, since that is where the verifier collects from; `main`'s
own instruction for the path stays, so the adapter still creates and widens it before the commands. Without shared
storage the task is left out at prepare naming the setting, like a named volume: every service the volume is
mounted in needs the storage of its own block, `main` (the task's block) as well as the binder (the
`gpu_sandbox_provider` block when it reserves GPUs), and the refusal names the block and service that lack it.
A target that nests its source fails prepare, however the target is spelled (`/logs/verifier:/logs/.` names
`/logs`), and so does a logs path with a `..` segment (`/logs/../etc`), on
`main` or a sidecar, since the path is used as written (the adapter's `mkdir -p`, the shared volume's name and
target); a sidecar's logs path with a character outside letters, digits, `_`, `.` and `-` (`/logs/my dir`) fails
prepare too, since the shared volume's name is made from it and the adapter mounts only Compose volume names.
Anonymous volumes, `tmpfs`, bind `propagation`, volume drivers and
other unsupported options fail prepare naming the field.

The `x-sandbox` key the planner writes for the sandbox adapter (`uploads`, and the sidecar adaptation's
`hosts`/`resolve_environment`) is Gym's output, never task input: a Compose file that carries one fails
prepare, and the server, naming `x-sandbox`. Upload sources are read in the server's process, so the adapter
only accepts sources under the task copy it is handed.

## Where settings live

Four homes, one owner each. A setting sits on this server's config only if every dataset on the same
deployment wants the same value; a unit test pins that list.

| Home | File | Owner | Holds |
|---|---|---|---|
| Task | `task.toml` | task author | image, resources, timeouts, env, verifier, agent user (read-only once fetched) |
| Dataset | `dataset.toml`, `[gym]` table | dataset author | `[gym]` `license` (one of Gym's dataset license values, required for `train` and `validation`) and `type` (`train`, `validation`, `example` or `benchmark`; unset means `validation` with a license, else `benchmark`); `[gym.defaults]` `timeout_multiplier`, `resource_multiplier`; `[gym.tasks."<id>"]` `environment` and `verifier` overrides in Harbor's own fields (`env` merges, other fields replace) and `artifacts` additions appended to the task's own |
| Deployment | the `sandbox:` provider block | operator | image rewrites and registry credentials (`images`), root setup commands (`setup`), networking, shared-memory labels, shared storage and `volume_sources` for Compose volumes |
| Run | run config | the run | provider selection, GPU routing, repeats, output paths, `sandbox_resources_override` for parity runs |

`gym dataset init <name>` writes a dataset in this shape, and `gym dataset validate <folder>` with no extra
config is the test that a dataset is self-contained.
