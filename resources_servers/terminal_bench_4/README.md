# TB4 resources server

## Contract

The Gym agent endpoint forwards `/run` to this server. One runner owns provisioning,
harness setup and execution, collection, separate verification, and cleanup.
It imports the generic mini-SWE harness and passes the existing `Environment.main`
sandbox directly. The harness never attaches, releases, grades, or destroys it.

`/run` accepts the Gym response parameters, `task_name`, `task_ref`, `dataset_ref`,
and `rollout_id`. The task must match the configured manifest; dataset rows cannot
select packages or override grading instructions. The adapter supplies a stable
`client_session_id` so HTTP retries share the same episode even before a cookie
response. Identical requests await one execution or replay its recorded result;
conflicting requests fail. Run one resources worker per artifact directory.

Harness setup, including working-directory discovery, has a separate 360-second
budget. Queueing and provisioning do not consume it. The runner supplies the
minimum of the official agent budget and any configured cap. The harness closes
pending I/O and joins its synchronous worker before the runner quiesces sandbox
process groups, collects artifacts, and starts the separate verifier. Official
zero and nonzero grades survive agent failure or timeout.

`/cancel_session` cancels setup or agent execution and waits for cleanup. An episode
that reached execution is still graded. Once finalization starts, cancellation
waits for that finalizer. `/verify` only replays an already recorded result; it
cannot initiate grading or supply an alternative agent result. Completed results
can also be replayed after restart. Version-2 records contain the complete run
request and response; records from the former split runner cannot resume here.

Shutdown cancels active harnesses and drains finalization for the configured grace
period, then interrupts grading and awaits cleanup. HTTP disconnection does not
cancel the runner. Abrupt process death stops renewal; provider TTL is the cleanup
fallback. Active records cannot resume after restart. The runner retains the
Compose creator and its relay/volume ownership throughout the episode.

Model calls use Gym's model server with the incoming cookies and rollout/token
capture routing. Harness trajectories live in the trial's `harness/` directory;
remote `/logs/agent` files are collected separately into `agent/`.

## Optional local training packages and reference solutions

The official public-package path remains the default. Set `local_task_packages:
true` and `manifest_path` to a trusted host manifest to use prebuilt training
images. The manifest has `format: "gym-tb4-local-v1"`, a dataset `ref`, and
`tasks` entries containing `name`, content-pinned `ref` (`sha256:...`), and
absolute host `path`. Names are used verbatim in this mode. Each package uses
the same strict task schema and content-hash check as public packages; rows
cannot supply arbitrary package paths. Shared-verifier tasks remain unsupported.

Local packages stage their host-owned `tests/` tree into `/tests` in the fresh
verifier, as root, after restoring the declared artifacts. They do not inject
tests into the agent's working directory. The test command still runs as
`verifier.user`; when absent, the existing image-default behavior is preserved.
Images must support the requested identities and root setup operations.

### Root-bootstrap compatibility images

For single-container images republished with `USER root`, set
`environment.root_bootstrap_image_users` to a trusted mapping from each **new,
effective image reference** to the **original image's OCI User**. Include both
agent and verifier images; prefer immutable digest references. An empty original
OCI User is `""` (root), not missing metadata. For example:

```yaml
root_bootstrap_image_users:
  registry.example/task:root-bootstrap: agent
  registry.example/grader:root-bootstrap: grader
```

This option does not rebuild images or override the backend's startup user.
Each sandbox must already start as root; the runner checks this with an
image-default command and fails before task execution otherwise. Missing image
metadata, unavailable users, or failed switches are errors, never root fallbacks.
Omitting the option retains existing behavior; Compose is rejected in this mode.
Account names and numeric UIDs are supported. Explicit `USER user:group` values
are rejected rather than silently losing the group override.

| Operation | Execution identity |
| --- | --- |
| Stage trusted solution/tests | Root |
| Golden solution and live Mini-SWE commands | `agent.user`, otherwise original agent-image User |
| Run verifier tests | `verifier.user`, otherwise original verifier-image User |
| Healthcheck / collection hook without its own user | Original image User for that environment |

The resolved agent identity is passed to both harnesses, not just resource-owned
commands. Existing named-user switching remains: removing it would run agents
as the root bootstrap user. Log setup probes the role's UID/GID and prepares only
`/logs`, `/logs/agent`, `/logs/verifier`, and `/logs/artifacts` for that role,
including EFS mounts. No workspace ownership or permissions are repaired.
Per-role `sandbox/<session-id>.identity.json` records bootstrap UID, original
image User, configured/effective execution user, and observed UID/GID.

Keep the original USER record independently of the new image, preserve task
users, and perform a live golden/model smoke before qualifying a new image.
The image-startup metadata below can be combined with this mode; it preserves
the new image's entrypoint and does not impose a service account.

### Reference solution checks

Set `execution_mode: oracle` for a reference-solution check instead of Mini-SWE.
The runner stages the trusted `solution/` directory into `/solution` as root,
makes these reference assets readable, then executes `bash /solution/solve.sh`
as `agent.user` (image default when absent), with the same discovered working
directory and task timeout. It does not change workspace permissions or invoke
a model. Artifact collection, separate grading and cleanup use the normal
lifecycle. `oracle/identity.json`, stdout/stderr and `oracle_exit_code` distinguish
solution failures from verifier errors; reward 1 is still required for a golden
pass. `execution_mode: miniswe` is the default for ordinary rollouts.

## Image startup: Compose and standalone

`environment.compose_image_configs` is the shared OCI metadata catalog for both
Compose and standalone images. Despite its historical name, no Compose file or
conversion is needed for a standalone image. The runner automatically uses its
recorded `Entrypoint` followed by `sh -c "sleep infinity"`, matching Compose's
default `main` service. It does **not** run the image's raw `Cmd`: an inherited
`python3` or `bash` command can exit immediately. With no entrypoint, the startup
is just the keepalive. Compose's explicit overrides and sidecar commands retain
their existing behavior. Standalone agent and verifier roles each remain one
ordinary sandbox, with unchanged users, workdirs, health checks and networking.

The catalog maps image references to records with `image`, `os`, `architecture`,
and the OCI `config`. Paths are absolute or relative to the Gym root. Acquire
and digest-check the metadata upstream; the runner does not query a registry.
Standalone records must match the effective image after `image_rewrites` (or
pin the same SHA-256 image digest), describe Linux/amd64, and have string-list
or null `Entrypoint`/`Cmd` fields. Shell-form entrypoints must already contain
their image-recorded shell argv; they are not parsed again.

The original official catalog contains only Compose images. A standalone image
absent from that shared catalog retains the provider's old keepalive behavior,
with a warning; the runner cannot infer an unrecorded entrypoint. No catalog
also retains the old behavior. To preserve startup for new datasets, include
**both** their agent and verifier images in the catalog.

The previous `environment.single_container_image_configs` path remains accepted
as a standalone-only compatibility override; when provided, it still requires
every effective standalone image to be present and fails on missing records.
Its startup semantics now also match Compose `main` (ENTRYPOINT + keepalive,
not ENTRYPOINT + raw CMD). Neither metadata path is a boolean startup toggle.

## Non-root Compose services

When loading agent Compose YAML, two task-specific adaptations use the
[Compose extensions](../../fern/versions/latest/pages/infrastructure/sandbox/compose.mdx):

- `medical-claims-processing`: `playwright-mcp` keeps `pwuser`, disables host-file
  injection with `x-sandbox.hosts: []`, and resolves `BROWSER_URL` to the workspace
  sandbox IP via `x-sandbox.resolve_environment`.
- `payments-pipeline-fix`: `kafka` keeps `appuser` and disables host-file injection.
  Its single-broker controller uses localhost; clients retain the `kafka` alias
  needed by the advertised listener.

Both services use their image's default user, omitting the redundant explicit
`user` value copied from image metadata. This avoids the provider attempting
`su` from a non-root process to the same user.

These changes apply only to the generated runtime YAML. Pinned task packages,
other services, and verifier environments retain their original configuration.

## Shared EFS logs

The benchmark profile sets `environment.efs_logs_host_path` to
`/mnt/efs/data/shared`. Each episode creates a unique EFS directory with separate
agent and verifier subdirectories mounted read-write at `/logs`. The image's
default UID/GID owns its log root with mode `755`; workloads keep their original
execution user. With root-bootstrap identity metadata enabled, the configured
role's UID/GID owns these harness log directories instead of the bootstrap UID.
This allows non-root images to initialize their log directories
and keeps root verifier reward-directory protections effective. Compose mounts
these logs in `main`; sidecar mounts and collection order remain unchanged.

A helper (`environment.efs_logs_init_image`, configured as `python:3.13-slim`)
initializes ownership and remains alive until both workloads are stopped. It
reuses the collected `/logs/artifacts` archive through EFS after agent teardown,
avoiding its upload from the resources host to the verifier. The archive is
checked against its collected digest, data-filtered, and repacked just as in the
host transfer. Exclusions and the local artifact manifest/files remain intact.
Overlapping artifact declarations and unavailable snapshots use the existing
ordered host restore. Agent logs, undeclared files, and agent-written reward
files do not leak into the fresh verifier role.

With split endpoints, a GPU requirement in either the agent or verifier environment
places the entire task on the GPU deployment, including CPU-only roles, Compose
sidecars, and storage helpers. Tasks without a GPU requirement use the CPU deployment.
Individual containers retain their declared resource requests; helpers do not request
GPUs. This keeps each task on one EFS share and network even when the endpoint pools
use different storage. An endpoint that explicitly
rejects the host mount with `VOLUME::HOST_PATH_NOT_ALLOWED` uses the original
filesystem/transfer lifecycle, recording `efs_logs_fallback` in diagnostics.
This preserves existing healthy GPU tasks on deployments without EFS support;
it does not fix non-root log creation on those deployments. Other provisioning
errors remain errors. Set `efs_logs_host_path: null` to disable EFS explicitly.

Normal completion, cancellation, and handled failures remove the owned EFS
directory after workload teardown and then destroy the helper. If workload
deletion fails, EFS data is retained to avoid deleting a live mount. Persistent
session records include the helper ID and exact EFS host path/subdirectory for
recovery. Provider TTL expires sandboxes after abrupt process death, but EFS data
requires separate cleanup in that case.
