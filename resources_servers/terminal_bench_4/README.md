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
execution user. This allows non-root images to initialize their log directories
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
