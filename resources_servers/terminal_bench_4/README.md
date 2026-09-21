# TB4 resources server

## Contract

`/seed_session` provisions the pinned task and returns its sandbox descriptor,
connection configuration, instruction, execution user, MCP/skills metadata, and
official agent timeout. The mini-SWE agent server connects to that sandbox and
owns harness setup, model calls, and the rollout loop. `/verify` collects artifacts,
runs the separate official verifier, and cleans up the task's resources.

Seed requests include `task_name`, `task_ref`, `dataset_ref`, and `rollout_id`.
The task must match the configured manifest. The agent supplies a stable
`client_session_id`, so identical seed retries share provisioning before the first
cookie response. Verification retries share one finalizer and replay its result;
conflicting requests fail. Run one resources worker per artifact directory.

The resources server retains the Compose creator, TTL renewal, and shared-volume
ownership throughout the episode. The agent closes its attached transport after
joining the mini-SWE worker, then submits its response, termination, execution
status, timings, and harness metadata to `/verify`. Failed setup skips grading;
a started agent is graded even after failure, cancellation, or timeout. Official
zero and nonzero grades survive agent failure or timeout.

Completed version-3 records can replay after a restart. Active records cannot
resume after resources-process restart. Shutdown drains active verification for
the configured grace period, then interrupts grading and awaits cleanup, including
provisioned sessions that never reached verification. Abrupt resources-process
death stops renewal; provider TTL is the sandbox cleanup fallback.

Model-call capture and harness trajectories belong to the agent server. Resource
artifacts, including collected remote `/logs/agent` files, remain under the trial
directory. The seed's connection configuration is used only for the internal
agent/resource exchange and is not persisted in session records.

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
