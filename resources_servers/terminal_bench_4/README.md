# TB4 resources server

## Contract

1. `/seed_session` accepts `task_name`, `task_ref`, `dataset_ref`, and `rollout_id`.
   The task must match the configured manifest. Dataset rows cannot select local
   paths or override grading instructions. The response includes a session ID,
   main-sandbox descriptor, instruction, user, working directory, setup/agent
   budgets, MCP declarations, and a task skills directory.
2. The agent attaches with `owns_sandbox=False`, installs/configures its harness,
   and calls `/start_session`. Resources return the remaining official agent
   budget. Repeating this call does not restart the clock.
3. `/verify` accepts the session ID, Gym response/usage, termination reason, and
   artifact references. Resources stop tracked harness process groups, collect
   task-declared state, run the official verifier, and destroy the
   collection. Official zero and nonzero rewards survive agent failure/timeout.
4. `/cancel_session` ends an abandoned setup or running episode. A running
   cancellation still permits official grading. Closing a borrowed client never
   destroys the resources-owned sandbox.

The states are preparing, ready, agent running, verifying, and closed. Session
cookies bind access to the originating resources session. The agent supplies its
stable `client_session_id` so retries before the initial cookie response also reuse
the same episode. Its `execution_id` stays fixed across resources HTTP retries;
another worker invocation for the same rollout is rejected before attachment.
Run one resources worker per artifact directory. Concurrent duplicate
seeds share one attempt; conflicting identities fail. Concurrent verification
requests share one lifecycle and the first accepted agent result. Completed
verification retries return the recorded response, including after restart.

The server retains the Compose creator and its relay/volume ownership for the
whole lifecycle. Shutdown cancels preparation and drains or interrupts finalization, then awaits cleanup. Client
HTTP disconnection does not cancel the resources task. Abandoned setup expires
at the setup deadline; abandoned execution expires at the official agent deadline.
Abrupt process death stops renewal; provider TTL is the cleanup fallback. Active
persisted episodes are rejected after restart and cannot be resumed safely. Use a
new rollout identity for a new attempt, not a stale descriptor.

`evaluation_completed` means an official reward was retrieved. A scored negative
has `reward=0` with no infrastructure error. A missing verifier result or an agent
infrastructure failure adds `infrastructure_error` and `_ng_failure_class`, even
when a reward was retrieved. Artifacts include the compatible trial directory and
worker trajectory references. Failure diagnostics are written before teardown.

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
