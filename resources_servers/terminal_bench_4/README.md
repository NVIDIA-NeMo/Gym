# TB4 resources server

## Contract

`/seed_session` provisions the pinned task and returns its sandbox descriptor,
connection configuration, explicit working directory, `task_id`, execution user, MCP/skills metadata,
and official agent timeout. An agent server connects to that sandbox and owns
harness setup, model calls, and the rollout loop. `/verify` collects artifacts,
runs the separate official verifier, and cleans up the task's resources.

Seed requests include `task_name`, `task_ref`, and `dataset_ref`. Prepared rows carry the task instruction in `responses_create_params.input`.
The task must match the configured manifest. The agent supplies a stable `Idempotency-Key` header per seed attempt, so identical
transport retries share provisioning before the first cookie response. A new key
starts a new attempt; requests without a key always start a new attempt. Rollout
capture IDs are independent of this HTTP retry contract. Verification retries share one finalizer and replay its result;
conflicting requests fail. Run one resources worker per artifact directory.

A seeded session has a resource-owned 10-hour deadline covering agent setup and
execution. Configure it with `seeded_session_timeout_sec` or the benchmark override
`tb4_seeded_session_timeout_sec`. The clock starts when the sandbox is ready;
retries do not extend it. If `/verify` has not claimed the session by the deadline,
resources quiesces the agent, destroys the owned resources, and releases its
concurrency slot. Late seed/verify requests receive HTTP 410, including after
restart. Expiry does not grade an abandoned rollout. Once `/verify` starts, its
normal verifier timeout and cleanup lifecycle take over.

The resources server retains the Compose creator, TTL renewal, and shared-volume
ownership throughout the episode. The agent closes its attached transport after
joining its harness worker, then submits its response, termination, execution
status, timings, and harness metadata to `/verify`. Failed setup skips grading;
a started agent is graded even after failure, cancellation, or timeout. Official
zero and nonzero grades survive agent failure or timeout.

The wire models in `models.py` are resources-owned. Agent implementations use
their own HTTP models and need not import this server. `termination.reason` accepts
`completed`, `timeout`, `nonzero_exit`, `cancelled`, and `infrastructure_error`;
`agent_started` indicates whether execution began. `harness_metadata` is opaque
result metadata, independent of the selected harness.

### Shared sandbox access

Set `environment.sandbox_provider` to a named top-level Gym provider configuration
to return Gym's existing `SandboxAccess` in `sandbox_access`. For example, compose
the provider configuration named `sandbox` on both servers and override:

```yaml
terminal_bench_4:
  resources_servers:
    terminal_bench_4:
      environment:
        sandbox_provider: sandbox
```

For split deployments, also set `sandbox_provider_by_pool: {cpu: sandbox_cpu,
gpu: sandbox_gpu}` and define both named providers on the resources and agent
servers. The agent, verifier, and Compose services use the same pool, including
GPU verification of a CPU task. Named providers supply their own connection
settings; the legacy `OPENSANDBOX_DOMAIN_CPU/GPU` overrides do not replace them.
Missing selected pool references fail setup.

The access object contains the selected `provider_config_ref`, serialized
`descriptor`, and `workdir`. Resources resolves the working directory as the task
user before returning the seed. `environment.workdir` takes precedence over the
task's configured working directory, then the image's default applies.

Inline provider mappings remain supported and return the legacy connection fields
with an explicit `workdir`; they cannot advertise a shared named-provider reference.
Named configurations also retain those legacy fields for existing clients. New
clients should prefer `sandbox_access` and resolve its provider reference from their
own Gym configuration. The execution user, task timeout, and MCP/skills fields
remain alongside the access object in this legacy seed contract.

Agent close must finish before calling `/verify`. miniSWE refuses verification
after an unacknowledged process cleanup or transport close; the resource deadline
still disposes of abandoned sandboxes. The PID-file quiesce step remains for legacy
clients and expiry recovery, rather than establishing miniSWE's successful close.

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
