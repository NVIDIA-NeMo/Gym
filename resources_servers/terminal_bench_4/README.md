# TB4 resources server

## Contract

`/seed_session` provisions the pinned task and returns its sandbox descriptor,
connection configuration, instruction, `task_id`, execution user, MCP/skills metadata,
and official agent timeout. An agent server connects to that sandbox and owns
harness setup, model calls, and the rollout loop. `/verify` collects artifacts,
runs the separate official verifier, and cleans up the task's resources.

Seed requests include `task_name`, `task_ref`, `dataset_ref`, and `rollout_id`.
The task must match the configured manifest. The agent supplies a stable
`client_session_id`, so identical seed retries share provisioning before the first
cookie response. Verification retries share one finalizer and replay its result;
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

If an executing agent's sandbox dies because it exceeds its memory or local
ephemeral storage limit, verification records a completed `reward: 0.0` and
cleans up without starting a verifier. These failures count in the pass-rate
denominator. `sandbox_resource_limit` retains the resource and provider evidence
in the rollout and trial result; `verifier_result.source` is
`sandbox_resource_limit`, and test counts remain null. The original agent
termination and diagnostics are preserved, but the result carries no
infrastructure-failure marker. Recognition requires an explicit provider death
report. Setup failures, node-pressure evictions, unrelated sandbox losses, and a
verifier that fails to start retain their existing infrastructure-error handling.

A verifier that runs but times out, or that leaves no reward file, an empty one
or one that cannot be parsed, scores `reward: 0.0` and counts in the pass-rate
denominator. This is how Terminal-Bench scores it: Harbor never retries these
outcomes and counts a trial without a reward as 0. The response keeps
`evaluation_completed: false`, names the exception in `failure_reason`, and sets
`failure_kind: verifier_error`. Only the verifier's own budget makes a timeout: a
provider call that times out within it, or a sandbox that reports no command
outcome, keeps the infrastructure-error handling. So does a trial whose agent
ended on an infrastructure failure, whatever its verifier did next, because
Harbor decides a retry from the trial's first exception.

The verifier command runs under the per-command limit
`environment.default_exec_timeout_s`, and that limit must be at least the task's
verifier budget (`[verifier].timeout_sec`, or its entry in
`environment.verifier_timeout_overrides`). A shorter limit could end the
verifier inside its budget, and the sandbox reports no evidence of that, so the
missing reward would read as the task's own zero. The server therefore refuses
to start when an override is longer than the limit, and refuses a task whose
package budget is longer, when it loads and before any sandbox starts or the
agent runs. The failure is `ExecLimitConfigurationError`, an infrastructure
failure marked terminal so that a resume does not retry it, and its message
names both values. Neither value is changed: raise the limit to fix it. An
equal limit is supported, because the verifier's budget starts before the
command is submitted and so expires first.

The wire models in `models.py` are resources-owned. Agent implementations use
their own HTTP models and need not import this server. `termination.reason` accepts
`completed`, `timeout`, `nonzero_exit`, `cancelled`, and `infrastructure_error`;
`agent_started` indicates whether execution began. `harness_metadata` is opaque
result metadata, independent of the selected harness.

Completed version-3 records can replay after a restart. Active records cannot
resume after resources-process restart. Shutdown drains active verification for
the configured grace period, then interrupts grading and awaits cleanup, including
provisioned sessions that never reached verification. Abrupt resources-process
death stops renewal; provider TTL is the sandbox cleanup fallback.

Model-call capture and harness trajectories belong to the agent server. Resource
artifacts, including collected remote `/logs/agent` files, remain under the trial
directory. The seed's connection configuration is used only for the internal
agent/resource exchange and is not persisted in session records.

## Test counts

Rollout output includes `num_tests_passed`, `num_tests_total`, `num_tests_failed`,
and `num_tests_skipped` alongside the official `reward`. These top-level numeric
fields also participate in Gym's generic aggregate metrics. Counts are retained
under `verifier_result` in the trial's `result.json`; full reports remain in
`verifier/`.

Counts come from the canonical `verifier/ctrf.json` summary, or from
`passed_cases`/`total_cases` in `verifier/trace_results.json`. They describe the
tests or cases in that report, which may omit checks performed outside the test
runner. Numbered CTRF reports are not added together. CTRF totals can include
pending or other outcomes in addition to passed, failed, and skipped tests.
Missing, malformed, or unsupported reports yield null counts. Verifier timeouts
also yield null counts. Report parsing diagnostics identify the source file.
The official reward and evaluation-completion rules remain unchanged.

## Other pinned Harbor datasets

`manifest_path` selects the pinned dataset. It defaults to Terminal-Bench 4.0; a relative
path is read from the repository root. Task names take the dataset's organization prefix, so
[Terminal-Bench-Science](../../benchmarks/terminal_bench_science/README.md) serves
`terminal-bench-science/<task>`. Its packages pass the same content-hash check.

Packages that ship only Dockerfiles leave `docker_image` unset. Supply images through
`environment.task_images`:

- `agent` and `verifier`: templates that accept `{org}`, `{task}` and `{digest}`.
- `service`: a template for Compose services that the package builds from a Dockerfile. It
  also accepts `{service}`, the Compose service name.
- `by_task.<task>.agent|verifier` and `by_task.<task>.services.<service>`: explicit per-task
  images, which take precedence over the templates. The server refuses to start when a key
  names a task its manifest lacks.

An image declared by the package always wins. A separate verifier without
`[verifier.environment]` uses the agent image, as in Harbor. A role with no image fails at
seed, before allocation.

Such packages' Compose files can build sidecars (`build:`). The server then runs each
sidecar's configured image, and `main` runs the agent image. It also applies Harbor's `CPUS`
and `MEMORY` Compose variables from the task's resources. These images have no published
name, so their `compose_image_configs` records are keyed `<task>/<service>`. Records keyed by
image name take precedence. `environment.compose_ignored_settings` lists Compose service
settings, or `networks`, that the deployment cannot enforce; they are removed instead of
rejected. It is empty by default, and a benchmark that sets it documents what it drops.

Two opt-in verifier settings adapt packages to a deployment. Both default to the package
values:

- `environment.verifier_resources_from_agent: true` gives a separate verifier environment
  that declares no CPUs or memory the agent environment's values. Without it, the provider
  default applies.
- `environment.verifier_timeout_overrides: {<task>: <seconds>}` replaces a task's
  `[verifier].timeout_sec`. An applied override is recorded in diagnostics. As with
  `by_task`, a key that names no manifest task stops the server at startup.

For local checks on a workstation, a single-container task can also use Gym's `docker`
sandbox provider. An offline verifier then runs with `docker run --network none`. EFS logs
and Compose still require OpenSandbox.

`/verify` also reports `partial_credit`: the fraction of reported tests that passed
(`num_tests_passed / num_tests_total`, so skipped tests count as not passed), never below the
official reward. Without usable test counts it equals the official reward. `reward` stays the
official binary grade. When the official evaluation did not complete, both are 0, so a mean over
the same rows uses the same denominator for both. `evaluation_completed: false` and
`infrastructure_error` mark those rows, which are left out of the default aggregates unless
`count_failure_classes_as_zero` counts them.

## Non-root Compose services

When loading agent Compose YAML, three task-specific adaptations use the
[Compose extensions](../../fern/versions/latest/pages/infrastructure/sandbox/compose.mdx):

- `medical-claims-processing`: `playwright-mcp` keeps `pwuser`, disables host-file
  injection with `x-sandbox.hosts: []`, and resolves `BROWSER_URL` to the workspace
  sandbox IP via `x-sandbox.resolve_environment`.
- `payments-pipeline-fix`: `kafka` keeps `appuser` and disables host-file injection.
  Its single-broker controller uses localhost; clients retain the `kafka` alias
  needed by the advertised listener.
- Terminal-Bench-Science `protein-active-learning`: `submission` keeps its `submission`
  user, disables host-file injection, and receives its only peer URL as
  `ASSAY_STATUS=http://assay:8080/status`, resolved to the assay sandbox IP. The image's
  server must read that variable; see the
  [benchmark README](../../benchmarks/terminal_bench_science/README.md#protein-active-learning-submission).

These services use their image's default user, omitting the redundant explicit
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
