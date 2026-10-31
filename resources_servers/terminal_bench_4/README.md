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

An agent whose process was killed before it finished (`termination.reason:
killed`) also records a completed `reward: 0.0` without starting a verifier, with
`verifier_result.source` set to `agent_killed`. Once the process is gone, a kill by
the agent's own command and one by the per-process OOM killer look the same, and
either ends the attempt.

A completed verifier command that leaves no reward file, an empty one, or one
that cannot be parsed scores zero under the Terminal-Bench contract. Ordinary
completed zero rewards remain scored. A client deadline or a sandbox result
without a confirmed command outcome is inconclusive: neither proves that the
submitted code exhausted the verifier's execution budget.

Preparation, test execution, reward retrieval and optional log transfer share the
original total task verifier deadline. Each command receives only the remaining
allowance; no phase starts a fresh budget. Infrastructure failures do not become
protocol zeros. A confirmed completed command's
successfully retrieved authoritative reward survives failures while collecting
optional logs or cleaning up local staging; those failures remain in
`verifier_diagnostics`. No verifier command is repeated.

The current sandbox result does not separately attest a native execution timeout
versus a cancelled or lost status wait. Those ambiguous outcomes therefore remain
ungraded. Environment-session results use the existing mask and failure fields;
legacy post-answer verifier failures use the same ungraded outcome. Earlier agent
infrastructure failures retain their legacy failure metadata. Saved answers are preserved,
and unsupported TB4 reverification does not authorize another agent attempt.

The existing configuration preflight still requires
`environment.default_exec_timeout_s` to cover the effective task verifier budget.
A deterministic incompatible configuration is rejected before task execution;
other sandbox setup failures retain their existing retry behavior.

The wire models in `models.py` are resources-owned. Agent implementations use
their own HTTP models and need not import this server. `termination.reason` accepts
`completed`, `timeout`, `nonzero_exit`, `killed`, `cancelled`, and `infrastructure_error`;
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

## Environment Server sessions (SandboxAccess)

Harnesses that borrow a task sandbox through `SandboxAccess` (for example
`codex_agent` behind `single_agent_turn` or `single_agent_turn_legacy`) use the
shared resources-session protocol instead of the contract above, as
[Terminal-Bench 2.1](../../benchmarks/terminal_bench_2_1/README.md) does:

- `/seed_session` with a `ResourcesSeedSessionRequest` whose `task_data` is the task
  row (`task_name`, `task_ref`, `dataset_ref`) provisions the task as above, then
  returns `sandbox_access`: a direct connection through the top-level sandbox provider
  block named by `sandbox_provider_ref` (it must describe the same provider as
  `environment.sandbox_provider`), and the task's working directory as its user sees
  it. Seeds are idempotent per `resources_session_id`; a changed request is rejected.
- `/verify` with the task row, `responses_create_params`, and the agent's `response`
  grades once, as above; the session cookie names the episode. Termination comes from
  `response.status`: `completed`, `incomplete` (`timeout`, or `nonzero_exit` for
  `max_output_tokens`), and `nonzero_exit` for any other status, such as the `failed`
  response of an agent that died while its sandbox still runs (codex_agent's
  `verify_after_agent_error`). Identical retries replay the verdict. A verify for a
  closed or unknown session returns a masked sample (`mask_sample: true`, reward 0) with
  `failure_kind: session_lost`, or `session_expired` past the seeded-session deadline.
- `/close_session` fences the session first, so a seed still provisioning releases its
  sandboxes instead of lending them (HTTP 409), even if the close's caller stops waiting.
  It then waits for in-flight provisioning or grading, stops anything still running
  without grading it (finishing even if the caller disconnects), and fences late seeds.
  A sandbox that cannot be stopped fails the close and keeps the fence; a repeated close
  retries the deletion, and the provider TTL bounds it. A completed close is remembered
  for `session_close_retry_window_seconds` (default 300), so retried closes succeed and
  late seeds are rejected; set it to cover the caller's retry horizon.

The borrower runs as the sandbox's default user, so tasks that declare
`agent.user` are rejected with HTTP 422, and the access carries no agent timeout:
configure the harness timeout from the task's `agent.timeout_sec`. Split sandbox
endpoints are not supported in this mode.

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

`environment.compose_service_overrides` maps a short task name and a Compose service to a
[JSON merge patch](https://www.rfc-editor.org/rfc/rfc7396) for that service in the agent's
generated Compose YAML: mappings merge, other values replace, and `null` removes a setting.
As with `by_task`, a task name that the manifest lacks stops the server at startup, and a
service that the task's Compose file lacks fails the task at seed. The Terminal-Bench 4
profile uses it for two services that run as non-root users, with the
[Compose extensions](../../fern/versions/latest/pages/infrastructure/sandbox/compose.mdx):

- `medical-claims-processing`: `playwright-mcp` keeps `pwuser`, disables host-file
  injection with `x-sandbox.hosts: []`, and resolves `BROWSER_URL` to the workspace
  sandbox IP via `x-sandbox.resolve_environment`.
- `payments-pipeline-fix`: `kafka` keeps `appuser` and disables host-file injection.
  Its single-broker controller uses localhost; clients retain the `kafka` alias
  needed by the advertised listener.

```yaml
compose_service_overrides:
  medical-claims-processing:
    playwright-mcp:
      x-sandbox: {hosts: [], resolve_environment: [BROWSER_URL]}
      user: null
```

`user: null` makes these services use their image's default user, omitting the
redundant explicit `user` value copied from image metadata. This avoids the provider
attempting `su` from a non-root process to the same user. Terminal-Bench-Science's
profile patches its `protein-active-learning` `submission` service the same way; see the
[benchmark README](../../benchmarks/terminal_bench_science/README.md#protein-active-learning-submission).

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
