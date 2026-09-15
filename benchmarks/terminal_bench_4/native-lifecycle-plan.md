# TB4 native lifecycle implementation plan

Status: implemented on `flafrance/tb4-native-lifecycle`. See
[native lifecycle notes](native-lifecycle.md) for the implementation, validation
evidence, and existing deployment blockers. The sections below retain the
original design and review gates.

## Objective and scope

Replace Harbor's task lifecycle in the TB4 resources server with explicit Gym
operations. Keep the existing agent, model, and resources server HTTP interfaces.
Keep new implementation helpers under `resources_servers/terminal_bench_4` until
another benchmark demonstrates a concrete reuse opportunity.

The final TB4 preparation and execution path must work without the Harbor Python
package installed. It must preserve the pinned task packages, evaluation profile,
separate-verifier behavior, and resource ownership established by the split.
Other benchmarks and the general Harbor integrations remain supported by their
existing code. This is not a repository-wide removal of Harbor.

Do not introduce a workflow engine, plugin registry, generic benchmark schema,
new server, or common session-manager API. Do not change mini-SWE prompts,
tool behavior, model routing, or their borrowed-sandbox interfaces.

## 1. Compatibility contract

Preserve these endpoints and their request/response schemas:

| Endpoint | Behavior to preserve |
| --- | --- |
| `/seed_session` | Validate manifest identity; return the existing session ID, main-sandbox descriptor, instruction, user, budgets, MCP declarations, and skills directory. Matching retries share preparation. Conflicting worker executions are rejected. |
| `/start_session` | End harness setup and begin the authoritative agent budget once. Repeated calls return remaining time without extending the deadline. |
| `/verify` | Accept the first agent submission, preserve authoritative termination, wait for resource-owned evaluation, and return the existing verification response. Retries do not rerun evaluation. |
| `/cancel_session` | Cancel preparation/setup or terminate a running episode. Running cancellation still permits grading. Return the existing closed-session response. |

Retain `SandboxConnection`, `SandboxedSeedResponse`, `AgentTermination`,
`SandboxedVerifyRequest`, and `SandboxedVerifyResponse` in
`nemo_gym/sandbox/handoff.py`. Retain `run_borrowed()` and the mini-SWE TB4 agent path.
The agent continues to attach using `owns_sandbox=False`; only resources destroy
the sandbox collection. Provider credentials stay out of descriptors and records.

Preserve session cookies, `client_session_id`, rollout identity, `execution_id`,
one resources worker per artifact directory, and the configured concurrency cap.
Preserve the agent cap as a shortening of the official budget, never an extension.

Preserve public reward semantics: an official zero is a completed evaluation;
missing official reward is an infrastructure failure; agent infrastructure
failure remains marked even when an official reward was recovered. Continue
returning existing artifact and timing keys, including `artifacts.trial` and
`environment_setup`, `agent_setup`, `agent_execution`, and `verifier`.

Provenance must identify the native lifecycle accurately. Keep dataset/task
identity keys; record the native implementation version. Do not report Harbor
as the executing runtime. Retain a legacy `harbor_version` key as null if needed
by a known consumer, and identify the reference compatibility version separately.
Inventory consumers before choosing the representation; no typed wire-schema
change is needed because provenance is already a dictionary.

## 2. Explicit execution model

The resource server owns a concrete `Session` with a phase, sandbox collection,
deadlines, termination, first accepted agent submission, evaluation result, and
persisted response. Remove `ExternalAgent`, `ExternalEpisode`, `Trial`, and the
Harbor environment factory from this path.

```text
/seed_session -> prepare_session() -> ready
                                     |
                    agent configures its harness
                                     |
/start_session ----------------------> agent_running
                                     |
            /verify, /cancel_session, or resource deadline
                                     |
                              request_finish()
                                     |
                              finalize_session()
                                     |
                     collect -> verify -> cleanup -> closed
```

### Task ownership and synchronization

- `/seed_session` creates one resource-owned preparation task. HTTP handlers
  await it through shielding, so a disconnected request does not cancel it.
- Preparation starts a 360-second harness-setup watchdog when the session becomes
  ready. Queuing for capacity and task environment provisioning do not consume
  that budget. Characterize the existing boundary around descriptor creation and
  preserve it in tests.
- `/start_session` changes `ready` to `agent_running` under a session lock,
  records a monotonic deadline, and replaces the setup watchdog with an execution
  watchdog. Persist wall-clock timestamps for diagnostics; do not use persisted
  monotonic timestamps to resume work after a restart.
- `request_finish()` is the single arbitration point for worker submission,
  cancellation, and deadline expiration. Under the lock it records termination
  and creates or returns exactly one resource-owned finalization task.
- No slow sandbox I/O runs under the session lock. Handlers await the shared task
  outside the lock. Stop superseded watchdogs without allowing a watchdog to
  cancel itself or its newly created finalizer.
- Keep phase changes and errors directly visible in the session record. Use the
  existing phases (`preparing`, `ready`, `agent_running`, `verifying`, `closed`);
  an internal subphase can record collection or cleanup without changing APIs.

These are ordinary background tasks with explicit owners, not a simulated agent
or callbacks into another framework's execution loop.

### Termination and failure rules

| Trigger | Resource behavior |
| --- | --- |
| Preparation failure or cancellation | Record diagnostics, clean up every acquired resource, close without grading. |
| Harness setup timeout or cancellation | Stop harness work, clean up, close without grading. Reject premature `/verify`. |
| Worker completion/nonzero exit | Freeze submission, stop tracked agent processes, collect, grade, clean up. |
| Worker timeout or resource execution deadline | Record timeout, stop tracked agent processes, collect, grade, clean up even if `/verify` never arrives. |
| Running cancellation or agent infrastructure failure | Preserve termination and existing grading/failure-classification behavior. |
| Verification or collection infrastructure failure | Persist failure and available diagnostics; run cleanup; do not fabricate a successful evaluation. |
| Cleanup failure | Attempt remaining cleanup operations, record residual resource identities and errors, and retain the official reward if obtained. Match public failure classification to characterized behavior. |

Define first-winner ordering for simultaneous submission, timeout, and cancel
under the lock. A request handled after the execution deadline cannot extend the
budget because a watchdog was delayed. Once evaluation begins, a later request
cannot replace its termination or restart collection. Preserve late worker
artifact references without allowing a second evaluation.

Keep the evaluation result independent of the HTTP verification body: deadline
evaluation may complete before any `/verify` request exists. A later first
submission combines its Gym response/usage with that stored result, preserving
the resource-side termination. Persist the resulting response for exact retries.

### Cleanup, shutdown, and restart

Track resource ownership immediately on acquisition, including partial Compose
startup, verifier startup, relay/volume ownership, and provider transports.
Use existing sandbox/Compose cleanup methods. A failed operation must not skip
cleanup of unrelated acquired resources or leak a concurrency slot.

On graceful shutdown, reject new work, cancel unfinished preparation, and drain
or explicitly terminate resource-owned finalization tasks according to the host
shutdown allowance. Always attempt cleanup in `finally`; record interruption if
grading cannot finish. Abrupt process death retains provider TTL as the fallback.

Persist a versioned TB4 record atomically. Continue loading completed legacy
records and replaying saved responses. Add a small legacy-result reader for
closed records that have an evaluation result but no saved HTTP response.
Reject resumed execution of non-closed records after restart, as today. Do not
build distributed recovery or active-session resumption in this change.

## 3. Code organization

Proposed files are ownership boundaries, not new extension interfaces. Combine
small helpers if separate modules do not improve readability.

| Location under `resources_servers/terminal_bench_4` | Responsibility | Possible later extraction |
| --- | --- | --- |
| `app.py` | Existing routes, caller ownership, request validation, response construction | None planned |
| `lifecycle.py` | Concrete Session, preparation, deadline arbitration, finalization, shutdown | Deadline and finalization mechanics after a second benchmark needs them |
| `task.py` | TB4 task settings and pinned-package loading | Package transport/cache mechanics only |
| `environment.py` | Direct `SandboxSpec` construction, main/Compose provisioning, users/env, readiness | Small provisioning helpers already useful across benchmarks |
| `collection.py` | TB4 collect-hook sequence and artifact placement semantics | None of the TB4 ordering policy |
| `verifier.py` | Separate verifier creation, restoration, test execution, reward parsing | Small execution/output helpers if independently useful |
| `utils/transfers.py` | Directory/file transfer, exclusions, transfer reports | Transport-independent transfer helpers |
| `utils/state.py` | Atomic records and completed-record compatibility | Persistence primitives; no generic session abstraction yet |

Keep the sequence in `lifecycle.py` and `collection.py` readable as straight-line
async code. Avoid a base environment class mirroring Harbor. A concrete owned
collection may expose `main`, named services, and `stop()` using Gym objects.
Retain the Compose normalization logic currently in
`resources_servers/terminal_bench_4/compose_config.py`; it already has no
Harbor imports. Avoid duplicating it solely to change its directory. Relocate it
with compatibility imports only if the implementation establishes a clear need.

## 4. Task package and provisioning behavior

### Inventory first

Before implementation, obtain all 66 manifest-pinned packages and record the
fields and features actually used. The completed inventory is summarized in the
[native lifecycle notes](native-lifecycle.md#package-and-feature-inventory).
The inventory must include resolved defaults as well as explicitly present keys:

- Single-step status, image identities, environment files, Compose services.
- Agent/verifier CPU, memory, disk, GPU, users, working directories, startup env.
- Build/setup/execution/verifier budgets and health-check settings.
- Verifier mode, verifier image inheritance, and network policy resolution.
- Artifact source/destination/exclusions, service names, and collect hooks.
- MCP declarations, task-provided skills paths, and any injected assets in the
  actual configured profile.

Use that inventory to define a small internal schema. Preserve original task
files. Permit irrelevant descriptive metadata; reject unsupported settings that
would affect execution or grading before allocating resources. Explicitly reject
multi-step packages as today. If the inventory reveals shared-verifier tasks,
support their observed semantics rather than forcing separate verification.

### Harbor-free loading

Retain runtime resolution by trusted task name and ref and the configured
`task_download_dir` behavior. Implement only the package lookup, download, and
cache path needed for the pinned dataset. Use Gym's aiohttp request facilities;
do not import Harbor's registry/downloader indirectly or invoke its CLI.

Document the package content-hash algorithm and distinguish it from an archive
byte hash. Verify against the correct identity before marking a cache entry
complete; use temporary extraction and atomic promotion. Characterize cache
reuse, concurrent requests, archive layout, and package-relative path handling.
Existing Harbor caches can be accepted only after validating the required layout
and identity. A prefetch utility may call this same loader; it must not become a
new mandatory external preparation step for the existing server contract.

### Direct environment setup

Port the useful behavior from the current Harbor sandbox adapter and its base:

- Published-image startup and applicable environment-directory uploads.
- Compose image metadata resolution, entrypoint/command/workdir/user semantics.
- Per-task resources, independent agent/verifier CPU/GPU pool selection, existing
  request/limit policy, shared memory and capabilities through provider APIs.
- Startup environment and TB4 runtime overrides such as `nextjs-performance`.
- Log directories, task readiness checks, and task skills/MCP metadata handoff.
- Supported network isolation; retain rejection of unsupported allowlists or
  policy transitions. Do not quietly relax the separate verifier's policy.
- Process-group tracking and stopping using the existing agent PID-file contract.

Keep settings that currently control provider behavior and deployment overrides.
Replace the Harbor `environment.import_path` selection in the TB4-owned YAML with
concrete TB4 settings; do not add another dynamic environment factory. Update all
TB4 config wrappers and resolved-config checks. Preserve unrelated local provider
and configuration changes when implementing this plan.

## 5. Collection and verification sequence

For the separate-verifier path, write the following sequence explicitly:

1. Stop tracked harness process groups. If quiescence fails, do not proceed to
   normal state collection/grading; record the failure and clean up.
2. Save agent submission references and collect agent logs needed by existing
   diagnostics. Worker-local trajectory paths remain references, not assumed
   resource-server-local files.
3. Execute main-service collect hooks and collect declared main artifacts,
   including the conventional artifacts directory.
4. Stop the main service before sidecar evidence collection when sidecars are
   involved. Preserve characterized best-effort behavior for hooks and optional
   artifact failures; do not make every missing file fatal.
5. Execute sidecar hooks and collect their declared artifacts. Preserve source
   mapping, destination/exclusion rules, collision behavior, and transfer reports.
6. Stop the agent collection and release its owned auxiliary resources.
7. Create the verifier environment with its resolved image/resources/user/env and
   network policy. Preserve actual verifier startup behavior; do not add a new
   verifier health-check phase merely because agent setup has one.
8. Restore collected artifacts to their original verifier paths. A host artifact
   destination is not automatically a verifier destination. Preserve the
   conventional-directory mapping and required permissions.
9. Run the official verifier command under its deadline. For separate verifier
   images, use the baked-in tests; do not upload agent-side tests over them.
10. Retrieve verifier logs and parse the official reward. Preserve reward JSON/text
    precedence, valid zero, malformed/missing/non-finite value handling, and the
    distinction between test exit status and the emitted reward.
11. Clean up the verifier in `finally`, persist the evaluation result, and close.

Implement TB4 result records directly, rather than reproducing Harbor's complete
result model. Retain the artifact directory and public link keys; preserve any
on-disk fields consumed by current smoke/report tools or update those tools in
the same change. Store enough phase, resource, transfer, and error information
to diagnose setup, collection, grading, and teardown independently.

## 6. Implementation sequence and review gates

### Step 1 — Freeze the contract and inventory the pinned tasks

- Capture the current source/config baseline, including relevant uncommitted
  provider fixes when producing reference runs.
- Inventory all packages and current artifact/report consumers.
- Record the exact currently supported and blocked task categories; historical
  migration counts are evidence, not a fresh claim about today's deployment.
- Add contract fixtures for response semantics, termination precedence, timeout
  boundaries, artifact mapping, and result replay. Use Harbor 0.23.0 only in an
  isolated reference environment, outside the final TB4 dependency set.

Gate: every pinned execution-affecting feature has an implementation destination;
package retrieval and hash semantics are understood; compatibility decisions are
written down. This is the point to refine effort estimates.

### Step 2 — Native task loading and direct sandbox provisioning

- Implement `task.py`, environment settings, and direct main/Compose ownership.
- Reuse the existing sandbox/provider infrastructure and Compose normalization.
- Validate cold/warm package cache paths and resolved specs for the inventory.
- Exercise real setup/cleanup for representative CPU, Compose, and GPU packages.

Gate: native preparation yields the same instructions, handoff metadata, runtime
settings, and readiness outcomes; partial startup leaves no owned resource leaks.

### Step 3 — Native collection and official verification

- Implement the explicit collection sequence and separate verifier runner.
- Create deterministic fixtures with known files and sidecar state, including
  positive outputs that actually exercise verifier logic.
- Compare native and reference runs from equivalent seeded state. Do not compare
  two independently generated model trajectories as if they were identical input.

Gate: transfer manifests, restored bytes/paths, verifier settings, and official
rewards agree on supported fixtures; explained nondeterministic log fields are
excluded from comparisons. Failures retain the expected evaluation classification.

### Step 4 — Wire the native session lifecycle behind existing endpoints

- Replace the Harbor `_execute()` path with explicit preparation, watchdog, and
  single-finalizer operations. Remove `runtime.py` after references are migrated.
- Keep the agents and `nemo_gym/sandbox` handoff contract unchanged.
- Adapt existing handoff tests to fake sandbox boundaries, rather than fake
  `Trial` objects. Test observable behavior, not only phase assignments.
- Implement versioned persistence and closed legacy-record replay.

Gate: existing clients work without protocol changes; concurrent finish causes,
HTTP disconnects, setup abandonment, execution abandonment, and restart handling
all preserve ownership and produce at most one evaluation per live session.

### Step 5 — Prove parity and switch the TB4 configuration

- Point TB4 configs at the native implementation and remove TB4's Harbor runtime
  dependency. Update smoke source hashes, artifact readers, README, and migration
  notes. Keep the historical Harbor migration record identifiable as history.
- Run real HTTP rollouts using mini-SWE and the existing model
  routing, inspecting actual agent actions, verifier outputs, and cleanup.
- Cover the feature matrix below; run all 66 task preparations and supported
  task evaluations, documenting infrastructure blockers explicitly. A blocked
  category remains an unresolved validation limitation, not a passing check.
- Retain the reference implementation in a separate checkout for comparisons and
  rollback. Do not ship a permanent dual-runtime abstraction. Drain active
  sessions before deployment; roll back by restoring the previous code/config
  with a separate artifact directory, not by resuming native in-flight sessions.

Gate: no unexplained parity regressions; no benchmark-definition drift; tests and
live evidence support each supported task category. Resolve outstanding failures
or explicitly document their scope before declaring migration complete.

## 7. Validation matrix

| Area | Required evidence |
| --- | --- |
| Task loading | All 66 pinned identities, cold/warm cache, content-hash mismatch, malformed/unsupported execution fields, no mutable latest resolution |
| Provisioning | Single container, Compose dependencies/startup metadata, task runtime env, non-root user, health-check success/failure, partial startup cleanup |
| Resources | Independent agent/verifier CPU/GPU selection, memory/disk units, request/limit settings, shared memory/capabilities, supported offline verifier policy |
| Collection | Main file and directory, exclusions and collisions, optional missing artifact, hook failure, sidecar collection after main stop, restored path/permissions |
| Verification | Known positive and zero reward, malformed/missing/non-finite reward, timeout, baked-in tests, official reward after agent failure |
| Sessions | Duplicate seed/start/verify, conflicting execution IDs, cookie isolation, verify-vs-timeout/cancel races, disconnected requests, late first verify after deadline |
| Lifecycle | Worker disappears during setup/run, shutdown during every phase, provider cleanup failure, no leaked slots/transports/collections |
| Persistence | Exact completed-response replay, closed evaluation without submission, legacy completed records, rejection of interrupted sessions |
| Agents | Real mini-SWE episodes; task MCP and skills exercised where declared; unchanged model routing and usage/trajectory reporting |
| Independence | Fresh TB4 environment with no Harbor package; preparation, server import/start, and representative end-to-end evaluation succeed |

Use deterministic sandbox-boundary tests for races and collection behavior, plus
real provider/model rollouts for integration. Zero-reward smoke tests alone do
not prove state-transfer or grading parity. Run focused tests during development,
the repository-required coverage/lint checks for changed code, and pre-commit
checks before review. Do not run the unrelated all-server rollout suite.

## 8. Completion criteria

- The TB4 path imports and executes without Harbor, including package loading.
- A reader can follow preparation, start, finish, collection, verification, and
  cleanup through ordinary TB4 functions with explicit resource ownership.
- Existing server endpoints, schemas, and agent behavior remain compatible.
- Supported pinned tasks preserve task state, settings, and grading semantics.
- Deadline-triggered grading and retry/restart behavior are covered by tests.
- No new shared framework or server interface is introduced. Potential reusable
  helpers remain local and are named by the operation they perform.
- Other Harbor integrations remain functional; shared adapters are not deleted
  merely because TB4 stops using them.
- Validation evidence and remaining deployment limitations are recorded, with
  the native runtime identified accurately in provenance.
