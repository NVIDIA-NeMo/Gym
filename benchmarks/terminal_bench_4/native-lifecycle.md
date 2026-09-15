# TB4 native lifecycle

## Implementation and compatibility

The TB4 resources server uses `gym-tb4-native`, version `1`. Harbor is no longer a
TB4 dependency. The agent handoff schemas, mini-SWE/OpenCode profiles, model
routing, task pins, image pins, and official grading commands remain unchanged.
The earlier [agent/resources migration](migration.md) is a historical record of
the Harbor-backed reference implementation.

`app.py` validates caller/task identity and constructs HTTP responses. `lifecycle.py`
owns preparation, setup and execution deadlines, finish arbitration, collection,
verification, and cleanup. `environment.py` creates Gym sandboxes directly and
retains Compose ownership. `task.py` loads the pinned packages. `collection.py`,
`transfers.py`, and `verifier.py` implement state transfer and official grading.
These helpers are local to TB4.

A resource-owned preparation task survives HTTP disconnection. The 360-second
setup budget begins after provisioning and task health checks, before descriptor
creation. Starting an episode records a monotonic execution deadline once. A
submission handled after that deadline is a timeout even if the watchdog was
late. Otherwise, the first finish request under the session lock wins. One
resource-owned finalizer runs whether the worker submits, cancels, or disappears.
Setup cancellation closes without grading; running cancellation still grades.

Main hooks and artifacts precede main-service stop and sidecar collection. Hooks,
optional missing artifacts, and main-service stop errors retain the reference's
best-effort behavior. A failed harness-quiescence operation prevents grading.
Artifact host destinations do not change verifier destinations. Directory restores
empty their destination first and preserve the reference's permission setup.
The separate verifier uses its baked-in `/tests/test.sh`; tests are never uploaded
over it. Its startup uses the task's environment build budget. Its official
execution deadline starts after startup and restoration. Verifier health checks
are not added. Reward JSON takes precedence over reward text, including malformed
JSON; finite zero is an official grade, while missing/invalid reward is a failure.

Setup pre-creates the same host agent, verifier, and conventional artifact
directories as the reference. This matters when a provider rejects a directory
probe: an optional download must not replace the conventional host directory
with an empty file and then break verifier restoration.

Cleanup attempts both role environments. Compose cleanup also attempts remaining
services, volume helpers, and provider transports after another operation fails.
Records retain owned resource identities and errors. Cleanup failure does not
remove an already obtained official reward. Provider TTL remains the fallback
for abrupt process death and residual remote resources.

Shutdown rejects new seeds, cancels preparation/setup, and requests running
finalization. `shutdown_timeout_sec` gives finalizers a grace period (default 30
seconds); pending finalizers are then interrupted and their cleanup is awaited.
The process host must allow time for provider cleanup after that grace period.

## Package and feature inventory

All 66 pinned packages were obtained and their content hashes verified. The
inventory covers 52 CPU tasks, 11 CPU Compose tasks, and three H100 tasks. Every
package uses a separate Linux verifier image; shared verification and multi-step
packages are rejected before allocation. All agent budgets are 28,800 seconds;
the smoke cap can only shorten them.

The complete execution field inventory maps to:

| Package fields | Native implementation |
| --- | --- |
| Image, CPU, memory, storage, GPU count/type, build budget | `task.py`, `environment.py` |
| Environment variables, user, workdir, health check | `task.py`, `environment.py` |
| Compose commands, dependencies, image metadata, capabilities, shared memory | Existing Compose normalizer and Gym Compose/provider APIs |
| MCP declarations and task skills directory | Existing seed schema and unchanged agent implementations |
| Artifact source/destination/exclusions/service, collect hooks | `collection.py`, `transfers.py` |
| Separate verifier image/settings, network policy, execution budget, verifier env/user | `verifier.py`, `environment.py`, `lifecycle.py` |
| Descriptive task/author metadata | Accepted without changing execution |

Both role specifications for all 66 tasks (132 specifications) and all instructions
match the frozen reference after normalizing the deliberate ownership metadata
rename from `harbor-session`/`harbor-task` to `tb4-session`/`tb4-task`.

The cache layout remains `<task_download_dir>/terminal-bench/<task>/<digest>`;
the default is `~/.cache/harbor/tasks/packages` for compatibility. Cache hits are
revalidated. Cold downloads use the public registry's pinned-version RPC and
package object endpoint through Gym's shared aiohttp client. Extraction occurs
in a temporary sibling directory, followed by validation and atomic promotion.

The digest is SHA-256 over sorted UTF-8 records consisting of the package-relative
path, NUL, the file's SHA-256 hex digest, and LF. The publisher's file set is
`task.toml`, `instruction.md`, `README.md`, `trajectory.json`, and files under
`environment/`, `tests/`, `solution/`, and `steps/`. A package `.gitignore` replaces
the standard ignored temporary-file patterns. This is a content hash, not the
compressed archive's byte hash or `dirhash`.

## Records and consumers

Records have `record_version: 1` and are atomically replaced. They preserve
`artifacts.trial`, `gym-agent.json`, `result.json`, agent/verifier log directories,
`artifacts/manifest.json`, and the four public timing keys. The result is a small
TB4 record with the existing `verifier_result.rewards` and `exception_info` subset.
Completed legacy records replay saved HTTP responses exactly. A legacy closed
result without a saved response can be combined with its first later submission.
Interrupted active records are never resumed.

The consumer inventory covered the benchmark smoke runner, local health-report
and parity drivers, and EFB reporting scripts. No consumer requires a
`harbor_version` key in new responses. Native provenance therefore records
`runtime`, `runtime_version`, and `reference_harbor_version: 0.23.0`; legacy result
replay retains its original runtime provenance. Task and dataset identity keys
remain unchanged. Smoke provenance hashes cover the native modules, provider,
Compose adapter, manifest, and image/config files.

## Validation and baseline

The frozen source is `flafrance/terminal-bench-4` at `8b41d2fdf`, plus the existing
uncommitted SYS_PTRACE provider/config/test changes. It is preserved in the
separate `Gym-tb4-reference` checkout. The baseline health records are captured in
[health-baseline.json](health-baseline.json). OpenCode has 50/52 CPU, 9/11 Compose,
and 3/3 GPU healthy; mini-SWE has 5/5 sampled CPU, 8/11 Compose, and 3/3 GPU healthy.
The later OpenCode SYS_PTRACE follow-up is included in that baseline.

Pass `--baseline-health benchmarks/terminal_bench_4/health-baseline.json` to the
smoke runner. It records baseline status and regressions, and advances CPU →
Compose → GPU only when no newly unhealthy task appears. Without a baseline,
any unhealthy task blocks the next stage. Known unhealthy tasks are still run.

Validation artifacts for this migration are under
`../artifacts/native-lifecycle-20260915/` relative to the repository. They include
all package inventories, 132-role specification parity, cold/warm loading in a
fresh environment without Harbor, deterministic file-transfer/grading fixtures,
and live HTTP episodes. The official `ks-solver-cpp` solution receives reward 1.0
in native and isolated reference runs, with identical collected payload bytes
and semantically identical transfer manifests (JSON key order differs).

The focused suite passes 438 tests, with three expected skips. The native suite
passes 114 tests with 97.94% coverage in the Harbor-free environment. Live
abandonment canaries confirm setup timeout closes without grading, and execution
timeout grades the seeded solution at 1.0 before a late first submission arrives.
The recorded timeout and response survive retries. Pre-commit checks pass.

The first GPU sweep caught a conventional artifact-directory initialization
regression on `fp8-rmsnorm-gemm`. Restoring the reference directory layout fixed
both harnesses; a regression fixture reproduces the failed directory probe and
checks restoration. The original failed attempts remain in the validation
artifacts alongside the successful reruns.

Live category results and cleanup audit are recorded in `health-report.md` in
that artifact directory. Capped smoke rewards are not benchmark scores. The
standalone OpenCode smoke retains its public model endpoint mode and does not
certify remote Gym model routing. The official MCP task remains subject to its
existing deployment blocker; mocked routing or a different MCP task is not a
substitute for that missing live coverage.

## Rollout and rollback

Drain active sessions before switching code/configuration. Start native sessions
with a separate artifact directory. To roll back, restore the frozen reference
code/configuration and use another artifact directory. Never resume a native
in-flight session with the reference runtime or vice versa.
