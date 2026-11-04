# Full PR test workload comparison

This experiment compares the same sampled `main` commit under Gym's existing
execution and under Pants. Both lanes use clean source checkouts; experiment
instrumentation lives in a separate controller checkout. The branch is
`glovisotto/pants-pilot`.

## Workload

The baseline uses the source revision's CI commands. Both arms preserve its test scope, routing and gates:

- Four duration-balanced core shards on independent runners.
- A separate sandbox job and the combined coverage gate.
- Eight server shards for a full run, or the native changed-server selection.
- Session-routing end-to-end tests when selected, and lint.
- Existing runner types, system dependencies, test data, Python, uv and extras.

Use `force_full=true` to exercise the full test workload, including all server
shards. Use `false` to reproduce routing for a particular commit against its
parent. Both lanes receive the same routing decisions, before experiment files
could influence them.

PR metadata checks (copyright, secrets, guardwords and conditional documentation
checks) are outside the paired test-runtime measurement. Optional harness
conformance and scheduled container/GPU/release workloads are also outside it.
This measures the complete routed CPU test workload plus lint, not every GitHub
check or a release pipeline.

## Run

```bash
gh workflow run pants-full-ci.yml -R NVIDIA-NeMo/Gym \
  --ref glovisotto/pants-pilot \
  -f source_sha=<full-main-commit-sha> -f force_full=true
```

The workflow's push event only registers the branch workflow. Actual compute is
explicitly dispatched. Inspect the returned run, wait for completion, and retain
its artifacts:

```bash
gh run watch <run-id> -R NVIDIA-NeMo/Gym --interval 30 --exit-status
gh run download <run-id> -R NVIDIA-NeMo/Gym --dir /absolute/results/hosted/<run-id>
```

The authorized six-hour monitor refreshes the clean experiment branch against
main, selects fresh main commits, dispatches these comparisons, and collects the
results. It must avoid duplicate active runs, abort conflicting rebases, use
verified force-with-lease for updates, and retain signed DCO commits.

## Refresh dependency locks

When the sampled main commit changes `uv.lock`, `pyproject.toml` or
`.python-version`, regenerate from a clean checkout of that exact commit:

```bash
python3 scripts/pants/native_core/refresh_locks.py \
  --source /absolute/clean-sampled-main-checkout \
  --pants /absolute/pants-launcher \
  --uv /absolute/uv-0.11.29
```

The command exports both CI extras sets from uv, generates both Pants resolves,
and validates exact package pins and VCS integrity before publishing the files
together. It leaves the sampled checkout unchanged. Commit the validated
controller assets before dispatch; stale locks fail the comparison explicitly.

## What Pants does in this phase

Core and sandbox suites use native Pants Python targets, one per test file. The
four core jobs retain Gym's duration-balanced file assignments; the sandbox job
runs the complete sandbox selection. The adapter checks dependency parity against
`uv.lock` and preserves each job's JUnit and raw coverage inputs.

Server tests, session-routing E2E and lint retain their existing commands through
Pants workspace processes, preserving nested server environments and external
fixtures. The execution adapter records tracked-source inputs for both arms.

This phase establishes full-scope execution parity. Native tests currently use
`--test-force`, so recorded metadata labels them `forced-native-tests`; successful
test results are not reused. Workspace jobs use session-only result caching,
because server dependencies and external resources are not fully pinned. Engine
and dependency caches persist between runs. Do not claim selective test pruning
or cross-run test-result savings from these measurements.

The uv cache mirrors main's policy: one immutable OS/`uv.lock` key shared across
jobs, with an OS-level restore prefix. Baseline and Pants use separate namespaces
so one arm cannot warm the other. The `pants-full-v2-uv-` namespace starts cold
for both arms and excludes earlier experiment caches. Pants engine caches remain separate per job and
shard; their keys include the implementation/dependency contract and sampled
source commit, allowing repeated measurements of that source to reuse an exact
cache entry without uploading it again.

## Evidence

Require matching job selections, command success, testcase identities and outcomes,
and successful coverage gates before comparing speed. Missing results remain
failures. Report feedback time from shared workflow creation to each arm's final
required job, execution span, and total runner-minutes. Feedback includes shared
preparation, initial queue delay, setup and cache transfers. Compare fresh workflow
dispatches only; rerun attempts are rejected because they retain the original
creation timestamp. Preserve slower results and failures alongside faster runs.

The initial full-wrapper run ([41828904388](https://github.com/NVIDIA-NeMo/Gym/actions/runs/41828904388))
used per-job, per-run cache keys that forced cache uploads on every measurement.
Those uploads differ from main's shared immutable uv-cache policy. That run proves
test and coverage parity; its elapsed times are not production CI timing evidence.
Use fresh runs with the corrected cache policy for timing comparisons.

The historical six-file/103-test benchmark is documented in [SUBSET.md](SUBSET.md).
Its workflow is manual only and its measurements do not establish production CI
performance.
