# Full PR test workload comparison

This experiment compares the same sampled `main` commit under Gym's existing
execution and under Pants. Both lanes use clean source checkouts; experiment
instrumentation lives in a separate controller checkout. The branch is
`glovisotto/pants-pilot`.

## Workload

The comparison preserves the source revision's CI commands and routing:

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

## What Pants does in this phase

Pants orchestrates the complete existing CI commands through workspace processes.
This preserves Git-dependent tests, nested server environments and shared test
fixtures. The execution adapter records the full tracked-source input and keeps
coverage/JUnit artifacts available to the same gates.

This is a full-scope execution-parity phase, not yet a conversion of the complete
suite to native per-test Pants targets. Unlocked server dependencies, mutable
external resources and host tools make cross-session test-result reuse unsafe;
those results are not reused. Do not interpret orchestration overhead as a verdict
on native Pants caching or advertise selective test pruning that is absent.

## Evidence

Require matching job selections, command success, testcase identities and outcomes,
and successful coverage gates before comparing speed. Missing results remain
failures. Report both arm elapsed time and total runner-minutes, including setup
and cache transfer, and distinguish queued time from execution. Preserve slower
results and failures alongside faster runs.

The historical six-file/103-test benchmark is documented in [SUBSET.md](SUBSET.md).
Its workflow is manual only and its measurements do not establish production CI
performance.
