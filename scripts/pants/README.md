# Full PR test workload comparison

This experiment compares the same sampled `main` commit under Gym's existing
execution and under Pants. Both lanes use clean source checkouts; experiment
instrumentation lives in a separate controller checkout. The branch is
`glovisotto/pants-pilot`.

## Workload

The baseline uses the source revision's CI commands. Both arms preserve its test scope, routing and gates:

- Four duration-balanced core shards on independent runners, each with the sampled main's pytest worker count.
- A separate sandbox job and the combined coverage gate.
- Eight server shards for a full run, including main's four component workers per shard and their logs,
  or the native changed-server selection.
- Session-routing end-to-end tests when selected, and lint.
- Existing runner types, system dependencies, test data, Python, uv and extras.

The default `force_full=true` exercises the full test workload, including all server
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
  -f source_sha=<full-main-commit-sha> -f force_full=true -f execution_mode=incremental
```

The workflow's push event only registers the branch workflow. Actual compute is
explicitly dispatched. Inspect the returned run, wait for completion, and retain
its artifacts:

```bash
gh run watch <run-id> -R NVIDIA-NeMo/Gym --interval 30 --exit-status
gh run download <run-id> -R NVIDIA-NeMo/Gym --dir /absolute/results/hosted/<run-id>
```

The authorized three-hour monitor refreshes the clean experiment branch against
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
  --uv /absolute/uv-0.13.4
```

The command exports both CI extras sets from uv, generates both Pants resolves,
and validates exact package pins and VCS integrity before publishing the files
together. It leaves the sampled checkout unchanged. Commit the validated
controller assets before dispatch; stale locks fail the comparison explicitly.

## Full scope and incremental execution

`force_full` controls workload routing. `execution_mode` independently selects
`incremental` (default) or `forced` (a full execution control). The baseline
executes main's commands in either mode. Incremental Pants still accounts for
every selected test and preserves the complete JUnit and coverage gates; it may
reuse a successful result only for a target with declared inputs and a verified
runtime contract.

Core and sandbox suites use native Pants targets, one per test file. Core pytest
preserves main's worker count and file distribution (currently four workers with
`--dist=loadfile`), including numerical-library thread limits. Sandbox remains
serial. The planner verifies one batch per explicit execution group. The first
incremental candidate is `test_add_legacy_agent_environment_servers.py`; its
migration scanner inputs, configuration, dependencies and runtime fingerprint
form its cache key. Other native tests remain forced, including tests requiring
the actual Git tree. Failed runtime eligibility checks fall back to forced
execution. Server tests, session-routing E2E and lint retain their existing
commands through session-only Pants workspace processes.

Native artifacts include each batch's original Pants result-info, JUnit, raw
coverage and `test-accounting.json`. Reports distinguish executed, cached and
forced cases from actual Pants provenance, and merge collection skips and all
batch coverage exactly once. A request for incremental mode is not evidence of
a cache hit. Editable-install snapshots canonicalize temporary paths and restore
the genuine distribution and console scripts at runtime for forced tests.

For evidence, run a cold full comparison, repeat the same source to verify real
result hits, then sample a changed commit. A safe unrelated input change should
retain eligible results; a declared-input change must execute them again. Keep
`force_full=true` throughout this sequence. Use `execution_mode=forced` for an
explicit control, and compare whole-job costs, including collection, setup and
cache transfers. Hosted incremental timing remains unproven until that sequence
passes full outcome and coverage checks.

## Cache policy

The uv cache mirrors main's immutable OS/`uv.lock` key and OS-level restore
prefix, with separate baseline and Pants namespaces. Existing exact hits can
contain different dependencies because each arm's first successful writer may
differ; matching keys do not establish equal warmth.

Native reusable process stores use `pants-full-results-v1-` archives partitioned
by runner platform, job, shard, implementation contract and dependency digest.
The sampled source is the final key segment; a compatible prior-source archive
may restore before Pants evaluates its declared inputs. Every successful native
shard with an eligible batch owns its result archive, including sandbox if it
later gains eligible targets. Only primary-key misses save. Each store must fit
512 MiB by both apparent and allocated size. Forced process stores and native
named dependency trees are not archived; their repeated PEX/setup cost remains
part of measured CI time. Forced controls neither restore nor save result stores.

Workspace jobs retain the `pants-full-engine-v4-` dependency archives with shard
zero as writer. The launcher's `~/.cache/nce` has a separate exact-key ARC archive
keyed by platform, launcher checksum and Pants version; successful core shard
three is its sole writer. Routed runs without that shard do not refresh it.
Hosted lint and session-routing jobs bootstrap independently. Bootstrap retries
are limited to setup; test and lint commands execute once.

The monitor bounds branch-scoped experiment caches to 5 GiB: remove obsolete
contracts and older source generations per job/shard/key/version, then optional
workspace engine archives before reusable results. UV and shared launcher
archives retain their existing policy. The five possible native stores total at
most 2.5 GiB, but the monitor must still check actual aggregate usage and retain
cache inventories. Inspect actual restore/save outcomes for every comparison.

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

The [full native validation](https://github.com/NVIDIA-NeMo/Gym/actions/runs/41837880879)
passed all workload pairs and both combined coverage gates at 95.20%. Core and
sandbox used native Pants targets with forced execution; other jobs used workspace
commands. Native workload execution was slower, and dependency-cache warmth
differed between arms. This establishes full execution parity, not a causal
speedup. That run predates the shared job-type engine-cache keys described above.
