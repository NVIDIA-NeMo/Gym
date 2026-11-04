# Pants CI demo

This branch measures whether Pants can reduce feedback time for a declared subset
of Gym tests. Existing CI remains authoritative. It is an experiment, with no RFC
or migration of the user-facing Gym commands.

`pilot.json` declares the test files, target generators, Python version, and locked
requirements. Python imports are inferred. YAML, fixtures, scripts, and subprocess
inputs must be declared in BUILD dependencies. An unowned changed file forces a
full pilot run rather than silently skipping tests.

## Run one paired comparison

Provision `uv`, Python 3.13.14, and the scie-pants launcher. The launcher version in
`.github/workflows/pants-pilot.yml` is pinned; `pants.toml` pins the Pants engine.
Stage the experiment files first so the complete overlay is captured. Use an
output directory outside the checkout:

```bash
export PANTS_BIN="$PWD/.pants-bin/pants"
uv run --no-project --python 3.13.14 python scripts/pants/compare.py \
  --base <sampled-main-sha> --parent <sampled-main-parent-sha> \
  --repeats 3 --output /absolute/path/to/new-results
```

The checkout must contain the sampled main source plus the pilot overlay.
`monitor.py` prepares those checkouts automatically. Do not pass an unrelated main
SHA to an ordinary checkout. Capture dependencies with `refresh-locks.sh` after
changes to the authoritative dependency configuration; do not edit both lock
representations independently.

Parent and target dependency inputs must both validate against their own uv lock.
A dependency-change sample requiring a different parent lock fails explicitly;
this initial runner does not benchmark mismatched dependency transitions.

Each comparison retains JSON results, a Markdown summary, the exact experiment
patch and hash, main/parent/pilot SHAs, commands, stdout/stderr, and JUnit outcomes.
The runner checks exact test identity/status parity and withholds speed ratios
when a pair fails. Commands time out after 15 minutes and terminate their process
group. A comparison failure still writes its diagnostics and returns nonzero.

## What is measured

| Case | uv baseline | Pants |
| --- | --- | --- |
| Cold | New venv and empty download cache | Empty package/build/test caches |
| Warm dependencies | Recreate venv from cached downloads; rerun tests | Cached dependencies; force test execution |
| Warm results | Reuse venv; rerun tests | Reuse successful test results where valid |
| Changed commit | Warm on parent, then run full pilot subset | Warm on parent, then run selected affected targets |

Both engines use the same Python and pinned dependency versions. Tool binaries
and Python provisioning happen before timing. Runs are serial on the same runner,
with alternating engine order between repetitions. uv runs one pytest process;
Pants schedules test files with its default parallelism, using the same available
machine resources. CPU count and tool versions are recorded. Any speedup combines
scheduling, packaging and caching effects; it is not attributed solely to caching. Changed selection includes
transitive file dependencies in both parent and current graphs, so deleted inputs
remain visible. The native `--changed-diffspec=parent..main` selection time is included. A separate
full parent/current dependency-graph audit checks that selection and records its
own overhead outside the timing, alongside the full-run correctness checks.
Unknown input changes or disagreement with the graph audit force all pilot tests
to rerun without cached test results.

The changed baseline is the **full pilot subset**, not a reimplementation of
Gym's existing CI routing. These results do not measure whole-suite CI, its
parallel shards, queue delay, coverage, or GPU/live-service tests. Pants JUnit
includes cached results; its testcase count is not an executed-test count. Cached/executed target counts
come from the Pants summary (null when the output format is unrecognized); the
raw summary and process logs are retained. Existing coverage gates
are unchanged. Three repetitions yield a median but are still a small sample.

## Keep the experiment current

```bash
uv run --no-project --python 3.13.14 python scripts/pants/monitor.py \
  --once --max-samples 3 --repeats 3 --output /absolute/path/to/persistent-results
```

The monitor fetches main, rebases the clean pilot branch, samples recent real
commits across core/server/configuration changes, and runs the comparisons.
Reuse its output directory to preserve sampling state and daily scorecards. A
local Codex heartbeat calls it every six hours; workflow cron is deliberately
absent because GitHub schedules do not execute from an experimental branch.
The optional Actions workflow runs on pushes to this branch or manual dispatch,
uses read-only permissions, and uploads logs/artifacts for 14 days. Hosted runs
skip branch mutation; the local monitor owns rebasing. A newly added workflow may
need its branch push before it is available to dispatch.

Run the harness checks independently of Gym's test dependencies:

```bash
python -m unittest discover -s scripts/pants -p 'test_*.py'
```

Collect three to five days of comparisons initially. Judge correctness, timing
and maintenance effort together; a faster run with fewer or skipped tests is not
a valid result. Conflicting server dependency sets and the remaining Gym suite
need separate expansion and parity checks before this could replace existing CI.

Gym's lazy public exports currently connect much of the core dependency graph.
The pilot preserves those edges; it does not suppress dependencies to improve its
timings. Daily scorecards separate different test scopes and dependency locks.
