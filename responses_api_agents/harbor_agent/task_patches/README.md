# Terminal-Bench 2.1 reference-solution patches

Opt-in patches applied to a pinned TB 2.1 checkout by:

```bash
python responses_api_agents/harbor_agent/prepare_terminal_bench_2_1.py --apply-task-patches
```

**Off by default.** Without the flag the benchmark is prepared exactly as published upstream.

## Scope and why this is safe

Every patch here modifies `solution/solve.sh` and nothing else, and
`prepare_terminal_bench_2_1.py` *enforces* that — a patch touching `tests/` or `task.toml`
is refused, because those define what is asked and how it is graded.

`solution/` is uploaded only by Harbor's `OracleAgent`. A model agent (e.g. Terminus-2)
never sees it. **These patches therefore cannot affect a scored model run**; they exist so
that a *gold-patch run* — which executes each task's reference solution and should score
89/89 — measures the harness rather than upstream bit-rot.

A patch that does not apply cleanly raises instead of being skipped. The checkout is pinned
to `PINNED_COMMIT`, so a failure means the pin moved and the patch needs revisiting.

## Why each patch exists

All three were found by a gold-patch run of all 89 tasks (2026-07-27), which scored 85/89
before these fixes and 89/89 after. None of them is a harness defect; each is the outside
world drifting away from a task image that is pinned in time.

### `build-cython-ext.patch` — pin `planarity==0.6`

`planarity 1.0.0` (published 2026-06-29, ~8 months after the task image was built) renamed the
drawplanar graph attributes: node `pos`/`start`/`end` became `vertex_position`/`vertex_start`/
`vertex_end`. `pyknotid 0.5.3` reads the old keys verbatim in
`representations/representation.py`, so `pip install -e .` resolving planarity 1.x makes
`test_pyknotid_repository_tests` die with `KeyError: 'pos'`. The task's own `setup.py` leaves
every dependency unpinned. `peewee<4` is included as pre-emptive hardening for the same class
of drift.

### `caffe-cifar-10.patch` — fetch CIFAR-10 from a verified mirror

`data/cifar10/get_cifar10.sh` fetches `http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz`,
which 301-redirects to `cave.cs.toronto.edu`. That host TCP-RSTs connections from some
networks (reproduced from this cluster's egress on ports 80 and 443, with and without SNI,
while the same host served a third-party fetcher normally). `get_cifar10.sh` has no error
handling, so the run continued without data and caffe aborted on a glog `CHECK failed` —
which looks like a crash but is a missing-file cascade.

The replacement downloads a mirror and **verifies `md5 == c32a1d4ab5d03f1284b67883e8d87530`**,
which matches the official tarball byte-for-byte, so the task is not weakened. Drop this patch
if your network can reach the upstream host.

### `mcmc-sampling-stan.patch` — install `cmake`, pin `RcppParallel`

`solve.sh` installs `StanHeaders`, which needs `RcppParallel (>= 5.1.4)` but does not pin it.
`RcppParallel 6.0.0` (published 2026-07-23, four days before the run) switched to a
CMake-driven oneTBB build, and the task image ships no `cmake` — so the install failed with
`cmake was not found`, cascading to `there is no package called 'rstan'`.

The patch adds `cmake` **and** pins `RcppParallel` with `upgrade='never'`. Either alone would
work today; both together make the run independent of whatever CRAN publishes next (two runs
44 minutes apart resolved different RcppParallel versions). `MAKEFLAGS` is also raised from
`-j2` to `-j8` to match the sandbox CPU allocation.

## Upstreaming

The two dependency pins are not specific to this deployment and belong upstream in
`harbor-framework/terminal-bench-2-1`; the CIFAR-10 mirror is a network workaround and may not
be. Once upstream carries equivalent fixes, delete the corresponding patch here and bump
`PINNED_COMMIT`.
