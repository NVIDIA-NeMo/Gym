# CPU CI contract

These scripts are the portable execution contract shared by GitHub Actions and the internal
NeMo CI adapter. `contract-version` contains the interface version expected by the adapter.

- `lint.sh` installs the pinned pre-commit release and runs every hook against the repository.
- `core_unit_tests.sh` installs the `dev` extra and runs core tests with the `sandbox` marker
  excluded.
- `server_tests.sh SHARD_INDEX NUM_SHARDS` installs the `dev` extra and runs one zero-based shard
  of the full server suite.

All three entrypoints preserve ANSI result colors in non-interactive CI logs without requiring a
pseudo-terminal: lint passes `--color=always` to pre-commit, core passes `--color=yes` through
`PYTEST_ADDOPTS`, and server tests export `PY_COLORS=1` for every nested pytest process.

All scripts can be invoked from any working directory and propagate the underlying command's exit
status. Contract version 3 is CPU-only: it does not install the `sandbox` extra, run sandbox-marked
tests, require sandbox credentials, or infer GPU availability. When `GYM_CI_JUNIT_DIR` is set,
core tests write `core.xml` there and each server module writes a uniquely named pytest JUnit XML
report with a module-qualified class prefix. Lint remains a required pre-commit status because
pre-commit has no native JUnit output. GitHub's sandbox tests remain a separate public-only workflow
step outside this contract.

Contract version 3 keeps uv's resolved cache directory consistent across root and nested server
installs. A CI provider can set `UV_CACHE_DIR` to a persistent package cache and optionally set
`GYM_CI_UV_VENV_DIR` to place the server driver environment and disposable per-server environments
on faster local storage. Gym owns server discovery, removes every nested environment after its
tests finish, and removes the driver environment when the shard terminates normally. The venv root
must be an absolute, non-root path that is private and unique to the current CI job. When it is set,
installs use uv's copy link mode because the persistent cache and local venv root can be on
different filesystems.

Both CI providers run `core_unit_tests.sh` in its deterministic `dev`-only environment. GitHub then
installs its public-only sandbox dependencies and runs the sandbox-marked tests as a separate
coverage pass.

Before setup, the entrypoints remove inherited variables that can change the work selected by CI.
Lint clears `SKIP`; core and server tests clear external Gym roots, root configuration, and
`PYTHONPATH`; server tests also clear prerelease-install and inherited pytest-option overrides.
Resolver/index settings are intentionally preserved so internal package access remains available.
