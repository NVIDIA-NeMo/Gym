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
status. Contract version 2 is CPU-only: it does not install the `sandbox` extra, run sandbox-marked
tests, require sandbox credentials, or infer GPU availability. When `GYM_CI_JUNIT_DIR` is set,
core tests write `core.xml` there and each server module writes a uniquely named pytest JUnit XML
report with a module-qualified class prefix. Lint remains a required pre-commit status because
pre-commit has no native JUnit output. GitHub's sandbox tests remain a separate public-only workflow
step outside this contract.

Both CI providers run `core_unit_tests.sh` in its deterministic `dev`-only environment. GitHub then
installs its public-only sandbox dependencies and runs the sandbox-marked tests as a separate
coverage pass.

Before setup, the entrypoints remove inherited variables that can change the work selected by CI.
Lint clears `SKIP`; core and server tests clear external Gym roots, root configuration, and
`PYTHONPATH`; server tests also clear prerelease-install and inherited pytest-option overrides.
Resolver/index settings are intentionally preserved so internal package access remains available.
