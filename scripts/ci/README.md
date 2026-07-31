# CPU CI contract

These scripts are the portable execution contract shared by GitHub Actions and the internal
NeMo CI adapter. `contract-version` contains the interface version expected by the adapter.

- `lint.sh` installs the pinned pre-commit release and runs every hook against the repository.
- `core_unit_tests.sh` installs the `dev` extra and runs core tests with the `sandbox` marker
  excluded.
- `server_tests.sh SHARD_INDEX NUM_SHARDS` installs the `dev` extra and runs one zero-based shard
  of the full server suite.

All scripts can be invoked from any working directory and propagate the underlying command's exit
status. Contract version 1 is CPU-only: it does not install the `sandbox` extra, run sandbox-marked
tests, require sandbox credentials, or infer GPU availability. GitHub's sandbox tests remain a
separate public-only workflow step outside this contract.
