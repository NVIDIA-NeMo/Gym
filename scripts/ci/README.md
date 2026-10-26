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

The Gym container image (`docker/Dockerfile`, `release` target) seeds its `UV_CACHE_DIR` with the
root lock, so `lint.sh` and `core_unit_tests.sh` run without a package index; in the container,
`setup_dev.sh` syncs the driver environment with `--offline`. Every server module also installs its
own unlocked `requirements.txt` or `pyproject.toml`, which the root lock does not cover. Built with
`--build-arg NEMO_GYM_SEED_SERVER_ENVS=1`, the image also seeds every server's environment through
`ng_test_all +setup_only=true`, installed exactly as `server_tests.sh` installs it, so the server
suite runs offline too when the provider sets `UV_OFFLINE=1` for the nested installs. A server
whose Git source, direct or transitive, is not pinned to a full commit (a branch, a tag, an
abbreviated SHA, or no ref) still needs the network, because `uv` must fetch to resolve it.

## Shared server-test data

IFBench source and NLTK resources live under
`s3://nemo-ci-storage-31ff58e3-a555-4c8b-ad91-c453ce311bbd/nemo-fw/TestData/nemo_gym/v1/`.
GitHub test jobs use the existing `nemo-ci-aws-cpu-4-16` ARC runner pool, which already mounts
that bucket read-only at `/mnt/s3-data-ro`. Jobs set `NEMO_GYM_TEST_DATA_ROOT` to
`/mnt/s3-data-ro/nemo-fw/TestData/nemo_gym/v1`, set `NLTK_DATA` to its `nltk_data` child, and
run `scripts/ci/check_test_data.py`. The preflight requires 14 nonempty files and the pinned
IFBench revision before tests start. Its `--nltk-cache` option copies the NLTK resources into a
private cache and exports that path for the job: NLTK 3.10 refuses foreign-UID or writable shared
trees. Missing packages can download into the private cache. Compiler and browser dependencies
are installed by `install_test_dependencies.sh`; the S3 mount is supplied by ARC.

TF-States owns recurring S3 → GCS → Singapore GCS transfers. Its existing `gpu-datasets-sync`
CronJob copies the complete `nemo-fw/TestData` parent to the read-only dataset export and verifies
the Gym bundle's checksums. CI-Events' `nemo-gym` profile mounts the same versioned subtree at
`/mnt/gcs-data-ro/nemo-fw/TestData/nemo_gym/v1` and checks each required file during inventory,
before admitting test shards. The existing dataset root is mounted so inventory can also report
an absent Gym subtree.
These declarations require artifact seeding, infrastructure apply, and a pod-launcher release
before they can supply a running job.

Prepare the bundle locally while online:

```bash
uv run --no-project --with 'nltk>=3.10.0' --with regex --with defusedxml \
  python scripts/ci/seed_test_data.py cache/gym-test-data/v1
(cd cache/gym-test-data/v1 && sha256sum --strict -c SHA256SUMS)
```

Upload all files beneath that directory to the S3 prefix, with `SHA256SUMS` last. Preserve the
relative paths across both GCS mirrors and the dataset export. Use a new versioned prefix for
future incompatible bundles and update consumers together.

For a supplied filesystem tree, set `NEMO_GYM_TEST_DATA_ROOT` to the versioned root and `NLTK_DATA`
to its `nltk_data` child. IFBench copies mounted source into its private `.ifbench` cache before
patching; an absent source path falls back to its pinned online clone. Existing NLTK resources
are read from the supplied paths; missing ones download into `$XDG_CACHE_HOME/nltk_data` (default
`~/.cache/nltk_data`). Supplied directories may be read-only. Scoring checks use cached resources
without fetching NLTK's package index. Aviary lifecycle tests use local synthetic GSM8K rows.
Other servers can still require their own external data or tools.

Both CI providers run `core_unit_tests.sh` in its deterministic `dev`-only environment. GitHub then
installs its public-only sandbox dependencies and runs the sandbox-marked tests as a separate
coverage pass.

Before setup, the entrypoints remove inherited variables that can change the work selected by CI.
Lint clears `SKIP`; core and server tests clear external Gym roots, root configuration, and
`PYTHONPATH`; server tests also clear prerelease-install, inherited pytest-option, and
`PYTHONSAFEPATH` overrides. Clearing `PYTHONSAFEPATH` keeps nested module imports consistent with
GitHub when a CI image enables Python safe-path mode globally. Resolver/index settings are
intentionally preserved so internal package access remains available.
