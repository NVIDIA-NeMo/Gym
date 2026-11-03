# Shared CI contracts

## Package approvals

`check-review-approvals.sh` evaluates `.github/CODEOWNERS` on GitHub and GitLab. GitHub Teams
are the membership source. GitLab groups and GitLab CODEOWNERS sections are not used. The shell
entry point delegates matching and API calls to a Python standard-library helper; Gym is not installed.

```bash
bash scripts/ci/check-review-approvals.sh github --repository NVIDIA-NeMo/Gym --number 123
bash scripts/ci/check-review-approvals.sh gitlab --project-id "$CI_PROJECT_ID" --number "$CI_MERGE_REQUEST_IID"
```

Both modes require `GH_TOKEN` or `GITHUB_TOKEN` with organization-members read access. GitLab
also requires `GITLAB_TOKEN` and `CI_API_V4_URL`. Exit codes are 0 for satisfied ownership, 1 for
missing approvals, and 2 for API, configuration, or evidence errors. `--output FILE` saves the
structured verdict and its head SHA.

Each changed file uses the last matching standard CODEOWNERS rule. One member of any team on
that line must approve: the specialist team and `frontier-evals` are alternatives. Different
packages require their respective approvals; one Frontier Eval approval can cover every package.
Paths outside the package map retain a maintainer fallback with Frontier Eval as an alternative;
CI infrastructure and the existing BioNeMo overrides keep their dedicated owners.
`# suggest:` comments never gate approval. Renames check old and new paths. The checker rejects
truncated file lists, author approvals, and GitHub approvals for an older head. GitLab must reset
approvals on push, which is verified through its API. A changing evidence snapshot fails the check.

Policy and identity links come from the immutable target commit, not the submitted branch. The
checker itself must execute from trusted target-branch code. GitHub uses `pull_request_target`, plus
a secret-free review-event workflow followed by `workflow_run`. It posts the `Package approvals`
commit status against the actual PR head. Use its `workflow_dispatch` input to reevaluate a specific
PR after a team membership change, including removal of a previously eligible approver.

`.github/reviewer-identities.json` links stable GitLab and GitHub numeric account IDs; it does not
duplicate groups. Initial links were resolved through the corporate SSO audit and provider APIs.
A new team member without a verified identity link can approve on GitHub; GitLab ignores that
person's approval until the link is added. Display names and usernames are not authorization keys.

The GitLab adapter executes on a webhook-triggered pipeline of protected `nv-internal-main`. It
appends `Package approvals` to the exact MR head pipeline using `pipeline_id`. The existing
"pipelines must succeed" setting then enforces the verdict. Approval removal reruns the checker
and changes the job to failed. The trigger job needs protected, masked `GH_TOKEN` (members read)
and `GITLAB_TOKEN` (project API) variables. No organization-management PAT is needed by this check.

Deployment order:

1. Merge and mirror the policy, identity links, checker, and trusted CI configuration.
2. Register a GitLab merge-request webhook for the pipeline-trigger API on protected
   `nv-internal-main`, including approval, unapproval, and source-update events. Its payload is
   supplied through GitLab's `TRIGGER_PAYLOAD` file. Keep the trigger token outside the repository.
   Retry this trusted pipeline to refresh the verdict after a GitHub team membership change.
3. Verify missing approval, specialist approval, Frontier Eval approval, approval removal, and a
   source push on test PRs/MRs. Include fork PRs and renames.
4. Require the GitHub `Package approvals` commit status, restricted to the configured bot App.
   Keep GitLab's successful-pipeline requirement enabled. Disable native code-owner approval
   requirements only after both provider checks have been verified.

Local validation without the Gym runtime:

```bash
python tests/unit_tests/test_review_approvals.py
shellcheck scripts/ci/check-review-approvals.sh
actionlint .github/workflows/review-approval-event.yml .github/workflows/review-approvals.yml
```

## CPU CI contract

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

## Four-way core sharding on GitHub

`unit-tests.yml` runs `core_unit_tests.sh INDEX 4` on four independent runners.
The default invocation without arguments still runs every non-sandbox test for the internal adapter.
The opt-in pytest plugin assigns whole collected files, longest first, to the least-loaded shard.
Assignments use `core_test_durations.json`; new files use the median recorded duration.
Every collected file belongs to exactly one shard. Tests within a file stay serial.

The initial weights estimate file durations from timestamped verbose results in run `41587144451`.
They exclude collection and sandbox time; log buffering makes them approximate, not speedup measurements.
Each shard publishes JUnit durations with its coverage artifact for future weight updates.
Sum JUnit testcase times by source file to refresh the `files` map, and update the source metadata.
Keep all four shards on the same checked-in weights so they select disjoint partitions.

Sandbox tests run once on a separate runner. Coverage uses relative paths across runner checkouts.
The coverage job requires all four core files and the sandbox file before combining them.
Only the combined report enforces `pyproject.toml`'s coverage threshold and precision.
The report includes measured Git-tracked sources, not dependency-cache helpers or synthetic code from tests.
Missing tracked sources still fail reporting; the job does not ignore coverage errors.
The existing `Test` check requires every core shard, sandbox tests, combined coverage, and any required changed-server tests.
The eight server shards and the separate on-demand `full-test-suite.yml` keep their existing behavior.
