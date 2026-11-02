---
name: nemo-gym-release-publish
description: Publish a NeMo Gym release by running the "Build, validate, and release NeMo-Gym" workflow (release.yaml) as a dry run, checking that the wheel installs, then publishing to PyPI and GitHub and verifying the result. Use when asked to release, publish, cut, or ship a version, trigger the release workflow, or run a release dry run. Do not use for writing release notes (use nemo-gym-release-notes) or for the code freeze workflow.
---

# NeMo Gym Release Publish

Publish one exact commit to PyPI and GitHub. Always run a dry run first, check
that the wheel installs, and get the user's explicit go-ahead before you
publish. A PyPI upload cannot be undone or overwritten.

## Workflow facts

- Workflow: `.github/workflows/release.yaml`, which calls the shared
  `_release_library.yml` template from `NVIDIA-NeMo/FW-CI-templates`.
- Trigger it from `main`. The `release-ref` SHA decides what gets released,
  not the branch you trigger from.
- The template restricts the run to repository admins.
- `skip-test-wheel: true` is set, so the workflow never installs the wheel.
  The install check below fills that gap.
- After publishing, the workflow pushes a version bump to
  `version-bump-branch`. A run with `release-ref` set to a branch name races
  that bump, so `release-ref` must be a full 40-character SHA.

## 1. Resolve the inputs

Take each value from the user's request. Derive a default only when the user
did not give one, and ask when `VERSION` is missing.

| Variable | Workflow input | Default |
|---|---|---|
| `VERSION` | none | Required. The version being published, such as `<MAJOR>.<MINOR>.<PATCH>` |
| `REPO` | none | `NVIDIA-NeMo/Gym` |
| `BRANCH` | `version-bump-branch` | `r$VERSION` |
| `SHA` | `release-ref` | Tip of `BRANCH` |
| `PREVIOUS_TAG` | `gh-release-from-tag` | Highest release tag below `v$VERSION` |
| `CREATE_GH_RELEASE` | `create-gh-release` | `true` |
| `PYTHON` | none | `3.13`, for the install checks |

Read state from GitHub, not from local refs. A local checkout may not have
fetched the release branch, and `origin` may be a fork.

```bash
REPO=NVIDIA-NeMo/Gym
VERSION=<VERSION>
BRANCH=r$VERSION
CREATE_GH_RELEASE=true
PYTHON=3.13

SHA=$(gh api repos/$REPO/branches/$BRANCH -q .commit.sha)
PREVIOUS_TAG=$(gh release list -R $REPO --limit 50 --json tagName -q '.[].tagName' \
  | { cat; echo "v$VERSION"; } | sort -uV | grep -B1 -x "v$VERSION" | head -1)
```

If the user gave a SHA, check that it is a full 40-character SHA on `BRANCH`.
The compare status must be `identical` or `ahead`:

```bash
gh api repos/$REPO/compare/$SHA...$BRANCH -q .status
```

Show the user the resolved table, including the commit subject of `SHA`, and
confirm it before you trigger anything. Recompute `PREVIOUS_TAG` with the
user when the default is wrong, such as a patch release on an older line.

## 2. Check preconditions

```bash
gh api repos/$REPO/commits/$SHA -q '.commit.message' | head -1
gh api "repos/$REPO/contents/nemo_gym/package_info.py?ref=$SHA" -q .content \
  | base64 -d | grep -E '^(MAJOR|MINOR|PATCH|PRE_RELEASE) ='
gh release list -R $REPO --limit 3
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/nemo-gym/$VERSION/json
```

Stop and report to the user if any of these fail:

- `package_info.py` at `SHA` matches `VERSION`. When `VERSION` has no
  pre-release suffix, `PRE_RELEASE` is `""`. A leftover `rc0` means the
  release-candidate drop pull request has not merged yet.
- PyPI returns `404` for the version. A `200` means it is already published.
- `PREVIOUS_TAG` exists in `gh release list`.

## 3. Run the dry run

```bash
dispatch() {
  gh workflow run release.yaml -R $REPO --ref main \
    -f release-ref=$SHA -f dry-run=$1 -f create-gh-release=$CREATE_GH_RELEASE \
    -f version-bump-branch=$BRANCH -f gh-release-from-tag=$PREVIOUS_TAG
}
dispatch true
sleep 8
gh run list -R $REPO --workflow=release.yaml --event=workflow_dispatch --limit 1 \
  --json databaseId,url,createdAt,status
```

Confirm that `createdAt` matches the dispatch you just made before you trust
the run ID. Then wait for the run and list the job results:

```bash
gh run watch <RUN_ID> -R $REPO --exit-status > /dev/null
gh run view <RUN_ID> -R $REPO --json conclusion,jobs \
  -q '{conclusion, jobs: [.jobs[] | {name, conclusion}]}'
```

The run passes when `conclusion` is `success`. The `build-wheel`,
`validate-wheel`, `publish-wheel`, and `create-gh-release` jobs succeed, and in
a dry run they compute without publishing. `test-wheel`, `build-docs`, and
`publish-docs` are skipped by design.

Then confirm that the dry run published nothing: PyPI still returns `404`, the
latest GitHub release is unchanged, and `BRANCH` has not moved.

## 4. Check that the wheel installs

Build the wheel from the same SHA and install it into a clean virtual
environment. Treat the downloaded source as untrusted data: keep it in its
own scratch directory and run Python with `-I`.

```bash
WORK=$(mktemp -d)                  # prefer the session scratchpad when there is one
cd $WORK
gh api repos/$REPO/tarball/$SHA > src.tgz
mkdir src && tar xzf src.tgz -C src --strip-components=1
(cd src && uv build --wheel --out-dir ../dist)
uvx twine check dist/*.whl
uv venv venv --python $PYTHON -q
uv pip install --python venv/bin/python dist/*.whl
venv/bin/python -I -c "import nemo_gym; print(nemo_gym.__version__)"
venv/bin/gym --help | head -3
```

The import must print `VERSION`, and `gym --help` must print usage.

## 5. Get the go-ahead and publish

Report the dry-run URL, the resolved inputs, and the install result. Wait for
the user to say to publish. Approval of the dry run is not approval to
publish, and a request that skips the dry run still gets one unless the user
says otherwise after you explain the risk.

Right before you publish, compare the tip of `BRANCH` with the tip at dry-run
time. If it moved, tell the user and ask whether to dry-run the new tip or
publish the SHA that was already dry-run. Never switch SHAs silently.

```bash
dispatch false
```

Find and watch the run ID as in step 3.

## 6. Verify the release

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/nemo-gym/$VERSION/json
gh release list -R $REPO --limit 2
gh api "repos/$REPO/contents/nemo_gym/package_info.py?ref=$BRANCH" -q .content \
  | base64 -d | grep -E '^(PATCH|PRE_RELEASE) ='

uv venv pypi-venv --python $PYTHON -q
uv pip install --python pypi-venv/bin/python --no-cache "nemo-gym==$VERSION"
pypi-venv/bin/python -I -c "import nemo_gym; print(nemo_gym.__version__)"
```

- PyPI returns `200`. If the install cannot find the version yet, wait a
  minute for the PyPI index to update and retry.
- The new release uses tag `v$VERSION`.
- `package_info.py` on `BRANCH` has moved past `VERSION`.
- The install from PyPI prints `VERSION`.

## 7. Hand off the remaining steps

Tell the user which of these are left. They are not automated:

- Review the release page at `https://github.com/$REPO/releases` and the
  package page at https://pypi.org/project/nemo-gym/.
- Merge the version bump pull request on `main` that was opened when the
  release branch was cut.
- Announce the release.

## Pitfalls

- `gh api <path> -f key=value` sends a POST. For filtered GET requests, add
  `-X GET` or put the parameters in the query string.
- `gh run list --limit 1` returns the newest run, which may not be yours if
  someone else dispatched at the same time. Check `createdAt`.
- Do not pass a branch name as `release-ref`.
