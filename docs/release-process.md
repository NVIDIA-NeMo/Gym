# NeMo Gym Release Process

## Overview

Releases follow a three-phase flow: code freeze, release, and cleanup. All workflows are manually triggered via GitHub Actions.

## Prerequisites

- Admin access to the [Gym repo](https://github.com/NVIDIA-NeMo/Gym)
- Push access to `r*` release branches — the branch protection rule restricts pushes to specific users and apps. Check Settings → Branches → `r*` rule to verify you're on the allowlist before triggering a release.
- The `r*` branch protection rule must have valid required status checks (`Test`, `Lint check`, `DCO`) — not stale checks from other repos

## Phase 1: Code Freeze

**Workflow:** "Code freeze" (`release-freeze.yml`)

This creates a release branch and bumps the version on main for the next dev cycle.

### Automated (when branch protection allows it)

1. Go to **Actions → Code freeze → Run workflow**
2. Select `release-type`: `major` or `minor`
3. Set `freeze-commit` to the SHA you want to cut from (or leave as `main` for HEAD)
4. Run with `dry-run: true` first, verify output, then re-run with `dry-run: false`

### Manual (if branch protection blocks the workflow)

If the `r*` branch protection rule blocks the automated workflow (e.g. stale required status checks, restrictive push permissions), do it manually:

```bash
# 1. Create the release branch from the freeze commit
git fetch origin
git checkout <freeze-commit-sha>
git switch --force-create r<VERSION>
git push -u origin r<VERSION>

# 2. Create the cherry-pick label (used to backport fixes from r<VERSION> to main)
gh label create "cherry-pick-r<VERSION>" --repo NVIDIA-NeMo/Gym

# 3. Drop the pre-release tag on the release branch
# Edit nemo_gym/package_info.py: set PRE_RELEASE = ""
# Open a PR targeting r<VERSION>

# 4. Bump version on main for the next dev cycle
# Edit nemo_gym/package_info.py: increment MINOR (or MAJOR), keep PRE_RELEASE = "rc0"
# Open a PR targeting main
```

## Phase 2: Release

**Workflow:** "Build, validate, and release NeMo-Gym" (`release.yaml`)

This builds the wheel, publishes to PyPI, and creates a GitHub release.

### Inputs

| Input | Value | Notes |
|---|---|---|
| `release-ref` | **Full 40-char commit SHA** | The exact commit to release. **Must be a SHA, not a branch name.** Get it with `git rev-parse r<VERSION>`. Using a branch name causes a race condition where the version-bump job mutates the branch before the wheel is built. |
| `dry-run` | `true` / `false` | `true` computes everything but publishes nothing. Always dry-run first. |
| `create-gh-release` | `true` | Creates the GitHub release with auto-generated changelog. |
| `version-bump-branch` | `r<VERSION>` | The release branch. After publishing, the workflow bumps this branch to the next patch version (e.g. `0.3.0` → `0.3.1rc0`). |
| `gh-release-from-tag` | Previous release tag | e.g. `v0.2.1`. Used to generate the changelog diff. |

### Steps

1. Merge the RC-drop PR into `r<VERSION>` (so the version is clean, e.g. `0.3.0` not `0.3.0rc0`)
2. Get the SHA: `git rev-parse origin/r<VERSION>`
3. Go to **Actions → Build, validate, and release NeMo-Gym → Run workflow**
4. Run with `dry-run: true` first and verify the output
5. Re-run with `dry-run: false` to publish

### What the workflow does

1. **Validates** you have admin permission
2. **Builds** the wheel from the pinned SHA
3. **Validates** the wheel (twine check, wheel contents)
4. **Publishes** the wheel to PyPI
5. **Creates** a GitHub release with changelog (diff from `gh-release-from-tag`)
6. **Bumps** the version on `version-bump-branch` to the next patch pre-release
7. **Sends** a Slack notification

## Phase 3: Post-Release Cleanup

1. Verify the GitHub release at `https://github.com/NVIDIA-NeMo/Gym/releases`
2. Verify the PyPI package at `https://pypi.org/project/nemo-gym/`
3. Merge the version bump PR on main (from Phase 1, step 4)
4. Announce the release

## Cherry-Picking Fixes to a Release Branch

After the freeze, bug fixes land on main first, then get backported:

1. Merge the fix PR to main
2. Add the `cherry-pick-r<VERSION>` label to the PR
3. The `cherry-pick-release-commit.yml` workflow automatically creates a PR to backport the commit from `r<VERSION>` to main

## Common Pitfalls

| Pitfall | What goes wrong | How to avoid |
|---|---|---|
| Using a branch name as `release-ref` | Version-bump job mutates the branch before build-wheel reads it, publishing the wrong version | Always use a commit SHA |
| Stale `Nemo_CICD_Test` check on `r*` branch protection | PRs to the release branch can never be merged | Replace with `Test`, `Lint check`, `DCO` in branch protection settings |
| Forgetting to merge the RC-drop PR before releasing | Publishes `X.Y.Zrc0` instead of `X.Y.Z` | Merge the RC-drop PR first, then get the SHA |
| Running the release from `main` instead of the release branch | `run-on-version-tag-only` may silently no-op publish steps | Always trigger from the release branch context |

## Version Format

Versions are defined in `nemo_gym/package_info.py`:

```python
MAJOR = 0
MINOR = 3
PATCH = 0
PRE_RELEASE = "rc0"  # empty string for stable releases
```

- Release branches: `r<MAJOR>.<MINOR>.<PATCH>` (e.g. `r0.3.0`)
- Release tags: `v<MAJOR>.<MINOR>.<PATCH>` (e.g. `v0.3.0`)
- After release, the branch is bumped to `<MAJOR>.<MINOR>.<PATCH+1>rc0`
