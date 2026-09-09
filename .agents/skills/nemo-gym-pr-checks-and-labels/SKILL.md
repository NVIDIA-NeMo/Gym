---
name: nemo-gym-pr-checks-and-labels
description: Inspect and prepare NVIDIA-NeMo/Gym pull requests by selecting repository-specific local and CI checks, diagnosing missing or failed checks, and choosing current type, area, and state labels. Use for PR readiness, check selection, CI handoff, or PR labeling. Do not use for general code review or issue-triage sweeps.
---

# NeMo Gym PR Checks and Labels

Use this workflow for a local branch or a GitHub pull request in
`NVIDIA-NeMo/Gym`. Keep generic style rules in `AGENTS.md`, `pyproject.toml`,
and `.pre-commit-config.yaml`; this skill covers the decisions that depend on
Gym's change classifier, runtime contracts, and live PR taxonomy.

## Establish Scope

Read the artifact before choosing checks or labels:

```bash
git status --short
git diff --stat <base>...HEAD
git diff --name-only <base>...HEAD

gh pr view <PR> --repo NVIDIA-NeMo/Gym \
  --json number,title,body,isDraft,author,baseRefName,headRefName,headRefOid,labels,reviewDecision,mergeStateStatus
gh api repos/NVIDIA-NeMo/Gym/pulls/<PR> --jq .author_association
gh pr diff <PR> --repo NVIDIA-NeMo/Gym --name-only
gh pr checks <PR> --repo NVIDIA-NeMo/Gym
```

For local validation and CI routing, treat these repository files as
authoritative:

- `AGENTS.md` for the quality bar and required rollout evidence.
- `CONTRIBUTING.md` for contributor and environment requirements.
- `.pre-commit-config.yaml` and `pyproject.toml` for enforced style.
- `.github/actions/classify-changes/action.yml` for test-scope classification.
- `.github/workflows/unit-tests.yml` and `.github/workflows/cicd-main.yml` for
  actual CI behavior.
- `scripts/ci/cov_fail_under.py` for the coverage threshold used by CI.

When those files disagree with this skill, follow the repository files and
report the drift.

## Check PR Metadata

Apply the title and body contract from `AGENTS.md` before declaring a PR ready:

- Normal authored PR titles use `type(optional-scope): imperative summary`.
  Do not add Megatron Bridge's `[area]` prefix; Gym expresses that information
  with an `area:*` label. Preserve generated release and cherry-pick titles.
- The body explains what changed and why, links the relevant issue or explains
  why none is needed, lists exact validation, and provides rollout evidence or
  an explicit justified `N/A`.
- User-visible compatibility, migration, configuration, and benchmark-result
  effects are stated when applicable.
- Missing final validation is acceptable while the PR is a draft, but report it
  before changing the PR to ready for review.

## Select Checks

Run `pre-commit run --all-files` before handoff. Hooks can modify files; review
the resulting diff, stage only intended changes, and rerun until clean.

Then classify the complete PR diff using the same precedence as CI:

| Diff classification | Paths | CI test scope | Local evidence |
|---|---|---|---|
| Docs-only | Only `**.md`, `fern/**`, `LICENSE`, or `benchmarks/**` | Unit tests are skipped | For Fern changes, follow `fern/README.md`; otherwise run relevant docs/link checks |
| Server-only | One or more files in `resources_servers/**`, `responses_api_agents/**`, or `responses_api_models/**`, with no uncategorized files | Core and sandbox coverage tests plus tests for changed servers | Run `gym env test --resources-server <name>` for each changed server and targeted core tests when shared behavior is exercised |
| Full | Any uncategorized file, including core code, CI, scripts, test infrastructure, or native skill-discovery links | Core and sandbox coverage tests plus the eight-shard server suite | Run targeted tests for the changed contract; use `gym dev test` and `gym env test --all` when the change warrants the full local cost |

The precedence is full over server-only over docs-only. A Markdown file does
not make a mixed PR docs-only. A canonical `SKILL.md`-only edit matches the
Markdown rule, while adding a `.claude/skills` or `.codex/skills` symlink makes
the diff full-scope under the current classifier.

Additional evidence is required by behavior, not just path:

- For behavior-changing environment or agent code, run a representative real
  smoke rollout with a model and inspect both agent and verifier behavior.
- For a new resources server, satisfy the fixtures and validation contract in
  `CONTRIBUTING.md`, including server tests and example data/rollouts.
- For an AI-generated test, confirm it fails for the broken behavior and
  asserts an observable contract rather than generated wording or a
  pass-through mock.
- For docs, catalog, manifest, and other non-runtime changes, record model
  rollouts as not applicable rather than spending model compute.

Do not claim a check passed unless it ran against the final diff. If a check is
too expensive or unavailable, name it and explain the residual risk.

## Interpret GitHub Checks

Distinguish three states:

1. **Failed:** inspect the failing job and logs, reproduce the smallest relevant
   command, and determine whether the failure belongs to the PR or
   infrastructure.
2. **Pending:** report the running or queued check; do not treat it as success.
3. **Missing:** verify the path classification, draft state, head SHA, and trust
   gate before assuming CI is broken.

The main Gym pipeline runs from `pull-request/<number>`. An unverified or
external head commit may need a maintainer to comment `/ok to test <full-sha>`.
That comment authorizes repository CI to execute contributor code, so never post
it merely because a check is missing; do so only when the user explicitly asks
and has the needed maintainer authority.

Check DCO separately. Every commit must contain a `Signed-off-by` trailer;
cryptographic signing is optional.

## Choose Labels

Fetch the live labels and descriptions before making a decision:

```bash
gh label list --repo NVIDIA-NeMo/Gym --limit 200 \
  --json name,description --jq '.[] | [.name, .description] | @tsv'
```

Live repository metadata takes precedence over the examples below. For a
non-draft PR, prefer one label from each applicable family:

- **Type:** `bug`, `feature`, `docs`, `ci`, or `support`. Use `docs` when the
  deliverable is documentation or repository guidance even if CI classifies a
  skill file as a full-suite path. Preserve legacy synonyms unless the user
  explicitly asks for label hygiene.
- **Area:** one dominant `area:*` label. Choose the user-facing domain, not
  merely the file extension. For example, model-server docs are `area:model`,
  while generic contributor docs are `area:docs`.
- **State:** `needs-review` when a non-draft PR is ready for review;
  `ready-to-merge` only after approval and required checks; or `blocked`,
  `waiting-on-customer`, or `waiting-on-maintainers` when that state is
  supported by current evidence. Draft PRs normally have no state label.

Use the dominant changed behavior for area selection:

| Dominant scope | Area label |
|---|---|
| Agent harnesses and Responses API agent behavior | `area:agent` |
| Packaging, dependencies, containers, releases | `area:build` |
| CI, test routing, repository automation | `area:ci` |
| CLI commands and user-facing CLI behavior | `area:cli` |
| Hydra config, schemas, composition, compatibility | `area:config` |
| Shared APIs, servers, telemetry, registries | `area:core` |
| Datasets, preparation, task formats, materialization | `area:data` |
| Generic docs without a more specific domain | `area:docs` |
| Environment framework, lifecycle, scaffolding, validation | `area:env-infra` |
| Individual environments, benchmarks, verifiers | `area:environment` |
| Evaluation, rollouts, reward profiling, exporters | `area:evaluation` |
| Model servers, inference providers, adapters | `area:model` |
| Submission, deployment, services, lifecycle orchestration | `area:orchestration` |
| Sandbox providers and isolation/runtime contracts | `area:sandbox` |
| Training integrations and training-data interfaces | `area:training` |

Treat other labels as orthogonal and conservative:

- Add `benchmark-onboarding` only for a self-contained benchmark-onboarding PR.
- Add `community-request` only when external provenance is clear from GitHub
  association/permissions or a linked original contribution. Do not infer
  affiliation from a username or email address.
- Preserve release, cherry-pick, QA, runner, and partner labels unless the user
  asked for that workflow and the repository evidence supports the change.
- Leave `sla:*` labels to `scripts/pr_sla_tracker.py` and its workflow.
- Do not add `Run CICD` as a substitute for the `/ok to test` trust gate.

## Mutation Boundary

For requests to inspect checks, assess readiness, or recommend labels, remain
read-only and return the proposed changes. A request to label a PR authorizes
adding the supported labels to that PR, but not posting trust-gate comments,
submitting reviews, merging, or changing unrelated issues.

When applying labels, add only missing labels. Remove an existing label only
when it directly conflicts with the selected family and the user requested
normalization; otherwise report the conflict for maintainer judgment.

If `gh pr edit --add-label` fails while querying deprecated Projects Classic
fields, use the REST issue-label endpoint and then verify the PR:

```bash
gh api -X POST repos/NVIDIA-NeMo/Gym/issues/<PR>/labels \
  -f 'labels[]=LABEL'
gh pr view <PR> --repo NVIDIA-NeMo/Gym --json labels --jq '.labels[].name'
```

## Handoff

Report:

- the diff classification and why;
- checks passed, failed, pending, missing, or not run;
- rollout evidence, or why it is not applicable;
- current and proposed labels, including any ambiguity;
- the exact remaining maintainer or author action.
