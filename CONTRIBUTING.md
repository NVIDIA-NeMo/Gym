# Contributing to NeMo Gym

Welcome! We are excited to have you contribute to NeMo Gym. Whether you are adding new training environments, integrating RL frameworks, improving documentation, or fixing bugs, your contributions help advance RL training.

## High Priority Contributions

**New Environments**
- Novel training environments (coding, reasoning, tool use, games, and so on)
- Benchmark integrations (SWE-Bench, Tau Bench, and so on)

Refer to the [Environment Contribution Guide](https://docs.nvidia.com/nemo/gym/latest/contribute/environments) for detailed guidance.

**RL Framework Integrations**
- Integration for new RL training frameworks (TRL, SkyRL, and so on)

Refer to the [RL Framework Integration Guide](https://docs.nvidia.com/nemo/gym/latest/contribute/rl-framework-integration) for detailed guidance.

**Always Welcome**
- Documentation and Tutorials
- Bug Fixes
- Features and Enhancements

### Before Contributing

- **Bug reports**: Include reproduction steps and environment details
- **Features and breaking changes**: Open an issue to discuss before implementing
- **Environment behavior changes**: Require careful consideration as they affect versioning and result comparability

**Not sure where to start?** Refer to our [open issues](https://github.com/NVIDIA-NeMo/Gym/issues) or create a new issue to discuss your idea.

## Pull Requests

- For normal authored PRs, use a Conventional Commit-style title:
  `type(optional-scope): imperative summary`. Common types are `feat`, `fix`,
  `docs`, `test`, `refactor`, `perf`, `build`, `ci`, and `chore`. Use `design`
  for a design-only proposal, which normally carries the `docs` type label.
  Generated release and cherry-pick PRs may keep their generated title format.
- Explain what changed and why, link the relevant issue or explain why one is
  unnecessary, list the exact validation performed, and include rollout
  evidence or a justified `N/A`. State any compatibility, migration,
  configuration, or benchmark-result impact.
- Keep incomplete work as a draft. Mark it ready only after reviewing the final
  diff and recording the applicable local checks.
- Select one applicable type label and one dominant `area:*` label from the
  [repository's live labels](https://github.com/NVIDIA-NeMo/Gym/labels). Drafts normally have no state label; add
  `needs-review` when the PR is ready. If you cannot apply labels, list the
  proposed labels in the PR body for a maintainer.

## Use of AI and LLM Tools

We encourage contributors to use AI coding assistants (Cursor, Claude, Codex, OpenCode, and similar)
where they genuinely help, but AI assistance does not replace human understanding, judgment, and
accountability.

**Guiding principle:** Prefer contributions where your review and ownership clearly outweigh the
maintainer review burden. You are responsible for every line of code you submit, regardless of
whether you or an AI tool wrote it.

Refer to [`AGENTS.md`](./AGENTS.md) for the quality bar (shared by humans and coding agents), and to
[Use of AI and LLM Tools](https://docs.nvidia.com/nemo/gym/latest/contribute/development-setup#use-of-ai-and-llm-tools)
for how maintainers handle low-effort submissions.

## Licensing of Contributions

NeMo Gym is licensed under the **Apache License, Version 2.0** (see [`LICENSE`](./LICENSE)).
We accept contributions **only** under the terms of the Apache-2.0 license. By
submitting a contribution, you agree that:

- Your contribution is your own original work (or you have the right to submit it),
  and it is licensed to the project and its users under Apache-2.0.
- Every new source file you author carries the standard NVIDIA SPDX header:

  ```text
  # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  ```

- Do **not** introduce code under a license incompatible with Apache-2.0
  (e.g. GPL/LGPL/AGPL or a proprietary/custom source license) into the main tree.
- If you must vendor third-party code, it has to be under an Apache-2.0-compatible
  license, its original notices must be preserved, any file you modify must retain
  the upstream notice and add an NVIDIA `SPDX-License-Identifier: Apache-2.0`
  modifications block, and the component must be recorded in
  [`ATTRIBUTIONS.md`](./ATTRIBUTIONS.md). See
  `resources_servers/toolsandbox/tool_sandbox/VENDORING.md` for a worked example.

## Development Setup

For complete development setup, CI/CD requirements, DCO sign-off, and troubleshooting, refer to the [Development Setup Guide](https://docs.nvidia.com/nemo/gym/latest/contribute/development-setup.html).

External contributors should fork Gym, clone their fork as `origin`, and add
`NVIDIA-NeMo/Gym` as `upstream`. Contributors with repository write access may
clone it directly as `origin`. In either case, create a focused branch from the
current authoritative default branch (`upstream/main` for a fork or
`origin/main` for a direct clone) rather than developing on `main`.

**Quick Start:**

```bash
# Install uv 0.11.21 or newer (required for the repository's Python version).
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# External contributors: clone your fork, then add the upstream repository.
git clone git@github.com:YOUR-USERNAME/Gym.git
cd Gym
git remote add upstream git@github.com:NVIDIA-NeMo/Gym.git

# With write access, use git@github.com:NVIDIA-NeMo/Gym.git as the clone URL
# and omit the `git remote add upstream` step.
uv venv --python 3.13.14 && source .venv/bin/activate
uv sync --extra dev
pre-commit install
```

**Important:** All commits must be signed off under the DCO (`-s`).
Cryptographic signing (`-S`) is optional:

```bash
git commit -s -m "Your commit message"
```

If DCO checks fail after you have already pushed, see the [Development Setup Guide](https://docs.nvidia.com/nemo/gym/main/contribute/development-setup#dco-and-commit-signing). Force-pushing is disallowed on branches in the upstream repo; for fork branches, use `--force-with-lease` only if your fork allows it, otherwise push the signed history to a new branch.

## Code, Documentation, and Tests

- Follow the code-style rules in [`AGENTS.md`](./AGENTS.md). Ruff, not Black,
  owns Python linting and formatting; `pyproject.toml` and
  `.pre-commit-config.yaml` are the enforced configuration. Annotate new or
  changed public boundaries and document generated public APIs; mypy is useful
  for focused type-checkable areas but is not a repository-wide strict CI gate.
- Every change must assess whether tests and documentation need updating. State
  what was added or changed, or record a justified `N/A` in the PR body.
  Significant user-facing features should document their motivation,
  configuration or API, and an executable usage example.
- Put deterministic shared-library behavior in `tests/unit_tests/` and
  component behavior in that component's isolated test suite. Behavior-changing
  environment or agent work also requires a representative model rollout and
  inspection of agent and verifier behavior.
- When renaming a public symbol, configuration key, CLI flag, file, or path,
  search and update affected docs, docstrings, examples, scripts, configs, and
  tests. Add compatibility aliases, migration notes, or documentation redirects
  when users may still depend on the previous name.

## Dependencies

- Declare each new import in the narrowest appropriate dependency file. Keep
  environment-, agent-, and model-specific packages in that component's
  `pyproject.toml` or `requirements.txt`; add a root dependency only when shared
  Gym runtime code requires it.
- Root dependencies in `pyproject.toml` must record why they are needed, when
  they were updated, and their license. Do not introduce a license incompatible
  with Apache-2.0.
- When root dependency resolution changes, run `uv lock` and commit both
  `pyproject.toml` and `uv.lock`. Run the focused install and tests for the code
  path that imports the dependency.
- Guard or defer optional imports so the base package remains importable, and
  raise an actionable error when the integration is selected without its
  extra. Skip tests only when they genuinely require an unavailable optional
  dependency or external tool.

## Continuous Integration (CI)

Gym has three CI entry points:

1. **PR-triggered checks** — lint, copyright, secrets, DCO, and the Fern preview build/comment workflows run from pull-request activity outside the mirrored main CICD pipeline.
2. **Main Gym CICD** — `cicd-main.yml` runs on mirrored PR branches and pushes to `main`, and also on a four-hour schedule or manual dispatch. Its unit and component scope follows the change classifier; scheduled and manual runs add configured container and GPU E2E coverage.
3. **Unconditional full suite** — `full-test-suite.yml` runs every four hours or by manual dispatch, without change detection.

**Triggering trusted CI (`/ok to test`):** The mirrored Gym CICD and Fern
workflows run automatically for verified commits from NVIDIA-NeMo members. If
your commits are unverified, or you are an external contributor, these trusted
workflows do not start until a qualified NVIDIA-NeMo vetter comments
`/ok to test <full commit SHA>` on the PR. Approval applies to the current
head; after an unverified head changes, a qualified vetter must approve the new
full SHA.
GitHub-hosted checks such as lint, copyright, secrets, DCO, and the Fern preview
may still run directly. The trust gate prevents unreviewed contributor code
from executing on NVIDIA runners.

Run the checks applicable to your diff before you push:

```bash
pre-commit run --all-files                              # every change
gym dev test                                           # shared core behavior
gym env test +entrypoint=resources_servers/your_server # one component
gym env test +entrypoint=resources_servers/your_server \
  +should_validate_data=true                           # legacy resource-server data contract
gym env test --all                                     # all server components; expensive
python3 tests/unit_tests/test_fern_docs_links.py        # Fern internal links
make docs-check                                        # Fern configuration
```

The Fern commands require Node.js 22 or newer; see `fern/README.md` for setup
and preview instructions.

Use targeted tests for focused changes. The full local commands are appropriate
when the diff or risk warrants their cost; they are not required for a docs-only
change.

### Checks That Run on Every Pull Request

| Check | Workflow | What it enforces |
|-------|----------|------------------|
| **Copyright** | `copyright-check.yml` | New source files have the Apache 2.0 SPDX header (see below). |
| **Code linting** | `code-linting.yml` | `pre-commit run --all-files` passes for all configured hooks (see table below). |
| **Fern docs** | `fern-docs-ci.yml` | Checks internal links and runs `npm run check`; triggered by Fern content, its link test, or the workflow itself. |
| **Unit tests** | `unit-tests.yml` | Smart change detection selects the test scope, then runs it (see below). |

**Pre-commit hooks** (from `.pre-commit-config.yaml`):

| Hook | What it checks |
|------|----------------|
| `end-of-file-fixer` | Python files end with a newline |
| `trailing-whitespace` | No trailing whitespace in Python files |
| `ruff` (lint) | Python linting, with auto-fix |
| `ruff` (imports) | Import ordering (`--select I`), with auto-fix |
| `ruff-format` | Code formatting |
| `no-underscore-md` | No underscores in Markdown filenames (use hyphens) |
| `add-verified-flag` | New resources server YAML configs get `verified: false` injected automatically |
| `update-readme-table` | Root `README.md` environment table kept in sync |

Hooks that auto-modify files (`ruff`, `ruff-format`, `add-verified-flag`, `update-readme-table`) may fail the first run while they rewrite files — stage the changes and commit again.

**Every new source file must include the Apache 2.0 header using the comment syntax for that file type:**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

The implementation in `.github/actions/classify-changes/action.yml` is
authoritative. In summary, `unit-tests.yml` selects this scope:

| Changed files | Test scope |
|---------------|------------|
| Only `**.md`, `fern/**`, `LICENSE`, `benchmarks/**` | **CI docs-classified** — unit tests are skipped |
| Only `resources_servers/**`, `responses_api_agents/**`, `responses_api_models/**` | **Server-only** — core and sandbox coverage plus changed-component tests |
| Anything else (core library, CI, scripts, and so on) | **Full suite** — core and sandbox coverage plus all 8 server shards |

Priority is `other > server > doc`: a PR touching both a server file and a core
file triggers the full suite. The full suite splits resources servers, agent
servers, and model servers across **8 parallel shards**. Every discovered
component needs an isolated test suite; missing or failed tests fail the shard.
With `fail_on_total_and_test_mismatch=true`, CI also rejects modules that cannot
enter the test set, such as a component missing its `README.md`.

The CI docs-classified bucket includes executable files under `benchmarks/**`.
When a benchmark preparation script or test changes, run its targeted
preparation and tests locally even though the path classifier skips unit tests.
For a server-only change, use
`gym env test +entrypoint=<resources_servers|responses_api_agents|responses_api_models>/<name>`;
`--resources-server <name>` is shorthand for resources servers only.

### Checks a New Environment or Benchmark Must Satisfy

New complete environments and benchmarks use the manifest-backed onboarding
workflow documented in the
[Environment Contribution Guide](https://docs.nvidia.com/nemo/gym/latest/contribute/environments):

```bash
gym env validate --sync your_environment
gym env test your_environment
gym env publish your_environment
```

Before review, the manifest and its Gym configuration must agree, licensing
metadata must be complete, the resources server must export a passing verifier
fixture, and the workload must include at least one representative input. Each
new component also needs tests for its behavior.

For behavior-changing environment or agent code, run a representative real
smoke rollout with a model and inspect the agent and verifier behavior. Record
the evidence in the PR. Saving rollout artifacts in the repository is optional,
and a full evaluation, reward profile, or training run is not a merge gate for
training environments. Fixed evaluation benchmarks must follow the reward-
profiling requirements in the linked Environment Contribution Guide.

Standalone resources-server configs without a workload manifest remain on the
legacy validation path. Run:

```bash
gym env test +entrypoint=resources_servers/your_server \
  +should_validate_data=true
```

That legacy data validator requires exactly five rows in `data/example.jsonl`,
`data/example_metrics.json` with `"Number of examples": 5`, and exactly five
rows in `data/example_rollouts.jsonl`. It also rejects `*conflict*` artifacts
and requires dataset license fields and a config `domain`. New source files
need the SPDX header, and the server needs at least one test. New legacy configs
start with `verified: false`; maintainers change that only after baselining.

The manifest scaffold and the legacy five-row server validator currently use
different layouts. Do not create duplicate legacy artifacts solely to make a
manifest workload look like an unmigrated server; report a scaffold/CI mismatch
on the PR so the underlying validator contract can be corrected.

### Periodic and On-Demand Coverage

`cicd-main.yml` runs periodically and on demand in addition to its mirrored-PR
and `main` triggers. Its scheduled/manual path adds the configured container
build and GPU E2E coverage.

**Workflow:** `full-test-suite.yml` — runs every four hours and by manual
dispatch, with no change detection:

- **Core tests** — the same pytest markers as PR CI (`-m "not sandbox"` then `-m sandbox`).
- **Server suite** — the same 8-shard run with `fail_on_total_and_test_mismatch=true`.
- **Wheel install test** — builds the wheel, installs it in a fresh venv against a mock inference endpoint, and runs `ng_help`, `ng_dump_config`, `ng_init_resources_server`, `ng_run`, and `ng_collect_rollouts` end-to-end.
- **Slack notification** — reports failures for eligible `main` or release refs;
  a manually dispatched run may disable notification.
