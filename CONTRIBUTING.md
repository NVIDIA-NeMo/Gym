# Contributing To NeMo-Gym

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

## Development Setup

For complete development setup, CI/CD requirements, commit signing, and troubleshooting, refer to the [Development Setup Guide](https://docs.nvidia.com/nemo/gym/latest/contribute/development-setup.html).

**Quick start:**

```bash
git clone git@github.com:NVIDIA-NeMo/Gym.git
cd Gym
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra dev --group docs
pre-commit install
```

**Important:** All commits must be signed with DCO sign-off (`-s`):

```bash
git commit -s -m "Your commit message"
```

If DCO checks fail after you have already pushed, see the [Development Setup Guide](https://docs.nvidia.com/nemo/gym/main/contribute/development-setup#dco-and-commit-signing). Force-pushing is disallowed on branches in the upstream repo; for fork branches, use `--force-with-lease` only if your fork allows it, otherwise push the signed history to a new branch.

## Continuous Integration (CI)

CI runs in two clearly separated stages:

1. **Pre-merge checks** — run on every pull request. All must be green before a PR can merge.
2. **Post-merge full test suite** — runs after merge on `main`, with no change detection. It always exercises everything.

Reproduce the pre-merge checks locally before you push:

```bash
pre-commit run --all-files                          # linting + custom hooks
gym env test --resources-server your_server         # server tests + data validation
```

### Checks That Run on Every Pull Request

| Check | Workflow | What it enforces |
|-------|----------|------------------|
| **Copyright** | `copyright-check.yml` | Every new file has the Apache 2.0 SPDX header (see below). |
| **Code linting** | `code-linting.yml` | `pre-commit run --all-files` passes for all 8 hooks (see table below). |
| **Fern docs** | `fern-docs-ci.yml` | `npm run check` in `fern/` validates the docs config. Runs only when a PR touches `fern/**`. |
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

**Every new file must include the Apache 2.0 header:**

```python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

**Smart unit test selection** (`unit-tests.yml`) classifies your changed files and runs only the matching scope:

| Changed files | Test scope |
|---------------|------------|
| Only `**.md`, `docs/**`, `fern/**`, `LICENSE`, `benchmarks/**` | **Skip** — no tests run |
| Only `resources_servers/**`, `responses_api_agents/**`, `responses_api_models/**` | **Server-only** — tests run for the changed servers only |
| Anything else (core library, CI, scripts, and so on) | **Full suite** — all tests run |

Priority is `other > server > doc`: a PR touching both a server file and a core file triggers the full suite. When the full suite runs, the server tests are split across **8 parallel shards** with `fail_on_total_and_test_mismatch=true` — this means **every resources server must have at least one test**, or its shard fails.

### Checks a New Environment Must Satisfy

If you are contributing a new resources server (environment), it must pass all of the following before it can merge. Most are enforced inside `gym env test --resources-server your_server` (see `nemo_gym/cli/env.py`):

- **At least one test** — provide `tests/test_app.py`. A server with no tests fails the sharded suite (`fail_on_total_and_test_mismatch=true`).
- **Copyright header** — on every new file (enforced by `copyright-check.yml`).
- **`data/example.jsonl`** — must exist with **exactly 5 rows** (typically the first 5 rows of your train dataset).
- **`data/example_metrics.json`** — must exist and report `"Number of examples": 5`. Generate it with `gym dataset collate "+config_paths=[...]" +output_dirpath=... +mode=example_validation`.
- **`data/example_rollouts.jsonl`** — must exist with **exactly 5 rollouts** (run your example data through your agent to produce it).
- **No merge-conflict artifacts** — the `data/` directory must contain no `*conflict*` files.
- **Config `domain`** — every resources server YAML config must set `domain`.
- **Dataset `license`** — train and validation datasets must include a `license` field.
- **`verified: false`** — the `add-verified-flag` hook adds this to your config automatically. Leave it as-is; a maintainer flips it to `true` after baselining (see [Post-Merge](#post-merge-the-verified-flag)).

### Post-Merge: Full Test Suite

**Workflow:** `full-test-suite.yml` — runs on every push to `main` (no change detection; always runs everything):

- **Core tests** — the same pytest markers as PR CI (`-m "not sandbox"` then `-m sandbox`).
- **Server suite** — the same 8-shard run with `fail_on_total_and_test_mismatch=true`.
- **Wheel install test** — builds the wheel, installs it in a fresh venv against a mock inference endpoint, and runs `ng_help`, `ng_dump_config`, `ng_init_resources_server`, `ng_run`, and `ng_collect_rollouts` end-to-end.
- **Slack notification** — posts to the team channel if any job fails.

### Post-Merge: The `verified` Flag

New environments merge with `verified: false`. The environment is not surfaced as verified until a maintainer baselines the benchmark and flips `verified: true` in the YAML config. See the [Environment Contribution Guide](https://docs.nvidia.com/nemo/gym/latest/contribute/environments) for baselining requirements.
