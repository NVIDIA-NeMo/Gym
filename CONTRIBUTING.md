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

For complete development setup, commit signing, and troubleshooting, refer to the [Development Setup Guide](https://docs.nvidia.com/nemo/gym/latest/contribute/development-setup.html).

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

## CI Checks

All PRs must pass the following automated checks before merging.

### Code Linting (`.github/workflows/code-linting.yml`)

Runs `pre-commit run --all-files` across the full repository. Hooks enforced:

| Hook | What it checks |
|------|----------------|
| `end-of-file-fixer` | Python files end with a newline |
| `trailing-whitespace` | No trailing whitespace in Python files |
| `ruff` (lint + imports) | Python linting and import order, with auto-fix |
| `ruff-format` | Code formatting |
| `no-underscore-md` | No underscores in Markdown filenames |
| `add-verified-flag` | New resources server YAML configs get `verified: false` injected automatically |
| `update-readme-table` | Root `README.md` environment table kept in sync |

The `add-verified-flag` hook runs automatically — any new resources server YAML is committed with `verified: false`. A maintainer flips it to `true` after the benchmark has been baselined and reviewed.

### Copyright Check (`.github/workflows/copyright-check.yml`)

Every new file must include the Apache 2.0 copyright header (see the [Development Setup Guide](https://docs.nvidia.com/nemo/gym/latest/contribute/development-setup.html) for the exact text).

### DCO Sign-off

All commits must carry a `Signed-off-by` line via `git commit -s`.

### Fern Docs Check (`.github/workflows/fern-docs-ci.yml`)

Runs `npm run check` inside `fern/` to validate the documentation config. Triggers on PRs that touch `fern/**`.

### Smart Unit Test Selection (`.github/workflows/unit-tests.yml`)

The CI classifies changed files and runs the appropriate test scope:

| Changed files | Test scope |
|--------------|------------|
| Only `**.md`, `docs/**`, `fern/**`, `LICENSE`, `benchmarks/**` | **Skip** — no tests run |
| Only `resources_servers/**`, `responses_api_agents/**`, `responses_api_models/**` | **Server-only** — tests run for changed servers only |
| Anything else (core library, CI, scripts, etc.) | **Full suite** — all tests run |

Priority is `other > server > doc`: touching both a server file and a core file triggers the full suite.

**Core library tests** (full suite):
```bash
pytest tests/unit_tests/ -m "not sandbox"
pytest tests/unit_tests/ -m sandbox
```

**Server-only tests:**
- Extracts changed server names from the diff
- Runs `gym env test --resources-server <name>` for each with `delete_venvs_after_each_test=true`

### Server Suite — sharded (`.github/workflows/unit-tests.yml`)

When the full suite runs, server tests are split across **8 parallel shards**:

```
ng_test_all +fail_on_total_and_test_mismatch=true +delete_venvs_after_each_test=true +num_shards=8 +shard_index=<0-7>
```

`fail_on_total_and_test_mismatch=true` means **every resources server module must have at least one test** (`tests/test_app.py`). A server missing tests fails whichever shard picks it up. All 8 shards must pass for the PR to be green.

### Data Validation (enforced inside `gym env test`)

When `gym env test` runs — either via the server-only path or the sharded suite — it validates each resources server's example data:

| File | Requirement |
|------|-------------|
| `data/example.jsonl` | Must exist with **exactly 5 rows** |
| `data/example_metrics.json` | Must exist; `"Number of examples"` must equal `5` |
| `data/example_rollouts.jsonl` | Must exist with **exactly 5 rollouts** |
| `data/` directory | Must contain **no `*conflict*` files** (git merge artifacts) |

Missing files print an error with the exact command to generate them.

### Config Assertions

Checked when the server loads during `gym env test`:

- `domain` is required in every resources server YAML config
- Train and validation datasets must include a `license` field

---

### Post-merge (not a CI gate)

After a PR merges, the environment is not surfaced as verified until a maintainer baselines the benchmark (reward profiling + training validation) and flips `verified: true` in the YAML. See [Adding a Benchmark](https://docs.nvidia.com/nemo/gym/latest/contribute/environments/adding-a-benchmark) for baselining requirements.

---

### Full Test Suite on `main` (`.github/workflows/full-test-suite.yml`)

Runs on every push to `main` (no change detection — always runs everything):

- **Core tests** — same pytest markers as PR CI
- **Server suite** — same 8-shard run with `fail_on_total_and_test_mismatch=true`
- **Wheel install test** — builds the wheel, installs in a fresh venv, starts a mock inference endpoint, and runs `ng_help`, `ng_dump_config`, `ng_init_resources_server`, `ng_run`, and `ng_collect_rollouts` end-to-end
- **Slack notification** — posts to the team channel if any job fails
