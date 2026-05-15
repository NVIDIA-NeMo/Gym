# Contributing To NeMo-Gym

Welcome! We are excited to have you contribute to NeMo Gym. Whether you are adding new training environments, integrating RL frameworks, improving documentation, or fixing bugs, your contributions help advance RL training.

> By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before opening issues, PRs, or discussions.

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

## Finding a First Issue

New to the project? Start with one of these:

- **[`good first issue`](https://github.com/NVIDIA-NeMo/Gym/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)** — small, well-scoped tasks meant for newcomers.
- **[`help wanted`](https://github.com/NVIDIA-NeMo/Gym/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)** — issues where maintainers are actively looking for community help.
- **[Open issues](https://github.com/NVIDIA-NeMo/Gym/issues)** — full list. Filter by domain label (`coding`, `math`, `agent`, `safety`, etc.) if you have a focus area.

**Claiming an issue:** Comment on the issue saying you'd like to work on it before opening a PR. A maintainer will assign it to you (or let you know if it's already in flight). If your work stalls for more than two weeks, please leave a comment so the issue can be unassigned for someone else to pick up.

> **Placeholder:** the two-week stall window above is a proposed default and is pending confirmation by NeMo Gym product management.

If nothing on the tracker fits, the most accessible place to contribute is a new training environment under `resources_servers/` — each one is self-contained and doesn't require touching core infrastructure. See the [Environment Contribution Guide](https://docs.nvidia.com/nemo/gym/latest/contribute/environments).

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

**Important:** All commits must be signed with DCO sign-off (`-s`) and cryptographically signed (`-S`):

```bash
git commit -s -S -m "Your commit message"
```

## Contribution Workflow

The standard fork → branch → test → PR loop:

1. **Fork** the [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym) repository to your GitHub account.
2. **Clone** your fork locally and add the upstream remote:
   ```bash
   git clone git@github.com:<your-username>/Gym.git
   cd Gym
   git remote add upstream git@github.com:NVIDIA-NeMo/Gym.git
   ```
3. **Create a branch** off `main` with a descriptive name:
   ```bash
   git checkout -b my-feature-name
   ```
4. **Make your changes**, keeping commits small and focused.
5. **Run tests and linters** before pushing:
   ```bash
   ruff check --fix .
   ruff format .
   pre-commit run --all-files
   pytest tests/unit_tests/ -x
   # For changes to a specific resources/agent/model server:
   ng_test +entrypoint=resources_servers/<your_server>
   ```
6. **Push** to your fork and **open a Pull Request** against `NVIDIA-NeMo/Gym:main`.
7. **Respond to review feedback** by pushing additional commits to the same branch — please don't force-push during review unless asked.

To keep your branch up to date with `main`:

```bash
git fetch upstream
git rebase upstream/main
```

## Pull Request Expectations

To help reviewers turn your PR around quickly:

- **Tests are required.** New behavior needs new tests; bug fixes need a regression test that fails without the fix. Existing tests must pass. The project targets ≥95% coverage.
- **Update documentation.** If you change user-facing behavior, configuration, CLIs, or server contracts, update the relevant docs under `fern/` and any affected READMEs (root README, server README) in the same PR.
- **Keep PRs focused.** One logical change per PR. Split refactors from feature work where possible — small PRs get reviewed faster.
- **Write a clear description.** State what changed, why, and how you verified it. Link to the issue with `Fixes #123` or `Refs #123` so it auto-closes on merge.
- **Sign your commits.** Every commit needs DCO sign-off (`-s`) and a cryptographic signature (`-S`). PRs without sign-off cannot be merged.
- **Pass CI.** Lint, format, type checks, unit tests, and pre-commit hooks must be green. Fix failures locally before re-requesting review.
- **Be respectful and patient in review.** Reviewers may ask for changes — please assume good intent, ask for clarification when something is unclear, and remember our [Code of Conduct](CODE_OF_CONDUCT.md) applies to all interactions. Maintainers aim to provide an initial review within one week (placeholder target, pending PM confirmation); feel free to leave a polite ping if it's been longer.
- **Environment behavior changes** (new reward semantics, dataset changes) require explicit maintainer sign-off because they affect result comparability across releases.

**Not sure where to start?** Browse our [open issues](https://github.com/NVIDIA-NeMo/Gym/issues) or open a new one to discuss your idea before investing time in a large change.
