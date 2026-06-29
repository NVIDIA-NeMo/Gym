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

For environment setup, contribution workflow, testing requirements, commit signing, and troubleshooting, refer to the [Development Setup Guide](https://docs.nvidia.com/nemo/gym/main/contribute/development-setup).
