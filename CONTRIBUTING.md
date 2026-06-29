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

## Use of AI and LLM Tools

We encourage contributors to use AI coding assistants (Copilot, Cursor, Claude, ChatGPT, and so on)
where they genuinely help, but AI assistance does not replace human understanding, judgment, and
accountability.

**Guiding principle:** if the human effort required to create a pull request is less than the effort
required for maintainers to review it, that contribution should not be submitted. You are responsible
for every line of code you submit, regardless of whether you or an AI tool wrote it.

Refer to [Use of AI and LLM Tools](https://docs.nvidia.com/nemo/gym/latest/contribute/development-setup#use-of-ai-and-llm-tools)
for what we expect, how to attribute AI assistance, and the quality bar we enforce.

## Development Setup

For complete development setup, CI/CD requirements, commit signing, and troubleshooting, refer to the [Development Setup Guide](https://docs.nvidia.com/nemo/gym/latest/contribute/development-setup.html).

**Quick Start:**

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
