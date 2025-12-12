# Contributing To NeMo-Gym

Welcome to NeMo Gym! Help advance RL training for the community by contributing to NeMo Gym.

## High Priority Contributions

**New Environments**
- Novel training environments (coding, reasoning, tool use, games, etc.)
- Benchmark integrations (SWE-Bench, Tau Bench, etc.)

See [Environment Contribution Guide](https://docs.nvidia.com/nemo/gym/latest/contribute/environments) for detailed guidance.

**RL Framework Integrations**
- Integration for new RL training frameworks (TRL, SkyRL, etc.)

See [RL Framework Integration Guide](https://docs.nvidia.com/nemo/gym/latest/contribute/rl-framework-integration) for detailed guidance.

**Always Welcome**
- Documentation & Tutorials
- Bug Fixes
- Features & Enhancements

### Before Contributing

- **Bug reports**: Include reproduction steps and environment details
- **Features and breaking changes**: Open an issue to discuss before implementing
- **Environment behavior changes**: Require careful consideration as they affect versioning and result comparability

**Not sure where to start?** Check our [open issues](https://github.com/NVIDIA-NeMo/Gym/issues) or create a new issue to discuss your idea!

## Development Setup

For complete development setup, CI/CD requirements, commit signing, and troubleshooting, see the [Development Setup Guide](https://docs.nvidia.com/nemo/gym/latest/contribute/development-setup.html)

**Quick start:**

```bash
git clone git@github.com:NVIDIA-NeMo/Gym.git
cd Gym
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra dev --group docs
pre-commit install
```

**Important:** All commits must be signed with DCO sign-off (`-s`) and cryptographic signature (`-S`):

```bash
git commit -s -S -m "Your commit message"
```
