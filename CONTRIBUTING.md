# Contributing to NeMo Gym

Welcome to NeMo Gym! Help advance RL training for the community by contributing to NeMo Gym.

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
where they genuinely help. However, AI assistance does not replace human understanding, judgment, and
accountability.

### Guiding Principle

**If the human effort required to create a pull request is less than the effort required for
maintainers to review it, that contribution should not be submitted.**

You are responsible for every line of code you submit, regardless of whether you or an AI tool wrote it.

### What We Expect

- **Understand your changes**: You must be able to explain and debug every line in your PR. Treat AI
  output as code from an untrusted source that requires your review.
- **Self-review and test**: Before requesting review, read through the diff carefully, run the test
  suite (`pytest`), and run pre-commit checks locally. Never treat AI-generated code as ready to merge
  without your own verification.
- **Keep PRs focused**: AI tools sometimes make "drive-by improvements" to unrelated code. Strip out
  any changes that are not directly relevant to the task at hand.
- **Verify correctness of AI-generated tests**: AI-generated tests can appear to pass while testing
  nothing meaningful. Ensure assertions are substantive and cover the intended behavior.

### AI Attribution

When AI tools generate a substantial portion of your contribution, add an `Assisted-by:` trailer to
your commit message:

```bash
git commit -s -S -m "Add reward function for code evaluation

Assisted-by: GitHub Copilot"
```

This is not required for routine autocomplete suggestions, only for cases where AI generated
significant code blocks, logic, or documentation.

### What We Will Close

We will close pull requests and issues that appear to be low-effort, AI-generated submissions. Common
indicators include:

- Boilerplate or generic code that ignores project conventions
- PRs that do not pass CI or pre-commit checks
- Descriptions or comments that are clearly unreviewed LLM output
- Bulk "improvements" with no corresponding issue or discussion

This is not about policing tool usage. It is about maintaining the quality bar that the community and
maintainers depend on.

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

**Important:** All commits must be signed with DCO sign-off (`-s`) and cryptographic signature (`-S`):

```bash
git commit -s -S -m "Your commit message"
```
