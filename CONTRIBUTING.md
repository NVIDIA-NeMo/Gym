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
