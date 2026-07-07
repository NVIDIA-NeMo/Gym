# Vendored: apple/ToolSandbox

This directory is a **vendored, modified** copy of Apple's ToolSandbox.

- **Upstream project:** ToolSandbox — https://github.com/apple/ToolSandbox
- **Paper:** *ToolSandbox: A Stateful, Conversational, Interactive Evaluation
  Benchmark for LLM Tool Use Capabilities* — https://arxiv.org/abs/2408.04682
- **Upstream copyright:** Copyright (C) 2024 Apple Inc. All Rights Reserved.
- **License:** Apple custom source license — see [`LICENSE`](./LICENSE).
- **Subcomponent notices:** see [`ACKNOWLEDGEMENTS`](./ACKNOWLEDGEMENTS)
  (referenced by `LICENSE`; lists the licenses of bundled third-party
  dependencies such as `ccy`, `anthropic-sdk-python`, etc.).

The Apple license grants a personal, non-exclusive license to use, reproduce,
modify, and redistribute the software in source/binary form, with or without
modifications. It prohibits use of Apple's name/marks for endorsement and is
provided **AS IS**. This attribution is also recorded in the repository-level
`gym/ATTRIBUTIONS.md` under "Vendored Components (modified)".

## License-header policy

- Every **upstream** file retains its original
  `# Copyright (C) 2024 Apple Inc. All Rights Reserved.` header and the
  `# For licensing see accompanying LICENSE file.` notice.
- Files **modified** by NVIDIA additionally carry a
  `# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.` line
  beneath the Apple header.
- Files **authored** by NVIDIA carry a standard NVIDIA SPDX header
  (Apache-2.0) and no Apple header.

## NVIDIA modifications

The scoring engine, scenarios, tools, execution environment, and message/tool
conversion are used **unchanged** from upstream. The conversation *driver* was
re-wired so the benchmark runs natively inside NeMo Gym (agent-under-test = the
gym policy model, driven by an external agent harness; the user simulator and
Python execution environment run inside the gym resources server; scoring is a
pure `/verify`). The following files were modified from upstream:

| File | Change |
|------|--------|
| `roles/openai_api_agent.py` | Agent role talks to any OpenAI-compatible endpoint; sampling / reasoning-toggle handling. |
| `roles/openai_api_user.py` | User-simulator role talks to any OpenAI-compatible endpoint. |
| `cli/__init__.py` | CLI reworked for per-scenario runs, `--list-scenarios`, and OpenAI-compatible agent/user endpoints (retained for offline scoring-parity checks). |
| `cli/utils.py` | Scenario resolution / result-summary helpers for the reworked CLI. |
| `common/execution_context.py` | Added `new_context` / context-manager helpers for safe ambient-context binding. |
| `common/evaluation.py` | Replaced the `scipy.optimize.linear_sum_assignment` import with the vendored dependency-free solver below (scipy is excluded from the base install and cannot be installed under the gym tree via uv). No scoring change. |

NVIDIA-authored additions in this tree:

| File | Purpose |
|------|---------|
| `cli/__main__.py` | `python -m tool_sandbox.cli` entry point. |
| `common/_linear_assignment.py` | Dependency-free exact linear-sum-assignment (Hungarian / Jonker-Volgenant), a drop-in for the one `scipy.optimize.linear_sum_assignment` call in `evaluation.py`. Validated bit-identical to scipy across 20k random matrices (including `inf` / infeasible cases). |

The gym integration proper (resources server, agent harness, schemas, configs)
lives **outside** this vendored tree, under
`resources_servers/toolsandbox/` and `responses_api_agents/toolsandbox_agent/`.
