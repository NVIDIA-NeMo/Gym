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

This repository is Apache-2.0 licensed, so all NVIDIA-authored code and all
NVIDIA modifications to the vendored Apple sources are licensed under
**Apache-2.0**. The original Apple notices are always preserved.

- Files **modified** by NVIDIA retain Apple's original notice verbatim
  (`# Copyright (C) 2024 Apple Inc. All Rights Reserved.` +
  `# For licensing see accompanying LICENSE file.`) and add an NVIDIA
  modifications block below it:

  ```text
  # SPDX-FileCopyrightText: Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  # SPDX-License-Identifier: Apache-2.0
  ```

  The NVIDIA changes to that file are licensed under Apache-2.0; the original
  Apple-authored portions remain subject to the Apple license in
  [`LICENSE`](./LICENSE). Every source file under this directory has been
  touched by NVIDIA (at minimum a license-header and import-style change), so
  they all carry this dual header.
- Files **authored** by NVIDIA (e.g. `common/_linear_assignment.py`) carry a
  standard NVIDIA SPDX header (`SPDX-License-Identifier: Apache-2.0`) with the
  full Apache-2.0 boilerplate and no Apple header.
- The Apple custom source license itself is **not** removed or relicensed: the
  upstream [`LICENSE`](./LICENSE) and [`ACKNOWLEDGEMENTS`](./ACKNOWLEDGEMENTS)
  are kept in-tree and reproduced in `gym/ATTRIBUTIONS.md`. Only NVIDIA's own
  contributions are placed under Apache-2.0.

## NVIDIA modifications

The conversation *driver* was re-wired so the benchmark runs natively inside
NeMo Gym (agent-under-test = the gym policy model, driven by an external agent
harness; the user simulator and Python execution environment run inside the gym
resources server; scoring is a pure `/verify`). Everything else — scenarios,
tools, scoring engine, and validators — is carried over from upstream with only
copyright-header and import-style changes.

### Removed from upstream

| Removed | Reason |
|---------|--------|
| `analysis/` (entire directory: `analysis.py`, `data_loading.py`, `__init__.py`) | Offline result-analysis tooling; not needed inside NeMo Gym where scoring is a `/verify` call. |
| `cli/` (entire directory: `__init__.py`, `utils.py`, `__main__.py`) | Upstream CLI and entry point; NeMo Gym drives scenarios via the resources server API instead. |
| `roles/anthropic_api_agent.py` | Anthropic agent role; NeMo Gym uses an external OpenAI-compatible agent harness. |
| `roles/anthropic_tool_utils.py` | Anthropic tool helpers (companion to the removed Anthropic agent). |
| `roles/cli_role.py` | Interactive CLI role; not used in automated NeMo Gym runs. |
| `roles/cohere_agent.py` | Cohere agent role; not used. |
| `roles/gemini_agent.py` | Gemini agent role; not used. |
| `roles/gorilla_api_agent.py` | Gorilla API agent role; not used. |
| `roles/hermes_api_agent.py` | Hermes API agent role; not used. |
| `roles/hermes_prompts.yaml` | Hermes prompt templates (companion to removed Hermes agent). |
| `roles/mistral_api_agent.py` | Mistral agent role; not used. |
| `roles/mistral_tool_utils.py` | Mistral tool helpers (companion to the removed Mistral agent). |
| `roles/openai_api_agent.py` | Upstream synchronous OpenAI agent with hard-coded model subclasses; replaced by `roles/openai_api.py` (see below). |
| `roles/unhelpful_agent.py` | Unhelpful-agent role for refusal testing; not needed for standard NeMo Gym runs. |

### Modified from upstream

| File | Change |
|------|--------|
| `roles/openai_api.py` | **Replacement for upstream `openai_api_agent.py`.** Upstream contained a synchronous `OpenAIAPIAgent` class with hard-coded `GPT-*` model subclasses. NVIDIA replaced it with shared infrastructure only: `OpenAIRoleConfig` dataclass, `_is_openai_reasoning_model()`, `_sampling_kwargs()`, and an `openai_retry` tenacity decorator. The actual async agent class lives in the gym harness outside this tree. |
| `roles/openai_api_user.py` | Constructor now accepts `OpenAIRoleConfig` (any OpenAI-compatible endpoint) instead of hard-coding `api.openai.com`. `respond()` and `_model_inference()` made async (`AsyncOpenAI` client). Hard-coded model subclasses (`GPT_3_5_0125_User`, etc.) removed. `openai_retry` decorator applied for transient-error handling. |
| `roles/base_role.py` | `teardown()` and `respond()` converted to `async def` to match the async execution model. |
| `roles/execution_environment.py` | `respond()` converted to `async def`. |
| `common/execution_context.py` | Thread/async isolation model replaced: upstream used a module-level `globals()` singleton with manual save/restore; NVIDIA uses `contextvars.ContextVar` so context is automatically copied per asyncio task. Added `new_context()` context-manager helper. Also fixed polars Enum construction to use `.value` (required by polars ≥ 1.0). |
| `common/evaluation.py` | Replaced `from scipy.optimize import linear_sum_assignment` with the vendored `_linear_assignment` module (scipy is excluded from the base install). Also widened `Float32` → `Float64` in `map_rows` calls; added defensive column-drop guards before dropping `sandbox_message_index`; coerced bare `0`/`1` returns to `0.0`/`1.0`. No scoring change. |
| `common/message_conversion.py` | Added `sanitize_tool_call_id()` to coerce vLLM-emitted tool-call IDs (e.g. `chatcmpl-tool-<hex>`) into valid Python identifiers, preventing `SyntaxError` when they are interpolated as variable names. Removed `openai_messages_to_langchain_messages()` and the `langchain_core` import (langchain removed as a dependency). |
| `common/scenario.py` | `play()` and `play_and_evaluate()` converted to `async def`; all `roles[...].respond()` calls are now awaited. Added `progress_desc` parameter. Added an infinite-loop guard: if `sandbox_message_index` does not advance after `respond()` (e.g. vLLM emits `tool_calls=[]` with no content), a warning is logged and the loop breaks. |
| `common/tool_conversion.py` | `from langchain_core.pydantic_v1 import BaseModel` → `from pydantic import BaseModel` (langchain removed as a dependency). |
| `scenarios/`, `tools/` | No functional changes. NVIDIA copyright header added; multi-line import style applied. |

### NVIDIA-authored additions

| File | Purpose |
|------|---------|
| `common/_linear_assignment.py` | Dependency-free exact linear-sum-assignment (Hungarian / Jonker-Volgenant), a drop-in for the one `scipy.optimize.linear_sum_assignment` call in `evaluation.py`. Validated bit-identical to scipy across 20k random matrices (including `inf` / infeasible cases). |

The gym integration proper (resources server, agent harness, schemas, configs)
lives **outside** this vendored tree, under
`resources_servers/toolsandbox/` and `responses_api_agents/toolsandbox_agent/`.
