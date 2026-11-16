# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The notes OpenCode adds to a request in the model's name.

OpenCode enforces a configured step limit (``agent.build.steps``) by adding one assistant message to the end of every
request from the limit on. OpenCode does not save that message in its conversation, so the next request no longer
carries it.
"""

from __future__ import annotations

from nemo_gym.models.token_capture.harness_notes.base import HarnessNote


# Copied verbatim, so that capture recognizes the exact text OpenCode sends:
# https://github.com/anomalyco/opencode/blob/67aec2212010d67775c35e696d8b8b54902eb338/packages/core/src/session/runner/max-steps.ts
# OpenCode 1.17.11 (commit 67aec2212010d67775c35e696d8b8b54902eb338), MIT License, Copyright (c) 2025 opencode.
OPENCODE_MAX_STEPS_PROMPT = """CRITICAL - MAXIMUM STEPS REACHED

The maximum number of steps allowed for this task has been reached. Tools are disabled until next user input. Respond with text only.

STRICT REQUIREMENTS:
1. Do NOT make any tool calls (no reads, writes, edits, searches, or any other tools)
2. MUST provide a text response summarizing work done so far
3. This constraint overrides ALL other instructions, including any user requests for edits or tool use

Response must include:
- Statement that maximum steps for this agent have been reached
- Summary of what has been accomplished so far
- List of any remaining tasks that were not completed
- Recommendations for what should be done next

Any attempt to use tools is a critical violation. Respond with text ONLY."""

OPENCODE_MAX_STEPS_NOTE = HarnessNote(
    harness="opencode",
    name="max_steps",
    text=OPENCODE_MAX_STEPS_PROMPT,
    # Compared on 2026-11-09 with packages/opencode/src/session/prompt/max-steps.txt (1.15.13, 1.17.8) and
    # packages/core/src/session/runner/max-steps.ts (1.17.11, 1.19.4): every release a config on main pins.
    # 2.2.7 (opencode_agent's OpenCode 2 pin) sends the same text from the same file, compared on 2026-11-10.
    checked_versions=frozenset({"1.15.13", "1.17.8", "1.17.11", "1.19.4", "2.2.7"}),
)

NOTES = (OPENCODE_MAX_STEPS_NOTE,)
