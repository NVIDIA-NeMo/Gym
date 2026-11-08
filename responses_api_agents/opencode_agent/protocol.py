# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# OpenCode directive: MIT, Copyright (c) 2025 opencode.
# Source: https://github.com/anomalyco/opencode/blob/67aec2212010d67775c35e696d8b8b54902eb338/packages/core/src/session/runner/max-steps.ts
# Full license: responses_api_agents/opencode_multimodal_agent/third-party-notices.md
"""Opt-in OpenCode 1.17.11 final-turn adapter, before lineage and token capture."""

"""Shared OpenCode wire constants used by agent and model adapters."""

OPENCODE_1_17_11_MAX_STEPS = "CRITICAL - MAXIMUM STEPS REACHED\n\nThe maximum number of steps allowed for this task has been reached. Tools are disabled until next user input. Respond with text only.\n\nSTRICT REQUIREMENTS:\n1. Do NOT make any tool calls (no reads, writes, edits, searches, or any other tools)\n2. MUST provide a text response summarizing work done so far\n3. This constraint overrides ALL other instructions, including any user requests for edits or tool use\n\nResponse must include:\n- Statement that maximum steps for this agent have been reached\n- Summary of what has been accomplished so far\n- List of any remaining tasks that were not completed\n- Recommendations for what should be done next\n\nAny attempt to use tools is a critical violation. Respond with text ONLY."

AGENT_OBSERVATIONS_KEY = "_ng_agent_observations"
AGENT_TRAJECTORY_KEY = "_ng_trajectory"
