# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local probe adapters, importable by CI before Gym dependencies are installed."""

HARNESSES = {
    "simple": "SimpleAgent",
    "opencode": "OpenCodeAgent",
    "pi": "PiAgent",
    "codex": "CodexAgent",
    "hermes": "HermesAgent",
    "kilocode": "KiloCodeAgent",
    "stirrup": "StirrupAgentWrapper",
    "copilot": "CopilotAgent",
    "crewai": "CrewAIAgent",
    "openclaw": "OpenClawAgent",
    "nooa": "NOOAAgent",
    "miniswe_sandboxed": "MiniSWESandboxedAgent",
    "claude_code": "ClaudeCodeAgent",
    "terminus_2_sandboxed": "Terminus2Agent",
}
