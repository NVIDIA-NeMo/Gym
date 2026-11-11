# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local probe adapters, importable by CI before Gym dependencies are installed."""

HARNESSES = {
    "simple": {"agent": "SimpleAgent", "probe": ".provider:Probe"},
    "opencode": {"agent": "OpenCodeAgent", "probe": ".provider:Probe"},
    "pi": {"agent": "PiAgent", "probe": ".provider:Probe"},
    "codex": {"agent": "CodexAgent", "probe": ".provider:Probe"},
    "hermes": {"agent": "HermesAgent", "probe": ".provider:Probe"},
    "kilocode": {"agent": "KiloCodeAgent", "probe": ".provider:Probe"},
    "stirrup": {"agent": "StirrupAgentWrapper", "probe": ".provider:Probe"},
    "copilot": {"agent": "CopilotAgent", "probe": ".provider:Probe"},
    "crewai": {"agent": "CrewAIAgent", "probe": ".provider:Probe"},
    "openclaw": {"agent": "OpenClawAgent", "probe": ".provider:OpenClawProbe"},
    "nooa": {"agent": "NOOAAgent", "probe": ".nooa:NOOAProbe"},
    "miniswe_sandboxed": {"agent": "MiniSWESandboxedAgent", "probe": ".provider:Probe"},
    "claude_code": {"agent": "ClaudeCodeAgent", "probe": ".provider:Probe"},
    "terminus_2_sandboxed": {"agent": "Terminus2Agent", "probe": ".provider:Probe"},
}
