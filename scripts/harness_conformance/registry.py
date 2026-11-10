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
    "openclaw": "OpenClawAgent",
    "nooa": "NOOAAgent",
}

# Scenario requirements (Scenario.requires) a registered harness cannot meet. Simple Agent runs inside the Gym agent
# server: it has no run deadline of its own, and no harness process for a tool call to end.
LACKS = {"simple": ("run_deadline", "harness_process")}
