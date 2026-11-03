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
}


# Timing migration targets, independent of which local runtime probes are wired
# into HARNESSES above. A target's presence never establishes conformance.
P0_TOOL_TIMING_HARNESSES = {
    "opencode": "opencode_agent",
    "hermes": "hermes_agent",
    "openclaw": "openclaw_agent",
    "nooa": "nooa_agent",
    "claude_code": "claude_code_agent",
    "codex": "codex_agent",
    "terminus": "terminus_2_sandboxed_agent",
    "stirrup": "stirrup_agent",
    "miniswe": "miniswe_sandboxed_agent",
}


def tool_timing_inventory() -> dict[str, dict[str, object]]:
    """Report migration targets without claiming that probe registration proves timing."""
    return {
        name: {"component": component, "runtime_probe_available": name in HARNESSES, "timing_status": "unverified"}
        for name, component in P0_TOOL_TIMING_HARNESSES.items()
    }
