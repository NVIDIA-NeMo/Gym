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
    "miniswe_sandboxed": "MiniSWESandboxedAgent",
    "claude_code": "ClaudeCodeAgent",
    "terminus_2_sandboxed": "Terminus2Agent",
}

CONFIG_CLASSES = {"miniswe_sandboxed": "MiniSWESandboxedConfig"}
BORROWED_SANDBOX_HARNESSES = frozenset({"miniswe_sandboxed", "terminus_2_sandboxed"})
