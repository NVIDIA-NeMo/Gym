# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local probe adapters, importable by CI before Gym dependencies are installed."""

HARNESSES = {
    "opencode": "OpenCodeAgent",
    "opencode_sandboxed": "OpenCodeSandboxedAgent",
    "pi": "PiAgent",
    "codex": "CodexAgent",
    "hermes": "HermesAgent",
}
