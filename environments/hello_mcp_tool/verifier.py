# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verifier for the Hello MCP Tool task."""


async def verify(attempt, _verifier_input) -> float:
    """Require both the expected MCP call and its resulting workspace file."""
    expected = "Hello, NeMo Gym!"
    used_tool = any(
        call.server == "hello-tools"
        and call.name == "get_greeting"
        and call.arguments == {"name": "NeMo Gym"}
        and call.result == expected
        for call in attempt.tool_calls
    )
    if not used_tool:
        return 0.0

    try:
        actual = await attempt.workspace.read_text("/workspace/tool-greeting.txt")
    except (OSError, UnicodeError):
        return 0.0
    return float(actual.removesuffix("\n") == expected)
