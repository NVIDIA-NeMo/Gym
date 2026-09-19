# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-local verifier for Hello World."""


async def verify(attempt, _verifier_input) -> float:
    """Score whether the task's requested file has the expected text."""
    try:
        actual = await attempt.workspace.read_text("/workspace/hello-gym.txt")
    except (OSError, UnicodeError):
        return 0.0
    return float(actual.removesuffix("\n") == "Hello from NeMo Gym!")
