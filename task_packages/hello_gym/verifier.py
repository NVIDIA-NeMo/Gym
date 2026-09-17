# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package-local verifier for the Hello Gym tasks."""


async def verify(attempt, verifier_input) -> float:
    """Score whether the requested workspace file has the expected text."""
    try:
        actual = await attempt.workspace.read_text(verifier_input.path)
    except (OSError, UnicodeError):
        return 0.0
    matches = actual.removesuffix("\n") == verifier_input.content.removesuffix("\n")
    return float(matches)
