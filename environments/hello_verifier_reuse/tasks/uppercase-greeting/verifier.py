# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-owned verifier composed from a shared file helper."""

from nemo_gym.verifiers.files import read_text


async def verify(attempt, _verifier_input) -> float:
    """Require the requested greeting in uppercase."""
    actual = await read_text(attempt, "/workspace/shout.txt")
    if actual is None:
        return 0.0
    content = actual.removesuffix("\n")
    return float(content == "Hello from NeMo Gym!".upper() and content.isupper())
