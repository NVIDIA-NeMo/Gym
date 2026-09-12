# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep an explicit output allowance separate from the full context window."""


def token_limits(context: int, output: int | None = None) -> dict[str, int]:
    if output is None:
        # Preserve existing compositions that used one value for every limit.
        return {"context": context, "input": context, "output": context}
    if not 0 < output < context:
        raise ValueError("OpenCode output limit must be positive and smaller than its context window")
    return {"context": context, "input": context - output, "output": output}
