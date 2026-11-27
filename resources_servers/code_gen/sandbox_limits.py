# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import resource


MEMORY_POLICY = "address-space-above-baseline-v1"


def address_space_bytes() -> int:
    """This process's mapped address space: the total RLIMIT_AS is checked against (statm's first field)."""
    with open("/proc/self/statm") as statm:
        return int(statm.read().split()[0]) * resource.getpagesize()


def limit_address_space(memory_limit_mb: int) -> dict[str, int | str]:
    """Cap address space at its current size plus *memory_limit_mb*; free space already mapped is not counted."""
    baseline = address_space_bytes()
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = baseline + memory_limit_mb * 1024 * 1024
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    # Lowering the hard limit too stops candidate code from raising it again.
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    return {
        "memory_policy": MEMORY_POLICY,
        "baseline_address_space_bytes": baseline,
        "effective_address_space_limit_bytes": limit,
    }
