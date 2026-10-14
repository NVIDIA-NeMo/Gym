# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a Compose service may say, and the shape each field takes before the adapter reads it.

The field allow-list of the sandbox Compose adapter and its message per offending key, the
normalisation of deploy resources, devices and gpus, restart, pull_policy, platform, extra_hosts,
hooks, networks and the provider service options, and check_compose_fields, which runs them all
over a document and is a fixed point of its own output. The two Compose scalar and environment
shape helpers, the memory size parser and the interpolation error class live here because the
field checks, the resolver and the interpolation code in nemo_gym.tasks.harbor.compose_variables
all read them, and this module sits below both. Nothing here reads a task or a record.
"""

import re


def _bytes(value: str | int) -> int:
    if isinstance(value, int):
        return value
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt]?)(?:i?b)?", value.lower())
    if match is None:
        raise ValueError(f"Invalid Compose memory size: {value!r}")
    return int(float(match[1]) * 1024 ** " kmgt".index(match[2] or " "))
