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
from typing import Any


def _bytes(value: str | int) -> int:
    if isinstance(value, int):
        return value
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt]?)(?:i?b)?", value.lower())
    if match is None:
        raise ValueError(f"Invalid Compose memory size: {value!r}")
    return int(float(match[1]) * 1024 ** " kmgt".index(match[2] or " "))


class ComposeSubstitutionError(ValueError):
    """A Compose file names a variable or an ``env_file`` that prepare cannot resolve."""

    def __init__(self, name: str, message: str | None = None) -> None:
        self.name = name
        super().__init__(message or f"unresolved Compose variable ${{{name}}}")


def _compose_scalar(value: Any) -> str:
    """An environment value as Compose would emit it: booleans lower-case, ``None`` empty."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _environment_mapping(service_name: str, environment: Any, variables: dict[str, str]) -> dict[str, str | None]:
    """A service's ``environment`` as a mapping. A bare ``NAME`` in the list form, or a null value in the mapping
    form, stands for the task variable of that name and is ``None`` when there is none: Compose then leaves the
    variable unset in the container, even where an ``env_file`` sets it. ``NAME=`` and ``NAME: ''`` are empty."""
    if environment is None:
        return {}
    if isinstance(environment, list):
        pairs: dict[str, str | None] = {}
        for item in environment:
            key, separator, value = str(item).partition("=")
            pairs[key] = value if separator else variables.get(key)
        return pairs
    if isinstance(environment, dict):
        return {
            str(key): variables.get(str(key)) if value is None else _compose_scalar(value)
            for key, value in environment.items()
        }
    raise ComposeSubstitutionError("environment", f"service {service_name!r} environment must be a list or a mapping")
