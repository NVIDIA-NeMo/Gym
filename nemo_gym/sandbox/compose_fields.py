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
from collections.abc import Mapping
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


# The keys the sandbox Compose adapter (``nemo_gym.sandbox.adapters.docker_compose``) starts. Every other key
# of a document or a service is refused by name: at prepare, per task, and again by the adapter before it
# provisions anything.
SUPPORTED_COMPOSE_FIELDS = frozenset({"name", "services", "networks", "volumes"})
SUPPORTED_SERVICE_FIELDS = frozenset(
    {
        "image",
        "entrypoint",
        "command",
        "environment",
        "working_dir",
        "user",
        "depends_on",
        "healthcheck",
        "expose",
        "ports",
        "networks",
        "volumes",
        "cpus",
        "mem_limit",
        "labels",
        "restart",
        "network_mode",
        "cap_add",
        "shm_size",
        "x-sandbox",
    }
)


def unsupported_compose_fields(document: Any) -> list[str]:
    """The refusals the sandbox Compose adapter raises for the keys of ``document``: one message per offender in
    document order (the top level first, then each service), empty when every key is one the adapter starts.

    Pure, so prepare runs it on a task's resolved document and reports a refused field (``deploy``,
    ``extra_hosts``, ``platform``, a sidecar ``build:``) per task before any sandbox starts, and the adapter
    raises the first message at start. Only mappings are inspected: a document or service that is not one
    fails the adapter's shape checks instead.
    """
    messages: list[str] = []
    if not isinstance(document, Mapping):
        return messages
    unknown = sorted(str(key) for key in set(document) - SUPPORTED_COMPOSE_FIELDS)
    if unknown:
        messages.append(f"Unsupported Compose fields: {unknown}")
    services = document.get("services")
    if not isinstance(services, Mapping):
        return messages
    for name, service in services.items():
        if not isinstance(service, Mapping):
            continue
        unknown = sorted(str(key) for key in set(service) - SUPPORTED_SERVICE_FIELDS)
        if unknown:
            messages.append(f"Service {name!r}: unsupported Compose fields {unknown}")
    return messages
