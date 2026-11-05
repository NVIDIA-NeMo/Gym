# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate OpenSandbox resource quantities and inspection into shared Gym types."""

import ast
import math
import re
from collections.abc import Mapping
from decimal import Decimal
from urllib.parse import quote

from nemo_gym.sandbox.providers.base import SandboxResourceAmounts
from nemo_gym.sandbox.resource_telemetry import ComponentAllocation, ResourceAllocation, ResourceDesignator


_DECIMAL_EXPONENTS = {"n": -9, "u": -6, "m": -3, "": 0, "k": 3, "K": 3, "M": 6, "G": 9, "T": 12, "P": 15, "E": 18}
_BINARY_EXPONENTS = {"Ki": 10, "Mi": 20, "Gi": 30, "Ti": 40, "Pi": 50, "Ei": 60}


def _quantity(value: object) -> Decimal:
    """Decode Kubernetes decimal, binary and exponent quantities to base units."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("Invalid resource quantity")
    match = re.fullmatch(r"([+]?(?:\d+(?:\.\d*)?|\.\d+))([eE][+-]?\d+|[numkKMGTPE]|[KMGTPE]i)?", str(value))
    if not match:
        raise ValueError(f"Invalid resource quantity: {value!r}")
    amount, suffix = Decimal(match[1]), match[2] or ""
    if suffix in _BINARY_EXPONENTS:
        amount *= Decimal(2) ** _BINARY_EXPONENTS[suffix]
    elif suffix in _DECIMAL_EXPONENTS:
        amount *= Decimal(10) ** _DECIMAL_EXPONENTS[suffix]
    else:
        exponent = int(suffix[1:])
        if abs(exponent) > 308:
            raise ValueError("Resource quantity exponent out of range")
        amount *= Decimal(10) ** exponent
    if not math.isfinite(float(amount)):
        raise ValueError("Resource quantity is not finite")
    return amount


def resource_amounts(resources: Mapping[str, object]) -> SandboxResourceAmounts:
    """Normalize known OSB/Kubernetes resources; vendor resources are not guessed."""
    values = {}
    for resource, name, divisor in (
        ("cpu", "cpu", 1),
        ("memory", "memory_mib", 2**20),
        ("ephemeral-storage", "disk_gib", 2**30),
    ):
        if resource in resources:
            values[name] = float(_quantity(resources[resource]) / divisor)
    gpu_values = [_quantity(resources[key]) for key in ("gpu", "nvidia.com/gpu") if key in resources]
    if gpu_values:
        if any(value != gpu_values[0] or value != value.to_integral_value() for value in gpu_values):
            raise ValueError("GPU quantities must be integral and unambiguous")
        values["gpu"] = int(gpu_values[0])
    if "gpu_type" in resources:
        if not isinstance(resources["gpu_type"], str):
            raise ValueError("GPU type must be a string")
        values["gpu_type"] = resources["gpu_type"]
    return SandboxResourceAmounts(**values)


def parse_allocation(text: str, *, scope: str | None = None) -> ResourceAllocation:
    """Parse explicit inspection resources; never infer allocation from submission.

    Scope can be the OSB control-plane deployment when a cluster ID is unavailable.
    Only emit a pod designator when both namespace and pod are present.
    """
    placement = {}
    for label, key in (("Node", "node"), ("Namespace", "namespace"), ("Pod Name", "pod")):
        match = re.search(rf"^{label}:[ \t]*(.+)$", text, re.MULTILINE)
        if match:
            placement[key] = match[1].strip()
    _, separator, section = text.partition("\nResources:\n")
    containers: dict[str, dict[str, SandboxResourceAmounts]] = {}
    container = None
    found = False
    if separator:
        for line in section.splitlines():
            match = re.fullmatch(r"  ([\w.-]+):", line)
            if match:
                container = match[1]
                containers.setdefault(container, {})
                continue
            match = re.fullmatch(r"    (Requests|Limits):\s*(\{.*\})", line)
            if match and container:
                resources = ast.literal_eval(match[2])
                if not isinstance(resources, dict):
                    raise ValueError("Invalid resource mapping")
                amounts = resource_amounts(resources)
                containers[container][match[1].lower()] = amounts
                found |= any(value is not None for value in vars(amounts).values())
            elif line and not line.startswith(" "):
                break
    if not found:
        raise ValueError("Inspection did not contain supported effective resources")
    designator = None
    if placement.get("namespace") and placement.get("pod"):
        namespace, pod = (quote(placement[key], safe="") for key in ("namespace", "pod"))
        designator = ResourceDesignator(kind="kubernetes", scope=scope, id=f"namespaces/{namespace}/pods/{pod}")
    host = None
    if placement.get("node"):
        host = ResourceDesignator(kind="kubernetes-node", scope=scope, id=placement["node"])
    return ResourceAllocation(
        source="diagnostics/inspect",
        components=tuple(ComponentAllocation(name=name, **values) for name, values in sorted(containers.items())),
        designator=designator,
        host=host,
    )
