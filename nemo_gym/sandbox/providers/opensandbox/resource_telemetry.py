# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read the existing OpenSandbox inspection response without inferring allocations."""

import ast
import re

from nemo_gym.sandbox.resource_telemetry import PREFIX, Attribute


def allocation_attributes(text: str) -> dict[str, Attribute]:
    """Parse only the explicit Resources section of the text inspection API.

    Resource quantities remain strings in Kubernetes units. A changed/unsupported
    response format is a gap, never a fallback to submitted resources.
    """
    attributes: dict[str, Attribute] = {PREFIX + "allocation.source": "diagnostics/inspect"}
    for label, key in (("Node", "node"), ("Namespace", "namespace"), ("Pod Name", "pod")):
        match = re.search(rf"^{label}:\s*(.+)$", text, re.MULTILINE)
        if match:
            attributes[PREFIX + "allocation." + key] = match[1].strip()
    _, separator, section = text.partition("\nResources:\n")
    container = None
    found = False
    if separator:
        for line in section.splitlines():
            match = re.fullmatch(r"  ([\w.-]+):", line)
            if match:
                container = match[1]
                continue
            match = re.fullmatch(r"    (Requests|Limits):\s*(\{.*\})", line)
            if match and container:
                resources = ast.literal_eval(match[2])
                if not isinstance(resources, dict):
                    raise ValueError("Invalid resource mapping")
                for resource, value in resources.items():
                    if not isinstance(resource, str) or not isinstance(value, (str, int, float)):
                        raise ValueError("Invalid resource quantity")
                    attributes[PREFIX + f"allocation.{container}.{match[1].lower()}.{resource}"] = str(value)
                    found = True
            elif line and not line.startswith(" "):
                break
    if not found:
        raise ValueError("Inspection did not contain effective resources")
    return attributes
