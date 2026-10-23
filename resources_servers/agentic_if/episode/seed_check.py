# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The receipts a seed can check before the actor's request arrives.

An Environment Server seeds with the task data only: the actor's ``responses_create_params`` reach the Resources
Server at verify, where ``validate_runtime`` checks every receipt. At seed, the structure and the receipts that do not
depend on the actor's request are checked: the task, the installed interfaces and a continuation's sources.
"""

from collections.abc import Mapping

from resources_servers.agentic_if.task_variants.builder import digest
from resources_servers.agentic_if.task_variants.interfaces import installed_interfaces
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec, validate_runtime
from resources_servers.agentic_if.task_variants.tasks import backend_task_digest


def validate_seed_runtime(row: Mapping[str, object]) -> RuntimeSpec:
    """Validate a sealed row's configuration and its request-independent receipts; raise ``ValueError`` if one fails.

    A row that still carries its ``responses_create_params`` is checked in full.
    """
    if "responses_create_params" in row:
        return validate_runtime(row)
    spec = RuntimeSpec.model_validate(row.get("agentic_if"))
    if any(key in row for key in ("task_ref", "dataset_ref")):
        raise ValueError("package task references (task_ref, dataset_ref) are not a supported task source")
    if spec.receipt is None:
        raise ValueError("IF input receipt is missing")
    task = backend_task_digest(dict(row))
    if spec.receipt.task_sha256 != task:
        raise ValueError("IF input receipt does not match the task")
    if spec.receipt.interfaces_sha256 != digest(installed_interfaces(spec.tools.tool_interfaces)):
        raise ValueError("IF input receipt does not match the installed interfaces")
    if spec.continuation:
        if spec.continuation.source_task_sha256 != task:
            raise ValueError("continuation source task receipt does not match")
        if spec.continuation.source_tools_sha256 != digest(spec.tools.model_dump(mode="json")):
            raise ValueError("continuation source tools receipt does not match")
    return spec
