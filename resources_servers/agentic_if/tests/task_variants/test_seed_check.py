# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed checks the receipts that do not depend on the actor's request; verify checks them all."""

from copy import deepcopy

import pytest

from resources_servers.agentic_if.task_variants.builder import digest
from resources_servers.agentic_if.task_variants.runtime import (
    RuntimeSpec,
    seal_runtime_row,
    validate_runtime,
    validate_seed_runtime,
)
from resources_servers.agentic_if.tests.support import task_data


def sealed(**configuration):
    return seal_runtime_row(
        {"instance_id": "task-a", "responses_create_params": {"input": [{"role": "user", "content": "Fix it"}]}},
        RuntimeSpec(harness="opencode", task_backend="swe", **configuration),
    )


def test_a_sealed_row_passes_at_seed_and_its_full_check_at_verify():
    row = sealed(tools={"tool_names": {"bash": "bash", "read": "read"}, "tool_interfaces": {}})
    assert validate_seed_runtime(task_data(row)) == validate_runtime(row)


@pytest.mark.parametrize(
    "defect, message",
    [
        ("task", "does not match the task"),
        ("receipt", "receipt is missing"),
        ("interfaces", "installed interfaces"),
        ("package", "task_ref"),
        ("structure", "validation error"),
    ],
)
def test_each_request_independent_receipt_is_checked_at_seed(defect, message):
    data = task_data(sealed())
    if defect == "task":
        data["instance_id"] = "another-task"
    elif defect == "receipt":
        data["agentic_if"]["receipt"] = None
    elif defect == "interfaces":
        data["agentic_if"]["receipt"]["interfaces_sha256"] = "0" * 64
    elif defect == "package":
        data["task_ref"] = "registry/task"
    else:
        data["agentic_if"]["harness"] = "another-harness"
    with pytest.raises(ValueError, match=message):
        validate_seed_runtime(data)


def continuation_row(**changes):
    prefix = {
        "info": {"id": "ses_prefix"},
        "messages": [
            {"info": {"id": "u", "role": "user"}, "parts": [{"type": "text", "text": "Fix it"}]},
            {"info": {"id": "a", "role": "assistant"}, "parts": [{"type": "text", "text": "Working"}]},
        ],
    }
    tools = RuntimeSpec(harness="opencode", task_backend="swe").tools.model_dump(mode="json")
    continuation = {
        "mode": "restore",
        "prefix_export": prefix,
        "workspace_diff": "",
        "source_task_sha256": changes.get("task", "0" * 64),
        "source_tools_sha256": changes.get("tools", digest(tools)),
    }
    return continuation


def test_a_continuation_must_continue_this_task_with_these_tools():
    from resources_servers.agentic_if.task_variants.tasks import backend_task_digest

    task = backend_task_digest({"instance_id": "task-a"})
    data = task_data(sealed(continuation=continuation_row(task=task)))
    assert validate_seed_runtime(data).continuation is not None
    for change, message in ((dict(task="1" * 64), "source task"), (dict(task=task, tools="2" * 64), "source tools")):
        with pytest.raises(ValueError, match=message):
            sealed(continuation=continuation_row(**change))
        broken = deepcopy(data)
        broken["agentic_if"]["continuation"] = continuation_row(**change)
        with pytest.raises(ValueError):
            validate_seed_runtime(broken)
