# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from resources_servers.terminal_bench_4.resource_limits import sandbox_resource_limit


@pytest.mark.parametrize(
    "detail, resource",
    [
        (
            "SandboxApiException: [SANDBOX::ENDED] Pod sandbox entered Failed phase (Evicted): "
            "Pod ephemeral local storage usage exceeds the total limit of containers 15Gi.",
            "storage",
        ),
        (
            '[SANDBOX::ENDED] Container sandbox exceeded its local ephemeral storage limit "1Gi".',
            "storage",
        ),
        (
            "Sandbox was OOM-killed. OpenSandbox status: state='Failed', reason='FAILED', "
            "message='container sandbox terminated with OOMKilled (exit code 137)'",
            "memory",
        ),
        ("[SANDBOX::ENDED] container sandbox terminated with OOMKilled (exit code 137)", "memory"),
        (
            'Get command status failed: HTTP 410: {"code":"SANDBOX::ENDED",'
            '"message":"Pod sandbox entered Failed phase (Evicted): Pod ephemeral local storage '
            'usage exceeds the total limit of containers 1Gi."}',
            "storage",
        ),
        (
            "Sandbox is dead. OpenSandbox status: state='Failed', reason='Evicted', "
            "message='Pod ephemeral local storage usage exceeds the total limit of containers 4Gi.'",
            "storage",
        ),
    ],
)
def test_explicit_provider_limit_deaths(detail, resource):
    limit = sandbox_resource_limit(detail)
    assert limit.resource == resource
    assert limit.detail == detail


@pytest.mark.parametrize(
    "detail",
    [
        None,
        "",
        "Command exited with code 137",
        "MemoryError: out of memory",
        "OOMKilled (exit code 137)",
        "No space left on device",
        "Pod ephemeral local storage usage exceeds the total limit of containers 4Gi.",
        "[SANDBOX::ENDED] Pod sandbox entered Failed phase (Evicted): "
        "The node was low on resource: ephemeral-storage.",
        "[SANDBOX::ENDED] Pod sandbox entered Failed phase (Evicted): The node was low on resource: memory.",
        "[SANDBOX::ENDED] container sandbox terminated with Error (exit code 137)",
        "Sandbox is dead. OpenSandbox status: state='Failed', message='node was drained'",
        "[KUBERNETES::POD_READY_TIMEOUT] sandbox remained Pending/CREATING",
        "[KUBERNETES::SANDBOX_NOT_FOUND] sandbox disappeared",
        "Process cleanup failed: Timed out during OpenSandbox command run after 70s",
    ],
)
def test_ambiguous_or_unrelated_errors_are_not_resource_limit_grades(detail):
    assert sandbox_resource_limit(detail) is None
