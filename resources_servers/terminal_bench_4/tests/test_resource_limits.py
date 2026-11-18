# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from resources_servers.terminal_bench_4.resource_limits import sandbox_resource_limit


# The cleanup failures of a real run, verbatim: OpenSandbox's lifecycle report cuts the pod detail at 256 characters.
STORAGE_EVICTION_CLEANUP = (
    "Process cleanup failed: Sandbox is unavailable; lifecycle status: state='Failed', reason='FAILED', "
    "message='1/1 observed pods failed; primary reason=PodFailed; sample pod=1ab9122e-73b8-449d-bda7-98f6eebd28b4-0; "
    "sample detail=Pod 1ab9122e-73b8-449d-bda7-98f6eebd28b4-0 entered Failed phase (Evicted): "
    "Pod ephemeral local storage usage exceeds the total limit of con'"
)
CONTAINER_EXIT_CLEANUP = (
    "Process cleanup failed: Sandbox is unavailable; lifecycle status: state='Failed', reason='FAILED', "
    "message='1/1 observed pods failed; primary reason=ContainerFailed; sample pod=aa162baf-e2bb-477e-84c8-ea31d6b0216d-0; "
    "sample detail=Pod aa162baf-e2bb-477e-84c8-ea31d6b0216d-0 container sandbox exited with code 143 (Error)'"
)
_UNAVAILABLE = (
    "Sandbox is unavailable; lifecycle status: state='Failed', reason='FAILED', message='1/1 observed pods failed; "
)


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
        (STORAGE_EVICTION_CLEANUP, "storage"),
        (
            _UNAVAILABLE + "primary reason=PodFailed; sample pod=p-0; sample detail=Pod p-0 entered Failed phase "
            "(Evicted): Pod ephemeral local storage usage exceeds the total limit of containers 10Gi.'",
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
        CONTAINER_EXIT_CLEANUP,
        "Sandbox is unavailable; lifecycle status: state='Failed', reason='FAILED'",
        _UNAVAILABLE + "primary reason=PodFailed; sample pod=p-0; sample detail=Pod p-0 entered Failed phase "
        "(Evicted): The node was low on resource: ephemeral-storage. Container sandbox was using 9Gi.'",
        _UNAVAILABLE
        + "primary reason=PodFailed; sample pod=p-0; sample detail=Pod p-0 entered Failed phase (Evicted)'",
        # The lifecycle report names no container, so an OOMKilled reason there may be a sidecar's.
        "Sandbox is unavailable; lifecycle status: state='Failed', reason='OOMKilled'",
        "Process cleanup failed: No space left on device",
        "Sandbox is unavailable; lifecycle status: state='Failed', reason='FAILED', message='No space left on device'",
        "Command exited with code 143",
        "container sandbox exited with code 143 (Error)",
    ],
)
def test_ambiguous_or_unrelated_errors_are_not_resource_limit_grades(detail):
    assert sandbox_resource_limit(detail) is None
