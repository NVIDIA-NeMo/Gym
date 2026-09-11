# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attach an exec-only AsyncSandbox to a seeded docker sandbox.

The docker sandbox provider does not implement serialize()/connect(), but the
swebench resources server returns the seeded container id as
``sandbox_handle``. Both servers share one docker daemon, so the agent can
reconstruct an exec-only AsyncSandbox around that container: a DockerProvider
plus a SandboxHandle whose ``raw`` names the existing container. The lifecycle
stays with the resources server, which stops the container during
verification.
"""

from __future__ import annotations

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxHandle
from nemo_gym.sandbox.providers.docker.provider import DockerProvider, _DockerContainer


def attach_docker_sandbox(sandbox_handle: str) -> AsyncSandbox:
    """Return an AsyncSandbox that execs into the already-running container."""

    if not isinstance(sandbox_handle, str) or not sandbox_handle.strip():
        raise ValueError("sandbox_handle must be a non-empty container identifier")
    container_id = sandbox_handle.strip()
    provider = DockerProvider()
    sandbox = AsyncSandbox(provider)
    sandbox._handle = SandboxHandle(
        sandbox_id=container_id,
        provider_name=provider.name,
        raw=_DockerContainer(
            name=container_id,
            image="",
            shell=provider._exec_config.exec_shell or "bash",
        ),
    )
    sandbox._stopped = False
    return sandbox
