# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provision the sandboxes an episode runs in.

Building the SandboxSpec from the task and the server configuration, clamping resources to
what the provider allows, routing GPU tasks to the GPU provider, creating the sandbox inside
the ready deadline with the seed grace, resolving and caching the task's Compose document and
starting the Compose group, and creating the separate verifier sandbox. The seed grace
constant and the seed timeout error live here with the deadline that uses them. Everything in
the mixin resolves the provider configuration, the global config and the sandbox classes from
this module's globals, which is where tests patch them.
"""

import math
from typing import Any

from environment_servers.harbor.session import resolve_env_templates
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.tasks.harbor import HarborTask


def _sandbox_resources(task: HarborTask) -> dict[str, Any]:
    environment = task.config.environment
    resources: dict[str, Any] = {}
    if environment.cpus is not None:
        resources["cpu"] = environment.cpus
    if environment.memory_mb is not None:
        resources["memory_mib"] = environment.memory_mb
    if environment.storage_mb is not None:
        resources["disk_gib"] = max(1, math.ceil(environment.storage_mb / 1024))
    if environment.gpus:
        resources["gpu"] = environment.gpus
        if environment.gpu_types:
            resources["gpu_type"] = environment.gpu_types[0]
    return resources


class HarborSandboxMixin:
    """Provision the sandboxes an episode runs in.

    Building the SandboxSpec from the task and the server configuration, clamping resources to
    what the provider allows, routing GPU tasks to the GPU provider, creating the sandbox inside
    the ready deadline with the seed grace, resolving and caching the task's Compose document and
    starting the Compose group, and creating the separate verifier sandbox. The seed grace
    constant and the seed timeout error live here with the deadline that uses them. Everything in
    the mixin resolves the provider configuration, the global config and the sandbox classes from
    this module's globals, which is where tests patch them.

    """

    def _sandbox_spec(self, task: HarborTask, workdir: str) -> SandboxSpec:
        global_config_dict = get_global_config_dict()
        metadata = (
            resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
            | self.config.sandbox_metadata
            | {"nemo_gym_environment_server": self.config.name, "harbor_task": task.task_id[:63]}
        )
        ttl = (
            self.config.default_episode_timeout_seconds
            + self.config.cleanup_timeout_seconds
            + self.config.sandbox_ttl_slack_s
        )
        return SandboxSpec(
            image=task.image,
            ttl_s=ttl,
            ready_timeout_s=self.config.sandbox_ready_timeout_s,
            workdir=workdir,
            env=resolve_env_templates(task.env),
            metadata=metadata,
            resources=_sandbox_resources(task),
            provider_options=dict(self.config.sandbox_provider_options),
        )

    async def _create_sandbox(self, task: HarborTask, workdir: str, *, episode_key: str) -> AsyncSandbox:
        provider_config = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
        sandbox = AsyncSandbox(provider_config)
        self._sandboxes[episode_key] = sandbox
        await sandbox.start(self._sandbox_spec(task, workdir))
        return sandbox
