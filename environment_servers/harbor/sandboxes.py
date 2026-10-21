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

import asyncio
import math
from typing import Any

from environment_servers.harbor.session import resolve_env_templates
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.tasks.harbor import HarborTask


# Added to `sandbox_ready_timeout_s` for the seed's own deadline on the create: room for the provider to
# report and clean up a create that hit its (equal) budget before the seed gives up on it.
SEED_CREATE_GRACE_S = 30.0


class HarborSeedTimeoutError(RuntimeError):
    """The sandbox was not ready within `sandbox_ready_timeout_s`: a provider-side seed failure, retried."""


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

    def _sandbox_spec(self, task: HarborTask, workdir: str | None) -> SandboxSpec:
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
        # The override replaces only the keys it names, so a GPU task routed to the GPU provider keeps `gpu`.
        resources = _sandbox_resources(task) | (self.config.sandbox_resources_override or {})
        env = dict(self.config.sandbox_env) | resolve_env_templates(task.env)
        if self.config.derive_cpu_env:
            env = cpu_cap_env(resources.get("cpu")) | env
        return SandboxSpec(
            image=task.image,
            ttl_s=ttl,
            ready_timeout_s=self.config.sandbox_ready_timeout_s,
            workdir=workdir,
            env=env,
            metadata=metadata,
            resources=resources,
            provider_options=dict(self.config.sandbox_provider_options),
        )

    async def _create_sandbox(self, task: HarborTask, workdir: str | None, *, episode_key: str) -> AsyncSandbox:
        provider_config = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
        sandbox = AsyncSandbox(provider_config)
        self._sandboxes[episode_key] = sandbox
        await sandbox.start(self._sandbox_spec(task, workdir))
        return sandbox

    async def _create_sandbox_within_deadline(self, task: HarborTask, episode_key: str) -> AsyncSandbox:
        """``_create_sandbox`` bounded by the configured ready timeout, whatever the provider's retry budget.

        The provider retries a create on its own schedule and its request may block as long as the cluster
        takes to allocate the pod; `sandbox_ready_timeout_s` is this server's promise, so it is enforced
        here too. The timeout is a provider-side seed failure, retried like any other.
        """
        deadline_s = self.config.sandbox_ready_timeout_s + SEED_CREATE_GRACE_S
        deadline = asyncio.timeout(deadline_s)
        try:
            async with deadline:
                return await self._create_sandbox(task, task.workdir, episode_key=episode_key)
        except TimeoutError as exc:
            if not deadline.expired():
                raise
            raise HarborSeedTimeoutError(
                f"the sandbox was not ready within {deadline_s:g}s "
                f"(sandbox_ready_timeout_s={self.config.sandbox_ready_timeout_s:g} plus {SEED_CREATE_GRACE_S:g}s grace)"
            ) from exc
