# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor 0.23 environment backed by Gym's provider-neutral sandbox API."""

import math
import shlex
from dataclasses import replace

from harbor.environments.base import ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, resolve_provider_config
from responses_api_agents.harbor_agent.custom_envs.nemo_gym_sandbox.environment import NemoGymSandboxEnvironment


class HarborSandboxEnvironment(NemoGymSandboxEnvironment):
    """Keep the legacy adapter stable while implementing the current Harbor contract."""

    @staticmethod
    def type() -> str:
        return "nemo-gym-sandbox"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(gpus=True)

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    def _validate_definition(self) -> None:
        super()._validate_definition()
        if (self.environment_dir / "docker-compose.yaml").exists():
            raise ValueError("Compose task environments require the Gym Compose adapter")
        if any(not mount["target"].startswith("/logs/") for mount in self._mounts):
            raise ValueError("Only Harbor log mounts are supported; logs are transferred through the sandbox API")
        if len(self.task_env_config.gpu_types or []) > 1:
            raise ValueError("The sandbox API requires one explicit GPU type")

    def _build_spec(self) -> SandboxSpec:
        config = self.task_env_config
        resources = SandboxResources(
            cpu=self._effective_cpus,
            memory_mib=self._effective_memory_mb,
            disk_gib=math.ceil(config.storage_mb / 1024) if config.storage_mb else None,
            gpu=self._effective_gpus or None,
            gpu_type=config.gpu_types[0] if config.gpu_types else None,
        )
        return replace(
            super()._build_spec(),
            resources=resources,
            env={**self._startup_env(), **self._sandbox_env},
        )

    async def start(self, force_build: bool) -> None:
        if force_build:
            raise ValueError("Published task images are immutable; force_build is unsupported")
        self._sandbox = AsyncSandbox(resolve_provider_config(self._sandbox_provider), self._build_spec())
        try:
            await self._sandbox.start()
            targets = " ".join(
                shlex.quote(target)
                for target in {
                    "/logs/agent",
                    "/logs/verifier",
                    "/logs/artifacts",
                    *[m["target"] for m in self._mounts],
                }
            )
            result = await self.exec(f"mkdir -p {targets}", timeout_sec=60)
            if result.return_code:
                raise RuntimeError(f"Failed to initialize Harbor log directories: {result.stderr}")
            await self._upload_environment_dir_after_start()
        except BaseException:
            await self.stop(delete=True)
            raise

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if self._exec_shell:
            command = f"{self._exec_shell} {shlex.quote(command)}"
        result = await self._require_sandbox().exec(
            command,
            cwd=cwd,
            env=self._merge_env(env),
            timeout_s=timeout_sec if timeout_sec is not None else self._default_exec_timeout_s,
            user=self._resolve_user(user),
        )
        return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.return_code)
