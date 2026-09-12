# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor 0.23 environment backed by Gym's provider-neutral sandbox API."""

import json
import math
import os
import shlex
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import yaml
from harbor.environments.base import ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities

from nemo_gym.sandbox import AsyncSandbox, AsyncSandboxCompose, SandboxResources, SandboxSpec, resolve_provider_config
from responses_api_agents.harbor_agent.custom_envs.nemo_gym_sandbox.environment import NemoGymSandboxEnvironment
from responses_api_agents.harbor_agent_general.compose_config import resolve_compose


class HarborSandboxEnvironment(NemoGymSandboxEnvironment):
    """Keep the legacy adapter stable while implementing the current Harbor contract."""

    def __init__(self, *args, sandbox_provider=None, compose_image_configs=None, **kwargs):
        # Keep credentials in the process environment, out of Harbor's saved job
        # and trial configs. Providers receive a private copy at construction.
        provider = deepcopy(sandbox_provider)
        if provider and "opensandbox" in provider:
            connection = provider["opensandbox"].setdefault("connection", {})
            if "api_key" not in connection and os.environ.get("OPENSANDBOX_API_KEY"):
                connection["api_key"] = os.environ["OPENSANDBOX_API_KEY"]
        self._compose: AsyncSandboxCompose | None = None
        self._active_service = ContextVar("harbor_sandbox_service", default=None)
        self._compose_image_configs = Path(compose_image_configs) if compose_image_configs else None
        if self._compose_image_configs is not None and not self._compose_image_configs.is_absolute():
            self._compose_image_configs = Path(__file__).resolve().parents[2] / self._compose_image_configs
        super().__init__(*args, sandbox_provider=provider, **kwargs)

    @staticmethod
    def type() -> str:
        return "nemo-gym-sandbox"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(gpus=True, docker_compose=True)

    @property
    def _uses_compose(self) -> bool:
        return (self.environment_dir / "docker-compose.yaml").is_file()

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    def _validate_definition(self) -> None:
        super()._validate_definition()
        if self._uses_compose and self._compose_image_configs is None:
            raise ValueError("Compose environments require compose_image_configs with verified OCI startup metadata")
        if self.extra_docker_compose_paths:
            raise ValueError("Additional Compose overlays must be merged upstream")
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
        try:
            if self._uses_compose:
                await self._start_compose()
            else:
                self._sandbox = AsyncSandbox(resolve_provider_config(self._sandbox_provider), self._build_spec())
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

    async def _start_compose(self) -> None:
        document = resolve_compose(
            yaml.safe_load((self.environment_dir / "docker-compose.yaml").read_text()),
            self.task_env_config.docker_image,
            json.loads(self._compose_image_configs.read_text()),
        )
        output_dir = self.trial_paths.trial_dir / "sandbox"
        output_dir.mkdir(parents=True, exist_ok=True)
        compose_file = output_dir / f"compose-{uuid4().hex}.yaml"
        compose_file.write_text(yaml.safe_dump(document, sort_keys=False))
        main_spec = self._build_spec()
        specs = {
            name: main_spec if name == "main" else replace(main_spec, resources=SandboxResources(), env={})
            for name in document["services"]
        }
        self._compose = AsyncSandboxCompose(
            resolve_provider_config(self._sandbox_provider),
            compose_file,
            service_specs=specs,
            timeout_s=self._sandbox_ready_timeout_s or 1200,
        )
        await self._compose.start()
        self._sandbox = self._compose.services["main"]

    def _require_sandbox(self):
        service = self._active_service.get()
        if service is not None:
            return self._compose.services[service]
        return super()._require_sandbox()

    @contextmanager
    def _service_scope(self, service):
        if not self.is_main_service(service) and (self._compose is None or service not in self._compose.services):
            raise ValueError(f"Compose service {service!r} is unavailable")
        token = self._active_service.set(None if self.is_main_service(service) else service)
        try:
            yield
        finally:
            self._active_service.reset(token)

    async def stop(self, delete: bool):
        if self._compose is None:
            await super().stop(delete)
        else:
            compose, self._compose = self._compose, None
            self._sandbox = None
            await compose.stop()

    async def stop_service(self, service: str) -> None:
        with self._service_scope(service):
            await self._require_sandbox().stop()

    async def service_exec(self, command, *, service=None, cwd=None, env=None, timeout_sec=None, user=None):
        if self.is_main_service(service):
            return await self.exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)
        with self._service_scope(service):
            result = await self._require_sandbox().exec(
                f"sh -c {shlex.quote(command)}",
                cwd=cwd,
                env=env,
                timeout_s=timeout_sec if timeout_sec is not None else self._default_exec_timeout_s,
                user=user,
            )
        return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.return_code)

    async def service_download_file(self, source_path, target_path, *, service=None):
        with self._service_scope(service):
            await self.download_file(source_path, target_path)

    async def service_download_dir(self, source_dir, target_dir, *, service=None):
        with self._service_scope(service):
            await self.download_dir(source_dir, target_dir)

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
