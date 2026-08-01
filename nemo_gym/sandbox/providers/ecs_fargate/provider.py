# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ECS Fargate sandbox provider.

Adapts :mod:`engine` (one stateful ``EcsFargateSandbox`` per sandbox) to Gym's ``SandboxProvider``
contract: per-sandbox engine state lives in ``SandboxHandle.raw`` and the provider methods delegate
to it. The model endpoint is reached directly from the sandbox, or via the SSH reverse tunnel when
``outside_endpoints`` are supplied through ``spec.provider_options``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.ecs_fargate import engine


def _outside_endpoints(spec: SandboxSpec) -> list[engine.OutsideEndpoint]:
    raw = spec.provider_options.get("outside_endpoints") or []
    return [engine.OutsideEndpoint.model_validate(item) for item in raw]


def _volumes(spec: SandboxSpec) -> list[engine.VolumeMount]:
    raw = spec.provider_options.get("volumes") or []
    return [engine.VolumeMount.model_validate(item) for item in raw]


def _engine_spec(spec: SandboxSpec) -> engine.SandboxSpec:
    if spec.image is None:
        raise SandboxCreateError("ECS Fargate sandbox requires SandboxSpec.image")
    entrypoint = " ".join(spec.entrypoint) if spec.entrypoint else None
    return engine.SandboxSpec(
        image=spec.image,
        workdir=spec.workdir or "/workspace",
        env=dict(spec.env),
        files=dict(spec.files),
        entrypoint=entrypoint,
        volumes=_volumes(spec),
        environment_dir=spec.provider_options.get("environment_dir"),
    )


def _apply_spec_overrides(cfg: engine.EcsFargateConfig, spec: SandboxSpec) -> engine.EcsFargateConfig:
    """Apply per-sandbox ``SandboxSpec`` requests (readiness/TTL/resources) onto the provider config.

    ``SandboxResources.cpu`` is in vCPUs; Fargate task CPU is in 1024-unit increments. GPU is not
    supported on Fargate.
    """
    overrides: dict[str, Any] = {}
    if spec.ready_timeout_s is not None:
        overrides["startup_timeout_sec"] = float(spec.ready_timeout_s)
    if spec.ttl_s is not None:
        overrides["max_task_lifetime_sec"] = int(spec.ttl_s)
    resources = spec.resources
    if not isinstance(resources, SandboxResources):
        resources = SandboxResources.from_mapping(resources)
    if resources.gpu:
        raise SandboxCreateError("ECS Fargate does not support GPU sandboxes (spec.resources.gpu)")
    if resources.cpu is not None:
        overrides["cpu"] = str(int(resources.cpu * 1024))
    if resources.memory_mib is not None:
        overrides["memory"] = str(int(resources.memory_mib))
    if resources.disk_gib is not None:
        overrides["ephemeral_storage_gib"] = int(resources.disk_gib)
    return cfg.model_copy(update=overrides) if overrides else cfg


class EcsFargateProvider:
    """Run sandboxes as AWS ECS Fargate tasks behind an SSH sidecar."""

    name = "ecs_fargate"

    def __init__(self, **config: Any) -> None:
        self._cfg = engine_config_from_mapping(config)

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        cfg = _apply_spec_overrides(self._cfg, spec)
        sandbox = engine.EcsFargateSandbox(_engine_spec(spec), ecs_config=cfg)
        try:
            await sandbox.start(outside_endpoints=_outside_endpoints(spec))
        except Exception as e:  # noqa: BLE001 — uniform create failure surface
            raise SandboxCreateError(f"ECS Fargate create failed: {e}") from e
        return SandboxHandle(
            sandbox_id=sandbox.task_arn or "",
            provider_name=self.name,
            raw=sandbox,
        )

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        sandbox: engine.EcsFargateSandbox = handle.raw
        result = await sandbox.exec(
            command,
            timeout_sec=180 if timeout_s is None else float(timeout_s),
            cwd=cwd,
            env=env,
            user=user,
        )
        return SandboxExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.return_code,
        )

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        sandbox: engine.EcsFargateSandbox = handle.raw
        await sandbox.upload(Path(source_path), target_path)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        sandbox: engine.EcsFargateSandbox = handle.raw
        await sandbox.download(source_path, Path(target_path))

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        sandbox: engine.EcsFargateSandbox = handle.raw
        return SandboxStatus.RUNNING if sandbox.is_running else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        sandbox: engine.EcsFargateSandbox = handle.raw
        await sandbox.stop()

    async def aclose(self) -> None:
        return None


def engine_config_from_mapping(config: dict[str, Any]) -> engine.EcsFargateConfig:
    """Build an ``EcsFargateConfig`` from provider kwargs.

    When ``region`` is given but ``cluster`` is omitted, infrastructure is auto-discovered from SSM;
    explicit kwargs always win over SSM.
    """
    config = dict(config)
    region = config.get("region")
    cluster = config.get("cluster")
    ssm_project = config.get("ssm_project", engine.DEFAULT_SSM_PROJECT)

    ssm: dict[str, Any] = {}
    if region is not None and cluster is None:
        ssm = engine.resolve_ecs_config_from_ssm(region, ssm_project)

    def pick(key: str, default: Any = None) -> Any:
        val = config.get(key)
        if val is not None:
            return val
        return ssm.get(key, default)

    ssh_sidecar = _sidecar_config(config.get("ssh_sidecar"), ssm.get("ssh_sidecar", {}))

    resolved = {
        **config,
        "region": region,
        "cluster": pick("cluster", ""),
        "subnets": config.get("subnets") or ssm.get("subnets", []),
        "security_groups": config.get("security_groups") or ssm.get("security_groups", []),
        "assign_public_ip": pick("assign_public_ip", True),
        "cpu": str(config.get("cpu", "4096")),
        "memory": str(config.get("memory", "8192")),
        "execution_role_arn": pick("execution_role_arn"),
        "task_role_arn": pick("task_role_arn"),
        "log_group": pick("log_group"),
        "max_task_lifetime_sec": config.get("max_task_lifetime_sec") or 14400,
        "startup_timeout_sec": float(config.get("startup_timeout_sec", 300.0)),
        "ssh_sidecar": ssh_sidecar,
        "s3_bucket": pick("s3_bucket"),
        "ecr_repository": pick("ecr_repository"),
        "codebuild_service_role": pick("codebuild_service_role"),
        "codebuild_compute_type": config.get("codebuild_compute_type") or "BUILD_GENERAL1_MEDIUM",
        "codebuild_build_timeout": config.get("codebuild_build_timeout") or 60,
        "auto_mirror": config.get("auto_mirror", True),
        "dockerhub_secret_arn": pick("dockerhub_secret_arn"),
        "efs_filesystem_id": pick("efs_filesystem_id"),
        "efs_access_point_id": pick("efs_access_point_id"),
        "ssm_project": ssm_project,
    }
    return engine.EcsFargateConfig.model_validate(resolved)


def _sidecar_config(yaml_sidecar: Any, ssm_ssh: Mapping[str, Any]) -> engine.SshSidecarConfig | None:
    if yaml_sidecar is not None:
        if isinstance(yaml_sidecar, engine.SshSidecarConfig):
            return yaml_sidecar
        if not isinstance(yaml_sidecar, Mapping):
            raise TypeError("ssh_sidecar must be a mapping or SshSidecarConfig instance")

        resolved = dict(yaml_sidecar)
        pub = resolved.get("public_key_secret_arn") or ssm_ssh.get("public_key_secret_arn", "")
        priv = resolved.get("private_key_secret_arn") or ssm_ssh.get("private_key_secret_arn", "")
        if not pub or not priv:
            raise ValueError(
                "ssh_sidecar.public_key_secret_arn and ssh_sidecar.private_key_secret_arn "
                "are required (set explicitly or auto-discovered from SSM)."
            )
        resolved.update(
            {
                "sshd_port": resolved.get("sshd_port", engine.DEFAULT_SSHD_PORT),
                "ssh_ready_timeout_sec": resolved.get("ssh_ready_timeout_sec", 300.0),
                "public_key_secret_arn": pub,
                "private_key_secret_arn": priv,
                "exec_server_port": resolved.get("exec_server_port", engine.DEFAULT_EXEC_SERVER_PORT),
            }
        )
        return engine.SshSidecarConfig.model_validate(resolved)
    if ssm_ssh.get("public_key_secret_arn") and ssm_ssh.get("private_key_secret_arn"):
        return engine.SshSidecarConfig(
            sshd_port=ssm_ssh.get("sshd_port", engine.DEFAULT_SSHD_PORT),
            public_key_secret_arn=ssm_ssh["public_key_secret_arn"],
            private_key_secret_arn=ssm_ssh["private_key_secret_arn"],
            exec_server_port=ssm_ssh.get("exec_server_port", engine.DEFAULT_EXEC_SERVER_PORT),
        )
    return None
