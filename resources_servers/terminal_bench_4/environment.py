# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own the main sandbox or concrete Compose collection for one TB4 role."""

import asyncio
import json
import math
import os
import shlex
from copy import deepcopy
from dataclasses import replace
from pathlib import Path, PurePosixPath
from time import monotonic
from typing import Any, Literal

import yaml
from pydantic import Field

from nemo_gym.sandbox import (
    AsyncSandbox,
    AsyncSandboxCompose,
    SandboxResources,
    SandboxSpec,
    resolve_provider_config,
    resolve_provider_metadata,
    rewrite_image,
)
from resources_servers.terminal_bench_4.compose_config import resolve_compose
from resources_servers.terminal_bench_4.task import Settings, resolve_env


class EnvironmentConfig(Settings):
    cpu_enforcement_policy: Literal["limit"]
    memory_enforcement_policy: Literal["limit"]
    sandbox_provider: dict[str, Any]
    sandbox_metadata: dict[str, str]
    sandbox_provider_options: dict[str, Any]
    sandbox_env: dict[str, str]
    sandbox_env_by_task: dict[str, dict[str, str]]
    sandbox_request_gpu_type: bool
    sandbox_split_endpoints: bool
    compose_image_configs: Path | None
    sandbox_ttl_s: float = Field(gt=0)
    sandbox_ready_timeout_s: float = Field(gt=0)
    default_exec_timeout_s: float = Field(gt=0)
    exec_shell: str | None
    image_rewrites: list[dict[str, str]]
    workdir: str | None
    efs_logs_host_path: str | None
    efs_logs_init_image: str


class HealthcheckError(RuntimeError):
    pass


class Environment:
    def __init__(self, task, config: EnvironmentConfig, session_id, directory, *, verifier=False):
        self.task = task
        self.config = config
        self.session_id = session_id
        self.directory = Path(directory)
        self.settings = task.config.verifier_environment if verifier else task.config.environment
        self.environment_dir = task.path / ("tests" if verifier else "environment")
        self.provider_config = deepcopy(config.sandbox_provider)
        self.pool = "default"
        self.main = None
        self.compose = None
        self.closed = False
        self.cleanup_errors = []
        self.resources = []
        self._cleanup_task = None
        self.shared_logs = None
        self.efs_logs_fallback = None
        self.log_role = "verifier" if verifier else "agent"
        self.task_env = resolve_env(self.settings.env)
        self.startup_env = (
            self.task_env | config.sandbox_env_by_task.get(task.name.split("/")[-1], {}) | config.sandbox_env
        )
        self.uses_compose = (self.environment_dir / "docker-compose.yaml").is_file()
        if not verifier and not self.uses_compose:
            services = {a.service for a in task.config.artifacts} | {h.service for h in task.config.verifier.collect}
            if services - {None, "main"}:
                raise ValueError("Sidecar artifacts and hooks require Compose")
        if config.sandbox_split_endpoints:
            # Keep every role and its helpers on one deployment: endpoint pools
            # can have different EFS filesystems and inter-sandbox networks.
            self.pool = "gpu" if task.config.environment.gpus or task.config.verifier_environment.gpus else "cpu"
            if "opensandbox" not in self.provider_config:
                raise ValueError("Split endpoints require OpenSandbox")
            connection = self.provider_config["opensandbox"].setdefault("connection", {})
            for key, suffix in (("domain", "DOMAIN"), ("api_key", "API_KEY")):
                name = f"OPENSANDBOX_{suffix}_{self.pool.upper()}"
                if not os.environ.get(name):
                    raise ValueError(f"Missing environment variable: {name}")
                connection[key] = os.environ[name]
        elif "opensandbox" in self.provider_config and os.environ.get("OPENSANDBOX_API_KEY"):
            self.provider_config["opensandbox"].setdefault("connection", {}).setdefault(
                "api_key", os.environ["OPENSANDBOX_API_KEY"]
            )
        resolve_provider_config(self.provider_config)
        if config.efs_logs_host_path and "opensandbox" not in self.provider_config:
            raise ValueError("EFS logs require OpenSandbox")
        if self.settings.network_mode == "no-network" and (
            self.uses_compose or "opensandbox" not in self.provider_config
        ):
            raise ValueError("Offline verification requires a single OpenSandbox environment")
        if self.uses_compose and config.compose_image_configs is None:
            raise ValueError("Compose requires verified image startup metadata")

    def build_spec(self):
        settings, config = self.settings, self.config
        metadata = {
            "tb4-session": self.session_id,
            "tb4-task": self.task.name.split("/")[-1],
            **resolve_provider_metadata(self.provider_config),
            **config.sandbox_metadata,
        }
        if self.pool != "default":
            metadata["nemo-gym.nvidia.com/resource-pool"] = self.pool
        options = deepcopy(config.sandbox_provider_options)
        if self.shared_logs is not None:
            volumes = list(options.get("volumes") or [])
            for volume in volumes:
                target, logs = PurePosixPath(volume.get("mountPath", "")), PurePosixPath("/logs")
                if (
                    target == logs
                    or target in logs.parents
                    or logs in target.parents
                    or volume.get("name") == "tb4-logs"
                ):
                    raise ValueError("EFS logs conflict with a configured /logs mount")
            volumes.append(self.shared_logs.volume(self.log_role))
            options["volumes"] = volumes
        if settings.network_mode == "no-network":
            options["network_policy"] = {"defaultAction": "deny", "egress": []}
        return SandboxSpec(
            image=rewrite_image(settings.docker_image, config.image_rewrites),
            resources=SandboxResources(
                cpu=settings.cpus,
                memory_mib=settings.memory_mb,
                disk_gib=math.ceil(settings.storage_mb / 1024) if settings.storage_mb else None,
                gpu=settings.gpus or None,
                gpu_type=settings.gpu_types[0] if settings.gpu_types and config.sandbox_request_gpu_type else None,
            ),
            ttl_s=config.sandbox_ttl_s,
            ready_timeout_s=config.sandbox_ready_timeout_s,
            workdir=config.workdir,
            env=self.startup_env,
            metadata=metadata,
            provider_options=options,
        )

    async def start(self):
        spec = self.build_spec()
        if self.uses_compose:
            image_path = self.config.compose_image_configs
            if not image_path.is_absolute():
                image_path = Path(__file__).resolve().parents[2] / image_path
            document = resolve_compose(
                yaml.safe_load((self.environment_dir / "docker-compose.yaml").read_text()),
                self.settings.docker_image,
                json.loads(image_path.read_text()),
            )
            if self.log_role == "agent":
                if self.task.name == "terminal-bench/medical-claims-processing":
                    # pwuser cannot edit /etc/hosts; its only peer URL is the
                    # browser's initial workspace page (which uses relative URLs).
                    document["services"]["playwright-mcp"]["x-sandbox"] = {
                        "hosts": [],
                        "resolve_environment": ["BROWSER_URL"],
                    }
                    # Use the image's default pwuser without an explicit su.
                    document["services"]["playwright-mcp"].pop("user", None)
                elif self.task.name == "terminal-bench/payments-pipeline-fix":
                    # This single broker uses localhost for its controller.
                    # Clients still resolve its advertised kafka:9092 address.
                    document["services"]["kafka"]["x-sandbox"] = {"hosts": []}
                    # Use the image's default appuser without an explicit su.
                    document["services"]["kafka"].pop("user", None)
            if "opensandbox" in self.provider_config:
                for service in document["services"].values():
                    if service.get("shm_size") is not None:
                        service.setdefault("labels", {})["nemo.nvidia.com/shm"] = str(service["shm_size"])
            sidecars = {a.service for a in self.task.config.artifacts} | {
                h.service for h in self.task.config.verifier.collect
            }
            if sidecars - {None, "main"} - document["services"].keys():
                raise ValueError("Artifact or collect hook references an unavailable Compose service")
            path = self.directory / "sandbox" / f"{self.session_id}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(document, sort_keys=False))
            self.compose = AsyncSandboxCompose(
                resolve_provider_config(self.provider_config),
                path,
                service_specs={
                    name: spec
                    if name == "main"
                    else replace(
                        spec,
                        resources=SandboxResources(),
                        env={},
                        provider_options=deepcopy(self.config.sandbox_provider_options),
                    )
                    for name in document["services"]
                },
                timeout_s=self.config.sandbox_ready_timeout_s,
            )
            await self.compose.start()
            self.main = self.compose.services["main"]
        else:
            self.main = AsyncSandbox(resolve_provider_config(self.provider_config), spec)
            try:
                await self.main.start()
            except Exception as exc:
                # Older endpoints (including the current GPU deployment) reject
                # host mounts before allocating a sandbox. Keep their existing
                # lifecycle usable, but never mask other provisioning failures.
                if (
                    self.shared_logs is None
                    or "VOLUME::HOST_PATH_NOT_ALLOWED" not in str(exc)
                    or self.shared_logs.host_path not in str(exc)
                ):
                    raise
                await self.main.stop()
                self.efs_logs_fallback = str(exc)
                self.shared_logs = None
                self.main = AsyncSandbox(resolve_provider_config(self.provider_config), self.build_spec())
                await self.main.start()
        if self.shared_logs is not None:
            await self.shared_logs.initialize_role(self)
        result = await self.exec("mkdir -p /logs/agent /logs/verifier /logs/artifacts", timeout_sec=60)
        if result.return_code:
            raise RuntimeError(f"Failed to initialize task log directories: {result.stderr}")
        # Published images with a build spec already contain these files.
        if (
            not self.uses_compose
            and not (self.environment_dir / "Dockerfile").exists()
            and self.environment_dir.is_dir()
        ):
            from resources_servers.terminal_bench_4.transfers import upload_dir

            cwd = self.settings.workdir or (await self.exec("pwd")).stdout.strip()
            await upload_dir(self.main, self.environment_dir, cwd)

    def sandbox(self, service=None):
        if service not in (None, "main"):
            if self.compose is None or service not in self.compose.services:
                raise ValueError(f"Unavailable Compose service: {service}")
            return self.compose.services[service]
        if self.main is None:
            raise RuntimeError("Main sandbox is not running")
        return self.main

    async def exec(self, command, *, service=None, cwd=None, env=None, timeout_sec=None, user=None):
        main = service in (None, "main")
        shell = self.config.exec_shell if main else "sh -c"
        if shell:
            command = f"{shell} {shlex.quote(command)}"
        persistent = self.task_env if main and not self.uses_compose else {}
        return await self.sandbox(service).exec(
            command,
            cwd=cwd,
            env=(persistent | (env or {})) or None,
            timeout_s=timeout_sec if timeout_sec is not None else self.config.default_exec_timeout_s,
            user=user,
        )

    async def main_connection(self):
        descriptor = await self.main.serialize(scope="operate")
        if "sandbox_id" not in descriptor:
            raise ValueError("TB4 requires an ID-based connectable provider")
        cwd = await self.exec("pwd", timeout_sec=30, user=self.task.config.agent.user)
        if cwd.return_code:
            raise RuntimeError("Unable to determine the task working directory")
        return {"provider": self.pool, "sandbox_id": descriptor["sandbox_id"], "workdir": cwd.stdout.strip()}

    async def healthcheck(self):
        hc = self.settings.healthcheck
        if hc is None:
            return
        grace = monotonic() + hc.start_period_sec
        failures = 0
        while True:
            in_grace = monotonic() < grace
            result = await self.exec(hc.command, timeout_sec=int(hc.timeout_sec))
            if result.return_code == 0:
                return
            if not in_grace:
                failures += 1
                if failures >= hc.retries:
                    raise HealthcheckError(f"Healthcheck failed after {hc.retries} consecutive retries: {hc.command}")
            await asyncio.sleep(hc.start_interval_sec if in_grace else hc.interval_sec)

    async def quiesce_agent(self, session_id):
        pidfile = shlex.quote(f"/tmp/{session_id}.pids")
        result = await self.exec(
            f"if [ -f {pidfile} ]; then groups=$(cat {pidfile}); for p in $groups; do "
            "case $p in ''|*[!0-9]*) exit 1;; esac; "
            'kill -TERM -- -"$p" 2>/dev/null || true; done; sleep 1; '
            'for p in $groups; do if kill -0 -- -"$p" 2>/dev/null; then '
            'kill -KILL -- -"$p" 2>/dev/null || exit 1; fi; done; sleep 1; fi',
            timeout_sec=30,
            user=self.task.config.agent.user,
        )
        if result.return_code:
            raise RuntimeError("Could not stop the external agent before artifact collection")

    async def stop_main(self):
        await self.main.stop()

    def resource_identities(self):
        members = self.compose.services if self.compose else {"main": self.main}
        result = []
        for name, sandbox in members.items():
            handle = getattr(sandbox, "_handle", None)
            if handle is not None:
                result.append({"service": name, "provider": self.pool, "sandbox_id": handle.sandbox_id})
        if self.compose:
            result.append({"compose_project": self.compose.project, "provider": self.pool})
        return result

    async def stop(self):
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._stop())
        await asyncio.shield(self._cleanup_task)

    async def _stop(self):
        self.resources = self.resource_identities()
        try:
            if self.compose is not None:
                await self.compose.stop()
            elif self.main is not None:
                await self.main.stop()
            self.closed = True
        except Exception as exc:
            self.cleanup_errors.append({"error": str(exc), "resources": self.resources})
            raise
