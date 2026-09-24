# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Own the main sandbox or concrete Compose collection for one TB4 role."""

import asyncio
import json
import logging
import math
import os
import re
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
from resources_servers.terminal_bench_4.compose_config import MAIN_COMMAND, resolve_compose, resolve_image_startup
from resources_servers.terminal_bench_4.task import Settings, resolve_env


logger = logging.getLogger(__name__)


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
    compose_image_configs: Path | None  # Shared OCI metadata for Compose and standalone images.
    # Compatibility override for existing single-container run configurations.
    single_container_image_configs: Path | None = None
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


def execution_user(value: str | int | None) -> str | int | None:
    """Normalize a task account/UID; None delegates to the current image default."""
    if value is None:
        return None
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        value = int(value)
    if type(value) is int and 0 <= value < 2**32:
        return value
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*\$?", value):
        return value
    raise ValueError(
        "Task execution identities require an account name or unsigned UID; "
        "USER user:group overrides are not supported and must not be discarded"
    )


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
        self.bootstrap_uid: int | None = None
        # Validate task identities before provisioning, without consulting
        # historical OCI metadata. Preprocessing owns any original-user fallback.
        execution_user(self.configured_user)
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

    @property
    def configured_user(self) -> str | int | None:
        """The task-authored execution identity for this role."""
        settings = self.task.config.verifier if self.log_role == "verifier" else self.task.config.agent
        return settings.user

    @property
    def role_user(self) -> str | int | None:
        """Use the task's identity, or the current image default when omitted."""
        return execution_user(self.configured_user)

    def _single_container_entrypoint(self, image: str) -> list[str] | None:
        override = self.config.single_container_image_configs
        path = override if override is not None else self.config.compose_image_configs
        if self.uses_compose or path is None:
            return None
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        records = json.loads(path.read_text())
        if not isinstance(records, dict):
            raise ValueError("OCI startup metadata must be an image mapping")
        if image not in records and override is None:
            # The official catalog originally covered only Compose images. Do
            # not break those legacy standalone tasks or guess an entrypoint.
            logger.warning("No recorded OCI startup metadata for %r; retaining provider keepalive", image)
            return None
        record = records.get(image)
        if not isinstance(record, dict):
            raise ValueError(f"No recorded OCI startup metadata for {image!r}")
        if record.get("image") != image:
            # Compose catalogs can spell tag@digest keys as repo@digest in the
            # record. Accept that only when both refs pin identical content.
            digest = image.rsplit("@", 1)[-1]
            recorded_image = record.get("image")
            if not (
                "@" in image
                and re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
                and isinstance(recorded_image, str)
                and recorded_image.endswith("@" + digest)
            ):
                raise ValueError(f"OCI startup metadata image does not match {image!r}")
        if (record.get("os"), record.get("architecture")) != ("linux", "amd64"):
            raise ValueError(f"OCI startup metadata for {image!r} requires a supported Linux/amd64 image")
        config = record.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"OCI startup metadata for {image!r} requires an image config")
        for field in ("Entrypoint", "Cmd"):
            value = config.get(field)
            if value is None:
                continue
            if not isinstance(value, list) or any(not isinstance(arg, str) for arg in value):
                raise ValueError(f"OCI {field} for {image!r} must be a list of strings or null")
        # A standalone sandbox is the main service, not a sidecar. Match the
        # Compose main convention: preserve ENTRYPOINT, replace CMD with the
        # keepalive (an inherited python/bash CMD can exit immediately).
        service = {"command": list(MAIN_COMMAND)}
        resolve_image_startup(service, config)
        # OpenSandbox takes the complete argv rather than separate EP/CMD fields.
        return [*service["entrypoint"], *service["command"]]

    def build_spec(self) -> SandboxSpec:
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
        image = rewrite_image(settings.docker_image, config.image_rewrites)
        return SandboxSpec(
            image=image,
            entrypoint=self._single_container_entrypoint(image),
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
        # Probe the running sandbox, not the original image: root-started main
        # containers can prepare role-owned logs; non-root images retain their
        # existing setup path without an attempted privilege escalation.
        result = await self.main.exec("id -u", timeout_s=30)
        try:
            self.bootstrap_uid = int(result.stdout.strip())
            if result.return_code or not 0 <= self.bootstrap_uid < 2**32:
                raise ValueError("Invalid bootstrap UID")
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Unable to determine sandbox default execution UID: {result.stderr}") from exc
        if self.shared_logs is not None:
            await self.shared_logs.initialize_role(self)
        log_setup = "mkdir -p /logs/agent /logs/verifier /logs/artifacts"
        if self.bootstrap_uid == 0:
            log_setup = (
                'for p in /logs /logs/agent /logs/verifier /logs/artifacts; do test ! -L "$p" || exit 1; done && '
                + log_setup
            )
        result = await self.exec(
            log_setup,
            timeout_sec=60,
        )
        if result.return_code:
            raise RuntimeError(f"Failed to initialize task log directories: {result.stderr}")
        if self.bootstrap_uid == 0:
            await self._prepare_role_logs()
        # Published images with a build spec already contain these files.
        if (
            not self.uses_compose
            and not (self.log_role == "verifier" and getattr(self.task, "stage_tests", False))
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
        if main:
            user = execution_user(user)
        return await self.sandbox(service).exec(
            command,
            cwd=cwd,
            env=(persistent | (env or {})) or None,
            timeout_s=timeout_sec if timeout_sec is not None else self.config.default_exec_timeout_s,
            user=user,
        )

    async def _prepare_role_logs(self) -> None:
        expected_uid = (
            self.role_user if isinstance(self.role_user, int) else 0 if self.role_user in (None, "root") else None
        )
        command = "id -u && id -g"
        if expected_uid is None:
            command += f" && id -u -- {shlex.quote(self.role_user)}"
        identity = await self.exec(command, user=self.role_user, timeout_sec=30)
        try:
            values = [int(value) for value in identity.stdout.split()]
            if expected_uid is None:
                uid, gid, expected_uid = values
            else:
                uid, gid = values
            if identity.return_code or min(uid, gid) < 0 or uid != expected_uid:
                raise ValueError("Invalid role identity")
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Unable to execute as {self.log_role} user {self.role_user!r}: {identity.stderr}"
            ) from exc
        # Only harness-owned log directories are prepared. Never recursively
        # chown the task workspace, restored artifacts, or trusted test assets.
        paths = "/logs /logs/agent /logs/verifier /logs/artifacts"
        result = await self.exec(
            f'for p in {paths}; do test ! -L "$p" && test -d "$p" || exit 1; done && '
            f"chown {uid}:{gid} {paths} && chmod 755 {paths}",
            user="root",
            timeout_sec=60,
        )
        if result.return_code:
            raise RuntimeError(f"Unable to prepare {self.log_role} log directories: {result.stderr}")
        path = self.directory / "sandbox" / f"{self.session_id}.identity.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "role": self.log_role,
                    "image": rewrite_image(self.settings.docker_image, self.config.image_rewrites),
                    "bootstrap_uid": self.bootstrap_uid,
                    "configured_user": self.configured_user,
                    "execution_user": self.role_user,
                    "execution_uid": uid,
                    "execution_gid": gid,
                },
                indent=2,
            )
            + "\n"
        )

    async def agent_workdir(self):
        cwd = await self.exec("pwd", timeout_sec=30, user=self.task.config.agent.user)
        if cwd.return_code:
            raise RuntimeError("Unable to determine the task working directory")
        return cwd.stdout.strip()

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
