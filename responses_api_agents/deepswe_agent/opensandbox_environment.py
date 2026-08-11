# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pier environment backed by Gym's provider-neutral OpenSandbox API."""

import asyncio
import math
import os
import shlex
import tarfile
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from opensandbox.constants import DEFAULT_EGRESS_PORT
from opensandbox.models.sandboxes import NetworkRule
from pier.environments.base import BaseEnvironment, ExecResult
from pier.environments.capabilities import EnvironmentCapabilities

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.providers.opensandbox.provider import _is_retryable_sdk_operation_error
from nemo_gym.sandbox.utils import rewrite_image


OPENSANDBOX_API_KEY_ENV = "OPENSANDBOX_API_KEY"  # pragma: allowlist secret
INSTALL_EGRESS_TARGETS = (
    "*.github.com",
    "github.com",
    "*.githubusercontent.com",
    "*.pythonhosted.org",
    "pypi.org",
    "astral.sh",
    "*.astral.sh",
    "*.ubuntu.com",
    "*.debian.org",
    "*.nodesource.com",
    "*.microsoft.com",
    "*.postgresql.org",
    "download.docker.com",
    "dl.yarnpkg.com",
    "packages.cloud.google.com",
    "*.fedoraproject.org",
    "*.centos.org",
    "*.rockylinux.org",
    "*.almalinux.org",
    "*.amazonlinux.com",
    "dl-cdn.alpinelinux.org",
)
STARTUP_COMMAND_RETRIES = 5
STARTUP_RETRY_MAX_DELAY_S = 15.0


def _provider_with_runtime_secret(provider: Mapping[str, Any]) -> dict[str, Any]:
    restored = {key: value for key, value in provider.items()}
    opensandbox = dict(restored.get("opensandbox") or {})
    connection = dict(opensandbox.get("connection") or {})
    if not connection.get("api_key"):
        api_key = os.getenv(OPENSANDBOX_API_KEY_ENV)
        if api_key:
            connection["api_key"] = api_key
    opensandbox["connection"] = connection
    restored["opensandbox"] = opensandbox
    return restored


class PierOpenSandboxEnvironment(BaseEnvironment):
    """Run both Pier's agent and its pristine verifier in OpenSandbox."""

    def __init__(
        self,
        *args: Any,
        provider: Mapping[str, Any],
        spec: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._provider_config = dict(provider)
        self._spec_config = dict(spec or {})
        self._sandbox: AsyncSandbox | None = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "gym-opensandbox"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            gpus=True,
            disable_internet=True,
            filtered_egress=True,
            preinstall_agents=True,
            mounted=False,
        )

    def _validate_definition(self) -> None:
        if self.task_env_config.docker_image is None and not (self.environment_dir / "Dockerfile").exists():
            raise ValueError(
                "OpenSandbox tasks need environment.docker_image or a Dockerfile whose FROM image is prebuilt"
            )

    def _image(self) -> str:
        image = self.task_env_config.docker_image
        if image is None:
            dockerfile = self.environment_dir / "Dockerfile"
            for raw_line in dockerfile.read_text().splitlines():
                line = raw_line.strip()
                if line.upper().startswith("FROM "):
                    image = line.split()[1]
                    break
        if not image:
            raise ValueError(f"Could not resolve an image from {self.environment_dir}")
        return rewrite_image(image, self._spec_config.get("image_rewrites", []))

    def _sandbox_spec(self) -> SandboxSpec:
        config = dict(self._spec_config)
        config.pop("image_rewrites", None)
        config.pop("tmux_bundle_path", None)
        config.pop("agent_install_timeout_s", None)
        provider_options = dict(config.pop("provider_options", {}))
        egress_targets = set(self.network_allowlist.domains)
        if self.agent_install_spec is not None:
            egress_targets.update(INSTALL_EGRESS_TARGETS)
        network_policy: dict[str, Any] = {"defaultAction": "deny"}
        if egress_targets:
            network_policy["egress"] = [{"action": "allow", "target": target} for target in sorted(egress_targets)]
        # Temporary install access is removed before the agent runs.
        provider_options["network_policy"] = network_policy

        metadata = dict(config.pop("metadata", {}))
        metadata.update(
            {
                "benchmark": "deep-swe",
                "harness": "pier",
                "task": self.environment_name,
                "session": self.session_id,
            }
        )
        disk_gib = math.ceil(self._effective_storage_mb / 1024) if self._effective_storage_mb else None
        resources = SandboxResources(
            cpu=float(self._effective_cpus) if self._effective_cpus is not None else None,
            memory_mib=self._effective_memory_mb,
            disk_gib=disk_gib,
            gpu=self._effective_gpus or None,
        )
        return SandboxSpec(
            image=self._image(),
            ttl_s=config.pop("ttl_s", None),
            ready_timeout_s=config.pop("ready_timeout_s", None),
            workdir=config.pop("workdir", self.task_env_config.workdir or "/app"),
            env={**self._persistent_env, **config.pop("env", {})},
            metadata=metadata,
            resources=resources,
            entrypoint=config.pop("entrypoint", None),
            provider_options=provider_options,
        )

    async def ensure_tmux(self) -> None:
        """Install Harbor's terminal dependency from an offline driver bundle."""
        result = await self._exec_startup("tmux -V", user="root")
        if result.return_code == 0:
            return

        bundle_path = self._spec_config.get("tmux_bundle_path")
        if not bundle_path:
            raise RuntimeError("tmux is missing from the task image and no tmux_bundle_path is configured")
        bundle = Path(str(bundle_path))
        if not bundle.is_file():
            raise FileNotFoundError(f"DeepSWE tmux bundle does not exist: {bundle}")

        remote_bundle = "/tmp/deep-swe-tmux.tar.gz"
        await self.upload_file(bundle, remote_bundle)
        result = await self._exec_startup(
            "rm -rf /opt/deep-swe-tmux && "
            "mkdir -p /opt/deep-swe-tmux /usr/local/bin && "
            f"tar xzf {shlex.quote(remote_bundle)} -C /opt/deep-swe-tmux && "
            "cp /opt/deep-swe-tmux/bin/tmux /usr/local/bin/tmux && "
            "chmod 755 /usr/local/bin/tmux /opt/deep-swe-tmux/bin/tmux-real && "
            f"rm -f {shlex.quote(remote_bundle)} && "
            "/usr/local/bin/tmux -V",
            timeout_sec=120,
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or result.stdout or "failed to install offline tmux bundle")

    async def _exec_startup(self, command: str, **kwargs: Any) -> ExecResult:
        """Retry transient endpoint failures only for idempotent setup commands."""
        for attempt in range(STARTUP_COMMAND_RETRIES + 1):
            try:
                return await self.exec(command, **kwargs)
            except Exception as exc:
                if attempt >= STARTUP_COMMAND_RETRIES or not _is_retryable_sdk_operation_error(exc):
                    raise
                delay_s = min(2**attempt, STARTUP_RETRY_MAX_DELAY_S)
                self.logger.warning(
                    "Retrying idempotent OpenSandbox startup command after transient failure "
                    "(attempt %s/%s, delay_s=%s): %s",
                    attempt + 1,
                    STARTUP_COMMAND_RETRIES + 1,
                    delay_s,
                    exc,
                )
                await asyncio.sleep(delay_s)

        raise RuntimeError("OpenSandbox startup command retry loop did not run")

    async def _preinstall_agent(self) -> None:
        install = self.agent_install_spec
        if install is None:
            return

        timeout_sec = int(self._spec_config.get("agent_install_timeout_s", 1800))
        mini_prerequisites_present = False
        if install.agent_name == "mini-swe-agent":
            prerequisite = await self._exec_startup(
                "command -v curl >/dev/null && command -v git >/dev/null && "
                "command -v gcc >/dev/null && command -v g++ >/dev/null && "
                "command -v make >/dev/null",
                timeout_sec=120,
                user="root",
            )
            mini_prerequisites_present = prerequisite.return_code == 0

        for index, step in enumerate(install.steps, start=1):
            # Skip mini-swe-agent's package-manager step when its tools exist.
            if mini_prerequisites_present and step.user == "root":
                continue
            user = "root" if step.user == "root" else self._resolve_user(None)
            result = await self._exec_startup(
                f"bash -lc {shlex.quote(step.run)}",
                env=step.env,
                timeout_sec=timeout_sec,
                user=user,
            )
            if result.return_code != 0:
                output = result.stderr or result.stdout or "no output"
                raise RuntimeError(
                    f"{install.agent_name} install step {index} failed with code {result.return_code}: {output}"
                )

        if install.verification_command:
            result = await self._exec_startup(
                f"bash -lc {shlex.quote(install.verification_command)}",
                timeout_sec=120,
                user=self._resolve_user(None),
            )
            if result.return_code != 0:
                output = result.stderr or result.stdout or "no output"
                raise RuntimeError(f"{install.agent_name} install verification failed: {output}")

    async def _close_install_egress(self) -> None:
        if self.agent_install_spec is None:
            return
        sandbox = self._require_sandbox()
        handle = sandbox._require_handle()
        raw = handle.raw
        egress_service = getattr(raw, "_egress_service", None)
        sandbox_service = getattr(raw, "_sandbox_service", None)
        if egress_service is None or sandbox_service is None:
            raise RuntimeError("OpenSandbox SDK does not expose its egress endpoint services")

        # OpenSandbox 0.1.15 omits this header from proxied endpoints.
        auth_header = "OPENSANDBOX-EGRESS-AUTH"
        if auth_header not in egress_service.endpoint.headers:
            direct_endpoint = await sandbox_service.get_sandbox_endpoint(
                raw.id,
                DEFAULT_EGRESS_PORT,
                use_server_proxy=False,
            )
            token = direct_endpoint.headers.get(auth_header)
            if not token:
                raise RuntimeError("OpenSandbox did not return an egress-auth endpoint header")
            egress_service.endpoint.headers[auth_header] = token
            egress_service._httpx_client.headers[auth_header] = token

        patch_rules = getattr(handle.raw, "patch_egress_rules", None)
        if patch_rules is None:
            raise RuntimeError("OpenSandbox SDK cannot close temporary agent-install egress rules")
        # Updating each target avoids the optional egress DELETE endpoint.
        await patch_rules([NetworkRule(action="deny", target=target) for target in INSTALL_EGRESS_TARGETS])

    async def start(self, force_build: bool) -> None:
        del force_build
        self._sandbox = await AsyncSandbox(_provider_with_runtime_secret(self._provider_config)).start(
            self._sandbox_spec()
        )
        result = await self._exec_startup(
            "mkdir -p /logs/agent /logs/verifier /logs/artifacts /tests && "
            "chmod 777 /logs/agent /logs/verifier /logs/artifacts",
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or result.stdout or "failed to initialize Pier log directories")

        await self._preinstall_agent()
        await self._close_install_egress()

        # The verifier uses a fresh task image with only the hidden tests uploaded.
        if (self.environment_dir / "test.sh").exists():
            await self.empty_dirs([self.env_paths.tests_dir], chmod=False)
            await self.upload_dir(self.environment_dir, self.env_paths.tests_dir.as_posix())
            await self._exec_startup("chmod +x /tests/test.sh", user="root")

    async def stop(self, delete: bool) -> None:
        del delete
        if self._sandbox is not None:
            await self._sandbox.stop()
            self._sandbox = None

    def _require_sandbox(self) -> AsyncSandbox:
        if self._sandbox is None:
            raise RuntimeError("OpenSandbox environment has not been started")
        return self._sandbox

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        result = await self._require_sandbox().exec(
            command,
            cwd=cwd,
            env=self._merge_env(env),
            timeout_s=timeout_sec,
            user=self._resolve_user(user),
        )
        return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.return_code)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        parent = str(Path(target_path).parent)
        await self.exec(f"mkdir -p {shlex.quote(parent)}", user="root")
        await self._require_sandbox().upload(source_path, target_path)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await self._require_sandbox().download(source_path, target)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        archive_name = f"pier-upload-{uuid.uuid4().hex}.tar.gz"
        remote_archive = f"/tmp/{archive_name}"
        with tempfile.TemporaryDirectory(prefix="pier-opensandbox-upload-") as temp_dir:
            archive = Path(temp_dir) / archive_name
            with tarfile.open(archive, "w:gz") as tar:
                for child in source.iterdir():
                    tar.add(child, arcname=child.name)
            await self.upload_file(archive, remote_archive)
            result = await self.exec(
                f"mkdir -p {shlex.quote(target_dir)} && "
                f"tar xzf {shlex.quote(remote_archive)} -C {shlex.quote(target_dir)} && "
                f"rm -f {shlex.quote(remote_archive)}",
                timeout_sec=300,
                user="root",
            )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or result.stdout or f"failed to upload {source}")

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        archive_name = f"pier-download-{uuid.uuid4().hex}.tar.gz"
        remote_archive = f"/tmp/{archive_name}"
        result = await self.exec(
            f"tar czf {shlex.quote(remote_archive)} -C {shlex.quote(source_dir)} .",
            timeout_sec=300,
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(result.stderr or result.stdout or f"failed to archive {source_dir}")
        with tempfile.TemporaryDirectory(prefix="pier-opensandbox-download-") as temp_dir:
            archive = Path(temp_dir) / archive_name
            await self.download_file(remote_archive, archive)
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(target, filter="data")
        await self.exec(f"rm -f {shlex.quote(remote_archive)}", user="root")
