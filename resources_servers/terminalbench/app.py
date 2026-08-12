# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.server_utils import SESSION_ID_KEY


class TerminalBenchResourcesServerConfig(BaseResourcesServerConfig):
    sandbox_provider: str
    sandbox_config: Dict[str, Any] = Field(default_factory=dict)
    evaluation_timeout: int = 300


class TerminalBenchSeedSessionRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class TerminalBenchSeedSessionResponse(BaseSeedSessionResponse):
    sandbox_handle: Dict[str, Any]


class TerminalBenchVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class TerminalBenchVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    resolved: bool
    evaluation_timed_out: bool = False
    sandbox_failed: bool = False
    mask_sample: bool = False
    evaluation_time: float = 0.0
    test_stdout: str = ""


class TerminalBenchResourcesServer(SimpleResourcesServer):
    config: TerminalBenchResourcesServerConfig

    _sandboxes: Dict[str, AsyncSandbox] = PrivateAttr(default_factory=dict)
    _metadata: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)

    @staticmethod
    def _request_metadata(body: BaseRunRequest | BaseVerifyRequest) -> Dict[str, Any]:
        return dict(body.responses_create_params.metadata or {})

    def _sandbox_spec(self, metadata: Dict[str, Any]) -> SandboxSpec:
        config = self.config.sandbox_config
        resources = dict(config.get("resources") or {})
        if metadata.get("cpus") is not None:
            resources["cpu"] = float(metadata["cpus"])
        if metadata.get("memory_mb") is not None:
            resources["memory_mib"] = int(float(metadata["memory_mb"]))
        if metadata.get("storage_mb") is not None:
            storage_mb = int(float(metadata["storage_mb"]))
            resources["disk_gib"] = max(1, (storage_mb + 1023) // 1024)
        if metadata.get("gpus") is not None:
            resources["gpu"] = int(float(metadata["gpus"]))

        return SandboxSpec(
            image=str(metadata.get("docker_image") or "ubuntu:22.04").removeprefix("docker://"),
            ttl_s=config.get("ttl_s"),
            ready_timeout_s=config.get("ready_timeout_s"),
            workdir=metadata.get("workdir"),
            metadata=resolve_provider_metadata(self.config.sandbox_provider, get_global_config_dict())
            | dict(config.get("metadata") or {})
            | {
                "benchmark": "terminalbench",
                "instance_id": str(metadata.get("instance_id") or "unknown")[:63],
            },
            resources=SandboxResources.from_mapping(resources),
            provider_options=dict(config.get("provider_options") or {}),
        )

    async def seed_session(
        self,
        request: Request,
        body: TerminalBenchSeedSessionRequest,
    ) -> TerminalBenchSeedSessionResponse:
        metadata = self._request_metadata(body)
        provider = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
        sandbox = AsyncSandbox(provider, self._sandbox_spec(metadata))
        await sandbox.start()
        session_id = request.session[SESSION_ID_KEY]
        self._sandboxes[session_id] = sandbox
        self._metadata[session_id] = metadata
        return TerminalBenchSeedSessionResponse(sandbox_handle=await sandbox.serialize(scope="operate"))

    @staticmethod
    def _archive_tests(task_dir: Path) -> Path:
        tests = task_dir / "tests"
        if not tests.is_dir():
            raise FileNotFoundError(f"TerminalBench tests directory does not exist: {tests}")
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as stream:
            archive = Path(stream.name)
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(tests, arcname=".")
        return archive

    async def verify(
        self,
        request: Request,
        body: TerminalBenchVerifyRequest,
    ) -> TerminalBenchVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        sandbox = self._sandboxes.pop(session_id)
        metadata = self._metadata.pop(session_id, self._request_metadata(body))
        started = time.time()
        timed_out = False
        sandbox_failed = False
        stdout = ""
        resolved = False
        archive: Optional[Path] = None
        try:
            task_dir = Path(str(metadata["task_dir"])).expanduser().resolve()
            archive = self._archive_tests(task_dir)
            await sandbox.upload(archive, "/tmp/terminalbench-tests.tar.gz")
            setup = await sandbox.exec(
                "rm -rf /tests && mkdir -p /tests /logs/verifier && "
                "tar -xzf /tmp/terminalbench-tests.tar.gz -C /tests && "
                "printf '[pytest]\\naddopts =\\n' > /pytest.ini",
                timeout_s=300,
                user="root",
            )
            if setup.return_code != 0:
                raise RuntimeError(setup.stderr or "failed to stage TerminalBench tests")

            timeout = int(float(metadata.get("verifier_timeout_sec") or self.config.evaluation_timeout))
            result = await sandbox.exec(
                "bash /tests/test.sh > /logs/verifier/test-stdout.txt 2>&1",
                timeout_s=timeout,
                user="root",
            )
            timed_out = result.error_type == "timeout"
            sandbox_failed = result.error_type == "sandbox"
            output = await sandbox.exec(
                "cat /logs/verifier/test-stdout.txt 2>/dev/null || true",
                timeout_s=30,
                user="root",
            )
            stdout = output.stdout or ""
            reward = await sandbox.exec(
                "cat /logs/verifier/reward.txt 2>/dev/null || true",
                timeout_s=30,
                user="root",
            )
            try:
                resolved = float((reward.stdout or "").strip()) > 0
            except ValueError:
                resolved = False
        except Exception:
            sandbox_failed = True
        finally:
            if archive is not None:
                archive.unlink(missing_ok=True)
            try:
                await sandbox.stop()
            except Exception:
                sandbox_failed = True

        return TerminalBenchVerifyResponse(
            **body.model_dump(),
            reward=1.0 if resolved else 0.0,
            resolved=resolved,
            evaluation_timed_out=timed_out,
            sandbox_failed=sandbox_failed,
            mask_sample=timed_out or sandbox_failed,
            evaluation_time=time.time() - started,
            test_stdout=stdout,
        )


if __name__ == "__main__":
    TerminalBenchResourcesServer.run_webserver()
