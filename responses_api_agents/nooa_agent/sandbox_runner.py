# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side launcher for a strictly isolated NOOA sandbox runtime."""

from __future__ import annotations

import asyncio
import json
import shlex
import tempfile
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig, NOOASandboxRuntimeConfig
from responses_api_agents.nooa_agent.runner import NOOARunFailure, NOOARunRequest, NOOARunResult
from responses_api_agents.nooa_agent.sandbox_protocol import SandboxRunArtifact, SandboxRunRequest


class SandboxedNOOARunner:
    """Create one sandbox and execute the complete NOOA runtime inside it."""

    def __init__(
        self,
        *,
        invocation: NOOAInvocationConfig,
        runtime: NOOASandboxRuntimeConfig,
        global_config: Any,
        model_server_name: str,
        resources_server_name: str,
        model_base_url: str,
        resources_base_url: str,
        max_steps: int,
        default_timeout_secs: float,
    ) -> None:
        self._invocation = invocation
        self._runtime = runtime
        self._provider = resolve_provider_config(runtime.provider, global_config)
        self._provider_metadata = resolve_provider_metadata(runtime.provider, global_config)
        self._model_server_name = model_server_name
        self._resources_server_name = resources_server_name
        self._model_base_url = (runtime.model_base_url or model_base_url).rstrip("/")
        self._resources_base_url = (runtime.resources_base_url or resources_base_url).rstrip("/")
        self._max_steps = max_steps
        self._default_timeout_secs = default_timeout_secs
        self._runtime_archive = Path(runtime.runtime_archive).expanduser() if runtime.runtime_archive else None
        if self._runtime_archive is not None and not self._runtime_archive.is_file():
            raise ValueError(f"NOOA sandbox runtime archive not found: {self._runtime_archive}")

    def _spec(self, request: NOOARunRequest) -> SandboxSpec:
        values = dict(self._runtime.spec)
        resources = SandboxResources.from_mapping(values.pop("resources", {}))
        metadata = {
            **self._provider_metadata,
            **values.pop("metadata", {}),
            "nemo_gym_agent": "nooa_agent",
            "task_id": request.task_id[:63],
            "rollout_id": request.rollout_id[:63],
        }
        known = {
            "image",
            "ttl_s",
            "ready_timeout_s",
            "workdir",
            "env",
            "files",
            "entrypoint",
            "provider_options",
            "ports",
        }
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown NOOA sandbox spec keys: {sorted(unknown)}")
        return SandboxSpec(resources=resources, metadata=metadata, **values)

    async def run(self, request: NOOARunRequest) -> NOOARunResult:
        descriptor = request.sandbox_descriptor
        if self._runtime.source == "seeded" and descriptor is None:
            raise ValueError("NOOA sandbox runtime requires seed_session to return sandbox_descriptor")
        if self._runtime.source == "create":
            descriptor = None
        if descriptor is not None:
            sandbox = await AsyncSandbox.connect(descriptor, provider=create_provider(self._provider))
        else:
            sandbox = AsyncSandbox(self._provider, self._spec(request))
        try:
            if descriptor is None:
                await sandbox.start()
            result = await self._run_worker(sandbox, request)
        except BaseException:
            await asyncio.shield(sandbox.stop())
            raise
        result.cleanup = sandbox.stop
        return result

    async def _run_worker(self, sandbox: AsyncSandbox, request: NOOARunRequest) -> NOOARunResult:
        run_dir = self._runtime.run_dir.rstrip("/")
        remote_request = f"{run_dir}/request.json"
        remote_result = f"{run_dir}/result.json"
        remote_checkpoint = f"{run_dir}/checkpoint.json"
        payload = SandboxRunRequest(
            row=request.row.model_dump(mode="json"),
            invocation=self._invocation,
            endpoints={
                self._model_server_name: self._model_base_url,
                self._resources_server_name: self._resources_base_url,
            },
            model_server_name=self._model_server_name,
            resources_server_name=self._resources_server_name,
            model_url_path=request.model_url_path,
            model_cookies=request.model_cookies,
            resource_cookies=request.resource_cookies,
            max_steps=self._max_steps,
            task_id=request.task_id,
            rollout_id=request.rollout_id,
        )

        with tempfile.TemporaryDirectory(prefix="nemo-gym-nooa-sandbox-") as temporary_dir:
            temporary = Path(temporary_dir)
            local_request = temporary / "request.json"
            local_artifact = temporary / "artifact.json"
            local_request.write_text(payload.model_dump_json(), encoding="utf-8")
            await sandbox.exec(f"mkdir -p {shlex.quote(run_dir)}", timeout_s=60)
            await sandbox.upload(local_request, remote_request)

            setup = ""
            env_prefix = ""
            if self._runtime_archive is not None:
                remote_archive = f"{run_dir}/runtime.tar.gz"
                await sandbox.upload(self._runtime_archive, remote_archive)
                setup = (
                    f"mkdir -p {shlex.quote(self._runtime.runtime_dir)} && "
                    f"tar -xzf {shlex.quote(remote_archive)} -C {shlex.quote(self._runtime.runtime_dir)} && "
                )
                env_prefix = f"PYTHONPATH={shlex.quote(self._runtime.runtime_dir)}:${{PYTHONPATH:-}} "

            command = (
                setup + env_prefix + f"{shlex.quote(self._runtime.python_executable)} "
                "-m responses_api_agents.nooa_agent.sandbox_worker "
                f"--request {shlex.quote(remote_request)} "
                f"--result {shlex.quote(remote_result)} "
                f"--checkpoint {shlex.quote(remote_checkpoint)}"
            )
            try:
                execution = await sandbox.exec(
                    command,
                    timeout_s=self._runtime.exec_timeout_secs or self._default_timeout_secs,
                )
            except asyncio.CancelledError as error:
                artifact = await asyncio.shield(
                    self._download_artifact(
                        sandbox,
                        preferred=remote_result,
                        fallback=remote_checkpoint,
                        local_path=local_artifact,
                    )
                )
                if artifact is not None:
                    partial = artifact.to_result()
                    if partial is not None:
                        error.nooa_result = partial
                raise

            artifact = await self._download_artifact(
                sandbox,
                preferred=remote_result,
                fallback=remote_checkpoint,
                local_path=local_artifact,
            )
            if artifact is None:
                detail = execution.stderr or execution.stdout or execution.error_type or "worker produced no artifact"
                raise RuntimeError(f"Sandboxed NOOA worker failed: {detail}")

            result = artifact.to_result()
            if result is None:
                raise RuntimeError(
                    f"Sandboxed NOOA worker failed with {artifact.error_type or 'unknown error'}: {artifact.error or ''}"
                )
            if artifact.error is not None:
                raise NOOARunFailure(RuntimeError(f"{artifact.error_type}: {artifact.error}"), result)
            if execution.return_code != 0:
                raise NOOARunFailure(
                    RuntimeError(execution.stderr or execution.stdout or f"worker exited {execution.return_code}"),
                    result,
                )
            return result

    @staticmethod
    async def _download_artifact(
        sandbox: AsyncSandbox,
        *,
        preferred: str,
        fallback: str,
        local_path: Path,
    ) -> SandboxRunArtifact | None:
        for remote_path in (preferred, fallback):
            exists = await sandbox.exec(f"test -f {shlex.quote(remote_path)}", timeout_s=30)
            if exists.return_code != 0:
                continue
            await sandbox.download(remote_path, local_path)
            try:
                return SandboxRunArtifact.model_validate(json.loads(local_path.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                continue
        return None
