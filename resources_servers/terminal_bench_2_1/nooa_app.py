# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NOOA's borrowed Terminal-Bench sessions over the canonical Gym task helpers."""

import asyncio
import logging
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from shlex import join
from tempfile import NamedTemporaryFile
from time import monotonic

from fastapi import FastAPI, HTTPException, Request
from pydantic import Field, PrivateAttr

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseVerifyRequest,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
)
from nemo_gym.episode_types import EpisodeId
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from resources_servers.terminal_bench_2_1.app import (
    _BULLSEYE_SECURITY_SNAPSHOT_SETUP,
    TEST_SH_PATCHES,
    TerminalBench21ResourcesServer,
    TerminalBench21ResourcesServerConfig,
    TerminalBench21SeedSessionRequest,
    TerminalBench21VerifyResponse,
)
from resources_servers.terminal_bench_2_1.nooa_task_metadata import CanonicalImageStartup, read_image_startup


LOG = logging.getLogger(__name__)


class NOOATerminalBenchConfig(TerminalBench21ResourcesServerConfig):
    """Bounds for NOOA-owned native task sessions."""

    session_close_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)


class NOOATerminalBenchTask(TerminalBench21SeedSessionRequest):
    """Canonical per-task verifier budget and source-bound image startup."""

    verifier_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    image_startup: CanonicalImageStartup | None = None


class NOOATerminalBenchVerifyRequest(NOOATerminalBenchTask, BaseVerifyRequest):
    """Native NOOA verification request with the original benchmark fields."""


@dataclass
class _NativeSession:
    request: ResourcesSeedSessionRequest
    response: ResourcesSeedSessionResponse | None = None
    verification_started: bool = False
    verification_key: str | None = None
    verdict: TerminalBench21VerifyResponse | None = None


class NOOATerminalBenchResourcesServer(TerminalBench21ResourcesServer):
    """Keep the borrowed task sandbox alive until NOOA's post-verification close."""

    config: NOOATerminalBenchConfig
    _native_sessions: dict[str, _NativeSession] = PrivateAttr(default_factory=dict)
    _native_session_locks: dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)
    _closed_native_sessions: dict[str, EpisodeId] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: object, /) -> None:
        super().model_post_init(context)
        if self.config.num_workers not in (None, 1):
            raise ValueError("NOOA Terminal-Bench sessions require num_workers=1")
        if self.config.is_verifying_golden_patch:
            raise ValueError("NOOA borrowed sessions do not support golden-patch mode")

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                for session_id in list(self._session_id_to_sandbox):
                    try:
                        await self._stop_session_sandbox(session_id)
                    except Exception:
                        LOG.exception("Failed to stop Terminal-Bench session %s on shutdown", session_id)

        app.router.lifespan_context = lifespan
        return app

    async def _stop_session_sandbox(self, session_id: str) -> None:
        sandbox = self._session_id_to_sandbox.get(session_id)
        if sandbox is not None:
            async with asyncio.timeout(self.config.session_close_timeout_seconds):
                await sandbox.stop()
            # Retain ownership after a failed stop so close can retry.
            self._session_id_to_sandbox.pop(session_id, None)

    async def close_resources_session(self, body: ResourcesCloseSessionRequest) -> ResourcesCloseSessionResponse:
        """Close the owner's sandbox, including when a seed response was lost."""
        session_id = body.resources_session_id
        async with self._native_session_locks.setdefault(session_id, asyncio.Lock()):
            closed_episode = self._closed_native_sessions.get(session_id)
            session = self._native_sessions.get(session_id)
            expected = closed_episode or (session.request.episode_id if session is not None else None)
            if expected is not None and expected != body.episode_id:
                raise HTTPException(409, "episode_id does not match the resources session")
            await self._stop_session_sandbox(session_id)
            self._native_sessions.pop(session_id, None)
            # Fence a delayed seed even if close arrives first.
            self._closed_native_sessions[session_id] = body.episode_id
            return ResourcesCloseSessionResponse(resources_session_id=session_id)

    async def _create_sandbox(
        self, verify_request: NOOATerminalBenchTask, *, session_id: str | None = None
    ) -> AsyncSandbox:
        entrypoint = None
        if verify_request.image_startup is not None:
            task_folder = Path(verify_request.task_folder)
            if not task_folder.is_absolute():
                task_folder = PARENT_DIR / task_folder
            expected = read_image_startup(task_folder)
            if (
                expected is None
                or expected != verify_request.image_startup
                or expected.docker_image != verify_request.docker_image
            ):
                raise ValueError("Sandbox startup metadata does not match the canonical task image and Dockerfile")
            entrypoint = expected.command
        # TODO @bxyu-nvidia: Refactor this after Hemil's swap from Python dataclass to Pydantic BaseModel
        global_config_dict = get_global_config_dict()
        resolved_sandbox_provider = resolve_provider_config(self.config.sandbox_provider, global_config_dict)
        provider_default_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config_dict)
        resources = dict(self.config.sandbox_config.get("resources", {}))

        # Derive from the final resources map (after the multilingual bump);
        # explicit sandbox_config.env keys win over the derived caps.
        sandbox_resources = SandboxResources.from_mapping(resources)
        env = dict(self.config.sandbox_config.get("env", {}))
        if self.config.sandbox_config.get("derive_cpu_env", True):
            env = cpu_cap_env(sandbox_resources.cpu) | env

        provider_options = deepcopy(self.config.sandbox_config.get("provider_options") or {})
        self._patch_sandbox_provider_options_for_instances(
            verify_request.task_name, sandbox_resources, provider_options
        )

        eval_sandbox_spec = SandboxSpec(
            image=verify_request.docker_image,
            ttl_s=self.config.sandbox_config.get("ttl_s", None),
            ready_timeout_s=self.config.sandbox_config.get("ready_timeout_s", None),
            workdir=None,  # Default to container's WORKDIR
            env=env,
            files=dict(),
            metadata=provider_default_metadata
            | self.config.sandbox_config.get("metadata", {})
            | {
                "nemo_gym_agent": self.config.name,
                "instance_id": verify_request.task_name,
            },
            resources=SandboxResources.from_mapping(resources),
            entrypoint=entrypoint,
            provider_options=provider_options,
        )
        eval_sandbox = AsyncSandbox(resolved_sandbox_provider)
        if session_id is not None:
            # Keep ownership even if setup and its first teardown attempt both fail.
            self._session_id_to_sandbox[session_id] = eval_sandbox

        async def _run_setup(sandbox: AsyncSandbox) -> None:
            result = await sandbox.exec(
                join(
                    ["bash", "-c", _BULLSEYE_SECURITY_SNAPSHOT_SETUP, "--", "/etc/os-release", "/etc/apt/sources.list"]
                ),
                timeout_s=self.config.evaluation_timeout,
            )
            if result.return_code != 0:
                raise RuntimeError(f"Failed to prepare TerminalBench package sources: {result}")

            result = await sandbox.exec("apt-get update", timeout_s=self.config.evaluation_timeout)
            if result.return_code != 0:
                print(f"Failed to apt-get update: {result}")

        # start_with_setup stops the container if _run_setup raises, instead of
        # leaving it running until its TTL.
        await eval_sandbox.start_with_setup(eval_sandbox_spec, _run_setup)

        return eval_sandbox

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        return await self._seed_native_session(request, body)

    async def _seed_native_session(
        self, request: Request, body: ResourcesSeedSessionRequest
    ) -> ResourcesSeedSessionResponse:
        if self.config.is_verifying_golden_patch:
            raise HTTPException(422, "Golden-patch mode cannot be used with agent sandbox sessions")
        task = NOOATerminalBenchTask.model_validate(body.task_data)
        if body.task_id.task_id != task.task_name:
            raise HTTPException(422, "TaskId does not match the Terminal-Bench task_name")
        task_folder = Path(task.task_folder)
        if not task_folder.is_absolute():
            task_folder = PARENT_DIR / task_folder
        if not (task_folder / "tests/test.sh").is_file():
            raise HTTPException(422, f"Missing local task verifier: {task_folder / 'tests/test.sh'}")
        session_id = body.resources_session_id
        async with self._native_session_locks.setdefault(session_id, asyncio.Lock()):
            if session_id in self._closed_native_sessions:
                raise HTTPException(409, "Resources session is already closed")
            session = self._native_sessions.get(session_id)
            if session is not None:
                if session.request != body:
                    raise HTTPException(409, "resources_session_id is already bound to a different request")
                if session.response is None or session.verification_started:
                    raise HTTPException(409, "Resources session is no longer available for seeding")
                request.session[SESSION_ID_KEY] = session_id
                return session.response
            session = _NativeSession(request=body.model_copy(deep=True))
            self._native_sessions[session_id] = session
            try:
                sandbox = await self._create_sandbox(task, session_id=session_id)
                self._session_id_to_sandbox[session_id] = sandbox
                working_directory = await sandbox.exec("pwd", timeout_s=30)
                workdir = (working_directory.stdout or "").strip()
                if working_directory.return_code != 0 or not Path(workdir).is_absolute():
                    raise RuntimeError("Could not determine the task sandbox's absolute working directory")
                session.response = ResourcesSeedSessionResponse(
                    resources_session_id=session_id,
                    sandbox_access=SandboxAccess(
                        connection=DirectSandboxConnection(
                            provider_config_ref=self.config.sandbox_provider,
                            descriptor=await sandbox.serialize(),
                        ),
                        workdir=workdir,
                    ),
                )
            except BaseException:
                try:
                    await self._stop_session_sandbox(session_id)
                except Exception:
                    LOG.exception("Failed to stop partially seeded Terminal-Bench session %s", session_id)
                raise
            request.session[SESSION_ID_KEY] = session_id
            return session.response

    async def verify(self, request: Request, body: NOOATerminalBenchVerifyRequest) -> TerminalBench21VerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        if session_id in self._closed_native_sessions:
            raise HTTPException(409, "Resources session is already closed")
        if session_id not in self._native_sessions:
            raise HTTPException(409, "No ready NOOA Terminal-Bench session")
        async with self._native_session_locks[session_id]:
            session = self._native_sessions.get(session_id)
            if session is None or session.response is None:
                raise HTTPException(409, "Resources session is not available for verification")
            key = body.model_dump_json()
            if session.verification_key is not None and session.verification_key != key:
                raise HTTPException(409, "Verification request changed for this episode")
            if session.verdict is not None:
                return session.verdict
            if session.verification_started:
                raise HTTPException(409, "Resources session is not available for verification")
            task = NOOATerminalBenchTask.model_validate(session.request.task_data)
            if task != NOOATerminalBenchTask.model_validate(body.model_dump()):
                raise HTTPException(409, "Verification task does not match the seeded task")
            session.verification_key = key
            session.verification_started = True
            sandbox = self._session_id_to_sandbox[session_id]
            timeout = body.verifier_timeout_seconds or self.config.evaluation_timeout
            start = monotonic()
            test_output = ""
            reward = 0.0
            completed = False
            try:
                # Reuse Gym's canonical test patches; no verifier bytes enter the
                # task sandbox before the NOOA environment has finished the agent.
                await self._upload_folder(
                    sandbox, Path(body.task_folder) / "tests", "/tests", TEST_SH_PATCHES, body.task_name
                )
                setup = await sandbox.exec("mkdir -p /logs/verifier", timeout_s=timeout)
                if setup.return_code != 0:
                    raise RuntimeError("Failed to prepare Terminal-Bench verifier output directory")
                result = await sandbox.exec("bash /tests/test.sh", timeout_s=timeout)
                test_output = (result.stderr or "") + (result.stdout or "")
                with NamedTemporaryFile(mode="w+", suffix=".txt") as temporary:
                    await sandbox.download("/logs/verifier/reward.txt", temporary.name)
                    reward = float(Path(temporary.name).read_text())
                if not isfinite(reward) or not 0 <= reward <= 1:
                    raise ValueError("Terminal-Bench reward must be finite and between 0 and 1")
                completed = True
            except Exception:
                LOG.exception("NOOA Terminal-Bench verification failed for %s", body.task_name)
                reward = 0.0
            # Resources owns the sandbox until close, even after cancellation or
            # verifier failure. The environment confirms NOOA cleanup first.
            session.verdict = TerminalBench21VerifyResponse(
                **body.model_dump(),
                reward=reward,
                evaluation_completed=completed,
                mask_sample=not completed,
                failure_kind=None if completed else "verifier_error",
                failure_reason=None if completed else "Terminal-Bench verification produced no valid reward",
                verification_time_taken=monotonic() - start,
                test_output=test_output,
                golden_patch_output=None,
            )
            return session.verdict


if __name__ == "__main__":
    NOOATerminalBenchResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = NOOATerminalBenchResourcesServer.run_webserver()
