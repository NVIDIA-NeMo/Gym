# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same-sandbox SWE execution; held-out assets are injected only at verification."""

import asyncio
import re
import shlex
from dataclasses import asdict, dataclass
from time import monotonic
from typing import Any, ClassVar

from fastapi import HTTPException, Request
from pydantic import Field, model_validator

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec
from nemo_gym.sandbox.config import resolve_provider_config, resolve_provider_metadata
from nemo_gym.sandbox.utils import cpu_cap_env
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from resources_servers.swe_external1.task_data import TaskMetadata, TaskRow
from resources_servers.swe_external1.verification import VerificationResult, require_success, run_verification


class SweExternal1ResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.UNSUPPORTED
    sandbox_provider: str
    sandbox_config: dict[str, Any] = Field(default_factory=dict)
    is_verifying_golden_patch: bool = False
    evaluation_timeout: int = Field(default=3600, gt=0)
    max_concurrency: int = Field(default=8, gt=0)
    num_workers: int = 1

    @model_validator(mode="after")
    def one_stateful_worker(self):
        if self.num_workers != 1:
            raise ValueError("session state requires num_workers=1; use async concurrency within the worker")
        return self


class SweExternal1SeedRequest(TaskRow, BaseSeedSessionRequest):
    pass


class SweExternal1SeedResponse(BaseSeedSessionResponse):
    sandbox_handle: str | None = None
    sandbox_descriptor: dict[str, Any] | None = None


class SweExternal1VerifyRequest(TaskRow, BaseVerifyRequest):
    pass


class SweExternal1VerifyResponse(BaseVerifyResponse):
    evaluation_completed: bool
    task_id: str
    test_output: str
    solution_output: str
    exit_code: int | None
    error: str | None
    cleanup_error: str | None = None
    verification_time_taken: float


@dataclass
class Session:
    sandbox: AsyncSandbox
    fingerprint: str
    expiry: asyncio.Task | None = None


class SweExternal1ResourcesServer(SimpleResourcesServer):
    config: SweExternal1ResourcesServerConfig

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._sessions: dict[str, Session] = {}
        self._busy: set[str] = set()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

    def _ttl(self, task: TaskMetadata) -> int:
        return int(self.config.sandbox_config.get("ttl_s", task.agent_timeout_s + task.verifier_timeout_s + 1800))

    async def _stop(self, session: Session | None) -> str | None:
        if session is None:
            return None
        if session.expiry is not None and session.expiry is not asyncio.current_task():
            session.expiry.cancel()
        try:
            await session.sandbox.stop()
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        return None

    async def _expire(self, key: str, session: Session, ttl_s: int) -> None:
        await asyncio.sleep(ttl_s)
        if self._sessions.get(key) is session:
            self._sessions.pop(key)
            await self._stop(session)

    async def _create(self, task: TaskMetadata) -> AsyncSandbox:
        global_config = get_global_config_dict()
        config = self.config.sandbox_config
        resources = SandboxResources(cpu=task.cpu, memory_mib=task.memory_mib, disk_gib=task.disk_gib)
        sandbox = AsyncSandbox(resolve_provider_config(self.config.sandbox_provider, global_config))
        try:
            await sandbox.start(
                SandboxSpec(
                    image=task.image_ref,
                    workdir=task.workdir,
                    ttl_s=self._ttl(task),
                    ready_timeout_s=config.get("ready_timeout_s", 1200),
                    resources=resources,
                    env=cpu_cap_env(task.cpu) | config.get("env", {}),
                    provider_options=config.get("provider_options", {}),
                    metadata=resolve_provider_metadata(self.config.sandbox_provider, global_config)
                    | {
                        "benchmark": "swe_external1",
                        "instance_id": re.sub(r"[^a-zA-Z0-9_.-]", "-", task.task_id)[:63],
                    },
                )
            )
            if task.setup_script:
                setup = await sandbox.exec(
                    f"bash -lc {shlex.quote(task.setup_script)}",
                    cwd=task.workdir,
                    timeout_s=self.config.evaluation_timeout,
                )
                require_success(setup, "task setup")
        except BaseException:
            await self._stop(Session(sandbox, ""))
            raise
        return sandbox

    async def seed_session(self, request: Request, body: SweExternal1SeedRequest) -> SweExternal1SeedResponse:
        if self.config.is_verifying_golden_patch:
            return SweExternal1SeedResponse()
        key = request.session[SESSION_ID_KEY]
        if key in self._busy:
            raise HTTPException(409, "session already has an operation in progress")
        self._busy.add(key)
        session = None
        try:
            async with self._semaphore:
                await self._stop(self._sessions.pop(key, None))
                task = body.verifier_metadata
                sandbox = await self._create(task)
                session = Session(sandbox, task.fingerprint())
                descriptor = dict(await sandbox.serialize())
                self._sessions[key] = session
                session.expiry = asyncio.create_task(self._expire(key, session, self._ttl(task)))
                return SweExternal1SeedResponse(
                    sandbox_handle=str(descriptor["sandbox_id"]), sandbox_descriptor=descriptor
                )
        except BaseException:
            self._sessions.pop(key, None)
            await self._stop(session)
            raise
        finally:
            self._busy.discard(key)

    async def verify(self, request: Request, body: SweExternal1VerifyRequest) -> SweExternal1VerifyResponse:
        key = request.session[SESSION_ID_KEY]
        if key in self._busy:
            raise HTTPException(409, "session already has an operation in progress")
        self._busy.add(key)
        started = monotonic()
        session = None
        cleanup_error = None
        result = VerificationResult()
        try:
            async with self._semaphore:
                task = body.verifier_metadata
                if self.config.is_verifying_golden_patch:
                    session = Session(await self._create(task), task.fingerprint())
                else:
                    session = self._sessions.pop(key, None)
                    if session is None:
                        raise ValueError("no active task session; seed_session must run before verify")
                    if session.expiry is not None:
                        session.expiry.cancel()
                    if session.fingerprint != task.fingerprint():
                        raise ValueError("verification task differs from the seeded task")
                result = await run_verification(
                    session.sandbox,
                    task,
                    golden=self.config.is_verifying_golden_patch,
                    timeout_cap_s=self.config.evaluation_timeout,
                )
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            cleanup_error = await self._stop(session)
            self._busy.discard(key)
        return SweExternal1VerifyResponse.model_validate(
            body.model_dump()
            | asdict(result)
            | {
                "task_id": body.verifier_metadata.task_id,
                "cleanup_error": cleanup_error,
                "failure_reason": result.error,
                "verification_time_taken": monotonic() - started,
            }
        )


if __name__ == "__main__":
    SweExternal1ResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SweExternal1ResourcesServer.run_webserver()
