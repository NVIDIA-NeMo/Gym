# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox ownership and replayable patch verification shared by the native SWE datasets."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import ClassVar

from fastapi import FastAPI, Request
from pydantic import BaseModel, PrivateAttr, computed_field

from nemo_gym.base_resources_server import BaseVerifyRequest, BaseVerifyResponse
from nemo_gym.judge import judge_failsafe
from nemo_gym.sandbox.owner import SandboxOwnerConfig, SandboxOwnerResourcesServer
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.telemetry.endpoints import traced_verify_endpoint
from resources_servers.swebench.patch_capture import PatchCapture


class NativeSWESandboxConfig(SandboxOwnerConfig):
    is_verifying_golden_patch: bool = False


class NativePreparedPatch(BaseModel):
    """The complete patch-capture result; grading this needs no live agent sandbox."""

    capture: PatchCapture
    extraction_error: str | None = None

    @computed_field
    @property
    def model_patch(self) -> str:
        """Patch text shared with harnesses that also support the generic SWE-bench verifier."""
        return self.capture.patch


class NativeSWESandboxResourcesServer(SandboxOwnerResourcesServer):
    """Common sandbox lifecycle for native SWE servers with typed verification routes."""

    config: NativeSWESandboxConfig
    checkpoint_replayable_paths: ClassVar[tuple[str, ...]] = ("/verify_prepared",)
    _prepared_patches: dict[str, NativePreparedPatch] = PrivateAttr(default_factory=dict)
    _prepare_locks: dict[str, asyncio.Lock] = PrivateAttr(default_factory=dict)

    def lends_sandboxes(self) -> bool:
        return not self.config.is_verifying_golden_patch

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        # Subclasses keep their dataset's typed request and response models on both routes.
        app.post("/prepare_verify")(self.prepare_verify)
        app.post("/verify_prepared")(
            traced_verify_endpoint(
                judge_failsafe(self.verify_prepared),
                static_attributes={"nemo.gym.server.name": self.config.name},
            )
        )
        return app

    async def prepare_verification(
        self, request: Request, capture: Callable[[], Awaitable[NativePreparedPatch]]
    ) -> NativePreparedPatch:
        """Capture once, end the sandbox session, and return the data the agent bookmarks."""
        session_id = request.session[SESSION_ID_KEY]
        async with self._prepare_locks.setdefault(session_id, asyncio.Lock()):
            if session_id not in self._prepared_patches:
                self._prepared_patches[session_id] = await capture()
            await self.end_session_sandbox(session_id)
            prepared = self._prepared_patches[session_id]
        self.checkpoint_session_ended(request)
        return prepared

    def forget_prepared_verification(self, request: Request) -> None:
        """The caller now owns the captured patch, including across retries and restores."""
        session_id = request.session[SESSION_ID_KEY]
        self._prepared_patches.pop(session_id, None)
        self._prepare_locks.pop(session_id, None)

    async def prepare_verify(self, request: Request, body: BaseVerifyRequest) -> NativePreparedPatch:
        """Subclasses expose this route with their concrete dataset request type."""
        raise NotImplementedError

    async def verify_prepared(self, request: Request, body: BaseVerifyRequest) -> BaseVerifyResponse:
        """Subclasses expose this route with their concrete prepared request and verdict types."""
        raise NotImplementedError
