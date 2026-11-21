# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lend the original AGA task its image workspace and grade the final answer with the original graders."""

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

from fastapi import FastAPI, Request
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.client import SESSION_ID_KEY, get_response_json, raise_for_status
from nemo_gym.core import TaskId
from nemo_gym.sandbox import SandboxAccess, SandboxConfig
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    LentSandboxes,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ReverifyMode,
    SimpleResourcesServer,
    lost_session_verification,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.aga_visual_tasks.final_answer import final_answer


class Config(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    sandbox_provider: str
    sandbox_image: str | None
    sandbox_config: SandboxConfig = Field(default_factory=lambda: SandboxConfig(ttl_s=3600))
    image_root: str
    model_server: str = "policy_model"
    resources_server_by_agent: dict[str, str]


class Seed(BaseSeedSessionRequest):
    source_images: dict[str, str] = Field(default_factory=dict, description="Original /images path to SHA256")


class Verify(BaseVerifyRequest):
    # This bridge delegates to different original typed graders. Preserve their
    # private inputs and optional selectors without inventing shared defaults.
    model_config = ConfigDict(extra="allow")

    image_tools_base_agent_ref: dict[str, str]


class Result(BaseVerifyResponse):
    pass


class ResourcesServer(SimpleResourcesServer):
    config: Config
    seed_request_model = Seed
    _sandboxes: LentSandboxes = PrivateAttr()
    _access: dict[str, SandboxAccess] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._sandboxes = LentSandboxes(
            self.config.sandbox_provider,
            self.config.sandbox_config,
            image=self.config.sandbox_image,
            workdir="/workspace",
            network_access="model_only",
            model_server=self.config.model_server,
        )

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._access

    async def initialize_session_state(self, session_id: str, body: Seed) -> None:
        root = Path(self.config.image_root).resolve()
        files = {}
        for destination, digest in body.source_images.items():
            relative = PurePosixPath(destination).relative_to("/images")
            source = (root / str(relative)).resolve()
            if not source.is_relative_to(root):
                raise ValueError("Image path escapes the original image root")
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("Original AGA image identity mismatch")
            files[destination] = data
        self._access[session_id] = await self._sandboxes.lend(
            session_id,
            TaskId(taskset="aga_visual_tasks", task_id=session_id),
            files=files,
        )

    async def release_session_state(self, session_id: str) -> None:
        await self._sandboxes.release(session_id)
        self._access.pop(session_id, None)

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        reply = await super().seed_session(body, request=request)
        async with self.session_lock(body.resources_session_id):
            if self.session_is_closed(body.resources_session_id) or self.session_is_verified(
                body.resources_session_id
            ):
                raise ResourcesSessionConflictError("AGA resources session is closed or verified")
            return reply.model_copy(update={"sandbox_access": self._access[body.resources_session_id]})

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[Any]:
            try:
                async with parent(app) as state:
                    yield state
            finally:
                await self._sandboxes.release_all()

        app.router.lifespan_context = lifespan
        return app

    async def verify(self, request: Request, body: Verify) -> Result:
        async with self.verifying(request.session[SESSION_ID_KEY], allow_missing=True) as verification:
            if verification.is_lost:
                return Result(**(body.model_dump() | lost_session_verification("AGA session closed")))
            # All scoring inputs are retained in this request; no mutable policy files are graded.
            verification.retain(None)
            try:
                resource = self.config.resources_server_by_agent[body.image_tools_base_agent_ref["name"]]
                scoring_body = body.model_dump(mode="json")
                # Exactly the original scoring-only extraction; never replace the saved native response.
                scoring_body["response"]["output"] = [
                    {
                        "type": "message",
                        "id": "final",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "annotations": [],
                                "text": final_answer(scoring_body["response"]["output"], strip_reasoning=True),
                            }
                        ],
                    }
                ]
                reply = await self.server_client.post(server_name=resource, url_path="/verify", json=scoring_body)
                await raise_for_status(reply)
                scored = await get_response_json(reply)
                if scored.get("mask_sample"):
                    return Result(
                        **(
                            body.model_dump()
                            | {
                                "reward": 0.0,
                                "mask_sample": True,
                                "failure_kind": "verifier_error",
                                "failure_reason": "Original base verifier was inconclusive",
                            }
                        )
                    )
                return Result(**(body.model_dump() | {"reward": float(scored["reward"])}))
            except Exception as error:
                return Result(
                    **(
                        body.model_dump()
                        | {
                            "reward": 0.0,
                            "mask_sample": True,
                            "failure_kind": "verifier_error",
                            "failure_reason": "AGA verification failed: " + type(error).__name__,
                        }
                    )
                )


if __name__ == "__main__":
    ResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ResourcesServer.run_webserver()
