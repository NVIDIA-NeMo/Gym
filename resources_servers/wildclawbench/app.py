# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources-owned native task container and original WildClaw verifier."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import ClassVar

from fastapi import FastAPI, Request
from pydantic import ConfigDict, PrivateAttr

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedRejectedError,
    ResourcesSessionConflictError,
    SimpleResourcesServer,
    lost_session_verification,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.wildclawbench_agent.execution import (
    ExecutionConfig,
    NativeProcess,
    create_lease,
    episode_output,
    image_identity,
)
from responses_api_agents.wildclawbench_agent.worker import failure_fields


class WildClawResourcesConfig(BaseResourcesServerConfig):
    execution: ExecutionConfig


class WildClawSeedRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: dict
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    wildclaw_execution_key: str


class WildClawVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class WildClawResources(SimpleResourcesServer):
    grading_may_use_uninstrumented_model: ClassVar[bool] = True
    config: WildClawResourcesConfig
    seed_request_model = WildClawSeedRequest
    _owned: dict[str, NativeProcess] = PrivateAttr(default_factory=dict)
    _image_identity: str = PrivateAttr()

    def model_post_init(self, context: object) -> None:
        super().model_post_init(context)
        self._image_identity = image_identity(self.config.execution)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                results = await asyncio.gather(
                    *(self.release_session_state(key) for key in list(self._owned)), return_exceptions=True
                )
                if any(isinstance(result, BaseException) for result in results):
                    raise RuntimeError("Native resources shutdown cleanup failed; inspect private episode logs")

        app.router.lifespan_context = lifespan
        return app

    def multiple_workers_unsupported_reason(self) -> str:
        return "native task containers and their cleanup are owned by one process"

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._owned

    async def initialize_session_state(self, session_id: str, body: WildClawSeedRequest) -> None:
        inputs = body.responses_create_params.input
        if not isinstance(inputs, list) or len(inputs) != 1:
            raise ResourcesSeedRejectedError("Use the prepared single user prompt")
        item = inputs[0].model_dump() if hasattr(inputs[0], "model_dump") else inputs[0]
        if item.get("role") != "user" or not isinstance(item.get("content"), str):
            raise ResourcesSeedRejectedError("Use the prepared single user prompt")
        episode, task = self.session_identity(session_id)
        output = episode_output(self.config.execution, body.wildclaw_execution_key)
        if (output / "lease.json").exists():
            raise ResourcesSessionConflictError("Native execution key already belongs to another resources session")
        self._owned[session_id] = NativeProcess(self.config.execution, output, self._image_identity)
        try:
            create_lease(
                self.config.execution,
                body.wildclaw_execution_key,
                body.verifier_metadata,
                item["content"],
                episode.model_dump(mode="json"),
                task.model_dump(mode="json"),
            )
        except (ValueError, OSError):
            raise ResourcesSeedRejectedError(
                "Native lease validation failed; inspect the prepared source and task assets"
            ) from None

    async def release_session_state(self, session_id: str) -> None:
        process = self._owned.get(session_id)
        if process is not None:
            await process.invoke("cleanup", 60)
            del self._owned[session_id]

    async def verify(self, request: Request, body: BaseVerifyRequest) -> WildClawVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY, "")
        async with self.verifying(session_id) as verification:
            if verification.is_lost:
                return WildClawVerifyResponse(
                    **body.model_dump(), **lost_session_verification("Native task session was lost")
                )
            if not verification.is_live:
                result = verification.retained
                if not isinstance(result, dict):
                    return WildClawVerifyResponse(
                        **body.model_dump(), **lost_session_verification("Native verification receipt was lost")
                    )
                return WildClawVerifyResponse.model_validate(dict(**body.model_dump(), **result))
            process = self._owned[session_id]
            # Retain a failure before any await; a disconnected judge is never rerun after cleanup.
            worker = json.loads((process.output / "worker.json").read_text())
            result = dict(
                **worker["metadata"],
                evaluation_protocol=worker["evaluation_protocol"],
                reward=0.0,
                native_reward=None,
                mask_sample=True,
                errors=["verification_interrupted"],
                **failure_fields(["verification_interrupted"]),
                artifact_dir=str(process.output),
            )
            verification.retain(result)
            try:
                native = await process.invoke("verify", 240)
                native.pop("events", None)
                result.clear()
                result.update(native)
            except Exception:
                result["errors"] = ["native_verification_failed"]
                result.update(failure_fields(result["errors"]))
            return WildClawVerifyResponse.model_validate(dict(**body.model_dump(), **result))


if __name__ == "__main__":
    WildClawResources.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = WildClawResources.run_webserver()
