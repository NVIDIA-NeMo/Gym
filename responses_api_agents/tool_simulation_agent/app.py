# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass

from fastapi import Body, Request
from pydantic import ConfigDict, JsonValue, PrivateAttr, ValidationError

from nemo_gym._checkpoint.agent import ActivationOutOfOrderError, LegacyRun, RestoredAgentSession, require_rollout
from nemo_gym._checkpoint.steps import CHECKPOINT_CLOSE_FOLLOWS_HEADER, StepType, seed_restarts, seed_verify_step_type
from nemo_gym.base_resources_server import (
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
)
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import EpisodeId, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming


LOG = logging.getLogger(__name__)
_LEGACY_SESSION_PREFIX = "run:"


def _cookies(values: Mapping[str, object]) -> dict[str, str]:
    return {str(key): str(getattr(value, "value", value)) for key, value in values.items()}


class ToolSimulationAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef


class ToolSimulationAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class ToolSimulationAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class ToolSimulationAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


@dataclass
class _ModelReply:
    completed: bool = False
    response: NeMoGymResponse | None = None


class ToolSimulationAgent(SimpleResponsesAPIAgent):
    config: ToolSimulationAgentConfig
    checkpoint_mode = "exported"
    # A completed self-dispatch may still be on its way to /run when a checkpoint
    # exports it. Keep its reply until /run records the verification boundary.
    _model_replies: dict[str, _ModelReply] = PrivateAttr(default_factory=dict)

    async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
        states = {}
        for key in session_keys:
            reply = self._model_replies.get(key, _ModelReply())
            states[key] = {
                "model_completed": reply.completed,
                **({"response": reply.response.model_dump(mode="json")} if reply.response is not None else {}),
            }
        return states

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        # Validate every reply before installing any of them.
        restored = {
            session.session_key: _ModelReply(
                completed=session.session.get("model_completed", False),
                response=(
                    NeMoGymResponse.model_validate(session.session["response"])
                    if "response" in session.session
                    else None
                ),
            )
            for session in sessions
        }
        self._model_replies.update(restored)

    async def retire_agent_session(self, session_key: str) -> None:
        self._model_replies.pop(session_key, None)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        participant = self.checkpoint_participant
        if participant is None:
            return await self._responses(request, body)
        path_params = getattr(request, "path_params", None)
        capture_key = path_params.get("rollout_id") if isinstance(path_params, Mapping) else None
        episode_id = EpisodeId.from_capture_key(require_rollout(capture_key))
        session_key = f"{_LEGACY_SESSION_PREFIX}{episode_id.rollout_id}"
        if not participant.has_session(session_key):
            # Direct /v1/responses has no /run boundary to restore.
            async with participant.restart_call(capture_key):
                return await self._responses(request, body)
        participant.check_session(session_key, episode_id)
        # Capture the session's object before awaiting: a late reply may update
        # this object after /run removes it, but cannot recreate the cache entry.
        model_reply = self._model_replies.setdefault(session_key, _ModelReply())
        while True:
            if model_reply.response is not None:
                participant.answered_from_saved_reply(session_key, episode_id)
                return model_reply.response
            if model_reply.completed:
                raise ActivationOutOfOrderError("tool-simulation model reply was already consumed by /run")
            if not await participant.wait_activation(session_key, 1):
                break
        async with participant.activation(session_key, episode_id, index=1) as activation:
            # Reissuing this call after a restore lets the model participant
            # install its generation cut. A late reply remains parked here.
            await activation.boundary(lambda: {})
            async with activation.awaiting_model():
                reply = await self._responses(request, body)
            model_reply.response = reply
            model_reply.completed = True
            return reply

    async def _responses(self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=self.url_path_for_request("/v1/responses", request),
            json=body,
        )

        # Model calls are expected to always succeed.
        await raise_for_status(model_response)
        model_response_json = await model_response.json()

        try:
            return NeMoGymResponse.model_validate(model_response_json)
        except ValidationError as e:
            raise RuntimeError(
                f"Received an invalid response from the model server: {json.dumps(model_response_json)}"
            ) from e

    async def run(self, body: ToolSimulationAgentRunRequest = Body()) -> ToolSimulationAgentVerifyResponse:
        participant = self.checkpoint_participant
        if participant is None:
            return await self._run(body, legacy_run=None)
        episode_id = EpisodeId.from_capture_key(require_rollout(self.rollout_id_from_run(body)))
        session_key = f"{_LEGACY_SESSION_PREFIX}{episode_id.rollout_id}"
        async with participant.legacy_run(session_key, episode_id) as legacy_run:
            model_reply = self._model_replies.setdefault(session_key, _ModelReply())
            try:
                return await self._run(body, legacy_run=legacy_run)
            finally:
                if self._model_replies.get(session_key) is model_reply:
                    self._model_replies.pop(session_key)

    async def _run(
        self, body: ToolSimulationAgentRunRequest, *, legacy_run: LegacyRun | None
    ) -> ToolSimulationAgentVerifyResponse:
        config = self.config
        continuation = (legacy_run.continuation if legacy_run else None) or {}
        stage = continuation.get("next", "seed" if legacy_run and not config.skip_verification else "model")
        if stage not in {"seed", "model", "verify", "return"}:
            raise ValueError(f"unknown tool-simulation checkpoint stage {stage!r}")
        cookies = continuation.get("cookies", {})
        resources_verify_step_type: StepType = continuation.get("resources_verify_step_type", "wait")
        model_reply = self._model_replies[legacy_run.key] if legacy_run is not None else None

        @asynccontextmanager
        async def step(step_type: StepType) -> AsyncIterator[None]:
            async with legacy_run.step(step_type) if legacy_run else nullcontext():
                try:
                    yield
                except BaseException:
                    # A failed replay step can park before propagating its error,
                    # exporting a null boundary that restarts from input. Clear
                    # the completion marker before that snapshot can be taken.
                    if model_reply is not None:
                        model_reply.completed = False
                        model_reply.response = None
                    raise

        async def boundary(state: dict[str, JsonValue]) -> None:
            if legacy_run is not None:
                if state["next"] in {"verify", "return"}:
                    # No await between removing the cached reply and recording
                    # the boundary that now owns it.
                    model_reply.response = None
                await legacy_run.boundary(state)

        if stage == "seed":
            await boundary({"next": "seed"})
            async with step("replay"):
                seed = await self.server_client.post(
                    server_name=config.resources_server.name, url_path="/seed_session", json=body.model_dump()
                )
                await raise_for_status(seed)
            cookies = _cookies(seed.cookies)
            resources_verify_step_type = seed_verify_step_type(seed.headers)
            if seed_restarts(seed.headers):
                await legacy_run.mark_restart()
            stage = "model"

        if stage == "model":
            await boundary(
                {"next": "model", "cookies": cookies, "resources_verify_step_type": resources_verify_step_type}
            )
            async with step("replay"):
                response = await self.server_client.post(
                    server_name=config.name,
                    url_path=self.url_path_for_run("/v1/responses", body),
                    json=body.responses_create_params,
                )
                await raise_for_status(response)
                response_json = await get_response_json(response)
        else:
            response_json = continuation["response"]

        if stage == "return":
            result = continuation["result"]
        elif config.skip_verification:
            result = body.model_dump() | {
                "response": response_json,
                "reward": float(config.skip_verification_reward),
                "verification_skipped": True,
            }
        else:
            await boundary(
                {
                    "next": "verify",
                    "response": response_json,
                    "cookies": cookies,
                    "resources_verify_step_type": resources_verify_step_type,
                }
            )
            verify_request = ToolSimulationAgentVerifyRequest.model_validate(
                body.model_dump() | {"response": response_json}
            )
            kwargs = (
                {
                    "cookies": cookies,
                    "headers": {CHECKPOINT_CLOSE_FOLLOWS_HEADER: "1"}
                    if resources_verify_step_type == "replay"
                    else {},
                }
                if legacy_run
                else {}
            )
            async with step(resources_verify_step_type):
                verify_response = await self.server_client.post(
                    server_name=config.resources_server.name,
                    url_path="/verify",
                    json=verify_request.model_dump(),
                    **kwargs,
                )
                await raise_for_status(verify_response)
                result = await get_response_json(verify_response)

        await boundary(
            {
                "next": "return",
                "response": response_json,
                "result": result,
                "cookies": cookies,
                "resources_verify_step_type": resources_verify_step_type,
            }
        )
        if legacy_run is not None and resources_verify_step_type == "replay" and not config.skip_verification:
            episode_id = EpisodeId.from_capture_key(require_rollout(self.rollout_id_from_run(body)))
            try:
                close = await self.server_client.post(
                    server_name=config.resources_server.name,
                    url_path="/close_session",
                    json=ResourcesCloseSessionRequest(
                        resources_session_id=f"{_LEGACY_SESSION_PREFIX}{episode_id.capture_key}", episode_id=episode_id
                    ).model_dump(mode="json"),
                    cookies=cookies,
                )
                await raise_for_status(close)
            except Exception:
                LOG.warning("could not close resources session of %s", episode_id.capture_key, exc_info=True)

        return ToolSimulationAgentVerifyResponse.model_validate(result)


if __name__ == "__main__":
    ToolSimulationAgent.run_webserver()
