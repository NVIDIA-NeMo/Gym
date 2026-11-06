# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Optional

from fastapi import HTTPException, Request, Response
from PIL import Image
from pydantic import ConfigDict, PrivateAttr

from nemo_gym.agents import (
    ActivationConflictError,
    AgentCloseSessionResponse,
    AgentSeedRejectedError,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    Body,
    OrderedOperationLedger,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    accumulate_response_usage,
)
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from resources_servers.image_tools import (
    IMAGE_TOOLS_ROLLOUT_KEY,
    ImageToolsEpisode,
    ImageToolsGymToolLogic,
    ImageToolsSettings,
    assistant_text,
    extract_image_refs,
    final_assistant_response,
)


# Bound on stopping a session's tool loop that is still running when the session closes.
_SESSION_CLOSE_TIMEOUT_SECS = 30.0


class ImageToolsAgentConfig(ImageToolsSettings, BaseResponsesAPIAgentConfig):
    """The image tool settings come from ``ImageToolsSettings``, which the tool reward verifier replays."""

    model_server: ModelServerRef
    # Used by /run only; under an Environment Server the image tools reward server routes verification.
    resource_servers_by_agent: dict[str, ResourcesServerRef]
    max_steps: int = 5
    max_output_tokens: Optional[int] = 4096
    include_stop_str_in_output: bool = True


class ImageToolsAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    image_tools_base_agent_ref: Optional[dict[str, Any]] = None


class ImageToolsAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


@dataclass
class _ImageToolsSession(AgentSessionState):
    operations: OrderedOperationLedger[NeMoGymResponse] = field(default_factory=OrderedOperationLedger)


class ImageToolsAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    base_reward: float = 0.0
    image_tools_aux_reward: float = 0.0
    image_tools_call_count: int = 0
    image_tools_error_count: int = 0
    image_tools_output_paths: list[str] = []
    image_tools_generation_image_paths: list[list[str]] = []
    image_tools_base_agent_ref: Optional[dict[str, Any]] = None


def _pil_to_data_url(path: str, fmt: str = "JPEG") -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        buf = BytesIO()
        image.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def _with_rollout_files(response: NeMoGymResponse, rollout_info: dict[str, Any], agent: str) -> NeMoGymResponse:
    """Carry the agent's name and the crop files this rollout wrote on the response, for the tool reward verifier."""
    files = {
        "agent": agent,
        "image_tools_output_paths": rollout_info["image_tools_output_paths"],
        "image_tools_generation_image_paths": rollout_info["image_tools_generation_image_paths"],
    }
    return response.model_copy(update={IMAGE_TOOLS_ROLLOUT_KEY: files})


class ImageToolsAgent(SimpleResponsesAPIAgent):
    config: ImageToolsAgentConfig
    _logic: ImageToolsGymToolLogic = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._logic = ImageToolsGymToolLogic(self.config.tool_config())

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> _ImageToolsSession:
        # The image tools run in this process, so a session takes no granted tools and no sandbox.
        if body.sandbox_access is not None:
            raise AgentSeedRejectedError("Image tools agent does not support sandbox access")
        required = [access.name for access in self.effective_tool_accesses(body) if access.required]
        if required:
            raise AgentSeedRejectedError(
                f"Image tools agent does not support required tool access: {', '.join(required)}"
            )
        return _ImageToolsSession(request=body)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        if not isinstance(state, _ImageToolsSession):
            raise TypeError("Expected image tools session state")
        await state.operations.close(timeout=_SESSION_CLOSE_TIMEOUT_SECS)
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id)

    def _resource_name(self, body: ImageToolsAgentRunRequest) -> str:
        base_agent_ref = body.image_tools_base_agent_ref
        if not base_agent_ref:
            extra = body.model_extra or {}
            base_agent_ref = extra.get("image_tools_base_agent_ref")
        if not base_agent_ref:
            raise ValueError("image_tools_base_agent_ref is required")
        base_agent_name = base_agent_ref["name"]
        if base_agent_name not in self.config.resource_servers_by_agent:
            raise ValueError(f"No resource server mapping for base agent {base_agent_name!r}")
        return self.config.resource_servers_by_agent[base_agent_name].name

    def _tool_user_message(self, observation: dict[str, Any]) -> tuple[NeMoGymEasyInputMessage, list[str]]:
        content = observation.get("content", "")
        if isinstance(content, str):
            return NeMoGymEasyInputMessage(role="user", content=content), []

        converted = []
        crop_paths = []
        for part in content:
            part_type = part.get("type")
            if part_type == "text":
                converted.append({"type": "input_text", "text": part.get("text", "")})
            elif part_type == "image":
                path = part["image"]
                crop_paths.append(path)
                converted.append(
                    {
                        "type": "input_image",
                        "image_url": _pil_to_data_url(path, self.config.crop_format.upper()),
                        "detail": "auto",
                    }
                )
        return NeMoGymEasyInputMessage(role="user", content=converted), crop_paths

    async def _call_model(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        cookies: Any,
        model_url_path: str,
    ):
        response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path=model_url_path,
            json=body,
            cookies=cookies,
        )
        await raise_for_status(response)
        return NeMoGymResponse.model_validate(await get_response_json(response)), response.cookies

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        model_url_path = self.url_path_for_request("/v1/responses", request)
        agent_session_id = self._agent_session_id_from_request(request)
        if agent_session_id is None:
            result, model_cookies, rollout_info = await self._run_image_tools_loop(
                body=body,
                row_metadata={},
                initial_cookies=request.cookies,
                model_url_path=model_url_path,
            )
            for key, value in model_cookies.items():
                response.set_cookie(key, value)
            return _with_rollout_files(result, rollout_info, self.config.name)
        state = self._require_agent_session(agent_session_id)
        if not isinstance(state, _ImageToolsSession):
            raise TypeError("Expected image tools session state")

        async def activation() -> NeMoGymResponse:
            result, _, rollout_info = await self._run_image_tools_loop(
                body=body, row_metadata={}, initial_cookies={}, model_url_path=model_url_path
            )
            return _with_rollout_files(result, rollout_info, self.config.name)

        try:
            return await state.operations.execute(index=1, request=body, operation=activation)
        except ActivationConflictError as error:
            raise HTTPException(409, str(error)) from error

    async def _run_image_tools_loop(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        row_metadata: dict[str, Any],
        initial_cookies: Any,
        model_url_path: str = "/v1/responses",
    ) -> tuple[NeMoGymResponse, Any, dict[str, Any]]:
        body = body.model_copy(deep=True)
        if body.max_output_tokens is None and self.config.max_output_tokens is not None:
            body.max_output_tokens = self.config.max_output_tokens

        episode = ImageToolsEpisode(
            self._logic, extract_image_refs(body.input), str(row_metadata.get("dataset", "unknown"))
        )

        new_outputs = []
        usage = None
        model_cookies = initial_cookies
        generation_image_paths: list[list[str]] = [[]]
        final_response: Optional[NeMoGymResponse] = None

        for _ in range(self.config.max_steps):
            base_input = (
                [NeMoGymEasyInputMessage(role="user", content=body.input)]
                if isinstance(body.input, str)
                else list(body.input)
            )
            model_body = body.model_copy(update={"input": base_input + new_outputs})
            # The request body type has no top-level `stop` field (extra="forbid"),
            # so route stop strings to vLLM via metadata.extra_body, the same way
            # include_stop_str_in_output is passed.
            extra_body = {}
            if self.config.stop_strings:
                extra_body["stop"] = self.config.stop_strings
            if self.config.include_stop_str_in_output:
                extra_body["include_stop_str_in_output"] = True
            if extra_body:
                metadata_extra = dict(model_body.metadata or {})
                current_extra = json.loads(metadata_extra.get("extra_body", "{}"))
                current_extra.update(extra_body)
                metadata_extra["extra_body"] = json.dumps(current_extra)
                model_body.metadata = metadata_extra

            model_response, model_cookies = await self._call_model(model_body, model_cookies, model_url_path)
            new_outputs.extend(model_response.output)
            usage = accumulate_response_usage(usage, model_response.usage)
            final_response = model_response

            # The step executes the image tool: it decodes images, runs PIL
            # transforms and writes crops to disk, and for http(s) image refs
            # _open_image does a blocking requests.get. Calling it directly from
            # this async loop would stall the event loop for every other
            # concurrent rollout, so hand the whole synchronous call to a thread.
            # Keeping base.py synchronous matters -- the verifiers import it too.
            observation = await asyncio.to_thread(episode.step, assistant_text(model_response.output))
            if observation is None:
                break

            user_message, crop_paths = self._tool_user_message(observation)
            new_outputs.append(user_message)
            generation_image_paths.append(crop_paths)

        if final_response is None:
            raise RuntimeError("ImageToolsAgent did not receive a model response")

        final_response = final_response.model_copy(update={"output": new_outputs, "usage": usage})
        rollout_info = {
            "image_tools_aux_reward": episode.aux_reward,
            "image_tools_call_count": int(episode.metadata.get("tool_call_count", 0)),
            "image_tools_error_count": int(episode.metadata.get("tool_error_count", 0)),
            "image_tools_output_paths": list(episode.metadata.get("crop_paths", [])),
            "image_tools_generation_image_paths": generation_image_paths,
        }
        return final_response, model_cookies, rollout_info

    async def run(self, request: Request, body: ImageToolsAgentRunRequest) -> ImageToolsAgentVerifyResponse:
        cookies = request.cookies
        resource_name = self._resource_name(body)

        seed_session_response = await self.server_client.post(
            server_name=resource_name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        resource_cookies = seed_session_response.cookies

        model_response, model_cookies, rollout_info = await self._run_image_tools_loop(
            body=body.responses_create_params,
            row_metadata=body.model_dump(exclude={"responses_create_params", "response"}),
            initial_cookies=cookies,
            model_url_path=self.url_path_for_run("/v1/responses", body),
        )

        verify_request_body = body.model_dump()
        verify_request_body["response"] = final_assistant_response(model_response).model_dump()
        verify_request = ImageToolsAgentVerifyRequest.model_validate(verify_request_body)
        verify_response = await self.server_client.post(
            server_name=resource_name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=resource_cookies,
        )
        await raise_for_status(verify_response)
        verify_response_json = await get_response_json(verify_response)

        base_reward = float(verify_response_json.get("reward", 0.0))
        aux_reward = float(rollout_info["image_tools_aux_reward"])
        verify_response_json["base_reward"] = base_reward
        verify_response_json["image_tools_aux_reward"] = aux_reward
        # Add auxiliary rewards to the base score, capped above at 1.0.
        # Positive bonuses can raise any score below the cap; negative penalties
        # still apply at every base score, including a correct answer's 1.0.
        verify_response_json["reward"] = min(1.0, base_reward + aux_reward)
        verify_response_json["image_tools_base_agent_ref"] = body.image_tools_base_agent_ref
        verify_response_json.update(rollout_info)
        verify_response_json["response"] = model_response.model_dump()
        return ImageToolsAgentVerifyResponse.model_validate(verify_response_json)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return await super().aggregate_metrics(body)


if __name__ == "__main__":
    ImageToolsAgent.run_webserver()
